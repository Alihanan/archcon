#!/usr/bin/env Rscript

# =============================================================================
# ArchCon GEO aggregation + disk-backed global RMA
# =============================================================================
#
# Assumptions
# -----------
# Run this script from:
#   thesis_code/data/GEO_DWNLD
#
# Existing inputs:
#   GEO_RMA/GSE*_raw_pm_median_common.rds
#   GEO_RMA/GSE*_rma_common.rds
#   GEO_RAW/GSE*_RAW.tar
#   gsm_to_gse_mapping.csv
#   ../common_probes.pkl
#
# The existing per-GSE RDS matrices must have:
#   rows    = samples
#   columns = common probe-set IDs
#
# This script creates one Python-friendly HDF5 store:
#
#   GEO_MATRIX_STORE/geo_expression_store.h5
#
# with three aligned matrices:
#
#   /expression/raw_original
#       Aggregated *_raw_pm_median_common.rds matrices.
#       These are original CEL PM intensities summarized by probe-set median.
#       NO RMA background correction, NO quantile normalization, NO log2.
#
#   /expression/rma_per_gse
#       Aggregated existing *_rma_common.rds matrices.
#       Each GSE was RMA-normalized separately.
#
#   /expression/rma_global
#       Leakage-safe train-reference RMA. The quantile target and probe effects
#       are fitted only on the frozen pretraining TRAIN samples. Validation and
#       test arrays are transformed one at a time with those frozen parameters.
#
# All three matrices have identical orientation/order:
#   rows    = samples (GSM)
#   columns = common probe sets
#
# Samples are ordered by GSE and then GSM, so every GSE occupies one contiguous
# row interval. gse_index.csv stores both R-style 1-based and Python-style
# 0-based half-open row ranges.
#
# Why HDF5?
# ---------
# It is a dense-array format that can be sliced directly from Python with h5py.
# The final store is written with native=TRUE so matrix dimensions are portable
# to C/Python order. No duplicate per-GSE matrix files are needed: Python can
# slice the corresponding contiguous row interval.
#
# Global RMA implementation
# -------------------------
# A conventional in-memory AffyBatch containing ~13k PrimeView arrays can
# require tens of GB of RAM. To avoid that, global RMA is decomposed into the
# same mathematical stages but executed in streaming/disk-backed form:
#
#   PASS 1:
#     CEL -> PM values -> RMA background correction
#         -> sorted values
#         -> accumulate ONE global quantile target
#
#   PASS 2:
#     CEL -> PM values -> RMA background correction
#         -> quantile-normalize each array to the global target
#         -> retain PM rows belonging to the 42,917 common probe sets
#         -> write normalized probe-level values to persistent working HDF5
#
#   PASS 3:
#     read probe-set blocks from persistent working HDF5
#         -> preprocessCore::subColSummarizeMedianpolishLog()
#         -> write final sample x probe-set global-RMA matrix
#
# The decomposition is validated on a small GSE against your already-computed
# affy::rma() result before the full global run starts.
#
# Resume behavior
# ---------------
# Progress is checkpointed at GSE/block level. Ctrl+C or a crash is safe:
# rerun the same script and completed stages are skipped/reused.
#
# IMPORTANT:
# preprocessCore previously caused pthread_create() errors on this machine.
# Use the single-threaded preprocessCore build that fixed your per-GSE RMA.
#
# =============================================================================


# =============================================================================
# Configuration
# =============================================================================

RDS_DIR <- "GEO_RMA"
RAW_ARCHIVE_DIR <- Sys.getenv(
  "ARCHCON_GEO_RAW_DIR",
  unset = "GEO_RAW"
)
COMMON_PROBES_PKL <- Sys.getenv("ARCHCON_COMMON_PROBES", "../common_probes.pkl")
STADNIUK_MAPPING_CSV <- Sys.getenv(
  "ARCHCON_GSM_MAPPING",
  "gsm_to_gse_mapping.csv"
)
FROZEN_SPLIT_CSV <- Sys.getenv(
  "ARCHCON_FROZEN_SPLIT",
  "prepared_sample_index.csv"
)

OUT_DIR <- "GEO_MATRIX_STORE"
FINAL_H5 <- file.path(OUT_DIR, "geo_expression_store.h5")

WORK_DIR <- ".GLOBAL_RMA_WORK"
PROGRESS_DIR <- file.path(WORK_DIR, "progress")
EXTRACT_DIR <- file.path(WORK_DIR, "extract")
WORK_H5 <- file.path(WORK_DIR, "train_reference_rma_probe_level.h5")

# A small already-computed GSE used to verify that the streaming decomposition
# reproduces the existing per-GSE RMA result before running globally.
VALIDATION_GSE <- "GSE100003"
VALIDATION_MAX_ABS_TOL <- 1e-5

# Number of common probe sets summarized together in PASS 3.
# Increase if RAM allows; decrease if memory pressure occurs.
SUMMARY_PROBESET_BLOCK_SIZE <- 128L

# Number of arrays normalized and written to the probe-level HDF5 at once.
# This keeps RAM moderate while avoiding millions of one-column HDF5 chunks.
PASS2_ARRAY_BATCH_SIZE <- 8L

# Final HDF5 precision. float32 is normally appropriate for neural-network and
# web-interface use and cuts storage roughly in half. The original RDS files
# remain untouched as the double-precision source data.
#
# Change both to H5T_IEEE_F64LE if you want double-precision storage.
FINAL_H5_TYPE <- "H5T_IEEE_F32LE"
WORK_H5_TYPE <- "H5T_IEEE_F32LE"

# Compression level for HDF5 matrices (0..9). Higher is smaller/slower.
H5_COMPRESSION_LEVEL <- 4L

# Layout version 2 fixes creation of Python-oriented matrices with rhdf5 2.38.x.
# In that release h5createDataset(native=TRUE) still creates its dataspace with
# R-oriented dimensions.  We therefore create the final datasets through the
# low-level native API and verify their physical dimensions before every write.
FINAL_H5_LAYOUT_VERSION <- 2L

# Delete the large intermediate probe-level HDF5 after global RMA and
# validation have finished successfully.
DELETE_PROBE_LEVEL_SCRATCH_AFTER_SUCCESS <- identical(
  tolower(Sys.getenv("ARCHCON_DELETE_PROBE_SCRATCH", "false")),
  "true"
)

# Set FALSE if you want to create only the aggregated original/per-GSE-RMA
# matrices and metadata now, leaving global RMA for a later run.
RUN_GLOBAL_RMA <- TRUE


# =============================================================================
# Packages
# =============================================================================

bioc_packages <- c(
  "affy",
  "affxparser",
  "Biobase",
  "preprocessCore",
  "primeviewcdf",
  "rhdf5"
)

cran_packages <- c("R.utils")

missing_bioc <- bioc_packages[
  !vapply(bioc_packages, requireNamespace, quietly = TRUE, FUN.VALUE = logical(1))
]

missing_cran <- cran_packages[
  !vapply(cran_packages, requireNamespace, quietly = TRUE, FUN.VALUE = logical(1))
]

if (length(missing_bioc) > 0L || length(missing_cran) > 0L) {
  stop(
    paste0(
      "Missing packages.\n\n",
      if (length(missing_bioc) > 0L) {
        paste0(
          "Bioconductor: ",
          paste(missing_bioc, collapse = ", "),
          "\n"
        )
      } else {
        ""
      },
      if (length(missing_cran) > 0L) {
        paste0(
          "CRAN: ",
          paste(missing_cran, collapse = ", "),
          "\n"
        )
      } else {
        ""
      },
      "\nSuggested installation:\n",
      "if (!requireNamespace(\"BiocManager\", quietly = TRUE)) ",
      "install.packages(\"BiocManager\")\n",
      "BiocManager::install(c(",
      paste(sprintf("\"%s\"", bioc_packages), collapse = ", "),
      "), update = FALSE)\n",
      "install.packages(\"R.utils\")\n\n",
      "If preprocessCore again produces pthread_create() errors, rebuild it ",
      "with --disable-threading as done previously."
    ),
    call. = FALSE
  )
}


# =============================================================================
# Basic path checks
# =============================================================================

required_paths <- c(
  RDS_DIR,
  RAW_ARCHIVE_DIR,
  COMMON_PROBES_PKL,
  STADNIUK_MAPPING_CSV
)

missing_paths <- required_paths[!file.exists(required_paths) & !dir.exists(required_paths)]

if (length(missing_paths) > 0L) {
  stop(
    "Missing required path(s): ",
    paste(missing_paths, collapse = ", "),
    call. = FALSE
  )
}

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(WORK_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(PROGRESS_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(EXTRACT_DIR, recursive = TRUE, showWarnings = FALSE)


# =============================================================================
# Generic helpers
# =============================================================================

timestamp <- function() {
  format(Sys.time(), "%Y-%m-%d %H:%M:%S %z")
}

natural_numeric_id <- function(x, prefix) {
  as.numeric(sub(paste0("^", prefix), "", x, ignore.case = TRUE))
}

extract_gsm <- function(x) {
  x <- basename(as.character(x))
  hit <- regexpr("GSM[0-9]+", x, ignore.case = TRUE, perl = TRUE)

  out <- rep(NA_character_, length(x))
  ok <- hit > 0L

  out[ok] <- toupper(
    regmatches(x[ok], regexpr("GSM[0-9]+", x[ok], ignore.case = TRUE, perl = TRUE))
  )

  out
}

gse_from_filename <- function(path) {
  hit <- regexpr("GSE[0-9]+", basename(path), ignore.case = TRUE, perl = TRUE)

  if (hit < 0L) {
    return(NA_character_)
  }

  toupper(regmatches(basename(path), hit))
}

append_unique_line <- function(path, value) {
  existing <- if (file.exists(path)) readLines(path, warn = FALSE) else character()

  if (!(value %in% existing)) {
    writeLines(c(existing, value), path)
  }
}

read_done_lines <- function(path) {
  if (!file.exists(path)) {
    return(character())
  }

  unique(readLines(path, warn = FALSE))
}

atomic_save_rds <- function(object, path, compress = TRUE) {
  tmp <- paste0(path, ".part")
  unlink(tmp, force = TRUE)

  saveRDS(object, tmp, compress = compress)

  if (file.exists(path)) {
    unlink(path, force = TRUE)
  }

  if (!file.rename(tmp, path)) {
    stop("Could not atomically rename ", tmp, " -> ", path, call. = FALSE)
  }
}

atomic_write_csv <- function(x, path) {
  tmp <- paste0(path, ".part")
  unlink(tmp, force = TRUE)

  utils::write.csv(
    x,
    tmp,
    row.names = FALSE,
    na = ""
  )

  if (file.exists(path)) {
    unlink(path, force = TRUE)
  }

  if (!file.rename(tmp, path)) {
    stop("Could not atomically rename ", tmp, " -> ", path, call. = FALSE)
  }
}

h5_native_dataset_dims <- function(file, dataset) {
  fid <- rhdf5::H5Fopen(file, flags = "H5F_ACC_RDONLY", native = TRUE)
  on.exit(rhdf5::H5Fclose(fid), add = TRUE)

  did <- rhdf5::H5Dopen(fid, dataset)
  on.exit(rhdf5::H5Dclose(did), add = TRUE)

  sid <- rhdf5::H5Dget_space(did)
  on.exit(rhdf5::H5Sclose(sid), add = TRUE)

  as.integer(rhdf5::H5Sget_simple_extent_dims(sid)$size)
}

create_native_matrix_dataset <- function(
  file,
  dataset,
  dims,
  chunk,
  h5_type,
  compression_level,
  fill_value = NaN
) {
  dims <- as.integer(dims)
  chunk <- as.integer(pmin(chunk, dims))

  fid <- rhdf5::H5Fopen(file, flags = "H5F_ACC_RDWR", native = TRUE)
  on.exit(rhdf5::H5Fclose(fid), add = TRUE)

  # native=TRUE here is essential: it prevents the samples/probes dimensions
  # from being reversed in the physical HDF5 dataspace.
  sid <- rhdf5::H5Screate_simple(dims, maxdims = dims, native = TRUE)
  on.exit(rhdf5::H5Sclose(sid), add = TRUE)

  dcpl <- rhdf5::H5Pcreate("H5P_DATASET_CREATE", native = TRUE)
  on.exit(rhdf5::H5Pclose(dcpl), add = TRUE)

  rhdf5::H5Pset_chunk(dcpl, chunk)
  rhdf5::H5Pset_fill_time(dcpl, "H5D_FILL_TIME_ALLOC")
  rhdf5::H5Pset_fill_value(dcpl, fill_value)
  rhdf5::H5Pset_obj_track_times(dcpl, FALSE)

  if (compression_level > 0L) {
    rhdf5::H5Pset_shuffle(dcpl)
    rhdf5::H5Pset_deflate(dcpl, compression_level)
  }

  did <- rhdf5::H5Dcreate(
    fid,
    dataset,
    h5_type,
    sid,
    dcpl = dcpl
  )

  if (!methods::is(did, "H5IdComponent")) {
    stop("Could not create HDF5 dataset: ", dataset, call. = FALSE)
  }

  verify_sid <- rhdf5::H5Dget_space(did)
  actual <- as.integer(rhdf5::H5Sget_simple_extent_dims(verify_sid)$size)
  rhdf5::H5Sclose(verify_sid)
  rhdf5::H5Dclose(did)

  if (!identical(actual, dims)) {
    stop(
      "HDF5 dataset ", dataset, " has physical dimensions ",
      paste(actual, collapse = " x "), "; expected ",
      paste(dims, collapse = " x "), ".",
      call. = FALSE
    )
  }

  invisible(TRUE)
}

h5_write_native_block_checked <- function(value, file, dataset, start) {
  if (!is.matrix(value)) {
    value <- as.matrix(value)
  }

  start <- as.integer(start)
  count <- as.integer(dim(value))
  dataset_dims <- h5_native_dataset_dims(file, dataset)

  if (
    length(start) != 2L ||
    length(count) != 2L ||
    any(start < 1L) ||
    any(start + count - 1L > dataset_dims)
  ) {
    stop(
      "Refusing out-of-bounds HDF5 write to ", dataset,
      ": start=", paste(start, collapse = ","),
      ", count=", paste(count, collapse = ","),
      ", dataset=", paste(dataset_dims, collapse = " x "),
      call. = FALSE
    )
  }

  rhdf5::h5write(
    value,
    file,
    dataset,
    start = start,
    count = count,
    native = TRUE
  )

  # rhdf5 2.38.x can print an HDF5 write error without propagating it to R.
  # Read deterministic sentinels back before allowing a completion marker.
  check_rows <- unique(as.integer(round(seq(
    1L,
    nrow(value),
    length.out = min(5L, nrow(value))
  ))))
  check_cols <- unique(as.integer(round(seq(
    1L,
    ncol(value),
    length.out = min(7L, ncol(value))
  ))))

  observed <- rhdf5::h5read(
    file,
    dataset,
    index = list(
      start[[1L]] - 1L + check_rows,
      start[[2L]] - 1L + check_cols
    ),
    drop = FALSE,
    native = TRUE
  )
  expected <- value[check_rows, check_cols, drop = FALSE]

  if (!identical(dim(observed), dim(expected))) {
    stop("HDF5 read-back returned the wrong dimensions for ", dataset, call. = FALSE)
  }

  error <- max(abs(as.numeric(observed) - as.numeric(expected)))
  tolerance <- 5e-6 * max(1, max(abs(as.numeric(expected))))

  if (!is.finite(error) || error > tolerance) {
    stop(
      "HDF5 write verification failed for ", dataset,
      ": max sentinel error=", format(error, digits = 8L),
      ", tolerance=", format(tolerance, digits = 8L),
      call. = FALSE
    )
  }

  invisible(TRUE)
}

read_common_probes_pickle <- function(path) {
  python <- Sys.which("python3")

  if (!nzchar(python)) {
    stop(
      "python3 is required to read common_probes.pkl.",
      call. = FALSE
    )
  }

  py <- tempfile(fileext = ".py")
  txt <- tempfile(fileext = ".txt")

  on.exit(
    unlink(c(py, txt), force = TRUE),
    add = TRUE
  )

  code <- c(
    "import pickle",
    "import sys",
    "",
    "src, dst = sys.argv[1], sys.argv[2]",
    "with open(src, 'rb') as fh:",
    "    obj = pickle.load(fh)",
    "",
    "if hasattr(obj, 'tolist'):",
    "    obj = obj.tolist()",
    "if isinstance(obj, set):",
    "    obj = list(obj)",
    "if not isinstance(obj, (list, tuple)):",
    "    raise TypeError(f'Unsupported common-probes object: {type(obj).__name__}')",
    "",
    "vals = [str(x) for x in obj]",
    "with open(dst, 'w', encoding='utf-8') as fh:",
    "    for x in vals:",
    "        fh.write(x + '\\n')"
  )

  writeLines(code, py)

  output <- system2(
    python,
    args = c(shQuote(py), shQuote(path), shQuote(txt)),
    stdout = TRUE,
    stderr = TRUE
  )

  status <- attr(output, "status")
  if (is.null(status)) {
    status <- 0L
  }

  if (status != 0L) {
    stop(
      "Failed to read common_probes.pkl:\n",
      paste(output, collapse = "\n"),
      call. = FALSE
    )
  }

  probes <- readLines(txt, warn = FALSE)
  probes <- probes[nzchar(probes)]

  if (length(probes) == 0L) {
    stop("common_probes.pkl is empty.", call. = FALSE)
  }

  if (anyDuplicated(probes)) {
    stop("common_probes.pkl contains duplicate probe-set IDs.", call. = FALSE)
  }

  probes
}

clean_sample_rownames <- function(mat, gse, source_name) {
  if (is.null(rownames(mat))) {
    stop(source_name, " has no sample row names.", call. = FALSE)
  }

  gsm <- extract_gsm(rownames(mat))

  if (anyNA(gsm)) {
    bad <- rownames(mat)[is.na(gsm)]
    stop(
      source_name,
      " contains row names from which GSM could not be extracted. First few: ",
      paste(head(bad, 10L), collapse = ", "),
      call. = FALSE
    )
  }

  if (anyDuplicated(gsm)) {
    dup <- unique(gsm[duplicated(gsm)])
    stop(
      source_name,
      " contains duplicate GSMs. First few: ",
      paste(head(dup, 10L), collapse = ", "),
      call. = FALSE
    )
  }

  data.frame(
    GSM = gsm,
    GSE = rep(gse, length(gsm)),
    source_sample_name = rownames(mat),
    stringsAsFactors = FALSE
  )
}

validate_probe_columns <- function(mat, common_probes, source_name) {
  if (ncol(mat) != length(common_probes)) {
    stop(
      source_name,
      " has ",
      ncol(mat),
      " columns; expected ",
      length(common_probes),
      ".",
      call. = FALSE
    )
  }

  if (is.null(colnames(mat))) {
    stop(source_name, " has no probe-set column names.", call. = FALSE)
  }

  if (!identical(colnames(mat), common_probes)) {
    stop(
      source_name,
      " probe columns are not identical to common_probes.pkl in the same order.",
      call. = FALSE
    )
  }
}


# =============================================================================
# Load common probes + Stadniuk mapping
# =============================================================================

message("[", timestamp(), "] Reading common probes...")
common_probes <- read_common_probes_pickle(COMMON_PROBES_PKL)

message(
  "[",
  timestamp(),
  "] common probes: ",
  format(length(common_probes), big.mark = ",")
)

stadniuk_mapping <- utils::read.csv(
  STADNIUK_MAPPING_CSV,
  stringsAsFactors = FALSE,
  check.names = FALSE
)

if (!all(c("GSM", "GSE") %in% names(stadniuk_mapping))) {
  stop(
    STADNIUK_MAPPING_CSV,
    " must contain columns GSM and GSE.",
    call. = FALSE
  )
}

stadniuk_mapping$GSM <- toupper(trimws(stadniuk_mapping$GSM))
stadniuk_mapping$GSE <- toupper(trimws(stadniuk_mapping$GSE))

if (anyDuplicated(stadniuk_mapping$GSM)) {
  dup <- unique(stadniuk_mapping$GSM[duplicated(stadniuk_mapping$GSM)])
  stop(
    "Stadniuk mapping contains duplicate GSMs. First few: ",
    paste(head(dup, 10L), collapse = ", "),
    call. = FALSE
  )
}

message(
  "[",
  timestamp(),
  "] Stadniuk mapping: ",
  format(nrow(stadniuk_mapping), big.mark = ","),
  " GSMs across ",
  length(unique(stadniuk_mapping$GSE)),
  " GSEs."
)


# =============================================================================
# Discover paired per-GSE RDS outputs
# =============================================================================

raw_files <- list.files(
  RDS_DIR,
  pattern = "^GSE[0-9]+_raw_pm_median_common\\.rds$",
  full.names = TRUE,
  ignore.case = TRUE
)

per_gse_rma_files <- list.files(
  RDS_DIR,
  pattern = "^GSE[0-9]+_rma_common\\.rds$",
  full.names = TRUE,
  ignore.case = TRUE
)

raw_by_gse <- setNames(raw_files, vapply(raw_files, gse_from_filename, character(1)))
rma_by_gse <- setNames(
  per_gse_rma_files,
  vapply(per_gse_rma_files, gse_from_filename, character(1))
)

paired_gses <- intersect(names(raw_by_gse), names(rma_by_gse))
missing_raw <- setdiff(names(rma_by_gse), names(raw_by_gse))
missing_rma <- setdiff(names(raw_by_gse), names(rma_by_gse))

if (length(missing_raw) > 0L || length(missing_rma) > 0L) {
  stop(
    "Per-GSE RDS outputs are not fully paired.\n",
    "RMA without raw: ",
    paste(missing_raw, collapse = ", "),
    "\nRaw without RMA: ",
    paste(missing_rma, collapse = ", "),
    call. = FALSE
  )
}

paired_gses <- paired_gses[
  order(
    natural_numeric_id(paired_gses, "GSE"),
    paired_gses
  )
]

if (length(paired_gses) == 0L) {
  stop("No paired GEO_RMA RDS matrices found.", call. = FALSE)
}

message(
  "[",
  timestamp(),
  "] Found ",
  length(paired_gses),
  " GSEs with both raw-original and per-GSE-RMA matrices."
)


# =============================================================================
# Build canonical sample registry and validate all RDS pairs
# =============================================================================

registry_path <- file.path(OUT_DIR, "sample_index.csv")
gse_index_path <- file.path(OUT_DIR, "gse_index.csv")
probe_index_path <- file.path(OUT_DIR, "probe_index.csv")
stadniuk_copy_path <- file.path(OUT_DIR, "stadniuk_gsm_to_gse_mapping.csv")

registry_parts <- vector("list", length(paired_gses))

message("[", timestamp(), "] Validating per-GSE RDS matrices...")

for (i in seq_along(paired_gses)) {
  gse <- paired_gses[[i]]

  raw_path <- raw_by_gse[[gse]]
  rma_path <- rma_by_gse[[gse]]

  raw_mat <- readRDS(raw_path)
  rma_mat <- readRDS(rma_path)

  if (!is.matrix(raw_mat)) {
    raw_mat <- as.matrix(raw_mat)
  }

  if (!is.matrix(rma_mat)) {
    rma_mat <- as.matrix(rma_mat)
  }

  validate_probe_columns(raw_mat, common_probes, raw_path)
  validate_probe_columns(rma_mat, common_probes, rma_path)

  raw_samples <- clean_sample_rownames(raw_mat, gse, raw_path)
  rma_samples <- clean_sample_rownames(rma_mat, gse, rma_path)

  if (!setequal(raw_samples$GSM, rma_samples$GSM)) {
    stop(
      gse,
      ": raw and per-GSE-RMA sample sets differ.",
      call. = FALSE
    )
  }

  raw_samples <- raw_samples[
    order(natural_numeric_id(raw_samples$GSM, "GSM"), raw_samples$GSM),
    ,
    drop = FALSE
  ]

  mapping_pos <- match(raw_samples$GSM, stadniuk_mapping$GSM)
  mapped_gse <- stadniuk_mapping$GSE[mapping_pos]

  mismatch <- !is.na(mapped_gse) & mapped_gse != gse

  if (any(mismatch)) {
    stop(
      gse,
      ": ",
      sum(mismatch),
      " GSM(s) disagree with Stadniuk's GSM->GSE mapping. First few: ",
      paste(
        paste0(
          raw_samples$GSM[mismatch][seq_len(min(10L, sum(mismatch)))],
          " mapped to ",
          mapped_gse[mismatch][seq_len(min(10L, sum(mismatch)))]
        ),
        collapse = ", "
      ),
      call. = FALSE
    )
  }

  raw_samples$in_stadniuk_mapping <- !is.na(mapping_pos)
  raw_samples$stadniuk_gse <- mapped_gse

  registry_parts[[i]] <- raw_samples

  rm(raw_mat, rma_mat, raw_samples, rma_samples)
  gc()

  if (i %% 25L == 0L || i == length(paired_gses)) {
    message(
      "[",
      timestamp(),
      "] validated ",
      i,
      "/",
      length(paired_gses),
      " GSEs."
    )
  }
}

sample_index <- do.call(rbind, registry_parts)
rownames(sample_index) <- NULL

sample_index <- sample_index[
  order(
    natural_numeric_id(sample_index$GSE, "GSE"),
    natural_numeric_id(sample_index$GSM, "GSM"),
    sample_index$GSM
  ),
  ,
  drop = FALSE
]

if (anyDuplicated(sample_index$GSM)) {
  dup <- unique(sample_index$GSM[duplicated(sample_index$GSM)])
  stop(
    "The combined processed collection contains duplicate GSMs across GSEs. First few: ",
    paste(head(dup, 10L), collapse = ", "),
    call. = FALSE
  )
}

# Freeze the exact GEO train/validation/test membership used by the sweep.
# The file is copied from SWEEP_ROOT/prepared/sample_index.csv by the PBS
# wrapper.  Only GEO rows participate here; supervised/IKEM rows are ignored.
if (!file.exists(FROZEN_SPLIT_CSV)) {
  stop(
    "Frozen pretraining split does not exist: ",
    FROZEN_SPLIT_CSV,
    call. = FALSE
  )
}

frozen_split <- utils::read.csv(
  FROZEN_SPLIT_CSV,
  stringsAsFactors = FALSE,
  check.names = FALSE
)

if (!"split" %in% names(frozen_split)) {
  stop("Frozen split must contain a 'split' column.", call. = FALSE)
}

if ("source_kind" %in% names(frozen_split)) {
  frozen_split <- frozen_split[
    tolower(trimws(frozen_split$source_kind)) == "geo",
    ,
    drop = FALSE
  ]
}

split_id_column <- intersect(
  c("GSM", "sample_id", "Sample_ID", "sample", "id"),
  names(frozen_split)
)

if (length(split_id_column) == 0L) {
  stop(
    "Frozen split has no GSM/sample_id column for matching GEO arrays.",
    call. = FALSE
  )
}

split_ids <- toupper(trimws(as.character(frozen_split[[split_id_column[[1L]]]])))
split_labels <- tolower(trimws(as.character(frozen_split$split)))

if (anyDuplicated(split_ids)) {
  stop("Frozen GEO split contains duplicate sample identifiers.", call. = FALSE)
}

split_pos <- match(sample_index$GSM, split_ids)
if (anyNA(split_pos)) {
  stop(
    sum(is.na(split_pos)),
    " processed GEO samples are absent from the frozen split. First few: ",
    paste(head(sample_index$GSM[is.na(split_pos)], 10L), collapse = ", "),
    call. = FALSE
  )
}

sample_index$pretraining_split <- split_labels[split_pos]
unexpected_splits <- setdiff(
  unique(sample_index$pretraining_split),
  c("train", "validation", "test")
)

if (length(unexpected_splits) > 0L) {
  stop(
    "Unexpected frozen split labels: ",
    paste(unexpected_splits, collapse = ", "),
    call. = FALSE
  )
}

train_rows_r <- which(sample_index$pretraining_split == "train")
if (length(train_rows_r) == 0L) {
  stop("Frozen split contains no GEO training samples.", call. = FALSE)
}

message(
  "[",
  timestamp(),
  "] frozen GEO split: train=",
  sum(sample_index$pretraining_split == "train"),
  ", validation=",
  sum(sample_index$pretraining_split == "validation"),
  ", test=",
  sum(sample_index$pretraining_split == "test"),
  "."
)

sample_index$global_row_r <- seq_len(nrow(sample_index))
sample_index$global_row_python <- sample_index$global_row_r - 1L

# GSE row ranges are contiguous because sample_index is sorted by GSE.
gse_split <- split(sample_index$global_row_r, sample_index$GSE)

gse_index <- do.call(
  rbind,
  lapply(names(gse_split), function(gse) {
    rows <- gse_split[[gse]]

    data.frame(
      GSE = gse,
      n_samples = length(rows),
      start_row_r = min(rows),
      end_row_r = max(rows),
      start_row_python = min(rows) - 1L,
      stop_row_python = max(rows),  # half-open Python slice
      stringsAsFactors = FALSE
    )
  })
)

gse_index <- gse_index[
  order(natural_numeric_id(gse_index$GSE, "GSE")),
  ,
  drop = FALSE
]

probe_index <- data.frame(
  probe_index_r = seq_along(common_probes),
  probe_index_python = seq_along(common_probes) - 1L,
  probe_id = common_probes,
  stringsAsFactors = FALSE
)

atomic_write_csv(sample_index, registry_path)
atomic_write_csv(gse_index, gse_index_path)
atomic_write_csv(probe_index, probe_index_path)
atomic_write_csv(stadniuk_mapping, stadniuk_copy_path)

message(
  "[",
  timestamp(),
  "] canonical collection: ",
  format(nrow(sample_index), big.mark = ","),
  " samples x ",
  format(length(common_probes), big.mark = ","),
  " probes, ",
  nrow(gse_index),
  " GSEs."
)

message(
  "[",
  timestamp(),
  "] ",
  sum(sample_index$in_stadniuk_mapping),
  " samples are present in Stadniuk's mapping; ",
  sum(!sample_index$in_stadniuk_mapping),
  " are additional/recovered samples."
)


# =============================================================================
# HDF5 final-store helpers
# =============================================================================

final_dims <- c(nrow(sample_index), length(common_probes))
final_chunk <- c(min(32L, final_dims[[1L]]), min(1024L, final_dims[[2L]]))
final_layout_path <- file.path(OUT_DIR, "hdf5_layout_version.txt")

ensure_final_h5 <- function() {
  created_datasets <- character()

  if (!file.exists(FINAL_H5)) {
    rhdf5::h5createFile(FINAL_H5)
  }

  objects <- rhdf5::h5ls(FINAL_H5, recursive = TRUE)

  if (!any(objects$group == "/" & objects$name == "expression")) {
    rhdf5::h5createGroup(FINAL_H5, "expression")
  }

  if (!any(objects$group == "/" & objects$name == "metadata")) {
    rhdf5::h5createGroup(FINAL_H5, "metadata")
  }

  objects <- rhdf5::h5ls(FINAL_H5, recursive = TRUE)

  create_matrix_if_missing <- function(name) {
    full_name <- paste0("/expression/", name)
    path_name <- paste0("expression/", name)

    exists <- any(
      paste0(objects$group, "/", objects$name) == full_name
    )

    if (!exists) {
      message(
        "[",
        timestamp(),
        "] creating HDF5 dataset ",
        path_name,
        " dims=",
        paste(final_dims, collapse = " x ")
      )

      create_native_matrix_dataset(
        FINAL_H5,
        path_name,
        dims = final_dims,
        chunk = final_chunk,
        h5_type = FINAL_H5_TYPE,
        compression_level = H5_COMPRESSION_LEVEL,
        fill_value = NaN
      )

      created_datasets <<- c(created_datasets, name)
    } else {
      actual <- h5_native_dataset_dims(FINAL_H5, path_name)

      if (!identical(actual, as.integer(final_dims))) {
        stop(
          "Existing HDF5 dataset ", path_name,
          " has physical dimensions ", paste(actual, collapse = " x "),
          "; expected ", paste(final_dims, collapse = " x "), ".\n",
          "This is the incompatible pre-layout-v2 store. Move ", FINAL_H5,
          " and the aggregation/pass3 progress files aside, then rerun.",
          call. = FALSE
        )
      }
    }
  }

  create_matrix_if_missing("raw_original")
  create_matrix_if_missing("rma_per_gse")
  create_matrix_if_missing("rma_global")

  # Metadata are deliberately also written into the HDF5 file so Python can
  # inspect the store without separately opening CSV files.
  objects <- rhdf5::h5ls(FINAL_H5, recursive = TRUE)

  write_meta_if_missing <- function(name, value) {
    full_name <- paste0("/metadata/", name)

    exists <- any(
      paste0(objects$group, "/", objects$name) == full_name
    )

    if (!exists) {
      rhdf5::h5write(
        value,
        FINAL_H5,
        paste0("metadata/", name)
      )
    }
  }

  write_meta_if_missing("GSM", sample_index$GSM)
  write_meta_if_missing("GSE", sample_index$GSE)
  write_meta_if_missing("probe_id", common_probes)
  write_meta_if_missing("global_row_python", sample_index$global_row_python)
  write_meta_if_missing("pretraining_split", sample_index$pretraining_split)
  write_meta_if_missing("hdf5_layout_version", FINAL_H5_LAYOUT_VERSION)

  writeLines(as.character(FINAL_H5_LAYOUT_VERSION), final_layout_path)

  invisible(created_datasets)
}

created_final_datasets <- ensure_final_h5()

# A newly created final dataset cannot reuse completion markers referring to a
# previous HDF5 file.  Train-target and PASS2 checkpoints remain reusable.
if ("raw_original" %in% created_final_datasets) {
  unlink(file.path(PROGRESS_DIR, "aggregate_raw_done_gse.txt"), force = TRUE)
}
if ("rma_per_gse" %in% created_final_datasets) {
  unlink(file.path(PROGRESS_DIR, "aggregate_per_gse_rma_done_gse.txt"), force = TRUE)
}
if ("rma_global" %in% created_final_datasets) {
  unlink(file.path(PROGRESS_DIR, "train_reference_pass3_done_block.txt"), force = TRUE)
}


# =============================================================================
# Write existing raw-original and per-GSE RMA matrices to aligned HDF5
# =============================================================================

raw_done_path <- file.path(PROGRESS_DIR, "aggregate_raw_done_gse.txt")
rma_done_path <- file.path(PROGRESS_DIR, "aggregate_per_gse_rma_done_gse.txt")

raw_done <- read_done_lines(raw_done_path)
rma_done <- read_done_lines(rma_done_path)

message("[", timestamp(), "] Aggregating existing per-GSE matrices into HDF5...")

for (i in seq_along(paired_gses)) {
  gse <- paired_gses[[i]]
  idx <- gse_index[gse_index$GSE == gse, , drop = FALSE]

  rows <- idx$start_row_r:idx$end_row_r
  expected_gsms <- sample_index$GSM[rows]

  if (!(gse %in% raw_done)) {
    mat <- readRDS(raw_by_gse[[gse]])
    if (!is.matrix(mat)) {
      mat <- as.matrix(mat)
    }

    validate_probe_columns(mat, common_probes, raw_by_gse[[gse]])

    gsms <- extract_gsm(rownames(mat))
    order_pos <- match(expected_gsms, gsms)

    if (anyNA(order_pos)) {
      stop(gse, ": could not align raw matrix to canonical GSM order.", call. = FALSE)
    }

    mat <- mat[order_pos, , drop = FALSE]

    h5_write_native_block_checked(
      mat,
      FINAL_H5,
      "expression/raw_original",
      start = c(rows[[1L]], 1L)
    )

    append_unique_line(raw_done_path, gse)
    rm(mat)
    gc()
  }

  if (!(gse %in% rma_done)) {
    mat <- readRDS(rma_by_gse[[gse]])
    if (!is.matrix(mat)) {
      mat <- as.matrix(mat)
    }

    validate_probe_columns(mat, common_probes, rma_by_gse[[gse]])

    gsms <- extract_gsm(rownames(mat))
    order_pos <- match(expected_gsms, gsms)

    if (anyNA(order_pos)) {
      stop(gse, ": could not align RMA matrix to canonical GSM order.", call. = FALSE)
    }

    mat <- mat[order_pos, , drop = FALSE]

    h5_write_native_block_checked(
      mat,
      FINAL_H5,
      "expression/rma_per_gse",
      start = c(rows[[1L]], 1L)
    )

    append_unique_line(rma_done_path, gse)
    rm(mat)
    gc()
  }

  if (i %% 25L == 0L || i == length(paired_gses)) {
    message(
      "[",
      timestamp(),
      "] aggregated ",
      i,
      "/",
      length(paired_gses),
      " GSEs."
    )
  }
}

message(
  "[",
  timestamp(),
  "] aggregated raw-original and per-GSE-RMA matrices are ready."
)


# =============================================================================
# Global-RMA CEL manifest
# =============================================================================

cel_manifest_path <- file.path(OUT_DIR, "cel_manifest.csv")
CEL_MANIFEST_VERSION <- 2L

tar_bin <- Sys.which("tar")
if (!nzchar(tar_bin)) {
  stop("'tar' is required to validate and list GEO RAW archives.", call. = FALSE)
}

list_tar_members_checked <- function(archive, gse) {
  stderr_path <- tempfile(pattern = paste0(gse, "_tar_"), fileext = ".stderr")
  on.exit(unlink(stderr_path, force = TRUE), add = TRUE)

  members <- suppressWarnings(
    system2(
      tar_bin,
      args = c("-tf", shQuote(archive)),
      stdout = TRUE,
      stderr = stderr_path
    )
  )
  status <- attr(members, "status")
  if (is.null(status)) status <- 0L

  if (status != 0L) {
    detail <- if (file.exists(stderr_path)) {
      paste(readLines(stderr_path, warn = FALSE), collapse = "\n")
    } else {
      "tar returned no diagnostic text"
    }

    stop(
      gse, ": RAW TAR is truncated or otherwise invalid: ", archive, "\n",
      detail, "\n",
      "Move/remove this archive and its affected per-GSE checkpoints, then rerun ",
      "the rebuild so Stage 1 downloads it again.",
      call. = FALSE
    )
  }

  if (length(members) == 0L) {
    stop(gse, ": RAW TAR is valid but empty: ", archive, call. = FALSE)
  }

  members
}

build_cel_manifest <- function() {
  manifest_parts <- vector("list", nrow(gse_index))

  for (i in seq_len(nrow(gse_index))) {
    gse <- gse_index$GSE[[i]]
    archive <- file.path(RAW_ARCHIVE_DIR, paste0(gse, "_RAW.tar"))

    if (!file.exists(archive)) {
      # Normally stage 1 already downloaded it.  This path also makes a
      # resumed job self-healing when large archives were deliberately
      # not copied back to persistent storage.
      dir.create(dirname(archive), recursive = TRUE, showWarnings = FALSE)
      partial <- paste0(archive, ".part")
      url <- paste0(
        "https://www.ncbi.nlm.nih.gov/geo/download/?acc=",
        gse,
        "&format=file"
      )
      status <- system2(
        "curl",
        args = c(
          "--location", "--fail", "--retry", "8", "--retry-delay", "10",
          "--retry-connrefused", "--continue-at", "-", "--output",
          shQuote(partial), shQuote(url)
        )
      )
      if (status != 0L || !file.exists(partial) || !file.rename(partial, archive)) {
        stop("Missing/unavailable RAW archive needed for global RMA: ", archive)
      }
    }

    members <- list_tar_members_checked(archive, gse)
    cel_members <- members[
      grepl("\\.CEL(\\.gz)?$", members, ignore.case = TRUE, perl = TRUE)
    ]

    if (length(cel_members) == 0L) {
      stop(gse, ": RAW TAR contains no CEL/CEL.gz members.", call. = FALSE)
    }

    member_gsm <- extract_gsm(cel_members)

    if (anyNA(member_gsm)) {
      cel_members <- cel_members[!is.na(member_gsm)]
      member_gsm <- member_gsm[!is.na(member_gsm)]
    }

    rows <- gse_index$start_row_r[[i]]:gse_index$end_row_r[[i]]
    wanted_gsm <- sample_index$GSM[rows]

    duplicate_member_gsm <- unique(
      member_gsm[
        duplicated(member_gsm) | duplicated(member_gsm, fromLast = TRUE)
      ]
    )
    duplicate_wanted <- intersect(wanted_gsm, duplicate_member_gsm)

    if (length(duplicate_wanted) > 0L) {
      stop(
        gse,
        ": multiple CEL/CEL.gz TAR members correspond to the same wanted GSM. ",
        "First few: ",
        paste(head(duplicate_wanted, 10L), collapse = ", "),
        call. = FALSE
      )
    }

    match_pos <- match(wanted_gsm, member_gsm)

    if (anyNA(match_pos)) {
      missing <- wanted_gsm[is.na(match_pos)]
      stop(
        gse,
        ": ",
        length(missing),
        " already-processed GSM(s) cannot be found in its RAW TAR. First few: ",
        paste(head(missing, 10L), collapse = ", "),
        call. = FALSE
      )
    }

    selected <- cel_members[match_pos]

    if (anyDuplicated(selected)) {
      stop(gse, ": duplicate CEL TAR member selected.", call. = FALSE)
    }

    manifest_parts[[i]] <- data.frame(
      manifest_version = rep(CEL_MANIFEST_VERSION, length(rows)),
      global_row_r = rows,
      global_row_python = rows - 1L,
      GSM = wanted_gsm,
      GSE = rep(gse, length(rows)),
      archive = rep(archive, length(rows)),
      archive_size_bytes = rep(as.numeric(file.info(archive)$size), length(rows)),
      archive_mtime_utc = rep(
        format(file.info(archive)$mtime, tz = "UTC", usetz = TRUE),
        length(rows)
      ),
      member = selected,
      compressed = grepl("\\.gz$", selected, ignore.case = TRUE),
      stringsAsFactors = FALSE
    )

    if (i %% 25L == 0L || i == nrow(gse_index)) {
      message(
        "[",
        timestamp(),
        "] CEL manifest: ",
        i,
        "/",
        nrow(gse_index),
        " GSEs."
      )
    }
  }

  manifest <- do.call(rbind, manifest_parts)
  rownames(manifest) <- NULL

  if (!identical(manifest$GSM, sample_index$GSM)) {
    stop("CEL manifest GSM order does not match sample_index.", call. = FALSE)
  }

  manifest
}

if (file.exists(cel_manifest_path)) {
  cel_manifest <- utils::read.csv(
    cel_manifest_path,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )

  required_manifest_columns <- c(
    "manifest_version",
    "GSM",
    "GSE",
    "archive",
    "archive_size_bytes",
    "archive_mtime_utc",
    "member"
  )
  manifest_schema_ok <- all(required_manifest_columns %in% names(cel_manifest))
  manifest_files_ok <- FALSE

  if (manifest_schema_ok) {
    manifest_archives <- unique(cel_manifest[, c("archive", "archive_size_bytes")])
    current_sizes <- vapply(
      manifest_archives$archive,
      function(path) {
        if (!file.exists(path)) return(NA_real_)
        as.numeric(file.info(path)$size)
      },
      FUN.VALUE = numeric(1L)
    )
    manifest_files_ok <- all(
      is.finite(current_sizes) &
        current_sizes == as.numeric(manifest_archives$archive_size_bytes)
    )
  }

  # Version 2 records only archives whose complete TAR listing succeeded.
  if (
    !manifest_schema_ok ||
    any(as.integer(cel_manifest$manifest_version) != CEL_MANIFEST_VERSION) ||
    !manifest_files_ok ||
    nrow(cel_manifest) != nrow(sample_index) ||
    !identical(cel_manifest$GSM, sample_index$GSM) ||
    !identical(cel_manifest$GSE, sample_index$GSE)
  ) {
    message("[", timestamp(), "] Existing CEL manifest is stale; rebuilding.")
    cel_manifest <- build_cel_manifest()
    atomic_write_csv(cel_manifest, cel_manifest_path)
  }
} else {
  message("[", timestamp(), "] Building CEL manifest...")
  cel_manifest <- build_cel_manifest()
  atomic_write_csv(cel_manifest, cel_manifest_path)
}


# =============================================================================
# Temporary CEL extraction helpers
# =============================================================================

ensure_raw_archive <- function(gse, archive) {
  if (file.exists(archive) && file.info(archive)$size > 0) {
    return(invisible(archive))
  }

  dir.create(dirname(archive), recursive = TRUE, showWarnings = FALSE)
  partial <- paste0(archive, ".part")
  url <- paste0(
    "https://www.ncbi.nlm.nih.gov/geo/download/?acc=",
    gse,
    "&format=file"
  )

  message("[", timestamp(), "] downloading required RAW archive: ", gse)
  args <- c(
    "--location", "--fail", "--retry", "8", "--retry-delay", "10",
    "--retry-connrefused", "--continue-at", "-", "--output",
    shQuote(partial), shQuote(url)
  )
  status <- system2("curl", args = args)

  if (status != 0L || !file.exists(partial) || file.info(partial)$size <= 0) {
    stop("Could not download RAW archive for ", gse, call. = FALSE)
  }
  if (!file.rename(partial, archive)) {
    stop("Could not finalize RAW archive for ", gse, call. = FALSE)
  }

  invisible(archive)
}

prepare_gse_cels <- function(gse) {
  rows <- which(cel_manifest$GSE == gse)

  if (length(rows) == 0L) {
    stop("No CEL manifest rows for ", gse, call. = FALSE)
  }

  archive <- unique(cel_manifest$archive[rows])

  if (length(archive) != 1L) {
    stop(gse, ": expected exactly one RAW archive.", call. = FALSE)
  }

  ensure_raw_archive(gse, archive)

  expected_archive_size <- unique(as.numeric(cel_manifest$archive_size_bytes[rows]))
  actual_archive_size <- as.numeric(file.info(archive)$size)
  if (
    length(expected_archive_size) != 1L ||
    !is.finite(actual_archive_size) ||
    actual_archive_size != expected_archive_size
  ) {
    stop(
      gse,
      ": RAW archive changed after the validated CEL manifest was built. ",
      "Remove GEO_MATRIX_STORE/cel_manifest.csv and rerun.",
      call. = FALSE
    )
  }

  members <- cel_manifest$member[rows]
  gsms <- cel_manifest$GSM[rows]

  work <- file.path(EXTRACT_DIR, gse)
  unlink(work, recursive = TRUE, force = TRUE)
  dir.create(work, recursive = TRUE, showWarnings = FALSE)

  ok <- FALSE

  on.exit(
    {
      if (!ok) {
        unlink(work, recursive = TRUE, force = TRUE)
      }
    },
    add = TRUE
  )

  extract_status <- suppressWarnings(utils::untar(
    archive,
    files = members,
    exdir = work
  ))

  if (!is.null(extract_status) && !identical(as.integer(extract_status), 0L)) {
    stop(
      gse, ": tar extraction failed with status ", extract_status,
      ". The RAW archive must be redownloaded.",
      call. = FALSE
    )
  }

  cel_paths <- character(length(members))

  for (j in seq_along(members)) {
    extracted <- file.path(work, members[[j]])

    if (!file.exists(extracted)) {
      stop(
        gse,
        ": extracted TAR member not found: ",
        extracted,
        call. = FALSE
      )
    }

    if (grepl("\\.gz$", extracted, ignore.case = TRUE)) {
      dest <- sub("\\.gz$", "", extracted, ignore.case = TRUE)

      R.utils::gunzip(
        extracted,
        destname = dest,
        remove = TRUE,
        overwrite = TRUE
      )

      cel_paths[[j]] <- dest
    } else {
      cel_paths[[j]] <- extracted
    }
  }

  if (!all(file.exists(cel_paths))) {
    stop(gse, ": not all temporary CEL files exist.", call. = FALSE)
  }

  names(cel_paths) <- gsms
  ok <- TRUE

  list(
    gse = gse,
    work_dir = work,
    GSM = gsms,
    paths = cel_paths
  )
}

cleanup_gse_cels <- function(prepared) {
  if (!is.null(prepared$work_dir)) {
    unlink(prepared$work_dir, recursive = TRUE, force = TRUE)
  }

  invisible(gc())
}

read_one_cel_vector <- function(path) {
  x <- affxparser::readCelIntensities(path)

  if (is.matrix(x)) {
    if (ncol(x) != 1L) {
      stop(
        "Expected one CEL intensity column from ",
        path,
        "; got ",
        ncol(x),
        ".",
        call. = FALSE
      )
    }

    x <- x[, 1L]
  }

  as.numeric(x)
}

header_field <- function(header, candidates) {
  nms <- names(header)
  lower <- tolower(nms)

  for (candidate in candidates) {
    pos <- match(tolower(candidate), lower)

    if (!is.na(pos)) {
      value <- header[[pos]]

      if (length(value) > 0L) {
        return(value[[1L]])
      }
    }
  }

  NULL
}

scalar_integer_or_na <- function(x) {
  if (is.null(x) || length(x) == 0L) {
    return(NA_integer_)
  }

  suppressWarnings(as.integer(x[[1L]]))
}


# =============================================================================
# Build one PrimeView AffyBatch template to obtain CDF PM indices
# =============================================================================

template_info_path <- file.path(WORK_DIR, "template_probe_index_info.rds")
template_mapping_version <- 2L

if (file.exists(template_info_path)) {
  template_info <- readRDS(template_info_path)

  if (
    !identical(template_info$common_probes, common_probes) ||
    !identical(template_info$mapping_version, template_mapping_version)
  ) {
    message("[", timestamp(), "] Probe list or mapping logic changed; rebuilding template info.")
    unlink(template_info_path, force = TRUE)
  }
}

if (!file.exists(template_info_path)) {
  first_gse <- gse_index$GSE[[1L]]
  prepared <- prepare_gse_cels(first_gse)

  first_path <- prepared$paths[[1L]]
  first_gsm <- prepared$GSM[[1L]]

  message(
    "[",
    timestamp(),
    "] Building PrimeView CDF template from ",
    first_gsm,
    " (",
    first_gse,
    ")."
  )

  header <- affxparser::readCelHeader(first_path)
  chiptype <- as.character(
    header_field(header, c("chiptype", "chipType", "arrayType"))
  )

  if (
    length(chiptype) == 0L ||
    is.na(chiptype) ||
    !grepl("primeview", chiptype, ignore.case = TRUE)
  ) {
    stop(
      "Template CEL is not reported as PrimeView by affxparser. chiptype=",
      chiptype,
      call. = FALSE
    )
  }

  n_rows <- scalar_integer_or_na(
    header_field(header, c("rows", "nrows"))
  )
  n_cols <- scalar_integer_or_na(
    header_field(header, c("cols", "columns", "ncols"))
  )
  n_total <- scalar_integer_or_na(
    header_field(header, c("total", "ncells", "cells"))
  )

  intensity <- read_one_cel_vector(first_path)

  if (is.na(n_total) || length(n_total) == 0L) {
    n_total <- length(intensity)
  }

  if (
    is.na(n_rows) ||
    is.na(n_cols) ||
    n_rows * n_cols != length(intensity)
  ) {
    stop(
      "Could not establish consistent PrimeView CEL rows/columns from affxparser header.",
      call. = FALSE
    )
  }

  pheno <- Biobase::AnnotatedDataFrame(
    data = data.frame(
      sample_id = first_gsm,
      row.names = first_gsm,
      stringsAsFactors = FALSE
    )
  )

  template_abatch <- methods::new(
    "AffyBatch",
    exprs = matrix(
      intensity,
      ncol = 1L,
      dimnames = list(NULL, first_gsm)
    ),
    cdfName = "PrimeView",
    annotation = affy::cleancdfname("PrimeView", addcdf = FALSE),
    nrow = as.numeric(n_rows),
    ncol = as.numeric(n_cols),
    phenoData = pheno
  )

  methods::validObject(template_abatch)
  invisible(affy::getCdfInfo(template_abatch))

  pm_all_list <- affy::indexProbes(
    template_abatch,
    which = "pm"
  )

  if (is.null(names(pm_all_list))) {
    stop("indexProbes(template, 'pm') did not return named probe sets.", call. = FALSE)
  }

  empty_all <- lengths(pm_all_list) == 0L

  if (any(empty_all)) {
    pm_all_list <- pm_all_list[!empty_all]
  }

  if (anyDuplicated(names(pm_all_list))) {
    stop("PrimeView CDF returned duplicate probe-set names.", call. = FALSE)
  }

  duplicated_within_probe_set <- vapply(
    pm_all_list,
    function(cell_index) anyDuplicated(cell_index) != 0L,
    FUN.VALUE = logical(1L)
  )
  if (any(duplicated_within_probe_set)) {
    bad <- names(pm_all_list)[duplicated_within_probe_set]
    stop(
      "PrimeView CDF repeats a PM cell inside the same probe set. First few: ",
      paste(head(bad, 10L), collapse = ", "),
      call. = FALSE
    )
  }

  missing_common <- setdiff(common_probes, names(pm_all_list))

  if (length(missing_common) > 0L) {
    stop(
      length(missing_common),
      " common probe set(s) absent from PrimeView CDF. First few: ",
      paste(head(missing_common, 10L), collapse = ", "),
      call. = FALSE
    )
  }

  all_pm_counts <- as.integer(lengths(pm_all_list))
  all_pm_cell_index <- as.integer(unname(unlist(pm_all_list, use.names = FALSE)))
  repeated_memberships <- length(all_pm_cell_index) - length(unique(all_pm_cell_index))
  if (repeated_memberships > 0L) {
    message(
      "PrimeView CDF contains ", repeated_memberships,
      " repeated PM-cell membership(s) across probe sets; preserving every occurrence."
    )
  }

  common_set_positions <- match(common_probes, names(pm_all_list))
  if (anyNA(common_set_positions)) {
    stop("Could not map common probe sets into the PrimeView CDF.", call. = FALSE)
  }

  common_pm_counts <- all_pm_counts[common_set_positions]

  if (any(common_pm_counts == 0L)) {
    bad <- common_probes[common_pm_counts == 0L]
    stop(
      "Common probe sets with zero PM cells: ",
      paste(head(bad, 10L), collapse = ", "),
      call. = FALSE
    )
  }

  all_pm_ends <- cumsum(all_pm_counts)
  all_pm_starts <- all_pm_ends - all_pm_counts + 1L
  common_pm_position_list <- lapply(
    common_set_positions,
    function(set_position) {
      seq.int(
        all_pm_starts[[set_position]],
        all_pm_ends[[set_position]]
      )
    }
  )

  common_pm_positions_grouped <- as.integer(
    unname(unlist(common_pm_position_list, use.names = FALSE))
  )
  if (
    anyNA(common_pm_positions_grouped) ||
    length(common_pm_positions_grouped) != sum(common_pm_counts)
  ) {
    stop("Failed mapping common PM memberships into the all-PM vector.", call. = FALSE)
  }

  template_info <- list(
    mapping_version = template_mapping_version,
    common_probes = common_probes,
    chiptype = chiptype,
    n_rows = n_rows,
    n_cols = n_cols,
    n_total_cells = length(intensity),
    all_pm_cell_index = as.integer(all_pm_cell_index),
    n_all_pm = length(all_pm_cell_index),
    common_pm_counts = as.integer(common_pm_counts),
    common_pm_positions_grouped = as.integer(common_pm_positions_grouped),
    n_common_pm = length(common_pm_positions_grouped)
  )

  atomic_save_rds(template_info, template_info_path, compress = TRUE)

  rm(
    intensity,
    template_abatch,
    pm_all_list,
    all_pm_counts,
    all_pm_starts,
    all_pm_ends,
    common_set_positions,
    common_pm_position_list
  )

  cleanup_gse_cels(prepared)
  prepared <- NULL
  gc()
}

message(
  "[",
  timestamp(),
  "] PrimeView PM cells used for global quantile target: ",
  format(template_info$n_all_pm, big.mark = ","),
  "; common-probe PM cells stored in working HDF5: ",
  format(template_info$n_common_pm, big.mark = ","),
  "."
)

work_bytes_per_value <- if (WORK_H5_TYPE == "H5T_IEEE_F32LE") 4 else 8
final_bytes_per_value <- if (FINAL_H5_TYPE == "H5T_IEEE_F32LE") 4 else 8

work_uncompressed_gib <- (
  template_info$n_common_pm *
    nrow(sample_index) *
    work_bytes_per_value /
    1024^3
)

final_three_uncompressed_gib <- (
  3 *
    nrow(sample_index) *
    length(common_probes) *
    final_bytes_per_value /
    1024^3
)

message(
  "[",
  timestamp(),
  "] approximate uncompressed storage: probe-level working HDF5 ",
  sprintf("%.1f GiB", work_uncompressed_gib),
  "; final three-matrix HDF5 ",
  sprintf("%.1f GiB", final_three_uncompressed_gib),
  ". Actual HDF5 size depends on compression."
)


# =============================================================================
# Verify affxparser cell ordering against existing raw-PM-median RDS
# =============================================================================

cell_order_validation_path <- file.path(PROGRESS_DIR, "cell_order_validated.txt")

if (!file.exists(cell_order_validation_path)) {
  gse <- VALIDATION_GSE

  if (!(gse %in% gse_index$GSE)) {
    gse <- gse_index$GSE[[1L]]
  }

  prepared <- prepare_gse_cels(gse)

  first_gsm <- prepared$GSM[[1L]]
  first_path <- prepared$paths[[1L]]
  intensity <- read_one_cel_vector(first_path)

  raw_ref <- readRDS(raw_by_gse[[gse]])
  if (!is.matrix(raw_ref)) {
    raw_ref <- as.matrix(raw_ref)
  }

  ref_pos <- match(first_gsm, extract_gsm(rownames(raw_ref)))

  if (is.na(ref_pos)) {
    cleanup_gse_cels(prepared)
    stop(
      "Could not find validation GSM ",
      first_gsm,
      " in ",
      raw_by_gse[[gse]],
      call. = FALSE
    )
  }

  # Reconstruct ALL 42,917 raw PM medians directly from affxparser physical
  # CEL intensities using the CDF-derived PM positions.
  all_pm <- intensity[template_info$all_pm_cell_index]

  common_grouped <- all_pm[
    template_info$common_pm_positions_grouped
  ]

  starts <- cumsum(c(1L, head(template_info$common_pm_counts, -1L)))
  ends <- cumsum(template_info$common_pm_counts)

  reconstructed <- vapply(
    seq_along(common_probes),
    function(j) {
      stats::median(common_grouped[starts[[j]]:ends[[j]]])
    },
    FUN.VALUE = numeric(1)
  )

  ref <- as.numeric(raw_ref[ref_pos, common_probes])

  max_abs <- max(abs(reconstructed - ref), na.rm = TRUE)

  cleanup_gse_cels(prepared)
  rm(intensity, raw_ref, all_pm, common_grouped, reconstructed, ref)
  gc()

  message(
    "[",
    timestamp(),
    "] CEL/CDF cell-order validation max abs difference: ",
    format(max_abs, scientific = TRUE)
  )

  if (!is.finite(max_abs) || max_abs > 1e-8) {
    stop(
      "affxparser CEL cell order does not reproduce the existing raw PM-median matrix. ",
      "Max abs difference = ",
      max_abs,
      ". Global RMA was NOT started.",
      call. = FALSE
    )
  }

  writeLines(
    paste(timestamp(), gse, first_gsm, max_abs),
    cell_order_validation_path
  )
}


# =============================================================================
# Streaming-RMA validation against one existing per-GSE RMA matrix
# =============================================================================

validate_streaming_decomposition <- function(gse) {
  prepared <- prepare_gse_cels(gse)

  on.exit(
    cleanup_gse_cels(prepared),
    add = TRUE
  )

  n <- length(prepared$paths)
  m <- template_info$n_all_pm

  message(
    "[",
    timestamp(),
    "] Validating streaming RMA decomposition on ",
    gse,
    " with ",
    n,
    " sample(s)..."
  )

  bg <- matrix(NA_real_, nrow = m, ncol = n)

  for (j in seq_len(n)) {
    intensity <- read_one_cel_vector(prepared$paths[[j]])
    pm <- intensity[template_info$all_pm_cell_index]

    corrected <- preprocessCore::rma.background.correct(
      matrix(pm, ncol = 1L),
      copy = FALSE
    )

    bg[, j] <- corrected[, 1L]

    rm(intensity, pm, corrected)
  }

  # This is the global quantile target for this validation dataset.
  sorted_bg <- apply(bg, 2L, sort)
  target <- rowMeans(sorted_bg)

  common_norm <- matrix(
    NA_real_,
    nrow = template_info$n_common_pm,
    ncol = n
  )

  for (j in seq_len(n)) {
    normalized <- preprocessCore::normalize.quantiles.use.target(
      matrix(bg[, j], ncol = 1L),
      target = target,
      copy = TRUE
    )

    common_norm[, j] <- normalized[
      template_info$common_pm_positions_grouped,
      1L
    ]

    rm(normalized)
  }

  group_labels <- rep(
    seq_along(common_probes),
    times = template_info$common_pm_counts
  )

  summarized <- preprocessCore::subColSummarizeMedianpolishLog(
    common_norm,
    group_labels
  )

  if (
    nrow(summarized) != length(common_probes) ||
    ncol(summarized) != n
  ) {
    stop(
      "Streaming validation summarizer returned unexpected dimensions: ",
      paste(dim(summarized), collapse = " x "),
      call. = FALSE
    )
  }

  rownames(summarized) <- common_probes
  colnames(summarized) <- prepared$GSM

  reference <- readRDS(rma_by_gse[[gse]])
  if (!is.matrix(reference)) {
    reference <- as.matrix(reference)
  }

  ref_gsm <- extract_gsm(rownames(reference))
  row_pos <- match(prepared$GSM, ref_gsm)

  if (anyNA(row_pos)) {
    stop(
      gse,
      ": streaming-validation GSMs cannot be aligned to existing RMA RDS.",
      call. = FALSE
    )
  }

  reference <- reference[
    row_pos,
    common_probes,
    drop = FALSE
  ]

  candidate <- t(summarized)
  candidate <- candidate[
    prepared$GSM,
    common_probes,
    drop = FALSE
  ]

  delta <- candidate - reference

  result <- data.frame(
    GSE = gse,
    n_samples = n,
    max_abs = max(abs(delta), na.rm = TRUE),
    mean_abs = mean(abs(delta), na.rm = TRUE),
    rmse = sqrt(mean(delta^2, na.rm = TRUE)),
    stringsAsFactors = FALSE
  )

  rm(
    bg,
    sorted_bg,
    target,
    common_norm,
    summarized,
    reference,
    candidate,
    delta
  )

  gc()
  result
}

stream_validation_csv <- file.path(OUT_DIR, "global_rma_streaming_validation.csv")

if (!file.exists(stream_validation_csv)) {
  validation_gse <- VALIDATION_GSE

  if (!(validation_gse %in% gse_index$GSE)) {
    validation_gse <- gse_index$GSE[
      which.min(gse_index$n_samples)
    ]
  }

  validation_result <- validate_streaming_decomposition(validation_gse)
  atomic_write_csv(validation_result, stream_validation_csv)

  message(
    "[",
    timestamp(),
    "] streaming-RMA validation: max_abs=",
    format(validation_result$max_abs, scientific = TRUE),
    ", mean_abs=",
    format(validation_result$mean_abs, scientific = TRUE),
    ", RMSE=",
    format(validation_result$rmse, scientific = TRUE)
  )

  if (
    !is.finite(validation_result$max_abs) ||
    validation_result$max_abs > VALIDATION_MAX_ABS_TOL
  ) {
    stop(
      "Streaming RMA decomposition does not reproduce existing per-GSE RMA ",
      "within tolerance. max_abs=",
      validation_result$max_abs,
      ", tolerance=",
      VALIDATION_MAX_ABS_TOL,
      ". Full global RMA was NOT started.",
      call. = FALSE
    )
  }
} else {
  validation_result <- utils::read.csv(
    stream_validation_csv,
    stringsAsFactors = FALSE
  )

  if (
    nrow(validation_result) < 1L ||
    validation_result$max_abs[[1L]] > VALIDATION_MAX_ABS_TOL
  ) {
    stop(
      "Existing streaming-validation record does not satisfy current tolerance.",
      call. = FALSE
    )
  }
}


# =============================================================================
# Global RMA
# =============================================================================

if (RUN_GLOBAL_RMA) {
  message("")
  message("======================================================================")
  message("TRAIN-REFERENCE GLOBAL RMA")
  message("======================================================================")
  message(
    "Samples: ",
    format(nrow(sample_index), big.mark = ","),
    "; common probes: ",
    format(length(common_probes), big.mark = ",")
  )

  # ---------------------------------------------------------------------------
  # PASS 1: global quantile target
  # ---------------------------------------------------------------------------

  target_sum_path <- file.path(WORK_DIR, "train_reference_target_sum.rds")
  target_path <- file.path(WORK_DIR, "train_reference_quantile_target.rds")
  pass1_done_path <- file.path(PROGRESS_DIR, "train_reference_pass1_done_gse.txt")

  if (!file.exists(target_path)) {
    if (file.exists(target_sum_path)) {
      target_state <- readRDS(target_sum_path)

      if (
        length(target_state$sum) != template_info$n_all_pm ||
        target_state$n_samples < 0L
      ) {
        stop("Invalid saved global-target state.", call. = FALSE)
      }

      if (is.null(target_state$done_gses)) {
        target_state$done_gses <- read_done_lines(pass1_done_path)
      }
    } else {
      target_state <- list(
        sum = numeric(template_info$n_all_pm),
        n_samples = 0L,
        done_gses = character()
      )
    }

    for (i in seq_len(nrow(gse_index))) {
      gse <- gse_index$GSE[[i]]

      if (gse %in% target_state$done_gses) {
        next
      }

      prepared <- prepare_gse_cels(gse)

      train_local <- which(prepared$GSM %in% sample_index$GSM[train_rows_r])

      # Work on a GSE-local copy. The additive target state is committed in
      # ONE atomic RDS write only after the whole GSE succeeds. This avoids
      # double-counting if the process dies between separate checkpoint files.
      local_sum <- target_state$sum
      local_n <- target_state$n_samples

      for (j in train_local) {
        intensity <- read_one_cel_vector(prepared$paths[[j]])
        pm <- intensity[template_info$all_pm_cell_index]

        corrected <- preprocessCore::rma.background.correct(
          matrix(pm, ncol = 1L),
          copy = FALSE
        )[, 1L]

        if (any(!is.finite(corrected))) {
          cleanup_gse_cels(prepared)
          stop(
            gse,
            "/",
            prepared$GSM[[j]],
            ": non-finite RMA-background-corrected PM values.",
            call. = FALSE
          )
        }

        local_sum <- local_sum + sort(corrected)
        local_n <- local_n + 1L

        rm(intensity, pm, corrected)

        train_position <- match(j, train_local)
        if (train_position %% 25L == 0L || train_position == length(train_local)) {
          message(
            "[PASS1 ",
            gse,
            "] ",
            train_position,
            "/",
            length(train_local),
            " TRAIN arrays."
          )
        }
      }

      cleanup_gse_cels(prepared)
      rm(prepared)

      target_state$sum <- local_sum
      target_state$n_samples <- local_n
      target_state$done_gses <- c(target_state$done_gses, gse)

      atomic_save_rds(target_state, target_sum_path, compress = FALSE)

      # Human-readable mirror only; correctness depends on target_state$done_gses.
      append_unique_line(pass1_done_path, gse)

      rm(local_sum)
      gc()

      message(
        "[",
        timestamp(),
        "] PASS1 completed ",
        gse,
        " (",
        i,
        "/",
        nrow(gse_index),
        "), total arrays in target=",
        target_state$n_samples
      )
    }

    if (target_state$n_samples != length(train_rows_r)) {
      stop(
        "Train-reference target was built from ",
        target_state$n_samples,
        " arrays; expected ",
        length(train_rows_r),
        ".",
        call. = FALSE
      )
    }

    target <- target_state$sum / target_state$n_samples
    atomic_save_rds(target, target_path, compress = TRUE)

    message(
      "[",
      timestamp(),
      "] PASS1 global quantile target complete."
    )
  } else {
    target <- readRDS(target_path)

    if (length(target) != template_info$n_all_pm) {
      stop("Saved global quantile target has wrong length.", call. = FALSE)
    }

    message("[", timestamp(), "] PASS1 target already exists; reusing.")
  }

  # ---------------------------------------------------------------------------
  # PASS 2: normalize each array to the global target -> probe-level working H5
  # ---------------------------------------------------------------------------

  pass2_done_path <- file.path(PROGRESS_DIR, "train_reference_pass2_done_gse.txt")

  if (!file.exists(WORK_H5)) {
    rhdf5::h5createFile(WORK_H5)

    rhdf5::h5createDataset(
      WORK_H5,
      "normalized_common_pm",
      dims = c(template_info$n_common_pm, nrow(sample_index)),
      H5type = WORK_H5_TYPE,
      chunk = c(
        min(4096L, template_info$n_common_pm),
        min(PASS2_ARRAY_BATCH_SIZE, nrow(sample_index))
      ),
      level = H5_COMPRESSION_LEVEL,
      native = FALSE
    )
  }

  pass2_done <- read_done_lines(pass2_done_path)

  for (i in seq_len(nrow(gse_index))) {
    gse <- gse_index$GSE[[i]]

    if (gse %in% pass2_done) {
      next
    }

    prepared <- prepare_gse_cels(gse)

    gse_rows <- which(sample_index$GSE == gse)
    wanted_gsms <- sample_index$GSM[gse_rows]
    local_pos <- match(wanted_gsms, prepared$GSM)

    if (anyNA(local_pos)) {
      cleanup_gse_cels(prepared)
      stop(gse, ": temporary CEL order cannot be aligned to sample index.", call. = FALSE)
    }

    batch_starts <- seq(
      1L,
      length(gse_rows),
      by = PASS2_ARRAY_BATCH_SIZE
    )

    for (batch_start in batch_starts) {
      batch_end <- min(
        batch_start + PASS2_ARRAY_BATCH_SIZE - 1L,
        length(gse_rows)
      )

      batch_local <- batch_start:batch_end
      batch_global_rows <- gse_rows[batch_local]

      common_batch <- matrix(
        NA_real_,
        nrow = template_info$n_common_pm,
        ncol = length(batch_local)
      )

      for (k in seq_along(batch_local)) {
        j <- batch_local[[k]]
        path <- prepared$paths[[local_pos[[j]]]]

        intensity <- read_one_cel_vector(path)
        pm <- intensity[template_info$all_pm_cell_index]

        corrected <- preprocessCore::rma.background.correct(
          matrix(pm, ncol = 1L),
          copy = FALSE
        )

        normalized <- preprocessCore::normalize.quantiles.use.target(
          corrected,
          target = target,
          copy = FALSE
        )[, 1L]

        common_normalized <- normalized[
          template_info$common_pm_positions_grouped
        ]

        if (any(!is.finite(common_normalized))) {
          cleanup_gse_cels(prepared)
          stop(
            gse,
            "/",
            wanted_gsms[[j]],
            ": non-finite globally normalized PM values.",
            call. = FALSE
          )
        }

        common_batch[, k] <- common_normalized

        rm(
          intensity,
          pm,
          corrected,
          normalized,
          common_normalized
        )
      }

      rhdf5::h5write(
        common_batch,
        WORK_H5,
        "normalized_common_pm",
        index = list(
          seq_len(template_info$n_common_pm),
          batch_global_rows
        ),
        native = FALSE
      )

      rm(common_batch)
      gc()

      message(
        "[PASS2 ",
        gse,
        "] ",
        batch_end,
        "/",
        length(gse_rows),
        " arrays."
      )
    }

    cleanup_gse_cels(prepared)
    rm(prepared)
    gc()

    append_unique_line(pass2_done_path, gse)

    message(
      "[",
      timestamp(),
      "] PASS2 completed ",
      gse,
      " (",
      i,
      "/",
      nrow(gse_index),
      ")."
    )
  }

  # ---------------------------------------------------------------------------
  # PASS 3: frozen TRAIN median-polish probe effects, then independent apply
  # ---------------------------------------------------------------------------

  pass3_done_path <- file.path(PROGRESS_DIR, "train_reference_pass3_done_block.txt")
  pass3_done <- as.integer(read_done_lines(pass3_done_path))
  pass3_done <- pass3_done[!is.na(pass3_done)]

  probe_starts <- cumsum(
    c(1L, head(template_info$common_pm_counts, -1L))
  )
  probe_ends <- cumsum(template_info$common_pm_counts)

  block_starts <- seq(
    1L,
    length(common_probes),
    by = SUMMARY_PROBESET_BLOCK_SIZE
  )

  for (block_id in seq_along(block_starts)) {
    if (block_id %in% pass3_done) {
      next
    }

    first_probe <- block_starts[[block_id]]
    last_probe <- min(
      first_probe + SUMMARY_PROBESET_BLOCK_SIZE - 1L,
      length(common_probes)
    )

    probe_ids <- first_probe:last_probe

    first_pm_row <- probe_starts[[first_probe]]
    last_pm_row <- probe_ends[[last_probe]]
    pm_rows <- first_pm_row:last_pm_row

    normalized_block <- rhdf5::h5read(
      WORK_H5,
      "normalized_common_pm",
      index = list(
        pm_rows,
        seq_len(nrow(sample_index))
      ),
      native = FALSE
    )

    # Fit probe effects only on frozen pretraining TRAIN arrays.  Once those
    # effects are fixed, each array (including validation/test) is summarized
    # independently as median(log2(PM) - frozen_probe_effect).  Therefore no
    # held-out array influences another array or either fitted RMA parameter.
    summarized <- matrix(
      NA_real_,
      nrow = length(probe_ids),
      ncol = nrow(sample_index)
    )

    local_start <- 1L
    for (local_probe in seq_along(probe_ids)) {
      n_pm <- template_info$common_pm_counts[probe_ids[[local_probe]]]
      local_rows <- local_start:(local_start + n_pm - 1L)
      log_block <- log2(normalized_block[local_rows, , drop = FALSE])

      if (any(!is.finite(log_block))) {
        stop(
          "PASS3 encountered non-positive/non-finite normalized PM values in probe set ",
          common_probes[probe_ids[[local_probe]]],
          ".",
          call. = FALSE
        )
      }

      train_fit <- stats::medpolish(
        log_block[, train_rows_r, drop = FALSE],
        trace.iter = FALSE
      )
      frozen_probe_effect <- as.numeric(train_fit$row)

      summarized[local_probe, ] <- apply(
        sweep(log_block, 1L, frozen_probe_effect, FUN = "-"),
        2L,
        stats::median
      )

      # Algebraic self-check: applying the frozen effects back to training
      # arrays must reproduce the fitted training chip expressions.
      fitted_train <- as.numeric(train_fit$overall + train_fit$col)
      max_train_delta <- max(
        abs(summarized[local_probe, train_rows_r] - fitted_train)
      )
      if (!is.finite(max_train_delta) || max_train_delta > 1e-7) {
        stop(
          "Frozen median-polish self-check failed for probe set ",
          common_probes[probe_ids[[local_probe]]],
          ": max delta=",
          max_train_delta,
          call. = FALSE
        )
      }

      local_start <- local_start + n_pm
    }

    expected_dim <- c(length(probe_ids), nrow(sample_index))

    if (!identical(dim(summarized), expected_dim)) {
      stop(
        "PASS3 block ",
        block_id,
        " returned dimensions ",
        paste(dim(summarized), collapse = " x "),
        "; expected ",
        paste(expected_dim, collapse = " x "),
        ".",
        call. = FALSE
      )
    }

    # HDF5 final matrix is samples x probes.
    h5_write_native_block_checked(
      t(summarized),
      FINAL_H5,
      "expression/rma_global",
      start = c(1L, first_probe)
    )

    append_unique_line(pass3_done_path, as.character(block_id))

    rm(
      normalized_block,
      summarized
    )
    gc()

    message(
      "[",
      timestamp(),
      "] PASS3 block ",
      block_id,
      "/",
      length(block_starts),
      " completed (probe sets ",
      first_probe,
      "-",
      last_probe,
      ")."
    )
  }

  # ---------------------------------------------------------------------------
  # Final cross-checks
  # ---------------------------------------------------------------------------

  message("[", timestamp(), "] Running final HDF5 spot checks...")

  spot_sample_rows <- unique(
    pmax(
      1L,
      pmin(
        nrow(sample_index),
        c(
          1L,
          nrow(sample_index),
          round(nrow(sample_index) / 2)
        )
      )
    )
  )

  spot_probe_cols <- unique(
    pmax(
      1L,
      pmin(
        length(common_probes),
        c(
          1L,
          length(common_probes),
          round(length(common_probes) / 2)
        )
      )
    )
  )

  for (method in c("raw_original", "rma_per_gse", "rma_global")) {
    spot <- rhdf5::h5read(
      FINAL_H5,
      paste0("expression/", method),
      index = list(spot_sample_rows, spot_probe_cols),
      native = TRUE
    )

    if (any(!is.finite(spot))) {
      stop(
        "Final HDF5 spot check found non-finite values in ",
        method,
        ".",
        call. = FALSE
      )
    }
  }

  writeLines(
    paste(
      timestamp(),
      "TRAIN_REFERENCE_RMA_COMPLETE",
      "samples=",
      nrow(sample_index),
      "probes=",
      length(common_probes)
    ),
    file.path(OUT_DIR, "GLOBAL_RMA_COMPLETE.txt")
  )

  if (
    DELETE_PROBE_LEVEL_SCRATCH_AFTER_SUCCESS &&
    file.exists(WORK_H5)
  ) {
    message(
      "[",
      timestamp(),
      "] Removing large completed probe-level working HDF5: ",
      WORK_H5
    )

    unlink(WORK_H5, force = TRUE)
  }
}


# =============================================================================
# Store manifest / README
# =============================================================================

store_manifest <- data.frame(
  matrix_name = c(
    "raw_original",
    "rma_per_gse",
    "rma_global"
  ),
  hdf5_dataset = c(
    "/expression/raw_original",
    "/expression/rma_per_gse",
    "/expression/rma_global"
  ),
  rows = nrow(sample_index),
  columns = length(common_probes),
  row_unit = "GSM sample",
  column_unit = "PrimeView common probe set",
  description = c(
    paste0(
      "Aggregated original PM signal summarized by median within each probe set; ",
      "no RMA background correction, no quantile normalization, no log transform."
    ),
    paste0(
      "Aggregated per-GSE RMA log2 expression. Each GSE was independently ",
      "background corrected, quantile normalized, and median-polish summarized."
    ),
    paste0(
      "Train-reference RMA log2 expression. Background correction is per array; ",
      "the quantile target and median-polish probe effects are fitted only on ",
      "the frozen pretraining train arrays. Validation/test arrays are transformed ",
      "independently with those frozen parameters."
    )
  ),
  stringsAsFactors = FALSE
)

atomic_write_csv(
  store_manifest,
  file.path(OUT_DIR, "store_manifest.csv")
)

readme <- c(
  "ArchCon GEO expression store",
  "============================",
  "",
  paste0("Created: ", timestamp()),
  paste0("Samples: ", nrow(sample_index)),
  paste0("GSEs: ", nrow(gse_index)),
  paste0("Common probes: ", length(common_probes)),
  "",
  "Main file:",
  "  geo_expression_store.h5",
  "",
  "Expression datasets (all sample x probe):",
  "  /expression/raw_original",
  "  /expression/rma_per_gse",
  "  /expression/rma_global",
  "",
  "Metadata in HDF5:",
  "  /metadata/GSM",
  "  /metadata/GSE",
  "  /metadata/probe_id",
  "  /metadata/global_row_python",
  "  /metadata/pretraining_split",
  "",
  "Companion metadata:",
  "  sample_index.csv",
  "  gse_index.csv",
  "  probe_index.csv",
  "  stadniuk_gsm_to_gse_mapping.csv",
  "  cel_manifest.csv",
  "  store_manifest.csv",
  "",
  "Per-GSE access:",
  "  gse_index.csv contains Python half-open slices:",
  "      start_row_python : stop_row_python",
  "  so a Python h5py client can read one GSE without loading the global matrix.",
  "",
  "Example conceptual Python access (not part of this R script):",
  "  with h5py.File('geo_expression_store.h5', 'r') as h5:",
  "      X = h5['expression/rma_per_gse'][start:stop, :]",
  "",
  "Sample/GSE provenance:",
  paste0(
    "  Stadniuk mapping contains ",
    nrow(stadniuk_mapping),
    " GSM rows across ",
    length(unique(stadniuk_mapping$GSE)),
    " GSEs."
  ),
  paste0(
    "  Current processed store contains ",
    nrow(sample_index),
    " GSM rows across ",
    nrow(gse_index),
    " GSEs."
  ),
  paste0(
    "  ",
    sum(sample_index$in_stadniuk_mapping),
    " current samples occur in Stadniuk's mapping."
  ),
  paste0(
    "  ",
    sum(!sample_index$in_stadniuk_mapping),
    " current samples are additional/recovered relative to that mapping."
  )
)

writeLines(
  readme,
  file.path(OUT_DIR, "README.txt")
)

message("")
message("======================================================================")
message("DONE")
message("======================================================================")
message("Python-friendly store: ", FINAL_H5)
message("Sample index:          ", registry_path)
message("GSE row ranges:        ", gse_index_path)
message("Probe index:           ", probe_index_path)
message("Store manifest:        ", file.path(OUT_DIR, "store_manifest.csv"))
message("")
message(
  "Rerunning this script is safe: completed GSEs/blocks are checkpointed."
)
