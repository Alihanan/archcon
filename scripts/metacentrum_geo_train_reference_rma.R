#!/usr/bin/env Rscript

# =============================================================================
# ArchCon GEO matrices and leakage-free train-reference RMA
# =============================================================================
#
# Normal use
# ----------
# Run metacentrum_rebuild_geo_rma.pbs.sh. Its driver enters the persistent
# rebuild directory and supplies the frozen sweep split automatically.
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
#       are fitted on every molecular-pretraining TRAIN array: 10,522 GEO TRAIN
#       plus 24 donor-clean IKEM TRAIN biopsies. GEO validation/test, the six
#       IKEM validation biopsies, and every outcome-held-out IKEM biopsy never
#       fit them.
#
# All three matrices have identical orientation/order:
#   rows    = samples (GSM)
#   columns = common probe sets
#
# Samples are ordered by GSE and then GSM, so every GSE occupies one contiguous
# row interval. gse_index.csv stores both R-style 1-based and Python-style
# 0-based half-open row ranges.
#
# What the three passes do
# ------------------------
#   PASS 1 learns one quantile-normalization target from 10,522 GEO TRAIN plus
#          24 donor-clean IKEM TRAIN arrays.
#   PASS 2 applies that frozen target to every GEO array and also caches the
#          normalized donor-clean IKEM TRAIN arrays needed to fit probe
#          effects.
#   PASS 3 learns probe effects from the combined TRAIN columns, saves those
#          effects, and produces the final GEO sample-by-probe-set matrix.
#
# The reusable PASS 1 target and PASS 3 probe effects are exactly what Phase 4
# later applies to every IKEM array. Only the 24 frozen IKEM training rows
# participate in fitting; IKEM validation and outcome-held-out rows are
# transform-only.
#
# Why files are written in blocks
# ------------------------------
# The complete probe-level matrix is much too large for ordinary RAM. HDF5 lets
# the script read and write a small block at a time. Every finished block has a
# checkpoint, so rerunning after Ctrl+C or a wall-time kill resumes safely.
#
# PASS 3 compute modes
# --------------------
# The PBS wrapper exposes two simple environment variables:
#
#   ARCHCON_RMA_MODE=stream   Read one PASS 3 block from HDF5 at a time.
#                             This is the default and uses modest RAM.
#   ARCHCON_RMA_MODE=ram      Read the complete normalized probe-level matrix
#                             into RAM once, then reuse it for every block.
#   ARCHCON_RMA_CPUS=N        Summarize independent probe sets on N local
#                             workers. Workers receive only one small probe-set
#                             task at a time. HDF5 access and checkpoints always
#                             stay in the parent process.
#
# RAM mode accelerates the long 336-block PASS 3 calculation. PASS 1 and PASS 2
# remain disk-checkpointed so a wall-time stop never discards earlier work.
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
TARGET_RDS <- file.path(WORK_DIR, "train_reference_quantile_target.rds")
PARAMETER_H5 <- file.path(WORK_DIR, "train_reference_rma_parameters.h5")
PARAMETER_H5_PART <- paste0(PARAMETER_H5, ".part")
PARAMETER_SIGNATURE <- file.path(
  WORK_DIR,
  "train_reference_rma_parameter_signature.rds"
)
PARAMETER_COMPLETE <- file.path(
  WORK_DIR,
  "TRAIN_REFERENCE_PARAMETERS_COMPLETE.txt"
)
PARAMETER_PROGRESS <- file.path(
  PROGRESS_DIR,
  "train_reference_parameters_done_block.txt"
)

# Stage 2B prepares leakage-free IKEM-local RMA and a compact cache containing
# background-corrected PM values for the frozen no-eGFR IKEM TRAIN samples.
IKEM_PREP_SCRIPT <- Sys.getenv("ARCHCON_IKEM_SCRIPT", unset = "")
IKEM_WORK_DIR <- ".IKEM_CEL_WORK"
IKEM_GLOBAL_TRAIN_BG_H5 <- file.path(
  IKEM_WORK_DIR,
  "ikem_global_train_background_corrected_pm.h5"
)
IKEM_GLOBAL_TRAIN_CONTRIBUTION <- file.path(
  IKEM_WORK_DIR,
  "ikem_global_train_reference_contribution.rds"
)
IKEM_GLOBAL_TRAIN_READY <- file.path(
  IKEM_WORK_DIR,
  "IKEM_GLOBAL_TRAIN_REFERENCE_READY.txt"
)
IKEM_REFERENCE_CSV <- file.path(
  "IKEM_MATRIX_STORE",
  "ikem_rma_reference_samples.csv"
)
IKEM_BACKGROUND_DATASET <- "background_corrected_all_pm"

# A small already-computed GSE used to verify that the streaming decomposition
# reproduces the existing per-GSE RMA result before running globally.
VALIDATION_GSE <- "GSE100003"
VALIDATION_MAX_ABS_TOL <- 1e-5

# Number of common probe sets summarized together in PASS 3.
# Increase if RAM allows; decrease if memory pressure occurs.
SUMMARY_PROBESET_BLOCK_SIZE <- 128L

# PASS 3 can either stream blocks from HDF5 or cache the complete probe-level
# matrix in RAM. "disk" and "hdf5" are accepted as aliases for "stream".
RMA_COMPUTE_MODE <- tolower(trimws(Sys.getenv(
  "ARCHCON_RMA_MODE",
  unset = "stream"
)))
if (RMA_COMPUTE_MODE %in% c("disk", "hdf5", "streaming")) {
  RMA_COMPUTE_MODE <- "stream"
}
if (!(RMA_COMPUTE_MODE %in% c("stream", "ram"))) {
  stop(
    "ARCHCON_RMA_MODE must be 'stream' or 'ram'; received: ",
    RMA_COMPUTE_MODE,
    call. = FALSE
  )
}

RMA_WORKERS <- suppressWarnings(as.integer(Sys.getenv(
  "ARCHCON_RMA_CPUS",
  unset = "1"
)))
if (length(RMA_WORKERS) != 1L || is.na(RMA_WORKERS) || RMA_WORKERS < 1L) {
  stop("ARCHCON_RMA_CPUS must be a positive integer.", call. = FALSE)
}
# RAM is filled a few sample columns at a time. This avoids a second full-size
# temporary allocation while converting the float32 HDF5 data to R doubles.
RAM_LOAD_ARRAY_BATCH_SIZE <- suppressWarnings(as.integer(Sys.getenv(
  "ARCHCON_RMA_RAM_LOAD_BATCH",
  unset = "16"
)))
if (
  length(RAM_LOAD_ARRAY_BATCH_SIZE) != 1L ||
  is.na(RAM_LOAD_ARRAY_BATCH_SIZE) ||
  RAM_LOAD_ARRAY_BATCH_SIZE < 1L
) {
  stop("ARCHCON_RMA_RAM_LOAD_BATCH must be a positive integer.", call. = FALSE)
}

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
    tmp <- paste0(path, ".part")
    unlink(tmp, force = TRUE)
    writeLines(c(existing, value), tmp)
    if (!file.rename(tmp, path)) {
      stop("Could not save progress marker: ", path, call. = FALSE)
    }
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

atomic_write_lines <- function(x, path) {
  tmp <- paste0(path, ".part")
  unlink(tmp, force = TRUE)
  writeLines(x, tmp)

  if (file.exists(path)) {
    unlink(path, force = TRUE)
  }

  if (!file.rename(tmp, path)) {
    stop("Could not atomically rename ", tmp, " -> ", path, call. = FALSE)
  }
}

gib <- function(bytes) {
  as.double(bytes) / 1024^3
}

round_up_to <- function(value, multiple) {
  ceiling(value / multiple) * multiple
}

phase3_lapply <- function(items, fun, cluster = NULL) {
  if (length(items) == 0L) {
    return(list())
  }

  if (is.null(cluster)) {
    return(lapply(items, fun))
  }

  # Workers receive one ordinary R task at a time. They never open HDF5 files.
  # The parent receives their results and performs every write/checkpoint.
  parallel::parLapplyLB(
    cluster,
    items,
    fun
  )
}

summarize_probe_task <- function(task) {
  log_block <- log2(task$normalized_values)

  if (any(!is.finite(log_block))) {
    stop(
      "PASS 3 found a non-positive or non-finite normalized value in probe set ",
      task$probe_name, ".",
      call. = FALSE
    )
  }

  train_fit <- stats::medpolish(
    log_block[, task$train_rows, drop = FALSE],
    trace.iter = FALSE
  )
  frozen_probe_effect <- as.numeric(train_fit$row)
  sample_summary <- apply(
    sweep(log_block, 1L, frozen_probe_effect, FUN = "-"),
    2L,
    stats::median
  )

  fitted_train <- as.numeric(train_fit$overall + train_fit$col)
  max_train_delta <- max(abs(sample_summary[task$train_rows] - fitted_train))
  if (!is.finite(max_train_delta) || max_train_delta > 1e-7) {
    stop(
      "Frozen median-polish check failed for probe set ",
      task$probe_name,
      ": max delta=", max_train_delta,
      call. = FALSE
    )
  }

  list(
    local_probe = task$local_probe,
    local_rows = task$local_rows,
    probe_effect = frozen_probe_effect,
    sample_summary = as.numeric(sample_summary)
  )
}

# Keep the worker closure deliberately tiny. In RAM mode the surrounding
# script environment eventually contains a ~41 GiB matrix; a PSOCK worker must
# never serialize or receive that environment.
environment(summarize_probe_task) <- baseenv()

load_probe_matrix_into_ram <- function(path, dataset, n_pm, n_samples, batch_size) {
  payload_gib <- gib(as.double(n_pm) * as.double(n_samples) * 8.0)

  message(
    "[", timestamp(), "] RAM mode: allocating ",
    sprintf("%.2f GiB", payload_gib),
    " for the R double-precision probe-level matrix."
  )

  value <- tryCatch(
    matrix(NA_real_, nrow = n_pm, ncol = n_samples),
    error = function(e) {
      stop(
        "RAM mode could not allocate its ", sprintf("%.2f GiB", payload_gib),
        " matrix: ", conditionMessage(e),
        ". Resubmit with more memory or use ARCHCON_RMA_MODE=stream.",
        call. = FALSE
      )
    }
  )

  starts <- seq.int(1L, n_samples, by = batch_size)
  for (batch_number in seq_along(starts)) {
    first_sample <- starts[[batch_number]]
    last_sample <- min(n_samples, first_sample + batch_size - 1L)
    sample_columns <- first_sample:last_sample

    block <- rhdf5::h5read(
      path,
      dataset,
      index = list(seq_len(n_pm), sample_columns),
      drop = FALSE,
      native = FALSE
    )

    expected <- c(n_pm, length(sample_columns))
    if (!identical(dim(block), expected) || any(!is.finite(block))) {
      stop(
        "RAM loading received an invalid HDF5 block for samples ",
        first_sample, "-", last_sample, ".",
        call. = FALSE
      )
    }

    value[, sample_columns] <- block
    rm(block)

    if (
      batch_number == 1L ||
      batch_number %% 25L == 0L ||
      batch_number == length(starts)
    ) {
      message(
        "[", timestamp(), "] RAM load: samples ", last_sample, "/", n_samples,
        "."
      )
    }
  }

  rhdf5::H5close()
  gc()
  value
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

create_parameter_h5 <- function(path, n_values) {
  unlink(path, force = TRUE)
  rhdf5::h5createFile(path)
  rhdf5::h5createDataset(
    path,
    "probe_effect_common_pm",
    dims = as.integer(n_values),
    H5type = "H5T_IEEE_F64LE",
    chunk = as.integer(min(8192L, n_values)),
    level = H5_COMPRESSION_LEVEL,
    fillValue = NaN,
    native = TRUE
  )
}

read_parameter_h5 <- function(path) {
  if (!file.exists(path)) {
    stop("Frozen probe-effect file does not exist: ", path, call. = FALSE)
  }
  objects <- rhdf5::h5ls(path, recursive = TRUE)
  if (!any(objects$group == "/" & objects$name == "probe_effect_common_pm")) {
    stop(
      "Frozen probe-effect file lacks /probe_effect_common_pm: ",
      path,
      call. = FALSE
    )
  }
  as.numeric(rhdf5::h5read(path, "probe_effect_common_pm", native = TRUE))
}

validate_parameter_h5 <- function(path, expected_values) {
  values <- read_parameter_h5(path)
  if (length(values) != expected_values) {
    stop(
      "Frozen probe-effect file contains ", length(values),
      " values; expected ", expected_values, ".",
      call. = FALSE
    )
  }
  bad <- which(!is.finite(values))
  if (length(bad) > 0L) {
    stop(
      "Frozen probe-effect file is incomplete: ", length(bad),
      " value(s) are missing or non-finite. First position: ", bad[[1L]],
      ". Rerun this script to resume the backfill.",
      call. = FALSE
    )
  }
  invisible(TRUE)
}

write_parameter_block_checked <- function(path, rows, values) {
  rows <- as.integer(rows)
  values <- as.numeric(values)
  if (length(rows) != length(values) || any(!is.finite(values))) {
    stop("Refusing to write an invalid frozen probe-effect block.", call. = FALSE)
  }

  rhdf5::h5write(
    values,
    path,
    "probe_effect_common_pm",
    index = list(rows),
    native = TRUE
  )
  observed <- as.numeric(rhdf5::h5read(
    path,
    "probe_effect_common_pm",
    index = list(rows),
    native = TRUE
  ))
  error <- max(abs(observed - values))
  if (length(observed) != length(values) || !is.finite(error) || error > 1e-12) {
    stop(
      "Frozen probe-effect block failed its HDF5 read-back check; max error=",
      format(error, digits = 8L),
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
# IKEM membership is independently gated from egfr_data.xlsx.
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

# Older prepared files may not have source_kind. In that case, the GSM prefix
# is the unambiguous way to retain only GEO rows and ignore IKEM/supervised rows.
geo_id <- grepl("^GSM[0-9]+$", split_ids)
split_ids <- split_ids[geo_id]
split_labels <- split_labels[geo_id]

if (anyDuplicated(split_ids)) {
  stop("Frozen GEO split contains duplicate sample identifiers.", call. = FALSE)
}

# The processed GEO collection and the frozen sweep must describe exactly the
# same arrays. Checking both directions prevents a seemingly successful build
# from silently omitting a downloaded/processed sample.
missing_from_processed <- setdiff(split_ids, sample_index$GSM)
unexpected_processed <- setdiff(sample_index$GSM, split_ids)
if (length(missing_from_processed) > 0L || length(unexpected_processed) > 0L) {
  missing_gse <- stadniuk_mapping$GSE[
    match(missing_from_processed, stadniuk_mapping$GSM)
  ]
  missing_gse <- missing_gse[!is.na(missing_gse) & nzchar(missing_gse)]
  missing_gse_summary <- if (length(missing_gse) > 0L) {
    counts <- sort(table(missing_gse), decreasing = TRUE)
    paste(
      paste0(names(counts), "=", as.integer(counts)),
      collapse = ", "
    )
  } else {
    "not available from GSM mapping"
  }

  stop(
    "Processed GEO samples do not exactly match the frozen sweep split.\n",
    "Missing from processed matrices: ", length(missing_from_processed),
    if (length(missing_from_processed) > 0L) {
      paste0(" (first: ", paste(head(missing_from_processed, 10L), collapse = ", "), ")")
    } else {
      ""
    },
    "\nUnexpected processed samples: ", length(unexpected_processed),
    if (length(unexpected_processed) > 0L) {
      paste0(" (first: ", paste(head(unexpected_processed, 10L), collapse = ", "), ")")
    } else {
      ""
    },
    "\nAffected mapped GSEs: ", missing_gse_summary,
    "\nRepair the incomplete GSE checkpoint(s), then rerun. Phase 4 will not start.",
    call. = FALSE
  )
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

# Per-GSE RMA is safe for molecular validation only when an entire study stays
# in one frozen split. Refuse to continue if a GSE crosses a split boundary.
gse_split_count <- vapply(
  split(sample_index$pretraining_split, sample_index$GSE),
  function(labels) length(unique(labels)),
  integer(1)
)
mixed_split_gse <- names(gse_split_count)[gse_split_count != 1L]
if (length(mixed_split_gse) > 0L) {
  stop(
    "The frozen molecular split is not study-disjoint. These GSEs occur in ",
    "more than one of train/validation/test: ",
    paste(head(mixed_split_gse, 20L), collapse = ", "),
    ". Per-GSE RMA would then cross a split boundary.",
    call. = FALSE
  )
}
message(
  "[", timestamp(), "] verified study-disjoint molecular split across ",
  length(gse_split_count), " GSEs."
)

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

# Prepare IKEM before fitting the global reference. This computes the
# per-dataset IKEM RMA from the frozen no-eGFR TRAIN subset and creates the
# small TRAIN-only background-corrected PM cache used below. The child process
# uses the same R executable, package library, working directory, and split.
if (!nzchar(IKEM_PREP_SCRIPT) || !file.exists(IKEM_PREP_SCRIPT)) {
  stop(
    "ARCHCON_IKEM_SCRIPT does not point to metacentrum_ikem_cel.R: ",
    IKEM_PREP_SCRIPT,
    call. = FALSE
  )
}

message(
  "[", timestamp(),
  "] preparing IKEM no-eGFR TRAIN arrays before combined global RMA..."
)
old_ikem_mode <- Sys.getenv("ARCHCON_IKEM_MODE", unset = NA_character_)
Sys.setenv(ARCHCON_IKEM_MODE = "prepare")
ikem_prepare_status <- system2(
  file.path(R.home("bin"), "Rscript"),
  c("--vanilla", shQuote(IKEM_PREP_SCRIPT))
)
if (is.na(old_ikem_mode)) {
  Sys.unsetenv("ARCHCON_IKEM_MODE")
} else {
  Sys.setenv(ARCHCON_IKEM_MODE = old_ikem_mode)
}
if (!identical(as.integer(ikem_prepare_status), 0L)) {
  stop(
    "IKEM train/local-RMA preparation failed with status ",
    ikem_prepare_status,
    ". Stage 3 has not fitted a combined reference.",
    call. = FALSE
  )
}

required_ikem_reference <- c(
  IKEM_GLOBAL_TRAIN_BG_H5,
  IKEM_GLOBAL_TRAIN_CONTRIBUTION,
  IKEM_GLOBAL_TRAIN_READY,
  IKEM_REFERENCE_CSV
)
missing_ikem_reference <- required_ikem_reference[
  !file.exists(required_ikem_reference)
]
if (length(missing_ikem_reference) > 0L) {
  stop(
    "IKEM preparation did not create: ",
    paste(missing_ikem_reference, collapse = ", "),
    call. = FALSE
  )
}

ikem_contribution <- readRDS(IKEM_GLOBAL_TRAIN_CONTRIBUTION)
required_contribution_fields <- c(
  "format_version", "input_signature", "frozen_split_md5", "template_md5",
  "target_sum", "n_samples", "sample_ids", "GSM",
  "background_cache", "background_dataset"
)
missing_contribution_fields <- setdiff(
  required_contribution_fields,
  names(ikem_contribution)
)
if (length(missing_contribution_fields) > 0L) {
  stop(
    "IKEM global-training contribution is incomplete (missing: ",
    paste(missing_contribution_fields, collapse = ", "), ").",
    call. = FALSE
  )
}

ikem_reference <- utils::read.csv(
  IKEM_REFERENCE_CSV,
  stringsAsFactors = FALSE,
  check.names = FALSE
)
ikem_train_count <- as.integer(ikem_contribution$n_samples)
ikem_contribution_valid <- identical(
  as.integer(ikem_contribution$format_version), 1L
) && identical(
  as.character(ikem_contribution$frozen_split_md5),
  as.character(unname(tools::md5sum(FROZEN_SPLIT_CSV)))
) && identical(
  as.character(ikem_contribution$template_md5),
  as.character(unname(tools::md5sum(template_info_path)))
) && identical(
  as.character(ikem_contribution$background_cache),
  basename(IKEM_GLOBAL_TRAIN_BG_H5)
) && identical(
  as.character(ikem_contribution$background_dataset),
  IKEM_BACKGROUND_DATASET
) && length(ikem_contribution$target_sum) == template_info$n_all_pm &&
  all(is.finite(ikem_contribution$target_sum)) &&
  length(ikem_contribution$sample_ids) == ikem_train_count &&
  length(ikem_contribution$GSM) == ikem_train_count &&
  nrow(ikem_reference) == ikem_train_count &&
  identical(
    as.character(ikem_contribution$sample_ids),
    as.character(ikem_reference$sample_id)
  ) && identical(
    as.character(ikem_contribution$GSM),
    as.character(ikem_reference$GSM)
  )
if (!ikem_contribution_valid || ikem_train_count < 2L) {
  stop(
    "IKEM global-training cache does not match the current split/template.",
    call. = FALSE
  )
}

ikem_cache_spot <- rhdf5::h5read(
  IKEM_GLOBAL_TRAIN_BG_H5,
  IKEM_BACKGROUND_DATASET,
  index = list(
    unique(c(1L, template_info$n_all_pm)),
    unique(c(1L, ikem_train_count))
  ),
  native = FALSE
)
if (any(!is.finite(ikem_cache_spot)) || any(ikem_cache_spot <= 0)) {
  stop("IKEM TRAIN background-corrected PM cache failed validation.", call. = FALSE)
}

combined_work_sample_count <- nrow(sample_index) + ikem_train_count
combined_train_rows_r <- c(
  train_rows_r,
  nrow(sample_index) + seq_len(ikem_train_count)
)
message(
  "[", timestamp(), "] combined reference fit: ",
  length(train_rows_r), " GEO TRAIN + ", ikem_train_count,
  " IKEM no-eGFR TRAIN = ", length(combined_train_rows_r), " arrays."
)

# Never silently reuse a GEO-only global target/work matrix from pipeline v1.
# The compact contract makes every resume conditional on the same split,
# template, and exact ordered IKEM training rows.
GLOBAL_CONTRACT <- file.path(WORK_DIR, "combined_train_reference_contract.rds")
current_global_contract <- list(
  format_version = 3L,
  frozen_split_md5 = unname(tools::md5sum(FROZEN_SPLIT_CSV)),
  template_md5 = unname(tools::md5sum(template_info_path)),
  # The contribution file is rewritten atomically on a resumed prepare call.
  # Its semantic signature is stable even if compressed-file bytes differ.
  ikem_input_signature = as.character(ikem_contribution$input_signature),
  geo_train_ids = as.character(sample_index$GSM[train_rows_r]),
  ikem_train_sample_ids = as.character(ikem_contribution$sample_ids),
  ikem_train_gsms = as.character(ikem_contribution$GSM)
)
old_global_artifacts <- c(
  TARGET_RDS,
  file.path(WORK_DIR, "train_reference_target_sum.rds"),
  WORK_H5,
  PARAMETER_H5,
  PARAMETER_H5_PART,
  PARAMETER_SIGNATURE,
  PARAMETER_COMPLETE,
  PARAMETER_PROGRESS,
  file.path(PROGRESS_DIR, "train_reference_pass1_done_gse.txt"),
  file.path(PROGRESS_DIR, "train_reference_pass2_done_gse.txt"),
  file.path(PROGRESS_DIR, "train_reference_pass3_done_block.txt"),
  file.path(OUT_DIR, "GLOBAL_RMA_COMPLETE.txt")
)
if (file.exists(GLOBAL_CONTRACT)) {
  saved_global_contract <- readRDS(GLOBAL_CONTRACT)
  if (!identical(saved_global_contract, current_global_contract)) {
    message(
      "The ordered IKEM training set changed. Resetting only the combined ",
      "global-RMA target, work matrix, parameters, and progress. Existing ",
      "per-GSE RMA and GEO raw/per-GSE matrices remain reusable."
    )
    unlink(old_global_artifacts, force = TRUE)
    unlink(GLOBAL_CONTRACT, force = TRUE)
    atomic_save_rds(current_global_contract, GLOBAL_CONTRACT, compress = TRUE)
  }
} else {
  incompatible <- old_global_artifacts[file.exists(old_global_artifacts)]
  if (length(incompatible) > 0L) {
    message(
      "Found global-RMA outputs without a matching reference contract; ",
      "resetting only those global artifacts. Existing per-GSE RMA remains ",
      "reusable."
    )
    unlink(incompatible, force = TRUE)
  }
  atomic_save_rds(current_global_contract, GLOBAL_CONTRACT, compress = TRUE)
}

work_bytes_per_value <- if (WORK_H5_TYPE == "H5T_IEEE_F32LE") 4 else 8
final_bytes_per_value <- if (FINAL_H5_TYPE == "H5T_IEEE_F32LE") 4 else 8

work_uncompressed_gib <- (
  as.double(template_info$n_common_pm) *
    as.double(combined_work_sample_count) *
    as.double(work_bytes_per_value) /
    1024^3
)

final_three_uncompressed_gib <- (
  3.0 *
    as.double(nrow(sample_index)) *
    as.double(length(common_probes)) *
    as.double(final_bytes_per_value) /
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

ram_payload_gib <- (
  as.double(template_info$n_common_pm) *
    as.double(combined_work_sample_count) *
    8.0 /
    1024^3
)
ram_minimum_gib <- round_up_to(
  max(64, 1.35 * ram_payload_gib + 8 + 0.5 * RMA_WORKERS),
  16
)
ram_recommended_gib <- round_up_to(
  max(96, 2.0 * ram_payload_gib + 12 + 1.0 * RMA_WORKERS),
  32
)

message(
  "[", timestamp(), "] PASS 3 compute configuration: mode=", RMA_COMPUTE_MODE,
  ", workers=", RMA_WORKERS, "."
)
if (RMA_COMPUTE_MODE == "ram") {
  message(
    "[", timestamp(), "] Full RAM matrix payload: ",
    sprintf("%.2f GiB", ram_payload_gib),
    ". Approximate minimum job memory: ", ram_minimum_gib,
    " GiB; conservative request: ", ram_recommended_gib, " GiB."
  )
}


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
    "GEO output samples: ",
    format(nrow(sample_index), big.mark = ","),
    "; fit samples: ",
    format(length(combined_train_rows_r), big.mark = ","),
    " (", length(train_rows_r), " GEO + ", ikem_train_count, " IKEM)",
    "; common probes: ", format(length(common_probes), big.mark = ",")
  )

  # ---------------------------------------------------------------------------
  # PASS 1: global quantile target
  # ---------------------------------------------------------------------------

  target_sum_path <- file.path(WORK_DIR, "train_reference_target_sum.rds")
  target_path <- TARGET_RDS
  pass1_done_path <- file.path(PROGRESS_DIR, "train_reference_pass1_done_gse.txt")

  if (!file.exists(target_path)) {
    if (file.exists(target_sum_path)) {
      target_state <- readRDS(target_sum_path)

      if (
        !identical(as.integer(target_state$format_version), 2L) ||
        length(target_state$sum) != template_info$n_all_pm ||
        target_state$n_samples < 0L
      ) {
        stop("Invalid or obsolete saved combined-target state.", call. = FALSE)
      }

      if (is.null(target_state$done_gses)) {
        target_state$done_gses <- read_done_lines(pass1_done_path)
      }
    } else {
      target_state <- list(
        format_version = 2L,
        sum = numeric(template_info$n_all_pm),
        n_samples = 0L,
        done_gses = character(),
        ikem_added = FALSE,
        ikem_input_signature = as.character(ikem_contribution$input_signature)
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

    if (isTRUE(target_state$ikem_added)) {
      if (!identical(
        as.character(target_state$ikem_input_signature),
        as.character(ikem_contribution$input_signature)
      )) {
        stop("Saved target contains a different IKEM TRAIN contribution.", call. = FALSE)
      }
    } else {
      target_state$sum <- target_state$sum + ikem_contribution$target_sum
      target_state$n_samples <- target_state$n_samples + ikem_train_count
      target_state$ikem_added <- TRUE
      target_state$ikem_input_signature <- as.character(
        ikem_contribution$input_signature
      )
      atomic_save_rds(target_state, target_sum_path, compress = FALSE)
      message(
        "[", timestamp(), "] PASS1 added ", ikem_train_count,
        " IKEM no-eGFR TRAIN arrays; total arrays in target=",
        target_state$n_samples, "."
      )
    }

    if (target_state$n_samples != length(combined_train_rows_r)) {
      stop(
        "Train-reference target was built from ",
        target_state$n_samples,
        " arrays; expected ",
        length(combined_train_rows_r),
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
  pass2_done <- read_done_lines(pass2_done_path)

  # The large probe-level file may be intentionally removed after a completely
  # successful build. It is not needed merely to validate/reuse finished Phase
  # 3 and Phase 4 artifacts. If those artifacts are not complete, however, an
  # absent work file means PASS 2 must be rebuilt from the beginning.
  pass3_marker_path <- file.path(
    PROGRESS_DIR,
    "train_reference_pass3_done_block.txt"
  )
  expected_pass3_blocks <- seq_len(ceiling(
    length(common_probes) / SUMMARY_PROBESET_BLOCK_SIZE
  ))
  saved_pass3_blocks <- suppressWarnings(as.integer(
    read_done_lines(pass3_marker_path)
  ))
  finished_without_work_h5 <- all(c(
    file.exists(file.path(OUT_DIR, "GLOBAL_RMA_COMPLETE.txt")),
    file.exists(PARAMETER_H5),
    file.exists(PARAMETER_COMPLETE),
    file.exists(PARAMETER_SIGNATURE),
    setequal(saved_pass3_blocks[!is.na(saved_pass3_blocks)], expected_pass3_blocks)
  ))

  if (!file.exists(WORK_H5)) {
    if (finished_without_work_h5) {
      message(
        "[", timestamp(), "] Large probe-level work file was removed after an ",
        "earlier successful run; completed outputs will be reused."
      )
      pass2_done <- c(
        as.character(gse_index$GSE),
        "IKEM:PRIVATE_OR_GSE290167:NO_EGFR_TRAIN"
      )
    } else {
      if (length(pass2_done) > 0L) {
        message(
          "[", timestamp(), "] Probe-level work file is missing, so PASS 2 will ",
          "restart. Earlier PASS 1 and per-GSE files remain reusable."
        )
        unlink(pass2_done_path, force = TRUE)
        pass2_done <- character()
      }
      rhdf5::h5createFile(WORK_H5)
      rhdf5::h5createDataset(
        WORK_H5,
        "normalized_common_pm",
        dims = c(template_info$n_common_pm, combined_work_sample_count),
        H5type = WORK_H5_TYPE,
        chunk = c(
          min(4096L, template_info$n_common_pm),
          min(PASS2_ARRAY_BATCH_SIZE, combined_work_sample_count)
        ),
        level = H5_COMPRESSION_LEVEL,
        native = FALSE
      )
    }
  }

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

  # Add the frozen IKEM no-eGFR TRAIN columns to the probe-level working
  # matrix. They participate only in fitting the global reference; the final
  # IKEM matrix is produced later by the transform-only Stage 4.
  ikem_pass2_key <- "IKEM:PRIVATE_OR_GSE290167:NO_EGFR_TRAIN"
  if (!(ikem_pass2_key %in% pass2_done)) {
    ikem_batch_starts <- seq.int(
      1L,
      ikem_train_count,
      by = PASS2_ARRAY_BATCH_SIZE
    )
    for (batch_start in ikem_batch_starts) {
      batch_end <- min(
        ikem_train_count,
        batch_start + PASS2_ARRAY_BATCH_SIZE - 1L
      )
      batch <- batch_start:batch_end
      corrected_batch <- rhdf5::h5read(
        IKEM_GLOBAL_TRAIN_BG_H5,
        IKEM_BACKGROUND_DATASET,
        index = list(seq_len(template_info$n_all_pm), batch),
        native = FALSE
      )
      common_batch <- matrix(
        NA_real_,
        nrow = template_info$n_common_pm,
        ncol = length(batch)
      )
      for (k in seq_along(batch)) {
        normalized <- preprocessCore::normalize.quantiles.use.target(
          matrix(corrected_batch[, k], ncol = 1L),
          target = target,
          copy = FALSE
        )[, 1L]
        common_batch[, k] <- normalized[
          template_info$common_pm_positions_grouped
        ]
        rm(normalized)
      }
      if (any(!is.finite(common_batch)) || any(common_batch <= 0)) {
        stop(
          "PASS2 produced invalid normalized values for IKEM TRAIN arrays.",
          call. = FALSE
        )
      }
      rhdf5::h5write(
        common_batch,
        WORK_H5,
        "normalized_common_pm",
        index = list(
          seq_len(template_info$n_common_pm),
          nrow(sample_index) + batch
        ),
        native = FALSE
      )
      rm(corrected_batch, common_batch)
      gc()
      message(
        "[PASS2 IKEM no-eGFR TRAIN] ",
        batch_end,
        "/",
        ikem_train_count,
        " arrays."
      )
    }
    append_unique_line(pass2_done_path, ikem_pass2_key)
    pass2_done <- c(pass2_done, ikem_pass2_key)
    message(
      "[", timestamp(), "] PASS2 added all ", ikem_train_count,
      " IKEM no-eGFR TRAIN arrays to the combined fit matrix."
    )
  }

  # ---------------------------------------------------------------------------
  # PASS 3: save the TRAIN-fitted probe effects and build final expression
  # ---------------------------------------------------------------------------

  pass3_done_path <- file.path(PROGRESS_DIR, "train_reference_pass3_done_block.txt")
  probe_starts <- cumsum(c(1L, head(template_info$common_pm_counts, -1L)))
  probe_ends <- cumsum(template_info$common_pm_counts)
  block_starts <- seq(
    1L,
    length(common_probes),
    by = SUMMARY_PROBESET_BLOCK_SIZE
  )
  expected_blocks <- seq_along(block_starts)

  read_block_progress <- function(path, label) {
    text <- read_done_lines(path)
    if (length(text) == 0L) return(integer())
    values <- suppressWarnings(as.integer(text))
    if (anyNA(values) || any(!values %in% expected_blocks)) {
      stop(
        label, " contains an invalid block number: ", path,
        ". Move that progress file aside and rerun.",
        call. = FALSE
      )
    }
    unique(values)
  }

  pass3_done <- read_block_progress(pass3_done_path, "PASS 3 progress")
  parameter_done <- read_block_progress(
    PARAMETER_PROGRESS,
    "Frozen-parameter progress"
  )

  # The signature protects a resumed sidecar from being mixed with a different
  # split, target, probe order or sample order. The large probe-level HDF5 is
  # deliberately not hashed because it may be deleted after successful use.
  current_parameter_signature <- list(
    format_version = 2L,
    algorithm = paste0(
      "combined GEO TRAIN plus IKEM no-measured-eGFR quantile normalization ",
      "and frozen median-polish probe effects"
    ),
    frozen_split_md5 = unname(tools::md5sum(FROZEN_SPLIT_CSV)),
    template_md5 = unname(tools::md5sum(template_info_path)),
    target_md5 = unname(tools::md5sum(TARGET_RDS)),
    sample_ids = as.character(sample_index$GSM),
    sample_splits = as.character(sample_index$pretraining_split),
    geo_train_sample_ids = as.character(sample_index$GSM[train_rows_r]),
    ikem_train_sample_ids = as.character(ikem_contribution$sample_ids),
    ikem_train_gsms = as.character(ikem_contribution$GSM),
    geo_train_sample_count = as.integer(length(train_rows_r)),
    ikem_train_sample_count = as.integer(ikem_train_count),
    fit_sample_count = as.integer(length(combined_train_rows_r)),
    common_probes = as.character(common_probes),
    n_common_pm = as.integer(template_info$n_common_pm)
  )

  parameter_state_exists <- any(file.exists(c(
    PARAMETER_H5,
    PARAMETER_H5_PART,
    PARAMETER_COMPLETE,
    PARAMETER_PROGRESS
  )))
  if (file.exists(PARAMETER_SIGNATURE)) {
    saved_signature <- readRDS(PARAMETER_SIGNATURE)
    if (!identical(saved_signature, current_parameter_signature)) {
      stop(
        "The saved frozen-parameter backfill belongs to different inputs. ",
        "Move only these small files aside, then rerun: ",
        paste(
          c(PARAMETER_H5, PARAMETER_H5_PART, PARAMETER_COMPLETE,
            PARAMETER_SIGNATURE, PARAMETER_PROGRESS),
          collapse = ", "
        ),
        ". Keep ", WORK_H5, "; it can still be reused if it matches the current split.",
        call. = FALSE
      )
    }
  } else {
    if (parameter_state_exists) {
      stop(
        "Frozen-parameter files exist without their provenance signature. ",
        "Move the small parameter files/progress marker aside and rerun; do not remove ",
        WORK_H5, ".",
        call. = FALSE
      )
    }
    atomic_save_rds(current_parameter_signature, PARAMETER_SIGNATURE, compress = TRUE)
  }

  if (file.exists(PARAMETER_H5) && file.exists(PARAMETER_H5_PART)) {
    stop(
      "Both a completed and partial frozen-parameter file exist. Keep the completed ",
      "file and move ", PARAMETER_H5_PART, " aside before rerunning.",
      call. = FALSE
    )
  }
  if (file.exists(PARAMETER_COMPLETE) && !file.exists(PARAMETER_H5)) {
    stop(
      "The frozen-parameter completion marker exists, but its HDF5 file is missing. ",
      "Move ", PARAMETER_COMPLETE, " aside and rerun.",
      call. = FALSE
    )
  }

  parameter_ready <- file.exists(PARAMETER_H5)
  if (parameter_ready) {
    validate_parameter_h5(PARAMETER_H5, template_info$n_common_pm)
    parameter_done <- expected_blocks
    message("[", timestamp(), "] Frozen probe effects already exist; reusing them.")
  } else {
    if (!file.exists(PARAMETER_H5_PART)) {
      create_parameter_h5(PARAMETER_H5_PART, template_info$n_common_pm)
      unlink(PARAMETER_PROGRESS, force = TRUE)
      parameter_done <- integer()
      message("[", timestamp(), "] Created resumable frozen probe-effect file.")
    } else {
      partial_values <- read_parameter_h5(PARAMETER_H5_PART)
      if (length(partial_values) != template_info$n_common_pm) {
        stop(
          "Partial frozen probe-effect file has the wrong length. Move it and ",
          PARAMETER_PROGRESS, " aside, then rerun.",
          call. = FALSE
        )
      }
      for (done_block in parameter_done) {
        first_probe <- block_starts[[done_block]]
        last_probe <- min(
          first_probe + SUMMARY_PROBESET_BLOCK_SIZE - 1L,
          length(common_probes)
        )
        rows <- probe_starts[[first_probe]]:probe_ends[[last_probe]]
        if (any(!is.finite(partial_values[rows]))) {
          stop(
            "Frozen-parameter progress marks block ", done_block,
            " complete, but its values are missing. Move ", PARAMETER_PROGRESS,
            " aside and rerun; the HDF5 file can be overwritten safely.",
            call. = FALSE
          )
        }
      }
      rm(partial_values)
      gc()
      message(
        "[", timestamp(), "] Resuming frozen probe effects after ",
        length(parameter_done), "/", length(expected_blocks), " blocks."
      )
    }
  }

  missing_summary_blocks <- setdiff(expected_blocks, pass3_done)
  missing_parameter_blocks <- if (parameter_ready) {
    integer()
  } else {
    setdiff(expected_blocks, parameter_done)
  }
  blocks_to_process <- union(missing_summary_blocks, missing_parameter_blocks)

  if (length(blocks_to_process) > 0L && !file.exists(WORK_H5)) {
    stop(
      "PASS 3 still needs ", length(blocks_to_process),
      " block(s), but the probe-level work file is missing: ", WORK_H5,
      ". It must be restored before Phase 4 can be prepared.",
      call. = FALSE
    )
  }

  if (length(missing_summary_blocks) == 0L &&
      length(missing_parameter_blocks) > 0L) {
    message(
      "[", timestamp(), "] Final global RMA is already complete. Backfilling only ",
      "the missing frozen probe effects for Phase 4 (",
      length(missing_parameter_blocks), " blocks)."
    )
  }

  phase3_cluster <- NULL
  if (length(blocks_to_process) > 0L && RMA_WORKERS > 1L) {
    message(
      "[", timestamp(), "] Starting ", RMA_WORKERS,
      " local PASS 3 workers. Workers never open or write HDF5 files."
    )
    phase3_cluster <- tryCatch(
      parallel::makePSOCKcluster(
        RMA_WORKERS,
        useXDR = FALSE
      ),
      error = function(e) {
        stop(
          "Could not start ", RMA_WORKERS, " PASS 3 workers: ",
          conditionMessage(e),
          call. = FALSE
        )
      }
    )
  }

  ram_probe_matrix <- NULL
  if (length(blocks_to_process) > 0L && RMA_COMPUTE_MODE == "ram") {
    ram_probe_matrix <- load_probe_matrix_into_ram(
      path = WORK_H5,
      dataset = "normalized_common_pm",
      n_pm = template_info$n_common_pm,
      n_samples = combined_work_sample_count,
      batch_size = RAM_LOAD_ARRAY_BATCH_SIZE
    )
    message(
      "[", timestamp(), "] RAM mode is ready; all remaining PASS 3 blocks ",
      "will reuse the in-memory matrix."
    )
  }

  for (block_id in blocks_to_process) {
    first_probe <- block_starts[[block_id]]
    last_probe <- min(
      first_probe + SUMMARY_PROBESET_BLOCK_SIZE - 1L,
      length(common_probes)
    )
    probe_ids <- first_probe:last_probe
    pm_rows <- probe_starts[[first_probe]]:probe_ends[[last_probe]]

    normalized_block <- if (RMA_COMPUTE_MODE == "ram") {
      ram_probe_matrix[pm_rows, , drop = FALSE]
    } else {
      rhdf5::h5read(
        WORK_H5,
        "normalized_common_pm",
        index = list(pm_rows, seq_len(combined_work_sample_count)),
        drop = FALSE,
        native = FALSE
      )
    }

    expected_normalized_dim <- c(length(pm_rows), combined_work_sample_count)
    if (!identical(dim(normalized_block), expected_normalized_dim)) {
      stop(
        "PASS 3 read block ", block_id, " with dimensions ",
        paste(dim(normalized_block), collapse = " x "), "; expected ",
        paste(expected_normalized_dim, collapse = " x "), ".",
        call. = FALSE
      )
    }

    # Close any high-level HDF5 handles before dispatching worker tasks. Workers
    # receive ordinary R matrices; the parent remains the sole HDF5 writer.
    rhdf5::H5close()

    local_probe_starts <- cumsum(c(
      1L,
      head(template_info$common_pm_counts[probe_ids], -1L)
    ))

    probe_tasks <- lapply(seq_along(probe_ids), function(local_probe) {
      n_pm <- template_info$common_pm_counts[probe_ids[[local_probe]]]
      local_start <- local_probe_starts[[local_probe]]
      local_rows <- local_start:(local_start + n_pm - 1L)
      list(
        local_probe = local_probe,
        local_rows = local_rows,
        probe_name = common_probes[probe_ids[[local_probe]]],
        normalized_values = normalized_block[local_rows, , drop = FALSE],
        train_rows = combined_train_rows_r
      )
    })

    probe_results <- phase3_lapply(
      probe_tasks,
      summarize_probe_task,
      cluster = phase3_cluster
    )

    summarized <- matrix(
      NA_real_,
      nrow = length(probe_ids),
      ncol = combined_work_sample_count
    )
    parameter_block <- rep(NA_real_, length(pm_rows))

    for (result in probe_results) {
      summarized[result$local_probe, ] <- result$sample_summary
      parameter_block[result$local_rows] <- result$probe_effect
    }

    expected_dim <- c(length(probe_ids), combined_work_sample_count)
    if (!identical(dim(summarized), expected_dim) ||
        length(parameter_block) != length(pm_rows) ||
        any(!is.finite(parameter_block))) {
      stop("PASS 3 produced an invalid block ", block_id, ".", call. = FALSE)
    }

    # Only GEO rows belong in GEO_MATRIX_STORE. The appended IKEM TRAIN columns
    # are used solely to fit the shared probe effects.
    final_block <- t(summarized[, seq_len(nrow(sample_index)), drop = FALSE])
    if (block_id %in% pass3_done) {
      existing_block <- rhdf5::h5read(
        FINAL_H5,
        "expression/rma_global",
        index = list(seq_len(nrow(sample_index)), probe_ids),
        native = TRUE
      )
      difference <- max(abs(as.numeric(existing_block) - as.numeric(final_block)))
      tolerance <- 5e-6 * max(1, max(abs(as.numeric(final_block))))
      if (!is.finite(difference) || difference > tolerance) {
        stop(
          "Existing global-RMA block ", block_id,
          " does not match the recomputed frozen effects (max difference=",
          format(difference, digits = 8L), "). Stop and inspect this store.",
          call. = FALSE
        )
      }
    } else {
      h5_write_native_block_checked(
        final_block,
        FINAL_H5,
        "expression/rma_global",
        start = c(1L, first_probe)
      )
      append_unique_line(pass3_done_path, as.character(block_id))
      pass3_done <- c(pass3_done, block_id)
    }

    if (!parameter_ready && !(block_id %in% parameter_done)) {
      write_parameter_block_checked(PARAMETER_H5_PART, pm_rows, parameter_block)
      append_unique_line(PARAMETER_PROGRESS, as.character(block_id))
      parameter_done <- c(parameter_done, block_id)
    }

    rm(
      normalized_block,
      summarized,
      final_block,
      parameter_block,
      probe_tasks,
      probe_results,
      local_probe_starts
    )
    if (exists("existing_block", inherits = FALSE)) rm(existing_block)
    gc()

    message(
      "[", timestamp(), "] PASS 3 block ", block_id, "/",
      length(block_starts), " ready (probe sets ", first_probe, "-", last_probe, ")."
    )
  }

  if (!is.null(phase3_cluster)) {
    parallel::stopCluster(phase3_cluster)
    phase3_cluster <- NULL
  }

  if (!is.null(ram_probe_matrix)) {
    rm(ram_probe_matrix)
    gc()
  }

  if (!setequal(pass3_done, expected_blocks)) {
    stop("PASS 3 final expression matrix is still incomplete.", call. = FALSE)
  }

  if (!parameter_ready) {
    parameter_done <- read_block_progress(
      PARAMETER_PROGRESS,
      "Frozen-parameter progress"
    )
    if (!setequal(parameter_done, expected_blocks)) {
      stop("Frozen probe-effect backfill is still incomplete.", call. = FALSE)
    }
    validate_parameter_h5(PARAMETER_H5_PART, template_info$n_common_pm)
    if (!file.rename(PARAMETER_H5_PART, PARAMETER_H5)) {
      stop(
        "Could not finalize frozen probe effects: ", PARAMETER_H5,
        call. = FALSE
      )
    }
    parameter_ready <- TRUE
  }

  validate_parameter_h5(PARAMETER_H5, template_info$n_common_pm)
  parameter_marker <- sprintf(
    paste0(
      "%s TRAIN_REFERENCE_PARAMETERS_COMPLETE pm_effects=%d probes=%d ",
      "geo_samples=%d geo_train_samples=%d ikem_train_samples=%d ",
      "train_samples=%d signature_md5=%s"
    ),
    timestamp(),
    template_info$n_common_pm,
    length(common_probes),
    nrow(sample_index),
    length(train_rows_r),
    ikem_train_count,
    length(combined_train_rows_r),
    unname(tools::md5sum(PARAMETER_SIGNATURE))
  )
  atomic_write_lines(parameter_marker, PARAMETER_COMPLETE)
  message("[", timestamp(), "] Frozen Phase 4 parameters are complete: ", PARAMETER_H5)

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

  atomic_write_lines(
    sprintf(
      paste0(
        "%s TRAIN_REFERENCE_RMA_COMPLETE geo_samples=%d probes=%d ",
        "geo_train_samples=%d ikem_train_samples=%d train_samples=%d ",
        "frozen_parameters=%s"
      ),
      timestamp(),
      nrow(sample_index),
      length(common_probes),
      length(train_rows_r),
      ikem_train_count,
      length(combined_train_rows_r),
      basename(PARAMETER_H5)
    ),
    file.path(OUT_DIR, "GLOBAL_RMA_COMPLETE.txt")
  )

  if (
    DELETE_PROBE_LEVEL_SCRATCH_AFTER_SUCCESS &&
    file.exists(WORK_H5) &&
    file.exists(PARAMETER_H5) &&
    file.exists(PARAMETER_COMPLETE)
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
      "the quantile target and median-polish probe effects are fitted on frozen ",
      "GEO TRAIN plus IKEM no-eGFR TRAIN arrays. GEO validation/test, IKEM ",
      "validation, and all eGFR arrays are transformed with frozen parameters only."
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
  "Reusable frozen GEO + IKEM no-eGFR TRAIN reference:",
  "  ../.GLOBAL_RMA_WORK/train_reference_quantile_target.rds",
  "  ../.GLOBAL_RMA_WORK/train_reference_rma_parameters.h5",
  paste0(
    "  Fit arrays: ", length(train_rows_r), " GEO TRAIN + ", ikem_train_count,
    " IKEM no-eGFR TRAIN = ", length(combined_train_rows_r), "."
  ),
  "  Validation/test/eGFR arrays do not fit either parameter.",
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
