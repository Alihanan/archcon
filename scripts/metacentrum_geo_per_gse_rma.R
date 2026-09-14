#!/usr/bin/env Rscript

# Process GEO GSE*_RAW.tar archives one at a time with bounded RAM:
#   GSE*_RAW.tar -> CEL/CEL.gz -> affxparser streaming passes
#   -> raw PM summary + exact joint per-GSE RMA -> common probes -> RDS
#
# Resume behavior:
#   * a GSE is skipped only when BOTH raw-signal and RMA *.rds files exist
#   * output is first written as *.part and atomically renamed after success
#   * only selected CEL/CEL.gz members are extracted from each TAR
#   * only selected CEL.gz files are decompressed; unrelated TAR members are never extracted
#   * temporary extracted CEL files are removed after every GSE
#   * RMA target/HDF5 progress is retained, so a stopped large GSE resumes
#   * no full samples-by-cells AffyBatch is created
#
# Output matrix orientation for both files:
#   rows    = samples
#   columns = common probe-set IDs, in the order stored in common_probes.pkl
#
# Two outputs are saved per GSE:
#   *_raw_pm_median_common.rds : median of original PM CEL intensities for each probe set;
#                               NO background correction, NO normalization, NO log transform.
#   *_rma_common.rds           : standard RMA log2 expression.

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

RAW_DIR <- Sys.getenv("ARCHCON_GEO_RAW_DIR", unset = "./GEO_RAW")
COMMON_PROBES_PKL <- "../common_probes.pkl"

# Optional but recommended. If present, only mapped GSMs for a GSE are retained.
# Set to NULL if you do not want to use it.
GSM_TO_GSE_CSV <- "gsm_to_gse_mapping.csv"

OUT_DIR <- "GEO_RMA"
WORK_ROOT <- ".GEO_RMA_work"
MANIFEST_PATH <- file.path(OUT_DIR, "rma_manifest.csv")

# Optional: process only selected series while testing, e.g.
# ONLY_GSES <- c("GSE85268")
# Leave NULL to process every GSE*_RAW.tar archive.
ONLY_GSES <- NULL

# The script verifies CEL headers through affxparser and keeps PrimeView arrays only.
EXPECTED_CDF_PATTERN <- "primeview"

# The normalized common-PM working matrix is disk-backed.  float64 preserves
# the numerical behavior of affy::rma/preprocessCore; it is intentionally not
# reduced to float32 merely to save disk space.
STREAM_H5_TYPE <- "H5T_IEEE_F64LE"
STREAM_H5_COMPRESSION <- 1L
STREAM_ARRAY_CHUNK <- 4L
STREAM_PROBESET_BLOCK <- 256L

# This pipeline is intentionally single-CPU. Set limits again inside R so the
# stage remains safe when invoked directly rather than through the PBS wrapper.
Sys.setenv(
  R_THREADS = "1",
  OMP_NUM_THREADS = "1",
  OPENBLAS_NUM_THREADS = "1",
  MKL_NUM_THREADS = "1",
  BLIS_NUM_THREADS = "1",
  VECLIB_MAXIMUM_THREADS = "1",
  NUMEXPR_NUM_THREADS = "1"
)

# -------------------------------------------------------------------------
# Package checks
# -------------------------------------------------------------------------

required_packages <- c(
  "affy",
  "affxparser",
  "Biobase",
  "preprocessCore",
  "primeviewcdf",
  "rhdf5",
  "R.utils"
)

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, quietly = TRUE, FUN.VALUE = logical(1))
]

if (length(missing_packages) > 0L) {
  stop(
    paste0(
      "Missing R packages: ",
      paste(missing_packages, collapse = ", "),
      "\n\nInstall them with:\n",
      "if (!requireNamespace(\"BiocManager\", quietly = TRUE)) ",
      "install.packages(\"BiocManager\")\n",
      "BiocManager::install(c(\"affy\", \"affxparser\", \"Biobase\", ",
      "\"preprocessCore\", \"primeviewcdf\", \"rhdf5\"), update = FALSE)\n",
      "install.packages(\"R.utils\")\n"
    ),
    call. = FALSE
  )
}

# -------------------------------------------------------------------------
# Basic path checks
# -------------------------------------------------------------------------

if (!dir.exists(RAW_DIR)) {
  stop("RAW_DIR does not exist: ", RAW_DIR, call. = FALSE)
}

if (!file.exists(COMMON_PROBES_PKL)) {
  stop("common_probes.pkl does not exist: ", COMMON_PROBES_PKL, call. = FALSE)
}

if (!is.null(GSM_TO_GSE_CSV) && !file.exists(GSM_TO_GSE_CSV)) {
  message(
    "GSM mapping was configured but not found; continuing without it: ",
    GSM_TO_GSE_CSV
  )
  GSM_TO_GSE_CSV <- NULL
}

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(WORK_ROOT, recursive = TRUE, showWarnings = FALSE)

# Do not remove WORK_ROOT here.  It contains bounded-memory RMA checkpoints
# (target sums, HDF5 columns, and summarized blocks) used to resume a large GSE.
# Each GSE's extracted CEL directory is still removed by process_archive().

stale_parts <- list.files(
  OUT_DIR,
  pattern = "\\.part$",
  full.names = TRUE
)
if (length(stale_parts) > 0L) {
  message("Removing stale incomplete output files from a previous run...")
  unlink(stale_parts, force = TRUE)
}

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

format_time <- function(x = Sys.time()) {
  format(x, "%Y-%m-%d %H:%M:%S %z")
}

gse_from_archive <- function(path) {
  sub(
    "_RAW\\.tar$",
    "",
    basename(path),
    ignore.case = TRUE
  )
}

extract_gsm <- function(path) {
  s <- basename(path)

  if (!grepl("GSM[0-9]+", s, ignore.case = TRUE, perl = TRUE)) {
    return(NA_character_)
  }

  toupper(
    sub(
      ".*?(GSM[0-9]+).*",
      "\\1",
      s,
      ignore.case = TRUE,
      perl = TRUE
    )
  )
}

read_common_probes_pickle <- function(pkl_path) {
  python <- Sys.which("python3")

  if (!nzchar(python)) {
    stop(
      "python3 was not found. It is needed only to read common_probes.pkl.",
      call. = FALSE
    )
  }

  helper_py <- tempfile(fileext = ".py")
  helper_txt <- tempfile(fileext = ".txt")

  on.exit(
    unlink(c(helper_py, helper_txt), force = TRUE),
    add = TRUE
  )

  python_code <- c(
    "import pickle",
    "import sys",
    "",
    "pkl_path, out_path = sys.argv[1], sys.argv[2]",
    "",
    "with open(pkl_path, 'rb') as fh:",
    "    obj = pickle.load(fh)",
    "",
    "if hasattr(obj, 'tolist'):",
    "    obj = obj.tolist()",
    "",
    "if isinstance(obj, set):",
    "    obj = list(obj)",
    "",
    "if not isinstance(obj, (list, tuple)):",
    "    raise TypeError(",
    "        'common_probes.pkl must contain a 1-D list/tuple/array/Index-like object; '",
    "        f'got {type(obj).__name__}'",
    "    )",
    "",
    "values = [str(x) for x in obj]",
    "",
    "with open(out_path, 'w', encoding='utf-8') as fh:",
    "    for value in values:",
    "        fh.write(value + '\\n')"
  )

  writeLines(python_code, helper_py)

  command_output <- system2(
    python,
    args = c(
      shQuote(helper_py),
      shQuote(pkl_path),
      shQuote(helper_txt)
    ),
    stdout = TRUE,
    stderr = TRUE
  )

  status <- attr(command_output, "status")
  if (is.null(status)) {
    status <- 0L
  }

  if (status != 0L) {
    stop(
      "Failed to read common_probes.pkl:\n",
      paste(command_output, collapse = "\n"),
      call. = FALSE
    )
  }

  probes <- readLines(helper_txt, warn = FALSE)
  probes <- probes[nzchar(probes)]

  if (length(probes) == 0L) {
    stop("common_probes.pkl produced an empty probe list.", call. = FALSE)
  }

  if (anyDuplicated(probes)) {
    stop("common_probes.pkl contains duplicated probe IDs.", call. = FALSE)
  }

  probes
}

load_gsm_mapping <- function(path) {
  if (is.null(path)) {
    return(NULL)
  }

  mapping <- utils::read.csv(
    path,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )

  gsm_candidates <- grep(
    "gsm",
    names(mapping),
    ignore.case = TRUE,
    value = TRUE
  )

  gse_candidates <- grep(
    "gse",
    names(mapping),
    ignore.case = TRUE,
    value = TRUE
  )

  if (length(gsm_candidates) == 0L || length(gse_candidates) == 0L) {
    stop(
      "Could not auto-detect GSM/GSE columns in ",
      path,
      ". Column names are: ",
      paste(names(mapping), collapse = ", "),
      call. = FALSE
    )
  }

  gsm_col <- gsm_candidates[[1L]]
  gse_col <- gse_candidates[[1L]]

  data.frame(
    gsm = toupper(trimws(as.character(mapping[[gsm_col]]))),
    gse = toupper(trimws(as.character(mapping[[gse_col]]))),
    stringsAsFactors = FALSE
  )
}

empty_manifest <- function() {
  data.frame(
    gse = character(),
    archive = character(),
    status = character(),
    samples = integer(),
    probes = integer(),
    cdf = character(),
    output = character(),
    raw_output = character(),
    raw_summary = character(),
    started_at = character(),
    finished_at = character(),
    message = character(),
    stringsAsFactors = FALSE
  )
}

if (file.exists(MANIFEST_PATH)) {
  manifest <- utils::read.csv(
    MANIFEST_PATH,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )

  template <- empty_manifest()
  missing_columns <- setdiff(names(template), names(manifest))

  # The manifest may have been created by an older version of this script.
  # empty_manifest() has zero rows, so assigning template[[column]] directly
  # to a non-empty manifest would fail with:
  #   replacement has 0 rows, data has N
  # Add missing columns as correctly typed NA vectors instead.
  for (column in missing_columns) {
    prototype <- template[[column]]

    if (is.integer(prototype)) {
      manifest[[column]] <- rep(NA_integer_, nrow(manifest))
    } else if (is.numeric(prototype)) {
      manifest[[column]] <- rep(NA_real_, nrow(manifest))
    } else if (is.logical(prototype)) {
      manifest[[column]] <- rep(NA, nrow(manifest))
    } else {
      manifest[[column]] <- rep(NA_character_, nrow(manifest))
    }
  }

  manifest <- manifest[, names(template), drop = FALSE]
} else {
  manifest <- empty_manifest()
}

write_manifest <- function() {
  tmp <- paste0(MANIFEST_PATH, ".part")

  utils::write.csv(
    manifest,
    tmp,
    row.names = FALSE,
    na = ""
  )

  if (file.exists(MANIFEST_PATH)) {
    unlink(MANIFEST_PATH, force = TRUE)
  }

  if (!file.rename(tmp, MANIFEST_PATH)) {
    stop("Could not update manifest: ", MANIFEST_PATH, call. = FALSE)
  }
}

set_manifest <- function(
  gse,
  archive,
  status,
  samples = NA_integer_,
  probes = NA_integer_,
  cdf = NA_character_,
  output = NA_character_,
  raw_output = NA_character_,
  raw_summary = NA_character_,
  started_at = NA_character_,
  finished_at = NA_character_,
  msg = NA_character_
) {
  row <- data.frame(
    gse = as.character(gse),
    archive = as.character(archive),
    status = as.character(status),
    samples = as.integer(samples),
    probes = as.integer(probes),
    cdf = as.character(cdf),
    output = as.character(output),
    raw_output = as.character(raw_output),
    raw_summary = as.character(raw_summary),
    started_at = as.character(started_at),
    finished_at = as.character(finished_at),
    message = as.character(msg),
    stringsAsFactors = FALSE
  )

  existing <- match(gse, manifest$gse)

  if (is.na(existing)) {
    manifest <<- rbind(manifest, row)
  } else {
    manifest[existing, ] <<- row
  }

  write_manifest()
}

# -------------------------------------------------------------------------
# Read common-probe list and optional GSM mapping once
# -------------------------------------------------------------------------

message("Reading common probe IDs from: ", COMMON_PROBES_PKL)
common_probes <- read_common_probes_pickle(COMMON_PROBES_PKL)
message("Common probes: ", format(length(common_probes), big.mark = ","))

gsm_mapping <- load_gsm_mapping(GSM_TO_GSE_CSV)

if (!is.null(gsm_mapping)) {
  message(
    "Loaded GSM->GSE mapping with ",
    format(nrow(gsm_mapping), big.mark = ","),
    " rows."
  )
}

# -------------------------------------------------------------------------
# CEL reader: affxparser -> AffyBatch
# -------------------------------------------------------------------------

read_affybatch_affxparser <- function(cel_files, expected_cdf_pattern) {
  if (length(cel_files) == 0L) {
    stop("No CEL files supplied to affxparser reader.", call. = FALSE)
  }

  message(
    "Reading CEL headers with affxparser::readCelHeader()..."
  )

  n_files <- length(cel_files)
  chip_types <- character(n_files)
  chip_rows <- integer(n_files)
  chip_cols <- integer(n_files)
  chip_total <- integer(n_files)

  for (idx in seq_along(cel_files)) {
    header <- affxparser::readCelHeader(cel_files[[idx]])

    chip_types[[idx]] <- as.character(header$chiptype[[1L]])
    chip_rows[[idx]] <- as.integer(header$rows[[1L]])
    chip_cols[[idx]] <- as.integer(header$cols[[1L]])
    chip_total[[idx]] <- as.integer(header$total[[1L]])

    if (idx %% 25L == 0L || idx == n_files) {
      message(
        "Header check: ",
        idx,
        "/",
        n_files,
        " CEL file(s)."
      )
    }

    rm(header)
  }

  if (anyNA(chip_rows) || anyNA(chip_cols) || anyNA(chip_total)) {
    stop(
      "At least one CEL header is missing array dimensions.",
      call. = FALSE
    )
  }

  nonempty_chip_types <- chip_types[nzchar(chip_types)]

  if (length(nonempty_chip_types) == 0L) {
    stop(
      "affxparser did not report a chip type for any CEL file.",
      call. = FALSE
    )
  }

  expected <- grepl(
    expected_cdf_pattern,
    chip_types,
    ignore.case = TRUE
  )

  # An empty chip type is not silently accepted. These archives are intended
  # to be GPL15207/PrimeView, so an unreadable/unknown type should be examined.
  if (any(!expected)) {
    bad_files <- basename(cel_files[!expected])
    bad_types <- chip_types[!expected]

    preview <- paste(
      paste0(
        head(bad_files, 10L),
        " [",
        head(bad_types, 10L),
        "]"
      ),
      collapse = ", "
    )

    stop(
      sum(!expected),
      " CEL file(s) are not reported as PrimeView by affxparser. First few: ",
      preview,
      call. = FALSE
    )
  }

  if (length(unique(chip_rows)) != 1L ||
      length(unique(chip_cols)) != 1L ||
      length(unique(chip_total)) != 1L) {
    stop(
      "CEL files in this GSE do not all have identical array dimensions.",
      call. = FALSE
    )
  }

  array_rows <- chip_rows[[1L]]
  array_cols <- chip_cols[[1L]]
  array_total <- chip_total[[1L]]

  if (array_rows * array_cols != array_total) {
    stop(
      "CEL header dimensions are inconsistent: rows*cols = ",
      array_rows * array_cols,
      " but total = ",
      array_total,
      ".",
      call. = FALSE
    )
  }

  sample_ids <- basename(cel_files)
  sample_ids <- sub(
    "\\.CEL$",
    "",
    sample_ids,
    ignore.case = TRUE
  )

  if (anyDuplicated(sample_ids)) {
    stop(
      "Duplicate sample IDs after stripping the .CEL suffix.",
      call. = FALSE
    )
  }

  message(
    "Reading ",
    n_files,
    " CEL intensity vector(s) with affxparser::readCelIntensities()..."
  )

  intensities <- affxparser::readCelIntensities(cel_files)

  if (is.null(dim(intensities))) {
    intensities <- matrix(
      intensities,
      ncol = 1L
    )
  }

  if (nrow(intensities) != array_total) {
    stop(
      "affxparser returned ",
      nrow(intensities),
      " probe-cell intensities per sample; CEL header says ",
      array_total,
      ".",
      call. = FALSE
    )
  }

  if (ncol(intensities) != n_files) {
    stop(
      "affxparser returned ",
      ncol(intensities),
      " sample column(s) for ",
      n_files,
      " CEL files.",
      call. = FALSE
    )
  }

  storage.mode(intensities) <- "double"
  colnames(intensities) <- sample_ids

  pheno_frame <- data.frame(
    sample_id = sample_ids,
    row.names = sample_ids,
    stringsAsFactors = FALSE
  )

  pheno_data <- Biobase::AnnotatedDataFrame(
    data = pheno_frame
  )

  # All retained arrays were verified above as PrimeView. The CDF package
  # installed for this array is primeviewcdf; AffyBatch expects the chip/CDF
  # name rather than the package name.
  cdf_name <- "PrimeView"

  message(
    "Constructing AffyBatch manually from affxparser intensities: ",
    array_rows,
    " x ",
    array_cols,
    " cells, ",
    n_files,
    " sample(s)."
  )

  abatch <- methods::new(
    "AffyBatch",
    exprs = intensities,
    cdfName = cdf_name,
    annotation = affy::cleancdfname(
      cdf_name,
      addcdf = FALSE
    ),
    nrow = as.numeric(array_rows),
    ncol = as.numeric(array_cols),
    phenoData = pheno_data
  )

  methods::validObject(abatch)

  # Force CDF resolution now, so a missing/incompatible primeviewcdf package
  # fails here with a clear error instead of much later in RMA.
  invisible(affy::getCdfInfo(abatch))

  rm(intensities)
  gc()

  list(
    abatch = abatch,
    sample_ids = sample_ids,
    cdf = cdf_name,
    chip_types = unique(chip_types),
    rows = array_rows,
    cols = array_cols,
    total = array_total
  )
}

# -------------------------------------------------------------------------
# Raw probe-set signal helper
# -------------------------------------------------------------------------

summarize_raw_pm_common <- function(abatch, common_probes, sample_ids) {
  message(
    "Building raw common-probe signal matrix from original PM CEL intensities..."
  )

  # common_probes.pkl contains probe-SET IDs, while a CEL file contains
  # individual probe-cell intensities. Therefore a 42,917-column raw matrix
  # requires a within-probe-set summary. We use the median PM intensity:
  # no background correction, no quantile normalization, and no log transform.
  pm_indices <- affy::indexProbes(
    abatch,
    which = "pm",
    genenames = common_probes
  )

  if (length(pm_indices) != length(common_probes)) {
    stop(
      "indexProbes returned ",
      length(pm_indices),
      " probe sets for ",
      length(common_probes),
      " requested common probes.",
      call. = FALSE
    )
  }

  missing_index <- vapply(
    pm_indices,
    function(idx) length(idx) == 0L,
    FUN.VALUE = logical(1)
  )

  if (any(missing_index)) {
    bad <- common_probes[missing_index]
    stop(
      length(bad),
      " common probe set(s) have no PM probe-cell indices. First few: ",
      paste(head(bad, 10L), collapse = ", "),
      call. = FALSE
    )
  }

  n_samples <- length(sample_ids)

  raw_signal <- vapply(
    seq_along(pm_indices),
    function(j) {
      values <- affy::intensity(abatch)[
        pm_indices[[j]],
        ,
        drop = FALSE
      ]

      if (nrow(values) == 1L) {
        return(as.numeric(values[1L, ]))
      }

      apply(
        values,
        2L,
        stats::median,
        na.rm = TRUE
      )
    },
    FUN.VALUE = numeric(n_samples)
  )

  rownames(raw_signal) <- sample_ids
  colnames(raw_signal) <- common_probes

  raw_signal
}

# -------------------------------------------------------------------------
# Bounded-memory CEL/RMA helpers
# -------------------------------------------------------------------------

atomic_save_rds <- function(object, path, compress = FALSE) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  part <- paste0(path, ".part")
  unlink(part, force = TRUE)
  saveRDS(object, part, compress = compress)
  if (!file.rename(part, path)) {
    stop("Could not atomically finalize RDS checkpoint: ", path, call. = FALSE)
  }
  invisible(path)
}

read_one_cel_vector <- function(path) {
  x <- affxparser::readCelIntensities(path)
  if (is.matrix(x)) {
    if (ncol(x) != 1L) {
      stop("Expected one intensity column in: ", path, call. = FALSE)
    }
    x <- x[, 1L]
  }
  as.numeric(x)
}

header_value <- function(header, candidates) {
  header_names <- tolower(names(header))
  for (candidate in tolower(candidates)) {
    position <- match(candidate, header_names)
    if (!is.na(position) && length(header[[position]]) > 0L) {
      return(header[[position]][[1L]])
    }
  }
  NULL
}

build_streaming_template <- function(cel_files, common_probes, expected_pattern) {
  n_files <- length(cel_files)
  chip_types <- character(n_files)
  chip_rows <- integer(n_files)
  chip_cols <- integer(n_files)
  chip_total <- integer(n_files)

  message("Reading CEL headers one at a time...")
  for (idx in seq_along(cel_files)) {
    header <- affxparser::readCelHeader(cel_files[[idx]])
    chip_types[[idx]] <- as.character(
      header_value(header, c("chiptype", "chipType", "arrayType"))
    )
    chip_rows[[idx]] <- as.integer(header_value(header, c("rows", "nrows")))
    chip_cols[[idx]] <- as.integer(
      header_value(header, c("cols", "columns", "ncols"))
    )
    chip_total[[idx]] <- as.integer(
      header_value(header, c("total", "ncells", "cells"))
    )
    rm(header)

    if (idx %% 25L == 0L || idx == n_files) {
      message("Header check: ", idx, "/", n_files, " CEL file(s).")
    }
  }

  if (any(!grepl(expected_pattern, chip_types, ignore.case = TRUE))) {
    bad <- which(!grepl(expected_pattern, chip_types, ignore.case = TRUE))
    stop(
      length(bad),
      " CEL file(s) are not PrimeView. First few: ",
      paste(basename(head(cel_files[bad], 10L)), collapse = ", "),
      call. = FALSE
    )
  }
  if (
    anyNA(chip_rows) || anyNA(chip_cols) || anyNA(chip_total) ||
    length(unique(chip_rows)) != 1L ||
    length(unique(chip_cols)) != 1L ||
    length(unique(chip_total)) != 1L
  ) {
    stop("CEL files do not have one consistent array geometry.", call. = FALSE)
  }

  first_intensity <- read_one_cel_vector(cel_files[[1L]])
  if (
    chip_rows[[1L]] * chip_cols[[1L]] != length(first_intensity) ||
    chip_total[[1L]] != length(first_intensity)
  ) {
    stop("PrimeView CEL header geometry is inconsistent.", call. = FALSE)
  }

  first_id <- sub("\\.CEL$", "", basename(cel_files[[1L]]), ignore.case = TRUE)
  pheno <- Biobase::AnnotatedDataFrame(
    data = data.frame(
      sample_id = first_id,
      row.names = first_id,
      stringsAsFactors = FALSE
    )
  )
  template_abatch <- methods::new(
    "AffyBatch",
    exprs = matrix(first_intensity, ncol = 1L, dimnames = list(NULL, first_id)),
    cdfName = "PrimeView",
    annotation = affy::cleancdfname("PrimeView", addcdf = FALSE),
    nrow = as.numeric(chip_rows[[1L]]),
    ncol = as.numeric(chip_cols[[1L]]),
    phenoData = pheno
  )
  methods::validObject(template_abatch)
  invisible(affy::getCdfInfo(template_abatch))

  pm_all_list <- affy::indexProbes(template_abatch, which = "pm")
  if (is.null(names(pm_all_list))) {
    stop("PrimeView CDF returned unnamed probe sets.", call. = FALSE)
  }
  pm_all_list <- pm_all_list[lengths(pm_all_list) > 0L]

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
      " common probe set(s) are absent from PrimeView CDF. First few: ",
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
  all_pm_ends <- cumsum(all_pm_counts)
  all_pm_starts <- all_pm_ends - all_pm_counts + 1L
  common_pm_positions <- as.integer(
    unname(
      unlist(
        lapply(
          common_set_positions,
          function(set_position) {
            seq.int(
              all_pm_starts[[set_position]],
              all_pm_ends[[set_position]]
            )
          }
        ),
        use.names = FALSE
      )
    )
  )
  if (
    anyNA(common_pm_positions) ||
    length(common_pm_positions) != sum(common_pm_counts)
  ) {
    stop("Could not map common-probe PM memberships into the all-PM vector.", call. = FALSE)
  }

  sample_ids <- sub("\\.CEL$", "", basename(cel_files), ignore.case = TRUE)
  if (anyDuplicated(sample_ids)) {
    stop("Duplicate sample IDs after stripping .CEL.", call. = FALSE)
  }

  result <- list(
    sample_ids = sample_ids,
    cdf = "PrimeView",
    chip_types = unique(chip_types),
    total_cells = length(first_intensity),
    all_pm_cell_index = all_pm_cell_index,
    n_all_pm = length(all_pm_cell_index),
    common_pm_counts = common_pm_counts,
    common_pm_positions = common_pm_positions,
    n_common_pm = length(common_pm_positions)
  )

  rm(
    first_intensity,
    template_abatch,
    pm_all_list,
    all_pm_counts,
    all_pm_starts,
    all_pm_ends,
    common_set_positions
  )
  gc()
  result
}

stream_raw_pm_common <- function(cel_files, template, common_probes) {
  n_samples <- length(cel_files)
  n_probes <- length(common_probes)
  output <- matrix(NA_real_, nrow = n_samples, ncol = n_probes)
  starts <- cumsum(c(1L, head(template$common_pm_counts, -1L)))
  ends <- cumsum(template$common_pm_counts)

  for (sample_index in seq_along(cel_files)) {
    intensity <- read_one_cel_vector(cel_files[[sample_index]])
    all_pm <- intensity[template$all_pm_cell_index]
    grouped <- all_pm[template$common_pm_positions]
    output[sample_index, ] <- vapply(
      seq_len(n_probes),
      function(probe_index) {
        stats::median(grouped[starts[[probe_index]]:ends[[probe_index]]])
      },
      FUN.VALUE = numeric(1)
    )
    rm(intensity, all_pm, grouped)
    if (sample_index %% 25L == 0L || sample_index == n_samples) {
      message("Raw PM summary: ", sample_index, "/", n_samples, " arrays.")
      gc()
    }
  }

  rownames(output) <- template$sample_ids
  colnames(output) <- common_probes
  output
}

stream_joint_rma <- function(cel_files, template, common_probes, state_dir) {
  dir.create(state_dir, recursive = TRUE, showWarnings = FALSE)
  target_state_path <- file.path(state_dir, "target_state.rds")
  target_path <- file.path(state_dir, "quantile_target.rds")
  normalized_h5 <- file.path(state_dir, "normalized_common_pm.h5")
  normalized_progress_path <- file.path(state_dir, "normalized_progress.rds")
  summarized_h5 <- file.path(state_dir, "summarized_expression.h5")
  summarized_progress_path <- file.path(state_dir, "summarized_progress.rds")
  n_samples <- length(cel_files)
  n_probes <- length(common_probes)

  signature <- list(
    sample_ids = template$sample_ids,
    common_probes = common_probes,
    n_all_pm = template$n_all_pm,
    n_common_pm = template$n_common_pm
  )
  signature_path <- file.path(state_dir, "signature.rds")
  if (file.exists(signature_path)) {
    if (!identical(readRDS(signature_path), signature)) {
      stop(
        "Streaming RMA checkpoint metadata changed. Remove only this directory and rerun: ",
        state_dir,
        call. = FALSE
      )
    }
  } else {
    atomic_save_rds(signature, signature_path, compress = TRUE)
  }

  # PASS 1: RMA background correction is array-wise.  Accumulate the mean
  # order-statistic vector without retaining any other array in RAM.
  if (file.exists(target_path)) {
    target <- readRDS(target_path)
  } else {
    if (file.exists(target_state_path)) {
      target_state <- readRDS(target_state_path)
    } else {
      target_state <- list(sum = numeric(template$n_all_pm), completed = 0L)
    }
    if (
      !is.list(target_state) || length(target_state$sum) != template$n_all_pm ||
      length(target_state$completed) != 1L || is.na(target_state$completed) ||
      target_state$completed < 0L || target_state$completed > n_samples ||
      any(!is.finite(target_state$sum))
    ) {
      stop("Invalid RMA pass-1 progress checkpoint: ", target_state_path, call. = FALSE)
    }
    start_at <- target_state$completed + 1L
    if (start_at <= n_samples) {
      for (sample_index in seq.int(start_at, n_samples)) {
        intensity <- read_one_cel_vector(cel_files[[sample_index]])
        corrected <- preprocessCore::rma.background.correct(
          matrix(intensity[template$all_pm_cell_index], ncol = 1L),
          copy = FALSE
        )[, 1L]
        if (any(!is.finite(corrected))) {
          stop("Non-finite background-corrected PM values in ", template$sample_ids[[sample_index]], call. = FALSE)
        }
        target_state$sum <- target_state$sum + sort(corrected)
        target_state$completed <- sample_index
        rm(intensity, corrected)
        if (sample_index %% 10L == 0L || sample_index == n_samples) {
          atomic_save_rds(target_state, target_state_path, compress = FALSE)
          message("RMA pass 1/3 (quantile target): ", sample_index, "/", n_samples, " arrays.")
          gc()
        }
      }
    }
    if (target_state$completed != n_samples) {
      stop("Incomplete RMA target checkpoint.", call. = FALSE)
    }
    target <- target_state$sum / n_samples
    atomic_save_rds(target, target_path, compress = TRUE)
    unlink(target_state_path, force = TRUE)
    rm(target_state)
  }
  if (length(target) != template$n_all_pm || any(!is.finite(target))) {
    stop("Saved per-GSE quantile target is invalid.", call. = FALSE)
  }

  # PASS 2: normalize each array to the shared per-GSE target. Store only the
  # common-probe PM cells in HDF5, in grouped probe-set order.
  if (!file.exists(normalized_h5)) {
    unlink(normalized_progress_path, force = TRUE)
    rhdf5::h5createFile(normalized_h5)
    rhdf5::h5createDataset(
      normalized_h5,
      "normalized_common_pm",
      dims = c(template$n_common_pm, n_samples),
      H5type = STREAM_H5_TYPE,
      chunk = c(min(4096L, template$n_common_pm), min(STREAM_ARRAY_CHUNK, n_samples)),
      level = STREAM_H5_COMPRESSION,
      native = FALSE
    )
  }
  normalized_completed <- if (file.exists(normalized_progress_path)) {
    as.integer(readRDS(normalized_progress_path))
  } else {
    0L
  }
  if (
    length(normalized_completed) != 1L || is.na(normalized_completed) ||
    normalized_completed < 0L || normalized_completed > n_samples
  ) {
    stop("Invalid RMA pass-2 progress checkpoint: ", normalized_progress_path, call. = FALSE)
  }
  if (normalized_completed < n_samples) {
    batch_starts <- seq.int(normalized_completed + 1L, n_samples, by = STREAM_ARRAY_CHUNK)
    for (batch_start in batch_starts) {
      batch_end <- min(batch_start + STREAM_ARRAY_CHUNK - 1L, n_samples)
      batch_columns <- batch_start:batch_end
      common_batch <- matrix(
        NA_real_,
        nrow = template$n_common_pm,
        ncol = length(batch_columns)
      )
      for (local_index in seq_along(batch_columns)) {
        sample_index <- batch_columns[[local_index]]
        intensity <- read_one_cel_vector(cel_files[[sample_index]])
        corrected <- preprocessCore::rma.background.correct(
          matrix(intensity[template$all_pm_cell_index], ncol = 1L),
          copy = FALSE
        )
        normalized <- preprocessCore::normalize.quantiles.use.target(
          corrected,
          target = target,
          copy = FALSE
        )[, 1L]
        common_batch[, local_index] <- normalized[template$common_pm_positions]
        rm(intensity, corrected, normalized)
      }
      if (any(!is.finite(common_batch))) {
        stop("Non-finite normalized PM values in RMA pass 2.", call. = FALSE)
      }
      rhdf5::h5write(
        common_batch,
        normalized_h5,
        "normalized_common_pm",
        index = list(seq_len(template$n_common_pm), batch_columns),
        native = FALSE
      )
      atomic_save_rds(as.integer(batch_end), normalized_progress_path)
      rm(common_batch)
      gc()
      message("RMA pass 2/3 (disk-backed normalization): ", batch_end, "/", n_samples, " arrays.")
    }
  }

  # PASS 3: standard RMA median polish, but only a bounded probe-set block is
  # materialized. Probe sets are independent at this stage, so blocking does
  # not change the fitted values.
  if (!file.exists(summarized_h5)) {
    unlink(summarized_progress_path, force = TRUE)
    rhdf5::h5createFile(summarized_h5)
    rhdf5::h5createDataset(
      summarized_h5,
      "expression",
      dims = c(n_samples, n_probes),
      H5type = STREAM_H5_TYPE,
      chunk = c(min(64L, n_samples), min(STREAM_PROBESET_BLOCK, n_probes)),
      level = STREAM_H5_COMPRESSION,
      native = FALSE
    )
  }
  completed_blocks <- if (file.exists(summarized_progress_path)) {
    as.integer(readRDS(summarized_progress_path))
  } else {
    0L
  }
  probe_starts <- cumsum(c(1L, head(template$common_pm_counts, -1L)))
  probe_ends <- cumsum(template$common_pm_counts)
  block_starts <- seq.int(1L, n_probes, by = STREAM_PROBESET_BLOCK)
  if (
    length(completed_blocks) != 1L || is.na(completed_blocks) ||
    completed_blocks < 0L || completed_blocks > length(block_starts)
  ) {
    stop("Invalid RMA pass-3 progress checkpoint: ", summarized_progress_path, call. = FALSE)
  }
  if (completed_blocks < length(block_starts)) {
    for (block_id in seq.int(completed_blocks + 1L, length(block_starts))) {
      first_probe <- block_starts[[block_id]]
      last_probe <- min(first_probe + STREAM_PROBESET_BLOCK - 1L, n_probes)
      pm_rows <- probe_starts[[first_probe]]:probe_ends[[last_probe]]
      normalized_block <- rhdf5::h5read(
        normalized_h5,
        "normalized_common_pm",
        index = list(pm_rows, seq_len(n_samples)),
        native = FALSE
      )
      dim(normalized_block) <- c(length(pm_rows), n_samples)
      local_counts <- template$common_pm_counts[first_probe:last_probe]
      groups <- rep(seq_along(local_counts), times = local_counts)
      summarized <- preprocessCore::subColSummarizeMedianpolishLog(
        normalized_block,
        groups
      )
      expected <- c(length(local_counts), n_samples)
      if (!identical(dim(summarized), expected) || any(!is.finite(summarized))) {
        stop("Unexpected/non-finite median-polish result in block ", block_id, call. = FALSE)
      }
      rhdf5::h5write(
        t(summarized),
        summarized_h5,
        "expression",
        index = list(seq_len(n_samples), first_probe:last_probe),
        native = FALSE
      )
      atomic_save_rds(as.integer(block_id), summarized_progress_path)
      rm(normalized_block, summarized)
      gc()
      if (block_id %% 10L == 0L || block_id == length(block_starts)) {
        message("RMA pass 3/3 (median polish): block ", block_id, "/", length(block_starts), ".")
      }
    }
  }

  expression <- rhdf5::h5read(summarized_h5, "expression", native = FALSE)
  dim(expression) <- c(n_samples, n_probes)
  rownames(expression) <- template$sample_ids
  colnames(expression) <- common_probes
  expression
}

# -------------------------------------------------------------------------
# One-GSE processor
# -------------------------------------------------------------------------

process_archive <- function(
  archive,
  common_probes,
  gsm_mapping,
  out_dir,
  work_root
) {
  gse <- gse_from_archive(archive)
  out_file <- file.path(out_dir, paste0(gse, "_rma_common.rds"))
  part_file <- paste0(out_file, ".part")
  raw_out_file <- file.path(
    out_dir,
    paste0(gse, "_raw_pm_median_common.rds")
  )
  raw_part_file <- paste0(raw_out_file, ".part")
  work_dir <- file.path(work_root, paste0(gse, "_cel_extract"))
  state_dir <- file.path(work_root, paste0(gse, "_streaming_rma"))

  unlink(work_dir, recursive = TRUE, force = TRUE)
  unlink(part_file, force = TRUE)
  unlink(raw_part_file, force = TRUE)

  dir.create(work_dir, recursive = TRUE, showWarnings = FALSE)

  on.exit(
    {
      unlink(work_dir, recursive = TRUE, force = TRUE)
      gc()
    },
    add = TRUE
  )

  message("")
  message("============================================================")
  message("[", gse, "] Archive: ", archive)
  message("============================================================")

  members <- utils::untar(archive, list = TRUE)

  cel_members <- members[
    grepl(
      "\\.CEL(\\.gz)?$",
      members,
      ignore.case = TRUE,
      perl = TRUE
    )
  ]

  if (length(cel_members) == 0L) {
    stop("No .CEL or .CEL.gz files found inside archive.", call. = FALSE)
  }

  # Filter by the GSM mapping BEFORE extraction whenever possible. This matters
  # for very large mixed supplementary archives: unrelated members never touch
  # disk at all.
  if (!is.null(gsm_mapping)) {
    wanted_gsms <- unique(
      gsm_mapping$gsm[
        gsm_mapping$gse == toupper(gse)
      ]
    )
    wanted_gsms <- wanted_gsms[nzchar(wanted_gsms)]

    if (length(wanted_gsms) > 0L) {
      member_gsms <- vapply(
        cel_members,
        extract_gsm,
        FUN.VALUE = character(1)
      )

      mapped_member <- !is.na(member_gsms) & member_gsms %in% wanted_gsms

      missing_gsms <- setdiff(
        wanted_gsms,
        unique(member_gsms[!is.na(member_gsms)])
      )

      if (length(missing_gsms) > 0L) {
        message(
          "[",
          gse,
          "] Mapping contains ",
          length(missing_gsms),
          " GSM(s) with no matching CEL member in this TAR."
        )
      }

      if (!any(mapped_member)) {
        stop(
          "None of the CEL members match the mapped GSMs for ",
          gse,
          ".",
          call. = FALSE
        )
      }

      if (sum(mapped_member) < length(cel_members)) {
        message(
          "[",
          gse,
          "] Pre-extraction GSM filter: ",
          length(cel_members),
          " CEL member(s) -> ",
          sum(mapped_member),
          " selected."
        )
      }

      cel_members <- cel_members[mapped_member]
    } else {
      message(
        "[",
        gse,
        "] No rows for this GSE in GSM mapping; using all CEL members."
      )
    }
  }

  message(
    "[",
    gse,
    "] Extracting only ",
    length(cel_members),
    " selected CEL/CEL.gz member(s) into temporary work directory..."
  )

  utils::untar(
    archive,
    files = cel_members,
    exdir = work_dir
  )

  cel_files <- list.files(
    work_dir,
    pattern = "\\.CEL(\\.gz)?$",
    recursive = TRUE,
    full.names = TRUE,
    ignore.case = TRUE
  )

  if (length(cel_files) == 0L) {
    stop("No CEL/CEL.gz files are available after extraction.", call. = FALSE)
  }

  compressed <- grepl("\\.gz$", cel_files, ignore.case = TRUE)

  # affy/affyio nominally support CEL.gz directly, but some CEL variants can
  # trigger native-code crashes (for example "free(): invalid pointer") during
  # compressed header parsing. Native crashes cannot be caught by tryCatch().
  #
  # We therefore keep the important large-archive optimization -- only mapped
  # CEL members are extracted from the TAR -- but decompress those selected
  # CEL.gz files before passing them to affyio/affy. This does NOT unpack the
  # other supplementary files in a huge GSE RAW archive.
  if (any(compressed)) {
    gz_files <- cel_files[compressed]

    message(
      "[",
      gse,
      "] Decompressing ",
      length(gz_files),
      " selected CEL.gz file(s) before affxparser reading..."
    )

    for (idx in seq_along(gz_files)) {
      gz_file <- gz_files[[idx]]
      destination <- sub(
        "\\.gz$",
        "",
        gz_file,
        ignore.case = TRUE
      )

      # R.utils::gunzip() removes only the temporary extracted .gz copy.
      # The original GSE*_RAW.tar archive is never touched.
      R.utils::gunzip(
        gz_file,
        destname = destination,
        remove = TRUE,
        overwrite = TRUE
      )

      if (idx %% 25L == 0L || idx == length(gz_files)) {
        message(
          "[",
          gse,
          "] Decompressed ",
          idx,
          "/",
          length(gz_files),
          " CEL.gz file(s)."
        )
      }
    }
  }

  cel_files <- list.files(
    work_dir,
    pattern = "\\.CEL$",
    recursive = TRUE,
    full.names = TRUE,
    ignore.case = TRUE
  )

  if (length(cel_files) == 0L) {
    stop("No uncompressed CEL files are available after extraction/decompression.", call. = FALSE)
  }

  message(
    "[",
    gse,
    "] Parsing CEL files through affxparser (avoids affyio Calvin/AGCC parser)..."
  )

  template <- build_streaming_template(
    cel_files = cel_files,
    common_probes = common_probes,
    expected_pattern = EXPECTED_CDF_PATTERN
  )

  sample_ids <- template$sample_ids
  cdf_used <- template$cdf

  message(
    "[",
    gse,
    "] affxparser chip type(s): ",
    paste(template$chip_types, collapse = ", ")
  )

  # Save the raw-scale signal checkpoint before RMA. If RMA subsequently
  # fails, this completed raw file is kept and will not be recomputed.
  if (
    file.exists(raw_out_file) &&
    !is.na(file.info(raw_out_file)$size) &&
    file.info(raw_out_file)$size > 0
  ) {
    message(
      "[",
      gse,
      "] Raw signal checkpoint already exists; not recomputing: ",
      raw_out_file
    )
  } else {
    message(
      "[",
      gse,
      "] Summarizing original PM intensities by common probe set..."
    )

    raw_expression <- stream_raw_pm_common(
      cel_files = cel_files,
      template = template,
      common_probes = common_probes
    )

    if (!identical(colnames(raw_expression), common_probes)) {
      stop(
        "Common-probe column order was not preserved in raw matrix.",
        call. = FALSE
      )
    }

    message(
      "[",
      gse,
      "] Raw matrix: ",
      nrow(raw_expression),
      " sample(s) x ",
      ncol(raw_expression),
      " probe(s)."
    )

    message(
      "[",
      gse,
      "] Saving raw-signal checkpoint: ",
      raw_out_file
    )

    saveRDS(
      raw_expression,
      raw_part_file,
      compress = "gzip"
    )

    if (!file.rename(raw_part_file, raw_out_file)) {
      stop(
        "Could not rename completed raw output to: ",
        raw_out_file,
        call. = FALSE
      )
    }

    rm(raw_expression)
    gc()
  }

  if (
    file.exists(out_file) &&
    !is.na(file.info(out_file)$size) &&
    file.info(out_file)$size > 0
  ) {
    message(
      "[",
      gse,
      "] RMA checkpoint already exists; not recomputing: ",
      out_file
    )

    unlink(state_dir, recursive = TRUE, force = TRUE)

    return(
      list(
        output = out_file,
        raw_output = raw_out_file,
        raw_summary = "median PM intensity; no background correction; no normalization; no log transform",
        samples = length(sample_ids),
        probes = length(common_probes),
        cdf = cdf_used
      )
    )
  }

  message(
    "[",
    gse,
    "] Running exact joint per-GSE RMA in bounded-memory streaming mode..."
  )

  expression <- stream_joint_rma(
    cel_files = cel_files,
    template = template,
    common_probes = common_probes,
    state_dir = state_dir
  )

  if (!identical(colnames(expression), common_probes)) {
    stop("Common-probe column order was not preserved.", call. = FALSE)
  }

  n_samples <- nrow(expression)
  n_probes <- ncol(expression)

  message(
    "[",
    gse,
    "] Final matrix: ",
    n_samples,
    " sample(s) x ",
    n_probes,
    " probe(s)."
  )

  message(
    "[",
    gse,
    "] Saving checkpoint: ",
    out_file
  )

  saveRDS(
    expression,
    part_file,
    compress = "gzip"
  )

  if (!file.rename(part_file, out_file)) {
    stop(
      "Could not rename completed temporary output to: ",
      out_file,
      call. = FALSE
    )
  }

  # The final RDS is now durable. Remove only the large, reproducible HDF5
  # working state; completed raw/RMA outputs remain untouched.
  rhdf5::H5close()
  unlink(state_dir, recursive = TRUE, force = TRUE)

  # Release the large objects before processing the next GSE.
  rm(
    template,
    expression
  )
  gc()

  list(
    output = out_file,
    raw_output = raw_out_file,
    raw_summary = "median PM intensity; no background correction; no normalization; no log transform",
    samples = n_samples,
    probes = n_probes,
    cdf = cdf_used
  )
}

# -------------------------------------------------------------------------
# Main loop
# -------------------------------------------------------------------------

archives <- sort(
  list.files(
    RAW_DIR,
    pattern = "_RAW\\.tar$",
    full.names = TRUE,
    ignore.case = TRUE
  )
)

if (!is.null(ONLY_GSES)) {
  wanted <- toupper(ONLY_GSES)
  archive_gses <- toupper(vapply(archives, gse_from_archive, FUN.VALUE = character(1)))
  archives <- archives[archive_gses %in% wanted]
}

if (length(archives) == 0L) {
  stop("No GSE*_RAW.tar archives found in: ", RAW_DIR, call. = FALSE)
}

message("Found ", length(archives), " RAW archive(s).")
message("Output directory: ", OUT_DIR)
message("Temporary directory: ", WORK_ROOT)
message("Manifest: ", MANIFEST_PATH)
message("")
message("Press Ctrl+C at any time. A GSE is skipped once BOTH raw and RMA checkpoints exist.")
message("Only selected CEL members are extracted; CEL.gz files are decompressed in temporary work space and parsed with affxparser.")

for (archive in archives) {
  gse <- gse_from_archive(archive)
  out_file <- file.path(OUT_DIR, paste0(gse, "_rma_common.rds"))
  raw_out_file <- file.path(
    OUT_DIR,
    paste0(gse, "_raw_pm_median_common.rds")
  )

  rma_done <- (
    file.exists(out_file) &&
    !is.na(file.info(out_file)$size) &&
    file.info(out_file)$size > 0
  )

  raw_done <- (
    file.exists(raw_out_file) &&
    !is.na(file.info(raw_out_file)$size) &&
    file.info(raw_out_file)$size > 0
  )

  if (rma_done && raw_done) {
    message(
      "[SKIP] ",
      gse,
      " -> both raw-signal and RMA outputs already exist."
    )
    next
  }

  started_at <- format_time()

  set_manifest(
    gse = gse,
    archive = basename(archive),
    status = "processing",
    output = out_file,
    raw_output = raw_out_file,
    raw_summary = "median PM intensity; no background correction; no normalization; no log transform",
    started_at = started_at,
    msg = "Started"
  )

  result <- tryCatch(
    process_archive(
      archive = archive,
      common_probes = common_probes,
      gsm_mapping = gsm_mapping,
      out_dir = OUT_DIR,
      work_root = WORK_ROOT
    ),
    interrupt = function(e) {
      set_manifest(
        gse = gse,
        archive = basename(archive),
        status = "interrupted",
        output = out_file,
        raw_output = raw_out_file,
        raw_summary = "median PM intensity; no background correction; no normalization; no log transform",
        started_at = started_at,
        finished_at = format_time(),
        msg = "Interrupted by user. Safe to rerun."
      )

      message("")
      message("Interrupted. Temporary files were cleaned up.")
      message("Rerun the same script to resume from the first unfinished GSE.")
      stop(e)
    },
    error = function(e) {
      structure(
        list(message = conditionMessage(e)),
        class = "gse_processing_error"
      )
    }
  )

  if (inherits(result, "gse_processing_error")) {
    message("[FAILED] ", gse, ": ", result$message)

    set_manifest(
      gse = gse,
      archive = basename(archive),
      status = "failed",
      output = out_file,
      raw_output = raw_out_file,
      raw_summary = "median PM intensity; no background correction; no normalization; no log transform",
      started_at = started_at,
      finished_at = format_time(),
      msg = result$message
    )

    gc()
    next
  }

  set_manifest(
    gse = gse,
    archive = basename(archive),
    status = "done",
    samples = result$samples,
    probes = result$probes,
    cdf = result$cdf,
    output = result$output,
    raw_output = result$raw_output,
    raw_summary = result$raw_summary,
    started_at = started_at,
    finished_at = format_time(),
    msg = "Completed successfully"
  )

  message("[DONE] ", gse)
  gc()
}

message("")
message("All currently processable archives have been visited.")
message("See manifest for details: ", MANIFEST_PATH)
