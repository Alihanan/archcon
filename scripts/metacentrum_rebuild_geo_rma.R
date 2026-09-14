#!/usr/bin/env Rscript

# One-command, interruption-safe entry point for rebuilding the CEL-derived
# ArchCon GEO and IKEM matrices. Heavy stage implementations live beside this
# driver so the project root stays clean.

args <- commandArgs(trailingOnly = TRUE)

usage <- function(status = 0L) {
  cat(
    paste0(
      "Usage:\n",
      "  Rscript metacentrum_rebuild_geo_rma.R --work-root DIR --frozen-split CSV ",
      "[--numpy-store DIR] [--ikem-store DIR] [--keep-raw true|false]\n\n",
      "DIR must initially contain download_summary.csv and ",
      "gsm_to_gse_mapping.csv. common_probes.pkl must be in its parent.\n"
    )
  )
  quit(status = status)
}

value_after <- function(flag, default = NULL) {
  pos <- match(flag, args)
  if (is.na(pos)) return(default)
  if (pos == length(args)) stop("Missing value after ", flag, call. = FALSE)
  args[[pos + 1L]]
}

if ("--help" %in% args || "-h" %in% args) usage(0L)

work_root <- value_after("--work-root")
frozen_split <- value_after("--frozen-split")
numpy_store <- value_after("--numpy-store")
ikem_store <- value_after("--ikem-store")
keep_raw <- tolower(value_after("--keep-raw", "false")) %in% c("1", "true", "yes")

if (is.null(work_root) || is.null(frozen_split)) usage(2L)

work_root <- normalizePath(work_root, mustWork = FALSE)
frozen_split <- normalizePath(frozen_split, mustWork = TRUE)

if (!dir.exists(work_root)) dir.create(work_root, recursive = TRUE)

argv <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", argv, value = TRUE)
if (length(file_arg) != 1L) stop("Cannot locate this R script.", call. = FALSE)
script_dir <- dirname(normalizePath(sub("^--file=", "", file_arg)))

required_stages <- file.path(
  script_dir,
  c(
    "metacentrum_geo_download.R",
    "metacentrum_geo_per_gse_rma.R",
    "metacentrum_geo_train_reference_rma.R",
    "metacentrum_ikem_cel.R"
  )
)
missing_stages <- required_stages[!file.exists(required_stages)]
if (length(missing_stages) > 0L) {
  stop("Missing bundled stage file(s): ", paste(missing_stages, collapse = ", "))
}

required_inputs <- c(
  file.path(work_root, "download_summary.csv"),
  file.path(work_root, "gsm_to_gse_mapping.csv"),
  file.path(dirname(work_root), "common_probes.pkl")
)
missing_inputs <- required_inputs[!file.exists(required_inputs)]
if (length(missing_inputs) > 0L) {
  stop("Missing input file(s): ", paste(missing_inputs, collapse = ", "))
}

lock_dir <- file.path(work_root, ".rebuild_geo_rma.lock")
lock_pid_path <- file.path(lock_dir, "pid")
if (dir.exists(lock_dir)) {
  prior_pid <- if (file.exists(lock_pid_path)) {
    suppressWarnings(as.integer(readLines(lock_pid_path, n = 1L, warn = FALSE)))
  } else {
    NA_integer_
  }
  prior_active <- length(prior_pid) == 1L && !is.na(prior_pid) &&
    system2("kill", c("-0", as.character(prior_pid)), stdout = FALSE, stderr = FALSE) == 0L
  if (prior_active) {
    stop("Another rebuild is active with PID ", prior_pid, ".", call. = FALSE)
  }
  message("Removing stale lock from an interrupted earlier invocation.")
  unlink(lock_dir, recursive = TRUE, force = TRUE)
}
if (!dir.create(lock_dir, showWarnings = FALSE)) {
  stop("Could not create rebuild lock: ", lock_dir, call. = FALSE)
}
writeLines(as.character(Sys.getpid()), lock_pid_path)

setwd(work_root)

Sys.setenv(
  ARCHCON_FROZEN_SPLIT = frozen_split,
  ARCHCON_COMMON_PROBES = file.path(dirname(work_root), "common_probes.pkl"),
  ARCHCON_GSM_MAPPING = file.path(work_root, "gsm_to_gse_mapping.csv"),
  ARCHCON_GEO_RAW_DIR = Sys.getenv(
    "ARCHCON_GEO_RAW_DIR",
    unset = file.path(work_root, "GEO_RAW")
  )
)

message("Work root:       ", normalizePath(work_root))
message("Frozen split:    ", frozen_split)
message("Script directory: ", script_dir)
message("Resume policy:   atomic files plus per-GSE/per-block checkpoints")

per_gse_marker <- file.path(work_root, "GEO_RMA", "PER_GSE_RMA_COMPLETE.txt")

if (!file.exists(per_gse_marker)) {
  # Stage 1's historical CSV marker is repaired from actual disk state. This
  # matters after a scratch restart where large RAW archives were not persisted.
  progress_path <- file.path(work_root, "download_summary.csv")
  progress <- utils::read.csv(
    progress_path,
    stringsAsFactors = FALSE,
    check.names = FALSE,
    colClasses = "character"
  )
  marker_col <- if ("raw_downloaded" %in% names(progress)) {
    match("raw_downloaded", names(progress))
  } else if (ncol(progress) >= 5L) {
    5L
  } else {
    NA_integer_
  }
  if (!is.na(marker_col)) {
    for (i in seq_len(nrow(progress))) {
      gse <- toupper(trimws(progress$gse_id[[i]]))
      archive <- file.path(Sys.getenv("ARCHCON_GEO_RAW_DIR"), paste0(gse, "_RAW.tar"))
      if (!file.exists(archive)) progress[[marker_col]][[i]] <- ""
    }
    utils::write.csv(progress, progress_path, row.names = FALSE, na = "")
  }

  message("\n=== Stage 1/4: resumable GEO RAW download ===")
  local(source(required_stages[[1L]], local = TRUE, chdir = FALSE))

  message("\n=== Stage 2/4: exact per-GSE RMA ===")
  local(source(required_stages[[2L]], local = TRUE, chdir = FALSE))

  rma_files <- list.files(
    file.path(work_root, "GEO_RMA"),
    pattern = "^GSE[0-9]+_rma_common\\.rds$"
  )
  raw_files <- list.files(
    file.path(work_root, "GEO_RMA"),
    pattern = "^GSE[0-9]+_raw_pm_median_common\\.rds$"
  )
  if (length(rma_files) == 0L || length(rma_files) != length(raw_files)) {
    stop("Per-GSE stage did not produce a complete paired RDS collection.")
  }
  writeLines(
    paste(Sys.time(), "paired_GSEs=", length(rma_files)),
    per_gse_marker
  )
} else {
  message("Per-GSE RMA marker found; stages 1 and 2 are already complete.")
}

message("\n=== Stage 3/4: exact train-reference RMA and HDF5 store ===")
local(source(required_stages[[3L]], local = TRUE, chdir = FALSE))

local(source(required_stages[[4L]], local = TRUE, chdir = FALSE))

write_npy_float32 <- function(h5_path, dataset, output_path, n_rows, n_cols) {
  part_path <- paste0(output_path, ".part")
  if (file.exists(part_path)) unlink(part_path, force = TRUE)
  con <- file(part_path, open = "wb")
  ok <- FALSE
  on.exit({
    if (inherits(con, "connection") && isOpen(con)) close(con)
    if (!ok && file.exists(part_path)) unlink(part_path, force = TRUE)
  }, add = TRUE)

  magic <- as.raw(c(0x93, as.integer(charToRaw("NUMPY")), 0x01, 0x00))
  header_core <- sprintf(
    "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }",
    n_rows,
    n_cols
  )
  padding <- (16L - ((10L + nchar(header_core, type = "bytes") + 1L) %% 16L)) %% 16L
  header <- paste0(header_core, strrep(" ", padding), "\n")

  writeBin(magic, con)
  writeBin(as.integer(nchar(header, type = "bytes")), con, size = 2L, endian = "little")
  writeBin(charToRaw(header), con)

  batch_size <- 16L
  for (start in seq.int(1L, n_rows, by = batch_size)) {
    stop_row <- min(n_rows, start + batch_size - 1L)
    block <- rhdf5::h5read(
      h5_path,
      dataset,
      index = list(start:stop_row, seq_len(n_cols)),
      native = TRUE
    )
    writeBin(as.numeric(t(block)), con, size = 4L, endian = "little")
    if (start == 1L || stop_row %% 256L == 0L || stop_row == n_rows) {
      message("NumPy export: ", stop_row, "/", n_rows, " samples.")
    }
  }

  close(con)
  ok <- TRUE
  if (!file.rename(part_path, output_path)) {
    stop("Could not atomically install NumPy matrix: ", output_path)
  }
}

if (!is.null(numpy_store)) {
  numpy_store <- normalizePath(numpy_store, mustWork = FALSE)
  dir.create(numpy_store, recursive = TRUE, showWarnings = FALSE)

  built_samples <- utils::read.csv(
    file.path(work_root, "GEO_MATRIX_STORE", "sample_index.csv"),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  built_probes <- utils::read.csv(
    file.path(work_root, "GEO_MATRIX_STORE", "probe_index.csv"),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )

  target_sample_path <- file.path(numpy_store, "sample_index.csv")
  target_probe_path <- file.path(numpy_store, "probe_index.csv")
  if (file.exists(target_sample_path) && file.exists(target_probe_path)) {
    target_samples <- utils::read.csv(
      target_sample_path,
      stringsAsFactors = FALSE,
      check.names = FALSE
    )
    target_probes <- utils::read.csv(
      target_probe_path,
      stringsAsFactors = FALSE,
      check.names = FALSE
    )
    sample_columns <- intersect(c("GSM", "sample_id", "Sample_ID"), names(target_samples))
    probe_columns <- intersect(c("probe_id", "probe_set_id", "probeset_id"), names(target_probes))
    if (length(sample_columns) == 0L || length(probe_columns) == 0L) {
      stop("Existing GEO_NUMPY_STORE metadata lacks sample/probe identifier columns.")
    }
    sample_col <- sample_columns[[1L]]
    probe_col <- probe_columns[[1L]]
    if (!identical(
      toupper(trimws(as.character(target_samples[[sample_col]]))),
      toupper(trimws(as.character(built_samples$GSM)))
    )) {
      stop("Existing GEO_NUMPY_STORE sample order differs; refusing replacement.")
    }
    if (!identical(
      as.character(target_probes[[probe_col]]),
      as.character(built_probes$probe_id)
    )) {
      stop("Existing GEO_NUMPY_STORE probe order differs; refusing replacement.")
    }
  } else {
    utils::write.csv(built_samples, target_sample_path, row.names = FALSE, na = "")
    utils::write.csv(built_probes, target_probe_path, row.names = FALSE, na = "")
  }

  write_npy_float32(
    file.path(work_root, "GEO_MATRIX_STORE", "geo_expression_store.h5"),
    "expression/rma_global",
    file.path(numpy_store, "rma_global.npy"),
    nrow(built_samples),
    nrow(built_probes)
  )
  utils::write.csv(
    data.frame(
      method = "CEL-level train-reference RMA",
      frozen_split = frozen_split,
      train_samples = sum(built_samples$pretraining_split == "train"),
      validation_samples = sum(built_samples$pretraining_split == "validation"),
      test_samples = sum(built_samples$pretraining_split == "test"),
      created_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
      stringsAsFactors = FALSE
    ),
    file.path(numpy_store, "rma_global_train_reference_provenance.csv"),
    row.names = FALSE
  )
  message("Installed scratch NumPy matrix: ", file.path(numpy_store, "rma_global.npy"))
}

if (!is.null(ikem_store)) {
  ikem_store <- normalizePath(ikem_store, mustWork = FALSE)
  dir.create(ikem_store, recursive = TRUE, showWarnings = FALSE)

  built_samples <- utils::read.csv(
    file.path(work_root, "IKEM_MATRIX_STORE", "sample_index.csv"),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  built_probes <- utils::read.csv(
    file.path(work_root, "IKEM_MATRIX_STORE", "probe_index.csv"),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  target_sample_path <- file.path(ikem_store, "sample_index.csv")
  target_probe_path <- file.path(ikem_store, "probe_index.csv")

  if (file.exists(target_sample_path)) {
    target_samples <- utils::read.csv(
      target_sample_path,
      stringsAsFactors = FALSE,
      check.names = FALSE
    )
    sample_columns <- intersect(
      c("sample_id", "Sample_ID", "GSM", "sample", "id"),
      names(target_samples)
    )
    if (length(sample_columns) == 0L || !identical(
      toupper(trimws(as.character(target_samples[[sample_columns[[1L]]]]))),
      toupper(trimws(as.character(built_samples$sample_id)))
    )) {
      stop("Existing IKEM store sample order differs from GSE290167.")
    }
  }
  if (file.exists(target_probe_path)) {
    target_probes <- utils::read.csv(
      target_probe_path,
      stringsAsFactors = FALSE,
      check.names = FALSE
    )
    probe_columns <- intersect(
      c("probe_id", "probe", "probeset_id", "ID", "id"),
      names(target_probes)
    )
    if (length(probe_columns) == 0L || !identical(
      as.character(target_probes[[probe_columns[[1L]]]]),
      as.character(built_probes$probe_id)
    )) {
      stop("Existing IKEM store probe order differs from the common-probe order.")
    }
  }

  # Correspondence was validated before any expression matrix was installed.
  # Preserve the established local sample spelling while adding GSM provenance.
  utils::write.csv(built_samples, target_sample_path, row.names = FALSE, na = "")
  utils::write.csv(built_probes, target_probe_path, row.names = FALSE, na = "")
  file.copy(
    file.path(work_root, "IKEM_MATRIX_STORE", "ikem_gse290167_correspondence.csv"),
    file.path(ikem_store, "ikem_gse290167_correspondence.csv"),
    overwrite = TRUE
  )
  file.copy(
    file.path(work_root, "IKEM_MATRIX_STORE", "ikem_rma_reference_samples.csv"),
    file.path(ikem_store, "ikem_rma_reference_samples.csv"),
    overwrite = TRUE
  )

  ikem_h5 <- file.path(
    work_root,
    "IKEM_MATRIX_STORE",
    "ikem_expression_store.h5"
  )
  for (method in c(
    "raw_original",
    "rma_cohort_legacy",
    "rma_per_gse",
    "rma_global"
  )) {
    write_npy_float32(
      ikem_h5,
      paste0("expression/", method),
      file.path(ikem_store, paste0(method, ".npy")),
      nrow(built_samples),
      nrow(built_probes)
    )
  }
  message("Installed exact CEL-derived IKEM matrices in: ", ikem_store)
}

if (!keep_raw && file.exists(file.path("GEO_MATRIX_STORE", "GLOBAL_RMA_COMPLETE.txt"))) {
  raw_archives <- list.files(
    Sys.getenv("ARCHCON_GEO_RAW_DIR"),
    pattern = "_RAW\\.tar$",
    full.names = TRUE
  )
  if (length(raw_archives) > 0L) {
    unlink(raw_archives, force = TRUE)
    message("Removed ", length(raw_archives), " completed RAW archives (--keep-raw false).")
  }
}

message("\nRebuild complete. Main store: ", file.path(work_root, "GEO_MATRIX_STORE"))
unlink(lock_dir, recursive = TRUE, force = TRUE)
