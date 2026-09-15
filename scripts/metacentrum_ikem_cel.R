#!/usr/bin/env Rscript

# Build leakage-free IKEM encoder inputs from the original Stadniuk CEL files.
# Public GSE290167 CELs are used only when a matching private CEL is absent.
# The script works in small blocks and resumes after interruption.
#
# The driver calls this file twice:
#   prepare   checks the private CEL collection, fits IKEM-local RMA from every
#             biopsy without a measured longitudinal eGFR, and caches those
#             same TRAIN arrays for the
#             combined GEO+IKEM global reference;
#   finalize  applies the completed combined global reference to every IKEM
#             array without refitting it.
#
#   raw_original   original PM intensity, median within each common probe set
#   rma_per_gse    an IKEM reference fitted only on outcome-free training
#                  samples, then frozen for every held-out IKEM sample
#   rma_global     each IKEM CEL transformed with the target and probe effects
#                  learned from GEO TRAIN plus IKEM no-eGFR TRAIN arrays
#
# eGFR values are read only to make the binary TRAIN-versus-HELD-OUT gate.
# Their numeric values never enter normalization or model fitting.

IKEM_GSE <- "GSE290167"
IKEM_EXPECTED_PLATFORM <- "GPL15207"
IKEM_PUBLIC_EXPECTED_SAMPLES <- 276L
IKEM_EXPECTED_SAMPLES <- as.integer(Sys.getenv(
  "ARCHCON_IKEM_EXPECTED_SAMPLES",
  unset = "288"
))
IKEM_EXPECTED_NO_EGFR_TRAIN <- as.integer(Sys.getenv(
  "ARCHCON_IKEM_EXPECTED_NO_EGFR_TRAIN",
  unset = "34"
))
if (is.na(IKEM_EXPECTED_SAMPLES) || IKEM_EXPECTED_SAMPLES < 2L ||
    is.na(IKEM_EXPECTED_NO_EGFR_TRAIN) ||
    IKEM_EXPECTED_NO_EGFR_TRAIN < 2L ||
    IKEM_EXPECTED_NO_EGFR_TRAIN >= IKEM_EXPECTED_SAMPLES) {
  stop(
    "Invalid ARCHCON_IKEM_EXPECTED_SAMPLES or ",
    "ARCHCON_IKEM_EXPECTED_NO_EGFR_TRAIN.",
    call. = FALSE
  )
}
IKEM_URL <- paste0(
  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE290nnn/",
  IKEM_GSE,
  "/suppl/",
  IKEM_GSE,
  "_RAW.tar"
)
IKEM_METADATA_URL <- paste0(
  "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?targ=gsm&acc=",
  IKEM_GSE,
  "&form=text&view=brief"
)

IKEM_RAW_DIR <- Sys.getenv(
  "ARCHCON_IKEM_RAW_DIR",
  unset = file.path(dirname(getwd()), "IKEM_CEL")
)
IKEM_PRIVATE_CEL_DIR <- Sys.getenv(
  "ARCHCON_IKEM_PRIVATE_CEL_DIR",
  unset = file.path(IKEM_RAW_DIR, "STADNIUK_LEGACY_CEL")
)
IKEM_SAMPLE_METADATA <- Sys.getenv("ARCHCON_IKEM_SAMPLE_METADATA", unset = "")
IKEM_EGFR_TABLE <- Sys.getenv("ARCHCON_IKEM_EGFR_TABLE", unset = "")
IKEM_CLASSIFIER_TABLE <- Sys.getenv("ARCHCON_IKEM_CLASSIFIER_TABLE", unset = "")
IKEM_LEGACY_STORE <- Sys.getenv("ARCHCON_IKEM_LEGACY_STORE", unset = "")
IKEM_OUT_DIR <- "IKEM_MATRIX_STORE"
IKEM_WORK_DIR <- ".IKEM_CEL_WORK"
IKEM_PROGRESS_DIR <- file.path(IKEM_WORK_DIR, "progress")
IKEM_EXTRACT_DIR <- file.path(IKEM_WORK_DIR, "extracted")
IKEM_ARCHIVE <- file.path(IKEM_RAW_DIR, paste0(IKEM_GSE, "_RAW.tar"))
IKEM_METADATA <- file.path(IKEM_RAW_DIR, paste0(IKEM_GSE, "_samples_brief.soft"))
IKEM_FINAL_H5 <- file.path(IKEM_OUT_DIR, "ikem_expression_store.h5")
IKEM_PROBE_H5 <- file.path(IKEM_WORK_DIR, "ikem_normalized_common_pm.h5")
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
IKEM_LOCAL_PARAMETERS <- file.path(
  IKEM_WORK_DIR,
  "ikem_train_reference_rma_parameters.h5"
)
IKEM_COMPLETE <- file.path(IKEM_OUT_DIR, "IKEM_CEL_PREPROCESSING_COMPLETE.txt")
IKEM_LOCAL_COMPLETE <- file.path(
  IKEM_OUT_DIR,
  "IKEM_LOCAL_RMA_COMPLETE.txt"
)
IKEM_SIGNATURE_FILE <- file.path(IKEM_WORK_DIR, "input_signature.txt")
IKEM_GLOBAL_SIGNATURE_FILE <- file.path(
  IKEM_WORK_DIR,
  "combined_global_reference_signature.txt"
)
IKEM_GLOBAL_PROGRESS <- file.path(
  IKEM_PROGRESS_DIR,
  "global_done_sample.txt"
)
IKEM_PIPELINE_VERSION <- "5-private-cel-outcome-gated-reference"

IKEM_MODE <- tolower(trimws(Sys.getenv("ARCHCON_IKEM_MODE", unset = "all")))
if (!(IKEM_MODE %in% c("prepare", "finalize", "all"))) {
  stop(
    "ARCHCON_IKEM_MODE must be prepare, finalize, or all; received: ",
    IKEM_MODE,
    call. = FALSE
  )
}
DO_LOCAL_PREPARATION <- IKEM_MODE %in% c("prepare", "all")
DO_GLOBAL_FINALIZATION <- IKEM_MODE %in% c("finalize", "all")

GLOBAL_WORK_DIR <- ".GLOBAL_RMA_WORK"
GLOBAL_TEMPLATE <- file.path(GLOBAL_WORK_DIR, "template_probe_index_info.rds")
GLOBAL_TARGET <- file.path(GLOBAL_WORK_DIR, "train_reference_quantile_target.rds")
GLOBAL_PARAMETERS <- file.path(
  GLOBAL_WORK_DIR,
  "train_reference_rma_parameters.h5"
)
GLOBAL_PARAMETERS_COMPLETE <- file.path(
  GLOBAL_WORK_DIR,
  "TRAIN_REFERENCE_PARAMETERS_COMPLETE.txt"
)
GLOBAL_PARAMETER_SIGNATURE <- file.path(
  GLOBAL_WORK_DIR,
  "train_reference_rma_parameter_signature.rds"
)
FROZEN_SPLIT_CSV <- Sys.getenv("ARCHCON_FROZEN_SPLIT", unset = "")

IKEM_ARRAY_BATCH <- 8L
IKEM_PROBESET_BLOCK <- 128L
IKEM_H5_LEVEL <- 4L

required_packages <- c(
  "affxparser", "preprocessCore", "rhdf5", "R.utils", "readxl"
)
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, quietly = TRUE, FUN.VALUE = logical(1))
]
if (length(missing_packages) > 0L) {
  stop(
    "Missing R packages required for IKEM CEL preprocessing: ",
    paste(missing_packages, collapse = ", "),
    call. = FALSE
  )
}
if (!nzchar(Sys.which("curl")) || !nzchar(Sys.which("tar"))) {
  stop("Both curl and tar must be available on PATH.", call. = FALSE)
}

Sys.setenv(
  OMP_NUM_THREADS = "1",
  OPENBLAS_NUM_THREADS = "1",
  MKL_NUM_THREADS = "1",
  VECLIB_MAXIMUM_THREADS = "1",
  NUMEXPR_NUM_THREADS = "1"
)

dir.create(IKEM_RAW_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(IKEM_OUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(IKEM_PROGRESS_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(IKEM_EXTRACT_DIR, recursive = TRUE, showWarnings = FALSE)

ikem_time <- function() format(Sys.time(), "%Y-%m-%d %H:%M:%S %z")

atomic_save_rds_ikem <- function(object, path, compress = TRUE) {
  partial <- paste0(path, ".part")
  unlink(partial, force = TRUE)
  saveRDS(object, partial, compress = compress)
  if (!file.rename(partial, path)) {
    stop("Could not atomically install ", path, call. = FALSE)
  }
}

atomic_write_csv_ikem <- function(object, path) {
  partial <- paste0(path, ".part")
  unlink(partial, force = TRUE)
  utils::write.csv(object, partial, row.names = FALSE, na = "")
  if (!file.rename(partial, path)) {
    stop("Could not atomically install ", path, call. = FALSE)
  }
}

atomic_write_lines_ikem <- function(object, path) {
  partial <- paste0(path, ".part")
  unlink(partial, force = TRUE)
  writeLines(object, partial)
  if (!file.rename(partial, path)) {
    stop("Could not atomically install ", path, call. = FALSE)
  }
}

read_done_ikem <- function(path) {
  if (!file.exists(path)) return(character())
  unique(trimws(readLines(path, warn = FALSE)))
}

append_done_ikem <- function(path, value) {
  current <- read_done_ikem(path)
  if (!(value %in% current)) {
    atomic_write_lines_ikem(c(current, value), path)
  }
}

download_atomic <- function(url, destination) {
  if (file.exists(destination) && file.info(destination)$size > 0) {
    return(invisible(destination))
  }
  partial <- paste0(destination, ".part")
  for (attempt in seq_len(20L)) {
    args <- c(
      "--location", "--fail", "--retry", "4", "--retry-delay", "5",
      "--retry-connrefused", "--connect-timeout", "30",
      "--continue-at", "-", "--output", shQuote(partial), shQuote(url)
    )
    status <- system2("curl", args = args)
    if (identical(as.integer(status), 0L) && file.exists(partial) &&
        file.info(partial)$size > 0) {
      if (!file.rename(partial, destination)) {
        stop("Could not finalize download: ", destination, call. = FALSE)
      }
      return(invisible(destination))
    }
    # curl 33 means that this server rejected a ranged resume. Restart once
    # from byte zero rather than retrying the same impossible operation.
    if (identical(as.integer(status), 33L)) unlink(partial, force = TRUE)
    if (attempt < 20L) Sys.sleep(10)
  }
  stop("Download failed after 20 attempts: ", url, call. = FALSE)
}

tar_members_checked <- function(archive) {
  output <- suppressWarnings(
    system2("tar", c("-tf", shQuote(archive)), stdout = TRUE, stderr = TRUE)
  )
  status <- attr(output, "status")
  if (is.null(status)) status <- 0L
  if (status != 0L) {
    stop(
      "Downloaded IKEM RAW archive failed tar integrity validation: ",
      archive,
      "\n",
      paste(tail(output, 10L), collapse = "\n"),
      call. = FALSE
    )
  }
  output
}

soft_field <- function(lines, prefix, required = TRUE) {
  value <- sub(prefix, "", lines[grepl(prefix, lines, perl = TRUE)])
  if (length(value) != 1L && required) {
    stop("Expected exactly one GEO metadata field matching ", prefix, call. = FALSE)
  }
  if (length(value) == 0L) return(NA_character_)
  trimws(value[[1L]])
}

parse_ikem_metadata <- function(path) {
  lines <- readLines(path, warn = FALSE, encoding = "UTF-8")
  starts <- grep("^\\^SAMPLE = ", lines)
  if (length(starts) != IKEM_PUBLIC_EXPECTED_SAMPLES) {
    stop(
      "GEO metadata contains ", length(starts), " samples; expected ",
      IKEM_PUBLIC_EXPECTED_SAMPLES, ".", call. = FALSE
    )
  }
  ends <- c(starts[-1L] - 1L, length(lines))
  records <- lapply(seq_along(starts), function(i) {
    block <- lines[starts[[i]]:ends[[i]]]
    gsm <- soft_field(block, "^\\^SAMPLE = ")
    title <- soft_field(block, "^!Sample_title = ")
    sample_id <- sub(
      "^donor procurement biopsy \\[([^]]+)\\]$",
      "\\1",
      title,
      perl = TRUE
    )
    if (identical(sample_id, title)) {
      stop("Unexpected GSE290167 sample title: ", title, call. = FALSE)
    }
    platform <- soft_field(block, "^!Sample_platform_id = ")
    supplementary <- soft_field(block, "^!Sample_supplementary_file = ")
    patient_line <- soft_field(
      block,
      "^!Sample_characteristics_ch1 = patient id: ",
      required = FALSE
    )
    tissue <- soft_field(
      block,
      "^!Sample_characteristics_ch1 = tissue: ",
      required = FALSE
    )
    data.frame(
      sample_id_geo = sample_id,
      sample_key_upper = toupper(sample_id),
      GSM = toupper(gsm),
      platform = platform,
      patient_id_geo = patient_line,
      tissue_geo = tissue,
      supplementary_url = supplementary,
      raw_member = basename(sub("^ftp://", "https://", supplementary)),
      stringsAsFactors = FALSE
    )
  })
  result <- do.call(rbind, records)
  rownames(result) <- NULL
  if (anyDuplicated(result$GSM) || anyDuplicated(result$sample_key_upper)) {
    stop("GSE290167 metadata contains duplicate GSM or biopsy identifiers.", call. = FALSE)
  }
  if (any(result$platform != IKEM_EXPECTED_PLATFORM)) {
    stop("GSE290167 contains a non-PrimeView platform entry.", call. = FALSE)
  }
  result
}

normalize_legacy_id <- function(x) {
  x <- basename(trimws(as.character(x)))
  x <- sub("\\.CEL(\\.gz)?$", "", x, ignore.case = TRUE, perl = TRUE)
  x <- sub("^GSM[0-9]+[_-]*", "", x, ignore.case = TRUE, perl = TRUE)
  x <- sub(
    "[_[:space:]-]*\\(?PrimeView\\)?[_[:space:]-]*$",
    "",
    x,
    ignore.case = TRUE,
    perl = TRUE
  )
  x <- sub("_+$", "", x, perl = TRUE)
  toupper(trimws(x))
}

column_named <- function(table, wanted, description) {
  position <- match(tolower(wanted), tolower(names(table)))
  if (is.na(position)) {
    stop(description, " has no column named ", wanted, ".", call. = FALSE)
  }
  names(table)[[position]]
}

read_canonical_ikem_samples <- function(path) {
  if (!nzchar(path) || !file.exists(path)) {
    stop("Canonical IKEM sample metadata is missing: ", path, call. = FALSE)
  }
  table <- utils::read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  id_column <- column_named(table, "sample_ID", "sample_metadata.csv")
  ids <- normalize_legacy_id(table[[id_column]])
  keys <- toupper(ids)
  if (nrow(table) != IKEM_EXPECTED_SAMPLES || anyDuplicated(keys) ||
      any(!grepl("^D[0-9]+_[LP]$", keys))) {
    stop(
      "sample_metadata.csv must contain exactly ", IKEM_EXPECTED_SAMPLES,
      " unique biopsy IDs of the form D<number>_L or D<number>_P.",
      call. = FALSE
    )
  }
  data.frame(
    sample_id = keys,
    sample_key_upper = keys,
    canonical_row_python = seq_along(keys) - 1L,
    stringsAsFactors = FALSE
  )
}

check_classifier_ids <- function(path, canonical) {
  if (!nzchar(path) || !file.exists(path)) return(invisible(NULL))
  table <- as.data.frame(
    readxl::read_excel(path, sheet = 1L),
    stringsAsFactors = FALSE
  )
  id_column <- column_named(table, "Sample_ID", basename(path))
  keys <- normalize_legacy_id(table[[id_column]])
  if (anyDuplicated(keys) || !setequal(keys, canonical$sample_key_upper)) {
    stop(
      "Klasifikator workbook and sample_metadata.csv do not contain the same ",
      "biopsy IDs. This workbook is used only as an ID cross-check.",
      call. = FALSE
    )
  }
  invisible(NULL)
}

read_egfr_availability <- function(path, canonical) {
  if (!nzchar(path) || !file.exists(path)) {
    stop("IKEM eGFR workbook is missing: ", path, call. = FALSE)
  }
  table <- as.data.frame(
    readxl::read_excel(path, sheet = 1L),
    stringsAsFactors = FALSE
  )
  id_column <- column_named(table, "patient", basename(path))
  outcome_names <- c("egfr_7d", "egfr_3m", "egfr_6m", "egfr_12m")
  outcome_columns <- vapply(
    outcome_names,
    function(name) column_named(table, name, basename(path)),
    character(1)
  )
  outcome_ids <- normalize_legacy_id(table[[id_column]])
  if (anyDuplicated(outcome_ids)) {
    stop("egfr_data.xlsx contains duplicate biopsy IDs.", call. = FALSE)
  }
  unknown <- setdiff(outcome_ids, canonical$sample_key_upper)
  if (length(unknown) > 0L) {
    stop(
      "egfr_data.xlsx contains biopsy IDs absent from sample_metadata.csv: ",
      paste(head(unknown, 10L), collapse = ", "),
      call. = FALSE
    )
  }

  measured <- vapply(seq_len(nrow(table)), function(row) {
    values <- suppressWarnings(as.numeric(unlist(
      table[row, outcome_columns, drop = FALSE],
      use.names = FALSE
    )))
    any(is.finite(values))
  }, logical(1))
  has_egfr <- rep(FALSE, nrow(canonical))
  position <- match(outcome_ids, canonical$sample_key_upper)
  has_egfr[position] <- measured
  data.frame(
    sample_key_upper = canonical$sample_key_upper,
    appears_in_egfr_workbook = canonical$sample_key_upper %in% outcome_ids,
    has_measured_egfr = has_egfr,
    stringsAsFactors = FALSE
  )
}

find_private_cels <- function(directory, canonical) {
  if (!dir.exists(directory)) {
    stop("Private IKEM CEL directory is missing: ", directory, call. = FALSE)
  }
  paths <- list.files(
    directory,
    pattern = "\\.CEL(\\.gz)?$",
    recursive = TRUE,
    full.names = TRUE,
    ignore.case = TRUE
  )
  keys <- normalize_legacy_id(paths)
  keep <- keys %in% canonical$sample_key_upper
  ignored <- paths[!keep]
  paths <- paths[keep]
  keys <- keys[keep]
  if (anyDuplicated(keys)) {
    duplicated_keys <- unique(keys[duplicated(keys) | duplicated(keys, fromLast = TRUE)])
    stop(
      "The private CEL folder contains more than one CEL for: ",
      paste(head(duplicated_keys, 10L), collapse = ", "),
      call. = FALSE
    )
  }
  if (length(ignored) > 0L) {
    message(
      "Ignoring ", length(ignored),
      " CEL file(s) whose IDs are absent from sample_metadata.csv."
    )
  }
  aligned <- paths[match(canonical$sample_key_upper, keys)]
  data.frame(
    sample_key_upper = canonical$sample_key_upper,
    private_cel_path = aligned,
    has_private_cel = !is.na(aligned),
    stringsAsFactors = FALSE
  )
}

ikem_training_reference <- function(metadata) {
  positions <- which(!metadata$has_measured_egfr)
  if (length(positions) != IKEM_EXPECTED_NO_EGFR_TRAIN) {
    stop(
      "The outcome gate found ", length(positions),
      " IKEM samples without measured longitudinal eGFR; expected ",
      IKEM_EXPECTED_NO_EGFR_TRAIN, ". Do not continue until egfr_data.xlsx ",
      "and sample_metadata.csv are verified (or explicitly set ",
      "ARCHCON_IKEM_EXPECTED_NO_EGFR_TRAIN).",
      call. = FALSE
    )
  }
  data.frame(
    reference_order = seq_along(positions),
    sample_id = metadata$sample_id[positions],
    sample_key_upper = metadata$sample_key_upper[positions],
    source = metadata$cel_source[positions],
    # Stable internal keys are used because 12 private CELs are not public GSMs.
    GSM = paste0("IKEM_", metadata$sample_id[positions]),
    public_GSM = metadata$GSM[positions],
    gse290167_row_r = positions,
    gse290167_row_python = positions - 1L,
    training_role = "train_no_measured_egfr",
    outcome_status = "no measured longitudinal eGFR",
    stringsAsFactors = FALSE
  )
}

if (DO_LOCAL_PREPARATION) {
  message("\n=== Stage 2B/4: IKEM train-only per-dataset RMA ===")
} else {
  message("\n=== Stage 4/4: apply combined global RMA to IKEM ===")
}

# The local IKEM calculation only needs the shared PrimeView probe template.
# The combined target and probe effects do not exist yet at that point.  They
# are required only by the second (finalize) call.
required_geo_reference <- if (DO_GLOBAL_FINALIZATION) {
  c(
    GLOBAL_TEMPLATE,
    GLOBAL_TARGET,
    GLOBAL_PARAMETERS,
    GLOBAL_PARAMETERS_COMPLETE,
    GLOBAL_PARAMETER_SIGNATURE
  )
} else {
  GLOBAL_TEMPLATE
}
missing_geo_reference <- required_geo_reference[!file.exists(required_geo_reference)]
if (length(missing_geo_reference) > 0L) {
  stop(
    "IKEM preprocessing cannot continue because a required reference artifact is missing.\n",
    "Missing: ", paste(missing_geo_reference, collapse = ", "), "\n",
    if (DO_LOCAL_PREPARATION) {
      "The GEO stage must first create the common PrimeView probe template."
    } else {
      paste0(
        "Rerun Stage 3 to finish the combined GEO+IKEM TRAIN target and ",
        "probe effects before finalizing IKEM."
      )
    },
    call. = FALSE
  )
}
if (!nzchar(FROZEN_SPLIT_CSV) || !file.exists(FROZEN_SPLIT_CSV)) {
  stop(
    "Phase 4 needs the same frozen sample split used by Phase 3: ",
    FROZEN_SPLIT_CSV,
    call. = FALSE
  )
}

# The common template is shared by the local and global methods.
template <- readRDS(GLOBAL_TEMPLATE)
common_probes <- as.character(template$common_probes)

global_target <- NULL
global_effect <- NULL
global_signature <- NULL
if (DO_GLOBAL_FINALIZATION) {
  global_target <- readRDS(GLOBAL_TARGET)
  global_signature <- readRDS(GLOBAL_PARAMETER_SIGNATURE)
  required_signature_fields <- c(
    "format_version", "frozen_split_md5", "template_md5", "target_md5",
    "sample_ids", "sample_splits", "common_probes", "n_common_pm",
    "ikem_train_sample_ids", "ikem_train_gsms", "fit_sample_count"
  )
  missing_signature_fields <- setdiff(
    required_signature_fields,
    names(global_signature)
  )
  if (length(missing_signature_fields) > 0L) {
    stop(
      "The combined-reference provenance is incomplete (missing: ",
      paste(missing_signature_fields, collapse = ", "), "). Rerun Stage 3.",
      call. = FALSE
    )
  }

  reference_hashes_match <- identical(
    as.character(global_signature$frozen_split_md5),
    as.character(unname(tools::md5sum(FROZEN_SPLIT_CSV)))
  ) && identical(
    as.character(global_signature$template_md5),
    as.character(unname(tools::md5sum(GLOBAL_TEMPLATE)))
  ) && identical(
    as.character(global_signature$target_md5),
    as.character(unname(tools::md5sum(GLOBAL_TARGET)))
  )
  reference_shape_matches <- identical(
    as.integer(global_signature$format_version),
    2L
  ) && identical(
    as.integer(global_signature$n_common_pm),
    as.integer(template$n_common_pm)
  ) && identical(
    as.character(global_signature$common_probes),
    common_probes
  )
  if (!reference_hashes_match || !reference_shape_matches) {
    stop(
      "The combined GEO+IKEM TRAIN reference does not match the current ",
      "split/template/target. Rerun Stage 3 with this frozen split.",
      call. = FALSE
    )
  }

  global_effect <- tryCatch(
    as.numeric(rhdf5::h5read(
      GLOBAL_PARAMETERS,
      "probe_effect_common_pm",
      native = TRUE
    )),
    error = function(error) {
      stop(
        "Could not read the combined-reference probe effects: ",
        conditionMessage(error), ". Rerun Stage 3.",
        call. = FALSE
      )
    }
  )
  if (length(global_target) != template$n_all_pm ||
      length(global_effect) != template$n_common_pm ||
      any(!is.finite(global_target)) || any(!is.finite(global_effect))) {
    stop(
      "The combined TRAIN reference contains missing, non-finite, or wrongly ",
      "sized values. Rerun Stage 3.",
      call. = FALSE
    )
  }

  parameter_marker <- paste(
    readLines(GLOBAL_PARAMETERS_COMPLETE, warn = FALSE),
    collapse = " "
  )
  expected_marker_parts <- c(
    paste0("pm_effects=", template$n_common_pm),
    paste0("signature_md5=", unname(tools::md5sum(GLOBAL_PARAMETER_SIGNATURE)))
  )
  if (any(!vapply(
    expected_marker_parts,
    grepl,
    logical(1),
    x = parameter_marker,
    fixed = TRUE
  ))) {
    stop(
      "The combined-reference completion marker does not match its files. ",
      "Rerun Stage 3.",
      call. = FALSE
    )
  }
  message(
    "Combined GEO+IKEM TRAIN reference verified: ",
    format(global_signature$fit_sample_count, big.mark = ","),
    " training arrays and ",
    format(length(global_effect), big.mark = ","),
    " probe effects."
  )
}

# sample_metadata.csv defines the complete 288-biopsy collection and its order.
# The private Stadniuk CEL directory is authoritative. Public GSE290167 is a
# byte-level CEL fallback, not a second cohort and not a source of outcomes.
canonical <- read_canonical_ikem_samples(IKEM_SAMPLE_METADATA)
check_classifier_ids(IKEM_CLASSIFIER_TABLE, canonical)
private <- find_private_cels(IKEM_PRIVATE_CEL_DIR, canonical)
availability <- read_egfr_availability(IKEM_EGFR_TABLE, canonical)

public <- NULL
missing_private <- canonical$sample_key_upper[!private$has_private_cel]
if (length(missing_private) > 0L) {
  if (!file.exists(IKEM_METADATA)) {
    message(
      length(missing_private),
      " private CEL(s) are absent; downloading the small GSE290167 metadata file."
    )
    download_atomic(IKEM_METADATA_URL, IKEM_METADATA)
  }
  public <- parse_ikem_metadata(IKEM_METADATA)
} else if (file.exists(IKEM_METADATA)) {
  public <- tryCatch(
    parse_ikem_metadata(IKEM_METADATA),
    error = function(error) {
      message(
        "Optional public metadata could not be parsed and will be ignored ",
        "because all private CELs are present: ", conditionMessage(error)
      )
      NULL
    }
  )
}

metadata <- canonical
metadata$sample_id_geo <- NA_character_
metadata$GSM <- NA_character_
metadata$patient_id_geo <- NA_character_
metadata$tissue_geo <- NA_character_
metadata$platform <- IKEM_EXPECTED_PLATFORM
metadata$supplementary_url <- NA_character_
metadata$raw_member <- NA_character_
if (!is.null(public)) {
  public_position <- match(metadata$sample_key_upper, public$sample_key_upper)
  matched <- which(!is.na(public_position))
  for (column in c(
    "sample_id_geo", "GSM", "patient_id_geo", "tissue_geo", "platform",
    "supplementary_url", "raw_member"
  )) {
    metadata[matched, column] <- public[public_position[matched], column]
  }
}
metadata$private_cel_path <- private$private_cel_path
metadata$has_private_cel <- private$has_private_cel
metadata$cel_source <- ifelse(
  metadata$has_private_cel,
  "STADNIUK_LEGACY_CEL",
  IKEM_GSE
)

# Download and validate the public RAW archive only if it is actually needed.
members <- character()
if (length(missing_private) > 0L) {
  not_public <- missing_private[
    !(missing_private %in% public$sample_key_upper)
  ]
  if (length(not_public) > 0L) {
    stop(
      "Private CELs are missing for IDs that GSE290167 cannot supply: ",
      paste(head(not_public, 10L), collapse = ", "),
      call. = FALSE
    )
  }
  download_atomic(IKEM_URL, IKEM_ARCHIVE)
  members <- tryCatch(
    tar_members_checked(IKEM_ARCHIVE),
    error = function(error) {
      message(
        "Existing GSE290167 RAW archive is incomplete/corrupt; deleting it and ",
        "downloading one clean copy. Reason: ", conditionMessage(error)
      )
      unlink(c(IKEM_ARCHIVE, paste0(IKEM_ARCHIVE, ".part")), force = TRUE)
      download_atomic(IKEM_URL, IKEM_ARCHIVE)
      tar_members_checked(IKEM_ARCHIVE)
    }
  )
  cel_members <- members[grepl("\\.CEL(\\.gz)?$", members, ignore.case = TRUE)]
  if (length(cel_members) != IKEM_PUBLIC_EXPECTED_SAMPLES ||
      anyDuplicated(cel_members)) {
    stop(
      "GSE290167 RAW archive contains ", length(cel_members),
      " unique CEL members; expected ", IKEM_PUBLIC_EXPECTED_SAMPLES, ".",
      call. = FALSE
    )
  }
  member_gsm <- toupper(sub(".*?(GSM[0-9]+).*", "\\1", cel_members, perl = TRUE))
  public_member_position <- match(public$GSM, member_gsm)
  if (anyNA(public_member_position)) {
    stop("At least one GSE290167 GSM has no CEL member in its RAW archive.", call. = FALSE)
  }
  public$raw_member <- cel_members[public_member_position]
  expected_basename <- basename(sub("^ftp://", "https://", public$supplementary_url))
  if (!identical(public$raw_member, expected_basename)) {
    stop("GSE290167 metadata and RAW TAR member names disagree.", call. = FALSE)
  }
  fallback_rows <- which(!metadata$has_private_cel)
  public_position <- match(metadata$sample_key_upper[fallback_rows], public$sample_key_upper)
  metadata$raw_member[fallback_rows] <- public$raw_member[public_position]
  metadata$GSM[fallback_rows] <- public$GSM[public_position]
}

metadata$appears_in_egfr_workbook <- availability$appears_in_egfr_workbook
metadata$has_measured_egfr <- availability$has_measured_egfr
metadata$training_role <- ifelse(
  metadata$has_measured_egfr,
  "held_out_measured_egfr",
  "train_no_measured_egfr"
)
metadata$donor_id <- sub("_[LP]$", "", metadata$sample_key_upper)
donor_role_count <- vapply(
  split(metadata$training_role, metadata$donor_id),
  function(roles) length(unique(roles)),
  integer(1)
)
donors_crossing_outcome_gate <- names(donor_role_count)[donor_role_count > 1L]
metadata$donor_crosses_outcome_gate <- metadata$donor_id %in%
  donors_crossing_outcome_gate
metadata$legacy_row_index_python <- NA_integer_
legacy_index_path <- file.path(IKEM_LEGACY_STORE, "sample_index.csv")
if (nzchar(IKEM_LEGACY_STORE) && file.exists(legacy_index_path)) {
  legacy_index <- utils::read.csv(
    legacy_index_path,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  id_candidates <- intersect(
    c("sample_id", "Sample_ID", "GSM", "sample", "id"),
    names(legacy_index)
  )
  if (length(id_candidates) > 0L) {
    legacy_keys <- normalize_legacy_id(legacy_index[[id_candidates[[1L]]]])
    if (anyDuplicated(legacy_keys)) {
      stop("Existing IKEM sample index contains duplicate IDs.", call. = FALSE)
    }
    metadata$legacy_row_index_python <-
      match(metadata$sample_key_upper, legacy_keys) - 1L
  }
}
metadata$row_index_python <- seq_len(nrow(metadata)) - 1L
local_reference <- ikem_training_reference(metadata)
local_reference_positions <- as.integer(local_reference$gse290167_row_r)
metadata$is_local_rma_reference_train <- seq_len(nrow(metadata)) %in%
  local_reference_positions

message(
  "IKEM CEL sources: ", sum(metadata$has_private_cel), " private; ",
  sum(!metadata$has_private_cel), " public GSE290167 fallback."
)
message(
  "Outcome gate: ", length(local_reference_positions),
  " no-measured-eGFR TRAIN; ", sum(metadata$has_measured_egfr),
  " measured-eGFR HELD OUT."
)
message(
  "Outcome gate unit: biopsy. Donors represented on both sides: ",
  length(donors_crossing_outcome_gate),
  if (length(donors_crossing_outcome_gate) > 0L) {
    paste0(" (", paste(donors_crossing_outcome_gate, collapse = ", "), ").")
  } else {
    "."
  }
)

# A compact manifest detects private CEL replacement without hashing many GB of
# CEL data on every restart. Small metadata/reference files are hashed fully.
private_info <- file.info(metadata$private_cel_path[metadata$has_private_cel])
private_signature <- rep("", nrow(metadata))
private_signature[metadata$has_private_cel] <- paste(
  basename(metadata$private_cel_path[metadata$has_private_cel]),
  private_info$size,
  as.numeric(private_info$mtime),
  sep = ":"
)
public_archive_signature <- if (length(missing_private) > 0L) {
  info <- file.info(IKEM_ARCHIVE)
  paste(basename(IKEM_ARCHIVE), info$size, as.numeric(info$mtime), sep = ":")
} else {
  "public-archive-not-used"
}
manifest_lines <- paste(
  metadata$sample_key_upper,
  metadata$cel_source,
  private_signature,
  ifelse(metadata$has_private_cel, "", metadata$raw_member),
  metadata$training_role,
  sep = "|"
)
manifest_path <- file.path(IKEM_WORK_DIR, "current_input_manifest.txt")
atomic_write_lines_ikem(c(public_archive_signature, manifest_lines), manifest_path)
signature_paths <- c(
  IKEM_SAMPLE_METADATA,
  IKEM_EGFR_TABLE,
  FROZEN_SPLIT_CSV,
  GLOBAL_TEMPLATE,
  manifest_path
)
input_signature <- paste(
  IKEM_PIPELINE_VERSION,
  paste(
    basename(signature_paths),
    unname(tools::md5sum(signature_paths)),
    collapse = "|"
  ),
  sep = "|"
)
prior_signature <- if (file.exists(IKEM_SIGNATURE_FILE)) {
  paste(readLines(IKEM_SIGNATURE_FILE, warn = FALSE), collapse = "")
} else {
  ""
}
has_partial_state <- file.exists(IKEM_FINAL_H5) || file.exists(IKEM_PROBE_H5) ||
  file.exists(IKEM_LOCAL_PARAMETERS) || file.exists(IKEM_GLOBAL_TRAIN_BG_H5) ||
  file.exists(IKEM_GLOBAL_TRAIN_CONTRIBUTION) ||
  length(list.files(IKEM_PROGRESS_DIR, all.files = FALSE)) > 0L
if (has_partial_state && !identical(prior_signature, input_signature)) {
  message("IKEM CEL inputs/split changed; discarding incompatible IKEM checkpoints.")
  unlink(
    c(
      IKEM_FINAL_H5,
      IKEM_PROBE_H5,
      IKEM_LOCAL_PARAMETERS,
      IKEM_GLOBAL_TRAIN_BG_H5,
      IKEM_GLOBAL_TRAIN_CONTRIBUTION,
      IKEM_GLOBAL_TRAIN_READY,
      IKEM_LOCAL_COMPLETE,
      IKEM_COMPLETE,
      IKEM_GLOBAL_SIGNATURE_FILE
    ),
    force = TRUE
  )
  unlink(IKEM_PROGRESS_DIR, recursive = TRUE, force = TRUE)
  dir.create(IKEM_PROGRESS_DIR, recursive = TRUE, showWarnings = FALSE)
  unlink(
    c(
      file.path(IKEM_WORK_DIR, "ikem_target_state.rds"),
      file.path(IKEM_WORK_DIR, "ikem_train_quantile_target.rds")
    ),
    force = TRUE
  )
}
atomic_write_lines_ikem(input_signature, IKEM_SIGNATURE_FILE)

global_output_signature <- ""
if (DO_GLOBAL_FINALIZATION) {
  global_paths <- c(GLOBAL_TARGET, GLOBAL_PARAMETERS, GLOBAL_PARAMETER_SIGNATURE)
  global_output_signature <- paste(
    input_signature,
    paste(
      basename(global_paths),
      unname(tools::md5sum(global_paths)),
      collapse = "|"
    ),
    sep = "|"
  )
  previous_global_signature <- if (file.exists(IKEM_GLOBAL_SIGNATURE_FILE)) {
    paste(readLines(IKEM_GLOBAL_SIGNATURE_FILE, warn = FALSE), collapse = "")
  } else {
    ""
  }
  if (!identical(previous_global_signature, global_output_signature)) {
    # Local RMA remains valid. Only the global output/progress belongs to the
    # older combined reference and must be overwritten.
    unlink(c(IKEM_COMPLETE, IKEM_GLOBAL_PROGRESS), force = TRUE)
    atomic_write_lines_ikem(
      global_output_signature,
      IKEM_GLOBAL_SIGNATURE_FILE
    )
  }
}

metadata$private_cel_filename <- ifelse(
  metadata$has_private_cel,
  basename(metadata$private_cel_path),
  NA_character_
)
correspondence <- metadata[, c(
  "row_index_python", "sample_id", "sample_id_geo", "sample_key_upper",
  "donor_id", "donor_crosses_outcome_gate", "GSM", "patient_id_geo",
  "tissue_geo", "platform", "cel_source",
  "private_cel_filename", "raw_member", "appears_in_egfr_workbook",
  "has_measured_egfr", "training_role", "legacy_row_index_python",
  "is_local_rma_reference_train"
)]
correspondence$metadata_match <- TRUE
correspondence$cel_available <- TRUE
correspondence$legacy_id_match <- !is.na(correspondence$legacy_row_index_python)
atomic_write_csv_ikem(
  correspondence,
  file.path(IKEM_OUT_DIR, "ikem_cel_correspondence.csv")
)
# Compatibility alias for older consumers; rows now include private-only CELs.
atomic_write_csv_ikem(
  correspondence,
  file.path(IKEM_OUT_DIR, "ikem_gse290167_correspondence.csv")
)
atomic_write_csv_ikem(
  local_reference,
  file.path(IKEM_OUT_DIR, "ikem_rma_reference_samples.csv")
)

legacy_probe_path <- file.path(IKEM_LEGACY_STORE, "probe_index.csv")
if (nzchar(IKEM_LEGACY_STORE) && file.exists(legacy_probe_path)) {
  legacy_probes <- utils::read.csv(
    legacy_probe_path,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  candidates <- intersect(c("probe_id", "probe", "probeset_id", "ID", "id"), names(legacy_probes))
  if (length(candidates) == 0L) {
    message(
      "Historical IKEM probe index has no recognizable probe-ID column; ",
      "the optional historical comparison will be skipped."
    )
  } else {
    legacy_probe_ids <- trimws(as.character(legacy_probes[[candidates[[1L]]]]))
    common_keys <- toupper(trimws(common_probes))
    legacy_keys <- toupper(legacy_probe_ids)
    legacy_positions <- match(common_keys, legacy_keys)
    exact_legacy_order <- identical(legacy_keys, common_keys)
    shared_probe_count <- sum(!is.na(legacy_positions))

    atomic_write_csv_ikem(
      data.frame(
        new_probe_index_python = seq_along(common_probes) - 1L,
        probe_id = common_probes,
        legacy_probe_index_python = ifelse(
          is.na(legacy_positions),
          NA_integer_,
          legacy_positions - 1L
        ),
        present_in_legacy_store = !is.na(legacy_positions),
        stringsAsFactors = FALSE
      ),
      file.path(IKEM_OUT_DIR, "legacy_probe_correspondence.csv")
    )

    if (exact_legacy_order) {
      message(
        "Historical IKEM probe audit: all ", length(common_probes),
        " probes already have the frozen order."
      )
    } else {
      message(
        "Historical IKEM probe audit: ", shared_probe_count, "/",
        length(common_probes),
        " frozen probes are shared; historical columns will be matched by ",
        "probe ID for the optional comparison."
      )
    }
  }
}

atomic_write_csv_ikem(
  data.frame(
    probe_index_python = seq_along(common_probes) - 1L,
    probe_id = common_probes,
    stringsAsFactors = FALSE
  ),
  file.path(IKEM_OUT_DIR, "probe_index.csv")
)
atomic_write_csv_ikem(
  correspondence,
  file.path(IKEM_OUT_DIR, "sample_index.csv")
)

# During the prepare call the combined global target/effects do not exist yet.
# Finite placeholders let the local checkpoint code keep a single simple path;
# rma_global is overwritten and certified only during the finalize call.
if (!DO_GLOBAL_FINALIZATION) {
  global_target <- rep(1, template$n_all_pm)
  global_effect <- numeric(template$n_common_pm)
}

if (file.exists(IKEM_COMPLETE)) {
  message("IKEM CEL completion marker found; reusing completed matrices.")
} else {
  extracted_marker <- file.path(IKEM_WORK_DIR, "EXTRACTION_COMPLETE.txt")
  paths <- metadata$private_cel_path
  fallback_rows <- which(!metadata$has_private_cel)
  if (length(fallback_rows) > 0L) {
    extracted_before <- list.files(
      IKEM_EXTRACT_DIR,
      pattern = "\\.CEL(\\.gz)?$",
      recursive = TRUE,
      full.names = TRUE,
      ignore.case = TRUE
    )
    if (!file.exists(extracted_marker) ||
        length(extracted_before) != IKEM_PUBLIC_EXPECTED_SAMPLES) {
      unlink(IKEM_EXTRACT_DIR, recursive = TRUE, force = TRUE)
      dir.create(IKEM_EXTRACT_DIR, recursive = TRUE, showWarnings = FALSE)
      status <- system2(
        "tar",
        c("-xf", shQuote(IKEM_ARCHIVE), "-C", shQuote(IKEM_EXTRACT_DIR))
      )
      if (status != 0L) {
        stop("Could not extract the validated GSE290167 RAW archive.", call. = FALSE)
      }
      extracted <- list.files(
        IKEM_EXTRACT_DIR,
        pattern = "\\.CEL(\\.gz)?$",
        recursive = TRUE,
        full.names = TRUE,
        ignore.case = TRUE
      )
      if (length(extracted) != IKEM_PUBLIC_EXPECTED_SAMPLES) {
        stop(
          "GSE290167 extraction did not produce exactly ",
          IKEM_PUBLIC_EXPECTED_SAMPLES, " CEL files.",
          call. = FALSE
        )
      }
      writeLines(paste(ikem_time(), length(extracted)), extracted_marker)
    }

    extracted <- list.files(
      IKEM_EXTRACT_DIR,
      pattern = "\\.CEL(\\.gz)?$",
      recursive = TRUE,
      full.names = TRUE,
      ignore.case = TRUE
    )
    extracted_gsm <- toupper(sub(
      ".*?(GSM[0-9]+).*", "\\1", basename(extracted), perl = TRUE
    ))
    paths[fallback_rows] <- extracted[
      match(metadata$GSM[fallback_rows], extracted_gsm)
    ]
    if (anyNA(paths[fallback_rows]) || !all(file.exists(paths[fallback_rows]))) {
      stop("Public fallback CELs cannot be aligned to IKEM metadata.", call. = FALSE)
    }
  } else {
    # A previous public-only run may have left an extracted copy. It is derived
    # and no longer needed when the complete private collection is available.
    unlink(IKEM_EXTRACT_DIR, recursive = TRUE, force = TRUE)
    unlink(extracted_marker, force = TRUE)
  }
  if (anyNA(paths) || !all(file.exists(paths))) {
    stop("Not every canonical IKEM biopsy has a readable CEL input.", call. = FALSE)
  }

  read_one_ikem_cel <- function(path) {
    actual <- path
    temporary <- NULL
    if (grepl("\\.gz$", path, ignore.case = TRUE)) {
      temporary <- file.path(IKEM_WORK_DIR, "current_input.CEL")
      unlink(temporary, force = TRUE)
      R.utils::gunzip(
        path,
        destname = temporary,
        remove = FALSE,
        overwrite = TRUE
      )
      actual <- temporary
    }
    on.exit(if (!is.null(temporary)) unlink(temporary, force = TRUE), add = TRUE)
    value <- affxparser::readCelIntensities(actual)
    if (is.matrix(value)) value <- value[, 1L]
    value <- as.numeric(value)
    if (length(value) != template$n_total_cells || any(!is.finite(value))) {
      stop("Invalid PrimeView CEL intensity vector: ", path, call. = FALSE)
    }
    value
  }

  first_header_path <- paths[[1L]]
  first_temp <- NULL
  if (grepl("\\.gz$", first_header_path, ignore.case = TRUE)) {
    first_temp <- file.path(IKEM_WORK_DIR, "header_input.CEL")
    R.utils::gunzip(
      first_header_path,
      destname = first_temp,
      remove = FALSE,
      overwrite = TRUE
    )
    first_header_path <- first_temp
  }
  first_header <- affxparser::readCelHeader(first_header_path)
  if (!is.null(first_temp)) unlink(first_temp, force = TRUE)
  header_text <- tolower(paste(unlist(first_header), collapse = " "))
  if (!grepl("primeview", header_text, fixed = TRUE)) {
    stop("The first IKEM CEL header is not PrimeView.", call. = FALSE)
  }

  n_samples <- nrow(metadata)
  n_probes <- length(common_probes)
  final_dims <- c(n_samples, n_probes)
  if (!file.exists(IKEM_FINAL_H5)) rhdf5::h5createFile(IKEM_FINAL_H5)
  objects <- rhdf5::h5ls(IKEM_FINAL_H5, recursive = TRUE)
  if (!any(objects$group == "/" & objects$name == "expression")) {
    rhdf5::h5createGroup(IKEM_FINAL_H5, "expression")
  }
  objects <- rhdf5::h5ls(IKEM_FINAL_H5, recursive = TRUE)
  for (dataset in c(
    "raw_original",
    "rma_per_gse",
    "rma_global"
  )) {
    full <- paste0("/expression/", dataset)
    if (!any(paste0(objects$group, "/", objects$name) == full)) {
      rhdf5::h5createDataset(
        IKEM_FINAL_H5,
        paste0("expression/", dataset),
        dims = final_dims,
        H5type = "H5T_IEEE_F32LE",
        chunk = c(min(16L, n_samples), min(1024L, n_probes)),
        level = IKEM_H5_LEVEL,
        fillValue = NaN,
        native = TRUE
      )
    }
  }

  common_starts <- cumsum(c(1L, head(template$common_pm_counts, -1L)))
  common_ends <- cumsum(template$common_pm_counts)
  raw_for_batch <- function(common_pm) {
    n_batch <- ncol(common_pm)
    result <- vapply(seq_along(common_probes), function(j) {
      values <- common_pm[
        common_starts[[j]]:common_ends[[j]],
        ,
        drop = FALSE
      ]
      if (nrow(values) == 1L) return(as.numeric(values[1L, ]))
      apply(values, 2L, stats::median)
    }, FUN.VALUE = numeric(n_batch))
    result
  }

  # Cache background-corrected PM values for the frozen IKEM no-eGFR
  # training arrays. Stage 3 uses these columns together with GEO TRAIN arrays
  # when fitting the combined global quantile target and probe effects.
  pass1_state_path <- file.path(IKEM_WORK_DIR, "ikem_target_state.rds")
  local_target_path <- file.path(IKEM_WORK_DIR, "ikem_train_quantile_target.rds")
  pass1_done_path <- file.path(IKEM_PROGRESS_DIR, "pass1_done_sample.txt")

  # If the large cache was removed while the small PASS1 state survived, that
  # state cannot certify the training columns. Rebuild PASS1 automatically.
  if (!file.exists(IKEM_GLOBAL_TRAIN_BG_H5) && file.exists(pass1_state_path)) {
    message(
      "IKEM TRAIN PM cache is absent; rebuilding the resumable PASS1 state."
    )
    unlink(
      c(
        pass1_state_path,
        pass1_done_path,
        local_target_path,
        IKEM_GLOBAL_TRAIN_CONTRIBUTION,
        IKEM_GLOBAL_TRAIN_READY
      ),
      force = TRUE
    )
  }
  if (!file.exists(IKEM_GLOBAL_TRAIN_BG_H5)) {
    rhdf5::h5createFile(IKEM_GLOBAL_TRAIN_BG_H5)
    rhdf5::h5createDataset(
      IKEM_GLOBAL_TRAIN_BG_H5,
      "background_corrected_all_pm",
      dims = c(template$n_all_pm, length(local_reference_positions)),
      H5type = "H5T_IEEE_F32LE",
      chunk = c(
        min(4096L, template$n_all_pm),
        min(IKEM_ARRAY_BATCH, length(local_reference_positions))
      ),
      level = IKEM_H5_LEVEL,
      fillValue = NaN,
      native = FALSE
    )
  }

  if (file.exists(pass1_state_path)) {
    target_state <- readRDS(pass1_state_path)
    if (is.null(target_state$done_keys)) {
      target_state$done_keys <- read_done_ikem(pass1_done_path)
    }
  } else {
    target_state <- list(
      local_train_sum = numeric(template$n_all_pm),
      local_train_n = 0L,
      done_keys = character()
    )
  }
  pass1_done <- target_state$done_keys
  for (batch_start in seq.int(1L, n_samples, by = IKEM_ARRAY_BATCH)) {
    batch_end <- min(n_samples, batch_start + IKEM_ARRAY_BATCH - 1L)
    batch <- batch_start:batch_end
    keys <- metadata$sample_key_upper[batch]
    if (all(keys %in% pass1_done)) next
    common_pm <- matrix(
      NA_real_, nrow = template$n_common_pm, ncol = length(batch)
    )
    train_sum <- target_state$local_train_sum
    train_n <- target_state$local_train_n
    for (k in seq_along(batch)) {
      intensity <- read_one_ikem_cel(paths[[batch[[k]]]])
      pm <- intensity[template$all_pm_cell_index]
      corrected <- preprocessCore::rma.background.correct(
        matrix(pm, ncol = 1L), copy = FALSE
      )[, 1L]
      common_pm[, k] <- pm[template$common_pm_positions_grouped]
      if (batch[[k]] %in% local_reference_positions) {
        sorted_corrected <- sort(corrected)
        reference_column <- match(batch[[k]], local_reference_positions)
        rhdf5::h5write(
          corrected,
          IKEM_GLOBAL_TRAIN_BG_H5,
          "background_corrected_all_pm",
          index = list(seq_len(template$n_all_pm), reference_column),
          native = FALSE
        )
        train_sum <- train_sum + sorted_corrected
        train_n <- train_n + 1L
        rm(sorted_corrected)
      }
      rm(intensity, pm, corrected)
    }
    raw_block <- raw_for_batch(common_pm)
    rhdf5::h5write(
      raw_block,
      IKEM_FINAL_H5,
      "expression/raw_original",
      index = list(batch, seq_len(n_probes)),
      native = TRUE
    )
    target_state$local_train_sum <- train_sum
    target_state$local_train_n <- train_n
    target_state$done_keys <- c(target_state$done_keys, keys)
    atomic_save_rds_ikem(target_state, pass1_state_path, compress = FALSE)
    for (key in keys) append_done_ikem(pass1_done_path, key)
    pass1_done <- c(pass1_done, keys)
    rm(common_pm, raw_block, train_sum)
    gc()
    message("[IKEM PASS1] ", batch_end, "/", n_samples, " CELs.")
  }
  if (length(unique(target_state$done_keys)) != n_samples) {
    stop("IKEM PASS1 checkpoint is internally inconsistent.", call. = FALSE)
  }
  if (target_state$local_train_n != length(local_reference_positions)) {
    stop("IKEM train-reference target sample count is incomplete.", call. = FALSE)
  }
  local_target <- target_state$local_train_sum / target_state$local_train_n
  if (length(local_target) != template$n_all_pm ||
      any(!is.finite(local_target))) {
    stop("IKEM PASS1 produced an invalid quantile target.", call. = FALSE)
  }
  atomic_save_rds_ikem(local_target, local_target_path, compress = TRUE)

  global_train_contribution <- list(
    format_version = 1L,
    input_signature = input_signature,
    frozen_split_md5 = unname(tools::md5sum(FROZEN_SPLIT_CSV)),
    template_md5 = unname(tools::md5sum(GLOBAL_TEMPLATE)),
    target_sum = as.numeric(target_state$local_train_sum),
    n_samples = as.integer(target_state$local_train_n),
    sample_ids = as.character(local_reference$sample_id),
    GSM = as.character(local_reference$GSM),
    background_cache = basename(IKEM_GLOBAL_TRAIN_BG_H5),
    background_dataset = "background_corrected_all_pm"
  )
  atomic_save_rds_ikem(
    global_train_contribution,
    IKEM_GLOBAL_TRAIN_CONTRIBUTION,
    compress = TRUE
  )

  # Validate every cached training value so an
  # interrupted/corrupted cache can never enter the combined global fit.
  for (cache_start in seq.int(
    1L,
    length(local_reference_positions),
    by = IKEM_ARRAY_BATCH
  )) {
    cache_end <- min(
      length(local_reference_positions),
      cache_start + IKEM_ARRAY_BATCH - 1L
    )
    cache_block <- rhdf5::h5read(
      IKEM_GLOBAL_TRAIN_BG_H5,
      "background_corrected_all_pm",
      index = list(seq_len(template$n_all_pm), cache_start:cache_end),
      native = FALSE
    )
    if (any(!is.finite(cache_block)) || any(cache_block <= 0)) {
      stop(
        "IKEM global-training PM cache is incomplete or invalid in columns ",
        cache_start, "-", cache_end, ". Remove .IKEM_CEL_WORK and rerun.",
        call. = FALSE
      )
    }
    rm(cache_block)
  }
  atomic_write_lines_ikem(
    paste(
      ikem_time(),
      "IKEM_GLOBAL_TRAIN_REFERENCE_READY",
      "samples=", length(local_reference_positions),
      "signature=", input_signature
    ),
    IKEM_GLOBAL_TRAIN_READY
  )

  if (!file.exists(IKEM_PROBE_H5)) {
    rhdf5::h5createFile(IKEM_PROBE_H5)
  }
  probe_objects <- rhdf5::h5ls(IKEM_PROBE_H5, recursive = TRUE)
  for (dataset in "local_normalized_common_pm") {
    if (any(probe_objects$group == "/" & probe_objects$name == dataset)) next
    rhdf5::h5createDataset(
      IKEM_PROBE_H5,
      dataset,
      dims = c(template$n_common_pm, n_samples),
      H5type = "H5T_IEEE_F32LE",
      chunk = c(min(4096L, template$n_common_pm), min(IKEM_ARRAY_BATCH, n_samples)),
      level = IKEM_H5_LEVEL,
      native = FALSE
    )
  }

  global_for_batch <- function(common_normalized) {
    n_batch <- ncol(common_normalized)
    result <- vapply(seq_along(common_probes), function(j) {
      rows <- common_starts[[j]]:common_ends[[j]]
      adjusted <- log2(common_normalized[rows, , drop = FALSE]) - global_effect[rows]
      if (nrow(adjusted) == 1L) return(as.numeric(adjusted[1L, ]))
      apply(adjusted, 2L, stats::median)
    }, FUN.VALUE = numeric(n_batch))
    result
  }

  pass2_done <- read_done_ikem(file.path(IKEM_PROGRESS_DIR, "pass2_done_sample.txt"))
  for (batch_start in seq.int(1L, n_samples, by = IKEM_ARRAY_BATCH)) {
    batch_end <- min(n_samples, batch_start + IKEM_ARRAY_BATCH - 1L)
    batch <- batch_start:batch_end
    keys <- metadata$sample_key_upper[batch]
    if (all(keys %in% pass2_done)) next
    local_common <- matrix(
      NA_real_, nrow = template$n_common_pm, ncol = length(batch)
    )
    for (k in seq_along(batch)) {
      intensity <- read_one_ikem_cel(paths[[batch[[k]]]])
      pm <- intensity[template$all_pm_cell_index]
      corrected <- preprocessCore::rma.background.correct(
        matrix(pm, ncol = 1L), copy = FALSE
      )
      local_norm <- preprocessCore::normalize.quantiles.use.target(
        corrected, target = local_target, copy = FALSE
      )[, 1L]
      local_common[, k] <- local_norm[template$common_pm_positions_grouped]
      rm(intensity, pm, corrected, local_norm)
    }
    if (any(!is.finite(local_common))) {
      stop("Non-finite IKEM RMA value in PASS2.", call. = FALSE)
    }
    rhdf5::h5write(
      local_common,
      IKEM_PROBE_H5,
      "local_normalized_common_pm",
      index = list(seq_len(template$n_common_pm), batch),
      native = FALSE
    )
    for (key in keys) append_done_ikem(
      file.path(IKEM_PROGRESS_DIR, "pass2_done_sample.txt"), key
    )
    pass2_done <- c(pass2_done, keys)
    rm(local_common)
    gc()
    message("[IKEM PASS2] ", batch_end, "/", n_samples, " CELs.")
  }
  if (!all(metadata$sample_key_upper %in% read_done_ikem(
    file.path(IKEM_PROGRESS_DIR, "pass2_done_sample.txt")
  ))) {
    stop("IKEM PASS2 checkpoint is incomplete.", call. = FALSE)
  }

  probe_starts <- seq.int(1L, n_probes, by = IKEM_PROBESET_BLOCK)
  pass3_path <- file.path(IKEM_PROGRESS_DIR, "pass3_done_block.txt")
  pass3_done <- as.integer(read_done_ikem(pass3_path))
  pass3_done <- pass3_done[!is.na(pass3_done)]
  if (!file.exists(IKEM_LOCAL_PARAMETERS)) {
    rhdf5::h5createFile(IKEM_LOCAL_PARAMETERS)
    rhdf5::h5createDataset(
      IKEM_LOCAL_PARAMETERS,
      "probe_effect_common_pm",
      dims = template$n_common_pm,
      H5type = "H5T_IEEE_F64LE",
      chunk = min(8192L, template$n_common_pm),
      level = IKEM_H5_LEVEL,
      fillValue = NaN,
      native = TRUE
    )
  } else {
    parameter_objects <- rhdf5::h5ls(IKEM_LOCAL_PARAMETERS, recursive = TRUE)
    if (!any(parameter_objects$group == "/" &
             parameter_objects$name == "probe_effect_common_pm")) {
      stop(
        "Existing IKEM train-reference parameter file is incomplete; remove ",
        IKEM_LOCAL_PARAMETERS,
        " and rerun.",
        call. = FALSE
      )
    }
  }
  for (block_id in seq_along(probe_starts)) {
    if (block_id %in% pass3_done) next
    first_probe <- probe_starts[[block_id]]
    last_probe <- min(n_probes, first_probe + IKEM_PROBESET_BLOCK - 1L)
    probe_ids <- first_probe:last_probe
    first_pm <- common_starts[[first_probe]]
    last_pm <- common_ends[[last_probe]]
    pm_rows <- first_pm:last_pm
    local_normalized <- rhdf5::h5read(
      IKEM_PROBE_H5,
      "local_normalized_common_pm",
      index = list(pm_rows, seq_len(n_samples)),
      native = FALSE
    )
    local_summarized <- matrix(
      NA_real_, nrow = length(probe_ids), ncol = n_samples
    )
    local_effect_block <- numeric(length(pm_rows))
    local_start <- 1L
    for (local_probe in seq_along(probe_ids)) {
      n_pm <- template$common_pm_counts[probe_ids[[local_probe]]]
      local_rows <- local_start:(local_start + n_pm - 1L)
      log_block <- log2(local_normalized[local_rows, , drop = FALSE])
      if (any(!is.finite(log_block))) {
        stop(
          "Non-positive/non-finite IKEM local-RMA PM value for probe set ",
          common_probes[probe_ids[[local_probe]]],
          call. = FALSE
        )
      }
      train_fit <- stats::medpolish(
        log_block[, local_reference_positions, drop = FALSE],
        trace.iter = FALSE
      )
      frozen_effect <- as.numeric(train_fit$row)
      local_effect_block[local_rows] <- frozen_effect
      local_summarized[local_probe, ] <- apply(
        sweep(log_block, 1L, frozen_effect, FUN = "-"),
        2L,
        stats::median
      )
      fitted_train <- as.numeric(train_fit$overall + train_fit$col)
      max_train_delta <- max(
        abs(local_summarized[local_probe, local_reference_positions] - fitted_train)
      )
      if (!is.finite(max_train_delta) || max_train_delta > 1e-7) {
        stop(
          "Frozen IKEM local-RMA self-check failed for probe set ",
          common_probes[probe_ids[[local_probe]]],
          ": max delta=", max_train_delta,
          call. = FALSE
        )
      }
      local_start <- local_start + n_pm
    }
    expected_dims <- c(length(probe_ids), n_samples)
    if (!identical(dim(local_summarized), expected_dims) ||
        any(!is.finite(local_summarized)) ||
        any(!is.finite(local_effect_block))) {
      stop("Invalid local-RMA summary in IKEM PASS3 block ", block_id, call. = FALSE)
    }
    rhdf5::h5write(
      t(local_summarized),
      IKEM_FINAL_H5,
      "expression/rma_per_gse",
      index = list(seq_len(n_samples), probe_ids),
      native = TRUE
    )
    rhdf5::h5write(
      local_effect_block,
      IKEM_LOCAL_PARAMETERS,
      "probe_effect_common_pm",
      index = list(pm_rows),
      native = TRUE
    )
    append_done_ikem(pass3_path, as.character(block_id))
    rm(
      local_normalized,
      local_summarized,
      local_effect_block
    )
    gc()
    message("[IKEM PASS3] block ", block_id, "/", length(probe_starts), ".")
  }
  completed_blocks <- as.integer(read_done_ikem(pass3_path))
  if (!all(seq_along(probe_starts) %in% completed_blocks)) {
    stop("IKEM PASS3 checkpoint is incomplete.", call. = FALSE)
  }

  for (method in c("raw_original", "rma_per_gse")) {
    spot <- rhdf5::h5read(
      IKEM_FINAL_H5,
      paste0("expression/", method),
      index = list(unique(c(1L, n_samples)), unique(c(1L, n_probes))),
      native = TRUE
    )
    if (any(!is.finite(spot))) {
      stop("IKEM final HDF5 contains non-finite values in ", method, call. = FALSE)
    }
  }
  effect_spot <- rhdf5::h5read(
    IKEM_LOCAL_PARAMETERS,
    "probe_effect_common_pm",
    index = list(unique(c(1L, template$n_common_pm))),
    native = TRUE
  )
  if (any(!is.finite(effect_spot))) {
    stop("Frozen IKEM train-reference probe effects are incomplete.", call. = FALSE)
  }

  atomic_write_lines_ikem(
    paste(
      ikem_time(), "IKEM_LOCAL_RMA_COMPLETE",
      "GSE=", IKEM_GSE, "samples=", n_samples, "probes=", n_probes,
      "local_reference_train_samples=", length(local_reference_positions),
      "signature=", input_signature
    ),
    IKEM_LOCAL_COMPLETE
  )

  if (DO_GLOBAL_FINALIZATION) {
    if (!file.exists(IKEM_GLOBAL_TRAIN_READY) ||
        !file.exists(IKEM_GLOBAL_TRAIN_CONTRIBUTION) ||
        !file.exists(IKEM_GLOBAL_TRAIN_BG_H5)) {
      stop(
        "The IKEM local stage is missing its global-training cache. ",
        "Rerun the prepare stage before finalizing global RMA.",
        call. = FALSE
      )
    }
    if (!identical(
      as.character(global_signature$ikem_train_sample_ids),
      as.character(local_reference$sample_id)
    ) || !identical(
      as.character(global_signature$ikem_train_gsms),
      as.character(local_reference$GSM)
    )) {
      stop(
        "The combined global reference was not fitted with exactly the current ",
        "frozen IKEM no-eGFR TRAIN rows.",
        call. = FALSE
      )
    }

    global_done <- read_done_ikem(IKEM_GLOBAL_PROGRESS)
    for (batch_start in seq.int(1L, n_samples, by = IKEM_ARRAY_BATCH)) {
      batch_end <- min(n_samples, batch_start + IKEM_ARRAY_BATCH - 1L)
      batch <- batch_start:batch_end
      keys <- metadata$sample_key_upper[batch]
      if (all(keys %in% global_done)) next

      global_common <- matrix(
        NA_real_,
        nrow = template$n_common_pm,
        ncol = length(batch)
      )
      for (k in seq_along(batch)) {
        intensity <- read_one_ikem_cel(paths[[batch[[k]]]])
        pm <- intensity[template$all_pm_cell_index]
        corrected <- preprocessCore::rma.background.correct(
          matrix(pm, ncol = 1L), copy = FALSE
        )
        normalized <- preprocessCore::normalize.quantiles.use.target(
          corrected, target = global_target, copy = FALSE
        )[, 1L]
        global_common[, k] <- normalized[
          template$common_pm_positions_grouped
        ]
        rm(intensity, pm, corrected, normalized)
      }

      global_block <- global_for_batch(global_common)
      if (any(!is.finite(global_block))) {
        stop("Non-finite combined-global IKEM RMA output.", call. = FALSE)
      }
      rhdf5::h5write(
        global_block,
        IKEM_FINAL_H5,
        "expression/rma_global",
        index = list(batch, seq_len(n_probes)),
        native = TRUE
      )
      for (key in keys) append_done_ikem(IKEM_GLOBAL_PROGRESS, key)
      global_done <- c(global_done, keys)
      rm(global_common, global_block)
      gc()
      message("[IKEM GLOBAL] ", batch_end, "/", n_samples, " CELs.")
    }

    if (!all(metadata$sample_key_upper %in% read_done_ikem(
      IKEM_GLOBAL_PROGRESS
    ))) {
      stop("IKEM combined-global progress is incomplete.", call. = FALSE)
    }
    global_spot <- rhdf5::h5read(
      IKEM_FINAL_H5,
      "expression/rma_global",
      index = list(
        unique(c(1L, n_samples)),
        unique(c(1L, n_probes))
      ),
      native = TRUE
    )
    if (any(!is.finite(global_spot))) {
      stop("IKEM combined-global matrix failed its final check.", call. = FALSE)
    }

    atomic_write_lines_ikem(
      paste(
        ikem_time(), "IKEM_CEL_PREPROCESSING_COMPLETE",
        "GSE=", IKEM_GSE, "samples=", n_samples, "probes=", n_probes,
        "local_reference_train_samples=", length(local_reference_positions),
        "global_reference_train_samples=", global_signature$fit_sample_count,
        "signature=", global_output_signature
      ),
      IKEM_COMPLETE
    )
    unlink(IKEM_EXTRACT_DIR, recursive = TRUE, force = TRUE)
    unlink(extracted_marker, force = TRUE)
  } else {
    message(
      "IKEM local RMA and the ", length(local_reference_positions),
      "-array global-training cache are ready. ",
      "Stage 3 can now fit the combined GEO+IKEM TRAIN reference."
    )
  }
}

message("IKEM CEL matrices: ", normalizePath(IKEM_FINAL_H5))
message("IKEM correspondence: ", normalizePath(file.path(
  IKEM_OUT_DIR, "ikem_cel_correspondence.csv"
)))
