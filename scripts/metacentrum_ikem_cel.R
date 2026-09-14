#!/usr/bin/env Rscript

# MetaCentrum: build leakage-free IKEM encoder inputs from public PrimeView CELs
# in GSE290167.  The implementation is disk-backed and resumable:
#
#   raw_original   original PM intensity, median within each common probe set
#   rma_per_gse    IKEM-specific RMA reference fitted only on outcome-free
#                  molecular-pretraining TRAIN samples; every other CEL is
#                  transformed independently with that frozen reference
#   rma_cohort_legacy
#                  all-cohort RMA used only to verify the historical matrix;
#                  it is never exposed to model training or eGFR evaluation
#   rma_global     each IKEM CEL transformed independently with the quantile
#                  target and probe effects fitted only on pretraining GEO train
#
# No eGFR or clinical outcome is read by this stage.

IKEM_GSE <- "GSE290167"
IKEM_EXPECTED_PLATFORM <- "GPL15207"
IKEM_EXPECTED_SAMPLES <- 276L
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
IKEM_LEGACY_STORE <- Sys.getenv("ARCHCON_IKEM_LEGACY_STORE", unset = "")
IKEM_OUT_DIR <- "IKEM_MATRIX_STORE"
IKEM_WORK_DIR <- ".IKEM_CEL_WORK"
IKEM_PROGRESS_DIR <- file.path(IKEM_WORK_DIR, "progress")
IKEM_EXTRACT_DIR <- file.path(IKEM_WORK_DIR, "extracted")
IKEM_ARCHIVE <- file.path(IKEM_RAW_DIR, paste0(IKEM_GSE, "_RAW.tar"))
IKEM_METADATA <- file.path(IKEM_RAW_DIR, paste0(IKEM_GSE, "_samples_brief.soft"))
IKEM_FINAL_H5 <- file.path(IKEM_OUT_DIR, "ikem_expression_store.h5")
IKEM_PROBE_H5 <- file.path(IKEM_WORK_DIR, "ikem_normalized_common_pm.h5")
IKEM_LOCAL_PARAMETERS <- file.path(
  IKEM_WORK_DIR,
  "ikem_train_reference_rma_parameters.h5"
)
IKEM_COMPLETE <- file.path(IKEM_OUT_DIR, "IKEM_CEL_PREPROCESSING_COMPLETE.txt")
IKEM_SIGNATURE_FILE <- file.path(IKEM_WORK_DIR, "input_signature.txt")
IKEM_PIPELINE_VERSION <- "2-inductive-no-egfr-leakage"

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
FROZEN_SPLIT_CSV <- Sys.getenv("ARCHCON_FROZEN_SPLIT", unset = "")

IKEM_ARRAY_BATCH <- 8L
IKEM_PROBESET_BLOCK <- 128L
IKEM_H5_LEVEL <- 4L

required_packages <- c("affxparser", "preprocessCore", "rhdf5", "R.utils")
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

read_done_ikem <- function(path) {
  if (!file.exists(path)) return(character())
  unique(trimws(readLines(path, warn = FALSE)))
}

append_done_ikem <- function(path, value) {
  current <- read_done_ikem(path)
  if (!(value %in% current)) write(value, file = path, append = TRUE)
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
  if (length(starts) != IKEM_EXPECTED_SAMPLES) {
    stop(
      "GEO metadata contains ", length(starts), " samples; expected ",
      IKEM_EXPECTED_SAMPLES, ".", call. = FALSE
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
  x <- sub("\\.CEL(\\.gz)?$", "", x, ignore.case = TRUE)
  x <- sub("_?\\(?PrimeView\\)?_?$", "", x, ignore.case = TRUE)
  x <- sub("^GSM[0-9]+_", "", x, ignore.case = TRUE)
  x
}

legacy_order <- function(metadata) {
  index_path <- file.path(IKEM_LEGACY_STORE, "sample_index.csv")
  if (!nzchar(IKEM_LEGACY_STORE) || !file.exists(index_path)) {
    metadata$sample_id <- toupper(metadata$sample_id_geo)
    metadata$legacy_row_index_python <- NA_integer_
    return(metadata)
  }
  index <- utils::read.csv(index_path, stringsAsFactors = FALSE, check.names = FALSE)
  candidates <- intersect(c("sample_id", "Sample_ID", "GSM", "sample", "id"), names(index))
  if (length(candidates) == 0L) {
    stop("Existing IKEM sample_index.csv has no sample identifier column.", call. = FALSE)
  }
  ids <- normalize_legacy_id(index[[candidates[[1L]]]])
  keys <- toupper(ids)
  if (anyDuplicated(keys)) {
    stop("Existing IKEM sample index contains duplicate biopsy IDs.", call. = FALSE)
  }
  if (length(keys) != nrow(metadata) || !setequal(keys, metadata$sample_key_upper)) {
    missing <- setdiff(keys, metadata$sample_key_upper)
    extra <- setdiff(metadata$sample_key_upper, keys)
    stop(
      "Existing IKEM store and GSE290167 do not contain the same biopsy IDs. ",
      "Only in existing store: ", paste(head(missing, 10L), collapse = ", "),
      "; only in GEO: ", paste(head(extra, 10L), collapse = ", "),
      call. = FALSE
    )
  }
  position <- match(keys, metadata$sample_key_upper)
  result <- metadata[position, , drop = FALSE]
  result$sample_id <- ids
  result$legacy_row_index_python <- seq_along(ids) - 1L
  rownames(result) <- NULL
  result
}

ikem_training_reference <- function(path, metadata) {
  if (!nzchar(path) || !file.exists(path)) {
    stop("Frozen pretraining sample index is missing: ", path, call. = FALSE)
  }
  split <- utils::read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  required <- c("sample_key", "split")
  missing <- setdiff(required, names(split))
  if (length(missing) > 0L) {
    stop(
      "Frozen pretraining sample index lacks: ",
      paste(missing, collapse = ", "),
      call. = FALSE
    )
  }
  sample_keys <- trimws(as.character(split$sample_key))
  is_supervised <- grepl("^SUPERVISED:", sample_keys, ignore.case = TRUE)
  is_train <- tolower(trimws(as.character(split$split))) == "train"
  selected <- split[is_supervised & is_train, , drop = FALSE]
  selected_keys <- sample_keys[is_supervised & is_train]
  if ("dataset_role" %in% names(selected)) {
    roles <- tolower(trimws(as.character(selected$dataset_role)))
    if (any(!grepl("no[ -]?egfr", roles))) {
      stop(
        "IKEM RMA reference includes a supervised row not explicitly marked no-eGFR.",
        call. = FALSE
      )
    }
  }
  if ("sample_id" %in% names(selected)) {
    ids <- normalize_legacy_id(selected$sample_id)
  } else {
    ids <- normalize_legacy_id(sub("^[^:]+:", "", selected_keys))
  }
  keys <- toupper(ids)
  if (length(keys) < 2L || anyDuplicated(keys)) {
    stop(
      "IKEM train-reference RMA requires at least two unique outcome-free ",
      "molecular-pretraining TRAIN samples.",
      call. = FALSE
    )
  }
  positions <- match(keys, metadata$sample_key_upper)
  if (anyNA(positions)) {
    stop(
      "Frozen IKEM training samples do not all occur in GSE290167: ",
      paste(head(ids[is.na(positions)], 10L), collapse = ", "),
      call. = FALSE
    )
  }
  data.frame(
    reference_order = seq_along(keys),
    sample_id = metadata$sample_id[positions],
    sample_key_upper = keys,
    GSE = IKEM_GSE,
    GSM = metadata$GSM[positions],
    gse290167_row_r = positions,
    gse290167_row_python = positions - 1L,
    frozen_split = "train",
    outcome_status = "no eGFR",
    stringsAsFactors = FALSE
  )
}

message("\n=== Stage 4/4: exact GSE290167 IKEM CEL preprocessing ===")
for (required in c(
  GLOBAL_TEMPLATE,
  GLOBAL_TARGET,
  GLOBAL_PARAMETERS,
  GLOBAL_PARAMETERS_COMPLETE
)) {
  if (!file.exists(required)) {
    stop(
      "Missing frozen GEO train-reference RMA artifact: ", required,
      ". Rerun stage 3 first.", call. = FALSE
    )
  }
}

download_atomic(IKEM_METADATA_URL, IKEM_METADATA)
download_atomic(IKEM_URL, IKEM_ARCHIVE)
metadata <- parse_ikem_metadata(IKEM_METADATA)
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
if (length(cel_members) != IKEM_EXPECTED_SAMPLES || anyDuplicated(cel_members)) {
  stop(
    "GSE290167 RAW archive contains ", length(cel_members),
    " unique CEL members; expected ", IKEM_EXPECTED_SAMPLES, ".", call. = FALSE
  )
}
member_gsm <- toupper(sub(".*?(GSM[0-9]+).*", "\\1", cel_members, perl = TRUE))
match_pos <- match(metadata$GSM, member_gsm)
if (anyNA(match_pos)) {
  stop("At least one GSE290167 GSM has no CEL member in the RAW archive.", call. = FALSE)
}
metadata$raw_member <- cel_members[match_pos]
expected_basename <- basename(sub("^ftp://", "https://", metadata$supplementary_url))
if (!identical(metadata$raw_member, expected_basename)) {
  stop("GEO sample metadata and RAW TAR member names do not agree exactly.", call. = FALSE)
}
metadata <- legacy_order(metadata)
metadata$row_index_python <- seq_len(nrow(metadata)) - 1L
local_reference <- ikem_training_reference(FROZEN_SPLIT_CSV, metadata)
local_reference_positions <- as.integer(local_reference$gse290167_row_r)
metadata$is_local_rma_reference_train <- seq_len(nrow(metadata)) %in%
  local_reference_positions

# A partial or completed run is reusable only for the exact same CEL archive,
# metadata, common-probe template, GEO-train quantile target and frozen probe
# effects.  This prevents a resumed run from silently mixing two frozen splits.
signature_paths <- c(
  IKEM_ARCHIVE,
  IKEM_METADATA,
  FROZEN_SPLIT_CSV,
  GLOBAL_TEMPLATE,
  GLOBAL_TARGET,
  GLOBAL_PARAMETERS
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
  file.exists(IKEM_LOCAL_PARAMETERS) ||
  length(list.files(IKEM_PROGRESS_DIR, all.files = FALSE)) > 0L
if (has_partial_state && !identical(prior_signature, input_signature)) {
  message("IKEM CEL inputs/reference changed; discarding incompatible partial matrices.")
  unlink(
    c(IKEM_FINAL_H5, IKEM_PROBE_H5, IKEM_LOCAL_PARAMETERS, IKEM_COMPLETE),
    force = TRUE
  )
  unlink(IKEM_PROGRESS_DIR, recursive = TRUE, force = TRUE)
  dir.create(IKEM_PROGRESS_DIR, recursive = TRUE, showWarnings = FALSE)
  unlink(
    c(
      file.path(IKEM_WORK_DIR, "ikem_target_state.rds"),
      file.path(IKEM_WORK_DIR, "ikem_cohort_quantile_target.rds"),
      file.path(IKEM_WORK_DIR, "ikem_train_quantile_target.rds")
    ),
    force = TRUE
  )
}
writeLines(input_signature, IKEM_SIGNATURE_FILE)

correspondence <- metadata[, c(
  "row_index_python", "sample_id", "sample_id_geo", "sample_key_upper",
  "GSM", "patient_id_geo", "tissue_geo", "platform", "raw_member",
  "legacy_row_index_python", "is_local_rma_reference_train"
)]
correspondence$metadata_match <- TRUE
correspondence$raw_tar_match <- TRUE
correspondence$legacy_id_match <- if (nzchar(IKEM_LEGACY_STORE)) TRUE else NA
atomic_write_csv_ikem(
  correspondence,
  file.path(IKEM_OUT_DIR, "ikem_gse290167_correspondence.csv")
)
atomic_write_csv_ikem(
  local_reference,
  file.path(IKEM_OUT_DIR, "ikem_rma_reference_samples.csv")
)

template <- readRDS(GLOBAL_TEMPLATE)
common_probes <- as.character(template$common_probes)
global_target <- readRDS(GLOBAL_TARGET)
global_effect <- rhdf5::h5read(
  GLOBAL_PARAMETERS,
  "probe_effect_common_pm",
  native = TRUE
)
if (length(global_target) != template$n_all_pm ||
    length(global_effect) != template$n_common_pm ||
    any(!is.finite(global_target)) || any(!is.finite(global_effect))) {
  stop("Frozen GEO train-reference RMA parameters are invalid.", call. = FALSE)
}

legacy_probe_path <- file.path(IKEM_LEGACY_STORE, "probe_index.csv")
if (nzchar(IKEM_LEGACY_STORE) && file.exists(legacy_probe_path)) {
  legacy_probes <- utils::read.csv(
    legacy_probe_path,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  candidates <- intersect(c("probe_id", "probe", "probeset_id", "ID", "id"), names(legacy_probes))
  if (length(candidates) == 0L ||
      !identical(as.character(legacy_probes[[candidates[[1L]]]]), common_probes)) {
    stop("Existing IKEM probe order does not equal the frozen common-probe order.", call. = FALSE)
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

if (file.exists(IKEM_COMPLETE)) {
  message("IKEM CEL completion marker found; reusing completed matrices.")
} else {
  extracted_marker <- file.path(IKEM_WORK_DIR, "EXTRACTION_COMPLETE.txt")
  extracted_before <- list.files(
    IKEM_EXTRACT_DIR,
    pattern = "\\.CEL(\\.gz)?$",
    recursive = TRUE,
    full.names = TRUE,
    ignore.case = TRUE
  )
  if (!file.exists(extracted_marker) ||
      length(extracted_before) != IKEM_EXPECTED_SAMPLES) {
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
    if (length(extracted) != IKEM_EXPECTED_SAMPLES) {
      stop("GSE290167 extraction did not produce exactly 276 CEL files.", call. = FALSE)
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
  extracted_gsm <- toupper(sub(".*?(GSM[0-9]+).*", "\\1", basename(extracted), perl = TRUE))
  paths <- extracted[match(metadata$GSM, extracted_gsm)]
  if (anyNA(paths) || !all(file.exists(paths))) {
    stop("Extracted IKEM CEL files cannot be aligned to GEO metadata.", call. = FALSE)
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
    stop("GSE290167 CEL header is not PrimeView.", call. = FALSE)
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
    "rma_cohort_legacy",
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

  pass1_state_path <- file.path(IKEM_WORK_DIR, "ikem_target_state.rds")
  cohort_target_path <- file.path(IKEM_WORK_DIR, "ikem_cohort_quantile_target.rds")
  local_target_path <- file.path(IKEM_WORK_DIR, "ikem_train_quantile_target.rds")
  pass1_done_path <- file.path(IKEM_PROGRESS_DIR, "pass1_done_sample.txt")
  if (file.exists(pass1_state_path)) {
    target_state <- readRDS(pass1_state_path)
    if (is.null(target_state$done_keys)) {
      target_state$done_keys <- read_done_ikem(pass1_done_path)
    }
  } else {
    target_state <- list(
      cohort_sum = numeric(template$n_all_pm),
      cohort_n = 0L,
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
    cohort_sum <- target_state$cohort_sum
    cohort_n <- target_state$cohort_n
    train_sum <- target_state$local_train_sum
    train_n <- target_state$local_train_n
    for (k in seq_along(batch)) {
      intensity <- read_one_ikem_cel(paths[[batch[[k]]]])
      pm <- intensity[template$all_pm_cell_index]
      corrected <- preprocessCore::rma.background.correct(
        matrix(pm, ncol = 1L), copy = FALSE
      )[, 1L]
      common_pm[, k] <- pm[template$common_pm_positions_grouped]
      sorted_corrected <- sort(corrected)
      cohort_sum <- cohort_sum + sorted_corrected
      cohort_n <- cohort_n + 1L
      if (batch[[k]] %in% local_reference_positions) {
        train_sum <- train_sum + sorted_corrected
        train_n <- train_n + 1L
      }
      rm(intensity, pm, corrected, sorted_corrected)
    }
    raw_block <- raw_for_batch(common_pm)
    rhdf5::h5write(
      raw_block,
      IKEM_FINAL_H5,
      "expression/raw_original",
      index = list(batch, seq_len(n_probes)),
      native = TRUE
    )
    target_state$cohort_sum <- cohort_sum
    target_state$cohort_n <- cohort_n
    target_state$local_train_sum <- train_sum
    target_state$local_train_n <- train_n
    target_state$done_keys <- c(target_state$done_keys, keys)
    atomic_save_rds_ikem(target_state, pass1_state_path, compress = FALSE)
    for (key in keys) append_done_ikem(pass1_done_path, key)
    pass1_done <- c(pass1_done, keys)
    rm(common_pm, raw_block, cohort_sum, train_sum)
    gc()
    message("[IKEM PASS1] ", batch_end, "/", n_samples, " CELs.")
  }
  if (target_state$cohort_n != length(unique(target_state$done_keys))) {
    stop("IKEM PASS1 checkpoint is internally inconsistent.", call. = FALSE)
  }
  if (target_state$cohort_n != n_samples) {
    stop("IKEM quantile target sample count is incomplete.", call. = FALSE)
  }
  if (target_state$local_train_n != length(local_reference_positions)) {
    stop("IKEM train-reference target sample count is incomplete.", call. = FALSE)
  }
  cohort_target <- target_state$cohort_sum / target_state$cohort_n
  local_target <- target_state$local_train_sum / target_state$local_train_n
  if (length(cohort_target) != template$n_all_pm ||
      length(local_target) != template$n_all_pm ||
      any(!is.finite(cohort_target)) || any(!is.finite(local_target))) {
    stop("IKEM PASS1 produced an invalid quantile target.", call. = FALSE)
  }
  atomic_save_rds_ikem(cohort_target, cohort_target_path, compress = TRUE)
  atomic_save_rds_ikem(local_target, local_target_path, compress = TRUE)

  if (!file.exists(IKEM_PROBE_H5)) {
    rhdf5::h5createFile(IKEM_PROBE_H5)
  }
  probe_objects <- rhdf5::h5ls(IKEM_PROBE_H5, recursive = TRUE)
  for (dataset in c("cohort_normalized_common_pm", "local_normalized_common_pm")) {
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
    cohort_common <- matrix(
      NA_real_, nrow = template$n_common_pm, ncol = length(batch)
    )
    local_common <- matrix(
      NA_real_, nrow = template$n_common_pm, ncol = length(batch)
    )
    global_common <- matrix(
      NA_real_, nrow = template$n_common_pm, ncol = length(batch)
    )
    for (k in seq_along(batch)) {
      intensity <- read_one_ikem_cel(paths[[batch[[k]]]])
      pm <- intensity[template$all_pm_cell_index]
      corrected <- preprocessCore::rma.background.correct(
        matrix(pm, ncol = 1L), copy = FALSE
      )
      cohort_norm <- preprocessCore::normalize.quantiles.use.target(
        corrected, target = cohort_target, copy = TRUE
      )[, 1L]
      local_norm <- preprocessCore::normalize.quantiles.use.target(
        corrected, target = local_target, copy = TRUE
      )[, 1L]
      global_norm <- preprocessCore::normalize.quantiles.use.target(
        corrected, target = global_target, copy = FALSE
      )[, 1L]
      cohort_common[, k] <- cohort_norm[template$common_pm_positions_grouped]
      local_common[, k] <- local_norm[template$common_pm_positions_grouped]
      global_common[, k] <- global_norm[template$common_pm_positions_grouped]
      rm(intensity, pm, corrected, cohort_norm, local_norm, global_norm)
    }
    global_block <- global_for_batch(global_common)
    if (any(!is.finite(cohort_common)) || any(!is.finite(local_common)) ||
        any(!is.finite(global_block))) {
      stop("Non-finite IKEM RMA value in PASS2.", call. = FALSE)
    }
    rhdf5::h5write(
      cohort_common,
      IKEM_PROBE_H5,
      "cohort_normalized_common_pm",
      index = list(seq_len(template$n_common_pm), batch),
      native = FALSE
    )
    rhdf5::h5write(
      local_common,
      IKEM_PROBE_H5,
      "local_normalized_common_pm",
      index = list(seq_len(template$n_common_pm), batch),
      native = FALSE
    )
    rhdf5::h5write(
      global_block,
      IKEM_FINAL_H5,
      "expression/rma_global",
      index = list(batch, seq_len(n_probes)),
      native = TRUE
    )
    for (key in keys) append_done_ikem(
      file.path(IKEM_PROGRESS_DIR, "pass2_done_sample.txt"), key
    )
    pass2_done <- c(pass2_done, keys)
    rm(cohort_common, local_common, global_common, global_block)
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
    cohort_normalized <- rhdf5::h5read(
      IKEM_PROBE_H5,
      "cohort_normalized_common_pm",
      index = list(pm_rows, seq_len(n_samples)),
      native = FALSE
    )
    local_normalized <- rhdf5::h5read(
      IKEM_PROBE_H5,
      "local_normalized_common_pm",
      index = list(pm_rows, seq_len(n_samples)),
      native = FALSE
    )
    labels <- rep(seq_along(probe_ids), times = template$common_pm_counts[probe_ids])
    cohort_summarized <- preprocessCore::subColSummarizeMedianpolishLog(
      cohort_normalized,
      labels
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
    if (!identical(dim(cohort_summarized), expected_dims) ||
        !identical(dim(local_summarized), expected_dims) ||
        any(!is.finite(cohort_summarized)) ||
        any(!is.finite(local_summarized)) ||
        any(!is.finite(local_effect_block))) {
      stop("Invalid cohort-RMA summary in IKEM PASS3 block ", block_id, call. = FALSE)
    }
    rhdf5::h5write(
      t(cohort_summarized),
      IKEM_FINAL_H5,
      "expression/rma_cohort_legacy",
      index = list(seq_len(n_samples), probe_ids),
      native = TRUE
    )
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
      cohort_normalized,
      local_normalized,
      cohort_summarized,
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

  for (method in c(
    "raw_original",
    "rma_cohort_legacy",
    "rma_per_gse",
    "rma_global"
  )) {
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
  writeLines(
    paste(
      ikem_time(), "IKEM_CEL_PREPROCESSING_COMPLETE",
      "GSE=", IKEM_GSE, "samples=", n_samples, "probes=", n_probes,
      "local_reference_train_samples=", length(local_reference_positions),
      "signature=", input_signature
    ),
    IKEM_COMPLETE
  )
  unlink(IKEM_EXTRACT_DIR, recursive = TRUE, force = TRUE)
  unlink(extracted_marker, force = TRUE)
}

message("IKEM CEL matrices: ", normalizePath(IKEM_FINAL_H5))
message("IKEM correspondence: ", normalizePath(file.path(
  IKEM_OUT_DIR, "ikem_gse290167_correspondence.csv"
)))
