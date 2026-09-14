# ============================================================
# Download raw GEO archives from download_summary.csv
# ============================================================

CSV_FILE <- "download_summary.csv"
OUT_DIR <- Sys.getenv("ARCHCON_GEO_RAW_DIR", unset = "GEO_RAW")

# Retry settings
MAX_ATTEMPTS <- 20L
RETRY_DELAY_SECONDS <- 10L
CONNECT_TIMEOUT_SECONDS <- 30L

# curl itself also performs several retries inside each attempt
CURL_RETRIES <- 3L


# ------------------------------------------------------------
# Checks
# ------------------------------------------------------------

if (Sys.which("curl") == "") {
  stop(
    "'curl' was not found on the system.\n",
    "On Pop!_OS / Ubuntu install it with:\n",
    "sudo apt install curl"
  )
}

if (!file.exists(CSV_FILE)) {
  stop("CSV file does not exist: ", CSV_FILE)
}

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

TAR_BIN <- Sys.which("tar")
if (!nzchar(TAR_BIN)) {
  stop("'tar' was not found; RAW archives cannot be validated.")
}

VALIDATION_DIR <- file.path(OUT_DIR, ".validated_tar")
dir.create(VALIDATION_DIR, recursive = TRUE, showWarnings = FALSE)


# ------------------------------------------------------------
# Read CSV
#
# Read everything as character so values such as "True" are not
# silently changed to TRUE when the CSV is written back.
# ------------------------------------------------------------

dat <- read.csv(
  CSV_FILE,
  stringsAsFactors = FALSE,
  check.names = FALSE,
  colClasses = "character"
)

if (!"gse_id" %in% names(dat)) {
  stop("CSV must contain a 'gse_id' column.")
}

# Your uploaded CSV has:
#
# gse_id,n_samples,n_genes,cached,
#
# i.e. the fifth column is the X / RAW-downloaded column.
#
# Give it a proper name.
if ("raw_downloaded" %in% names(dat)) {

  mark_col <- which(names(dat) == "raw_downloaded")[1]

} else if (ncol(dat) >= 5L) {

  mark_col <- 5L
  names(dat)[mark_col] <- "raw_downloaded"

} else {

  dat$raw_downloaded <- ""
  mark_col <- which(names(dat) == "raw_downloaded")
}

dat[[mark_col]][is.na(dat[[mark_col]])] <- ""


# ------------------------------------------------------------
# Save progress
# ------------------------------------------------------------

save_progress <- function() {

  write.table(
    dat,
    file = CSV_FILE,
    sep = ",",
    row.names = FALSE,
    col.names = TRUE,
    quote = FALSE,
    na = ""
  )
}


# Save once immediately so the previously unnamed column
# becomes "raw_downloaded".
save_progress()


# ------------------------------------------------------------
# Validate complete TAR structure before trusting/marking X
# ------------------------------------------------------------

archive_signature <- function(path) {
  info <- file.info(path)
  list(
    size = as.numeric(info$size),
    mtime = as.numeric(info$mtime)
  )
}

validation_marker <- function(gse) {
  file.path(VALIDATION_DIR, paste0(gse, ".rds"))
}

write_validation_marker <- function(gse, signature) {
  marker <- validation_marker(gse)
  tmp <- paste0(marker, ".part")
  unlink(tmp, force = TRUE)
  saveRDS(signature, tmp)
  if (file.exists(marker)) unlink(marker, force = TRUE)
  if (!file.rename(tmp, marker)) {
    stop("Could not save TAR-validation checkpoint for ", gse)
  }
}

validate_tar_archive <- function(path, gse, use_cache = TRUE) {
  if (!file.exists(path) || file.info(path)$size <= 0) {
    return(list(valid = FALSE, detail = "file is absent or empty", signature = NULL))
  }

  signature <- archive_signature(path)
  marker <- validation_marker(gse)

  if (use_cache && file.exists(marker)) {
    saved <- tryCatch(readRDS(marker), error = function(e) NULL)
    if (
      !is.null(saved) &&
      identical(saved$size, signature$size) &&
      identical(saved$mtime, signature$mtime)
    ) {
      return(list(valid = TRUE, detail = "cached complete-TAR check", signature = signature))
    }
  }

  stderr_path <- tempfile(pattern = paste0(gse, "_tar_"), fileext = ".stderr")
  on.exit(unlink(stderr_path, force = TRUE), add = TRUE)

  status <- suppressWarnings(
    system2(
      TAR_BIN,
      args = c("-tf", shQuote(path)),
      stdout = FALSE,
      stderr = stderr_path
    )
  )
  if (is.null(status)) status <- 0L

  if (identical(as.integer(status), 0L)) {
    return(list(valid = TRUE, detail = "complete TAR listing", signature = signature))
  }

  detail <- if (file.exists(stderr_path)) {
    paste(readLines(stderr_path, warn = FALSE), collapse = " | ")
  } else {
    paste0("tar exited with status ", status)
  }

  list(valid = FALSE, detail = detail, signature = signature)
}


# ------------------------------------------------------------
# Helper: perform one curl download
# ------------------------------------------------------------

curl_download <- function(url, partial_file) {

  args <- c(
    "--location",
    "--fail",

    "--connect-timeout",
    as.character(CONNECT_TIMEOUT_SECONDS),

    "--retry",
    as.character(CURL_RETRIES),

    "--retry-delay",
    "5",

    "--retry-connrefused",

    # Resume an interrupted .part file.
    "--continue-at",
    "-",

    "--output",
    shQuote(partial_file),

    # Let R see the final HTTP status code.
    "--write-out",
    shQuote("%{http_code}"),

    shQuote(url)
  )

  result <- suppressWarnings(
    system2(
      "curl",
      args = args,
      stdout = TRUE,
      stderr = ""
    )
  )

  exit_status <- attr(result, "status")

  if (is.null(exit_status)) {
    exit_status <- 0L
  }

  # --write-out should leave the final HTTP code in stdout.
  http_codes <- suppressWarnings(
    as.integer(result[grepl("^[0-9]{3}$", result)])
  )

  http_codes <- http_codes[!is.na(http_codes)]

  http_status <- if (length(http_codes) > 0L) {
    tail(http_codes, 1L)
  } else {
    NA_integer_
  }

  list(
    success = identical(as.integer(exit_status), 0L),
    exit_status = as.integer(exit_status),
    http_status = http_status
  )
}


# ------------------------------------------------------------
# Main loop
# ------------------------------------------------------------

n <- nrow(dat)

for (i in seq_len(n)) {

  gse <- trimws(dat$gse_id[i])

  if (
    is.na(gse) ||
    !nzchar(gse)
  ) {
    next
  }

  message("")
  message("============================================================")
  message(sprintf("[%d / %d] %s", i, n, gse))
  message("============================================================")

  # ----------------------------------------------------------
  # 1. Construct filenames and URL
  # ----------------------------------------------------------

  final_file <- file.path(
    OUT_DIR,
    paste0(gse, "_RAW.tar")
  )

  partial_file <- paste0(
    final_file,
    ".part"
  )

  url <- paste0(
    "https://www.ncbi.nlm.nih.gov/geo/download/?acc=",
    gse,
    "&format=file"
  )


  # ----------------------------------------------------------
  # 2. A historical X is trusted only with a matching validated TAR
  # ----------------------------------------------------------

  marked <- toupper(trimws(dat[[mark_col]][i])) == "X"

  if (marked && !file.exists(final_file)) {
    message("Marked X but archive is absent -- clearing the stale marker.")
    dat[[mark_col]][i] <- ""
    save_progress()
    marked <- FALSE
  }


  # ----------------------------------------------------------
  # 3. Validate any archive already present, including marked-X rows
  # ----------------------------------------------------------

  if (file.exists(final_file)) {

    size_mb <- file.info(final_file)$size / 1024^2

    message(sprintf("Validating existing file: %s (%.1f MB)", final_file, size_mb))
    existing_check <- validate_tar_archive(final_file, gse, use_cache = TRUE)

    if (existing_check$valid) {
      write_validation_marker(gse, existing_check$signature)
      message("Complete TAR confirmed -- keeping and marking X.")
      dat[[mark_col]][i] <- "X"
      save_progress()
      next
    }

    message("Existing TAR is incomplete/corrupt: ", existing_check$detail)
    message("It will be retried as an interrupted .part download.")
    dat[[mark_col]][i] <- ""
    save_progress()
    unlink(validation_marker(gse), force = TRUE)

    if (file.exists(partial_file)) unlink(partial_file, force = TRUE)
    if (!file.rename(final_file, partial_file)) {
      stop("Could not move corrupt archive to resumable path: ", partial_file)
    }
  }


  # ----------------------------------------------------------
  # 4. Download / resume
  # ----------------------------------------------------------

  message("URL: ", url)

  if (file.exists(partial_file)) {

    size_mb <- file.info(partial_file)$size / 1024^2

    message(
      sprintf(
        "Found partial download: %.1f MB",
        size_mb
      )
    )

    message("Trying to resume it...")
  }


  downloaded <- FALSE

  for (attempt in seq_len(MAX_ATTEMPTS)) {

    message(
      sprintf(
        "Download attempt %d / %d...",
        attempt,
        MAX_ATTEMPTS
      )
    )

    result <- curl_download(
      url,
      partial_file
    )


    # --------------------------------------------------------
    # Success
    # --------------------------------------------------------

    if (result$success) {

      if (
        !file.exists(partial_file) ||
        file.info(partial_file)$size <= 0
      ) {

        message(
          "Server returned success but no archive was downloaded."
        )

        break
      }

      completed_check <- validate_tar_archive(partial_file, gse, use_cache = FALSE)
      if (!completed_check$valid) {
        message("Downloaded response is not a complete TAR: ", completed_check$detail)
        message("Removing the unusable partial response and retrying from byte zero.")
        unlink(partial_file, force = TRUE)

        if (attempt < MAX_ATTEMPTS) {
          Sys.sleep(RETRY_DELAY_SECONDS)
          next
        }

        break
      }

      if (!file.rename(partial_file, final_file)) {
        stop(
          "Download succeeded, but could not rename:\n",
          partial_file,
          "\nto:\n",
          final_file
        )
      }

      write_validation_marker(gse, completed_check$signature)

      size_mb <- file.info(final_file)$size / 1024^2

      message(
        sprintf(
          "SUCCESS: %s (%.1f MB)",
          final_file,
          size_mb
        )
      )

      # Mark successful download immediately.
      dat[[mark_col]][i] <- "X"
      save_progress()

      downloaded <- TRUE
      break
    }


    # --------------------------------------------------------
    # HTTP 4xx:
    #
    # 404 etc. normally means there is no downloadable RAW
    # archive for this accession. Don't retry endlessly.
    #
    # 408 and 429 are temporary, so do retry those.
    # --------------------------------------------------------

    if (
      !is.na(result$http_status) &&
      result$http_status >= 400L &&
      result$http_status < 500L &&
      !result$http_status %in% c(408L, 429L)
    ) {

      message(
        sprintf(
          "HTTP %d -- RAW archive not available. Skipping %s.",
          result$http_status,
          gse
        )
      )

      break
    }


    # --------------------------------------------------------
    # curl exit code 33 = server rejected resume.
    #
    # In that case throw away the partial file and try that
    # accession again from the beginning.
    # --------------------------------------------------------

    if (
      result$exit_status == 33L &&
      file.exists(partial_file)
    ) {

      message(
        "Server rejected resume. Removing partial file and ",
        "retrying from the beginning."
      )

      unlink(partial_file)
    }


    # --------------------------------------------------------
    # Temporary/network failure
    # --------------------------------------------------------

    if (attempt < MAX_ATTEMPTS) {

      message(
        sprintf(
          "Download failed (curl=%d, HTTP=%s). Retrying in %d seconds...",
          result$exit_status,
          ifelse(
            is.na(result$http_status),
            "unknown",
            result$http_status
          ),
          RETRY_DELAY_SECONDS
        )
      )

      Sys.sleep(RETRY_DELAY_SECONDS)

    } else {

      message(
        "Maximum attempts reached. Leaving this GSE unmarked ",
        "so it can be retried next time the script is run."
      )
    }
  }


  if (!downloaded) {
    message("Not downloaded: ", gse)
  }
}


# ------------------------------------------------------------
# Final checkpoint
# ------------------------------------------------------------

save_progress()

message("")
message("============================================================")
message("Finished.")
message("RAW archives: ", normalizePath(OUT_DIR))
message("Progress CSV: ", normalizePath(CSV_FILE))
message("============================================================")
