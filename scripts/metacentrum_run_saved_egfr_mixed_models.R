# Flat source-tree helper for MetaCentrum.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) {
  stop("Usage: metacentrum_run_saved_egfr_mixed_models.R DESIGN SPECS METRICS_OUT PREDICTIONS_OUT")
}
if (!requireNamespace("lme4", quietly = TRUE)) {
  stop(
    paste0(
      "R package 'lme4' is unavailable. R_LIBS=", Sys.getenv("R_LIBS"),
      "; .libPaths()=", paste(.libPaths(), collapse = ":")
    )
  )
}

design_path <- normalizePath(args[[1]], mustWork = TRUE)
specs_path <- normalizePath(args[[2]], mustWork = TRUE)
metrics_path <- args[[3]]
predictions_path <- args[[4]]

design <- read.csv(design_path, stringsAsFactors = FALSE, check.names = FALSE)
specs <- read.csv(specs_path, stringsAsFactors = FALSE, check.names = FALSE)
design$time <- factor(design$time, levels = c("7d", "3m", "6m", "12m"))
design$patient <- factor(design$patient)

if (!"fit_id" %in% names(specs) || !"fit_id" %in% names(design)) {
  stop("The saved nested design and specifications must both contain fit_id.")
}
if (anyDuplicated(specs$fit_id)) {
  stop("The saved specifications contain duplicate fit_id values.")
}

# Resume only fit IDs for which both a metric and prediction rows were safely
# written. If a process stopped between the two writes, its unmatched metric is
# discarded and that fit is recomputed.
completed_ids <- character()
have_metrics <- file.exists(metrics_path) && file.info(metrics_path)$size > 0
have_predictions <- file.exists(predictions_path) && file.info(predictions_path)$size > 0
if (have_metrics && have_predictions) {
  previous_metrics <- read.csv(
    metrics_path,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  previous_predictions <- read.csv(
    predictions_path,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  if (
    "fit_id" %in% names(previous_metrics) &&
      "fit_id" %in% names(previous_predictions)
  ) {
    completed_ids <- intersect(
      unique(previous_metrics$fit_id),
      unique(previous_predictions$fit_id)
    )
    completed_ids <- intersect(specs$fit_id, completed_ids)
  }
  previous_metrics <- previous_metrics[
    previous_metrics$fit_id %in% completed_ids,
    ,
    drop = FALSE
  ]
  previous_metrics <- previous_metrics[
    !duplicated(previous_metrics$fit_id),
    ,
    drop = FALSE
  ]
  previous_predictions <- previous_predictions[
    previous_predictions$fit_id %in% completed_ids,
    ,
    drop = FALSE
  ]
  if (length(completed_ids) > 0) {
    write.table(
      previous_metrics,
      metrics_path,
      sep = ",",
      row.names = FALSE,
      col.names = TRUE,
      append = FALSE,
      qmethod = "double"
    )
    write.table(
      previous_predictions,
      predictions_path,
      sep = ",",
      row.names = FALSE,
      col.names = TRUE,
      append = FALSE,
      qmethod = "double"
    )
  }
  rm(previous_metrics, previous_predictions)
}
if (length(completed_ids) == 0) {
  if (file.exists(metrics_path)) file.remove(metrics_path)
  if (file.exists(predictions_path)) file.remove(predictions_path)
}
metrics_header <- length(completed_ids) == 0
predictions_header <- length(completed_ids) == 0

message(sprintf(
  "Starting %d lme4 fits with R %s and lme4 %s",
  nrow(specs),
  as.character(getRversion()),
  as.character(utils::packageVersion("lme4"))
))
if (length(completed_ids) > 0) {
  message(sprintf(
    "Resuming from %d completed fits; %d fits remain.",
    length(completed_ids),
    nrow(specs) - length(completed_ids)
  ))
}

for (i in seq_len(nrow(specs))) {
  spec <- specs[i, ]
  if (spec$fit_id %in% completed_ids) next
  block <- design[design$fit_id == spec$fit_id, , drop = FALSE]
  train <- block[block$partition == "train", , drop = FALSE]
  test <- block[block$partition == "test", , drop = FALSE]
  if (nrow(train) == 0 || nrow(test) == 0) {
    stop(sprintf("Fit %s has an empty train or test partition.", spec$fit_id))
  }

  n_features <- as.integer(spec$n_features)
  n_main_features <- as.integer(spec$n_main_features)
  n_time_interaction_features <- as.integer(spec$n_time_interaction_features)
  if (n_features != n_main_features + n_time_interaction_features) {
    stop(paste("Inconsistent feature counts for", spec$fit_id))
  }

  fixed_terms <- "time"
  if (n_main_features > 0) {
    fixed_terms <- c(fixed_terms, paste0("x", seq_len(n_main_features)))
  }
  if (n_time_interaction_features > 0) {
    interaction_indices <- n_main_features + seq_len(n_time_interaction_features)
    fixed_terms <- c(
      fixed_terms,
      paste0("x", interaction_indices, " * time")
    )
  }
  formula_text <- paste(
    "egfr ~",
    paste(fixed_terms, collapse = " + "),
    "+ (1 | patient)"
  )

  fit <- tryCatch(
    lme4::lmer(
      stats::as.formula(formula_text),
      data = train,
      REML = TRUE,
      control = lme4::lmerControl(
        check.conv.singular = lme4:::.makeCC(action = "ignore", tol = 1e-4)
      )
    ),
    error = function(error) {
      stop(sprintf("lme4 failed for fit %s: %s", spec$fit_id, conditionMessage(error)))
    }
  )

  prediction <- stats::predict(
    fit,
    newdata = test,
    re.form = NA,
    allow.new.levels = TRUE
  )
  error <- test$egfr - prediction
  convergence_messages <- fit@optinfo$conv$lme4$messages
  if (is.null(convergence_messages)) convergence_messages <- ""

  metadata <- data.frame(
    fit_id = spec$fit_id,
    stage = spec$stage,
    model_id = spec$model_id,
    model_label = spec$model_label,
    candidate_id = spec$candidate_id,
    `repeat` = spec[["repeat"]],
    fold = spec$fold,
    inner_fold = spec$inner_fold,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  metric_row <- cbind(
    metadata,
    data.frame(
      n_features = n_features,
      n_main_features = n_main_features,
      n_time_interaction_features = n_time_interaction_features,
      n_train_rows = nrow(train),
      n_test_rows = nrow(test),
      n_train_patients = length(unique(train$patient)),
      n_test_patients = length(unique(test$patient)),
      rmse = sqrt(mean(error^2)),
      mae = mean(abs(error)),
      singular = lme4::isSingular(fit),
      convergence_message = paste(convergence_messages, collapse = " | "),
      formula = formula_text,
      stringsAsFactors = FALSE,
      check.names = FALSE
    )
  )
  prediction_rows <- cbind(
    metadata[rep(1, nrow(test)), , drop = FALSE],
    data.frame(
      patient = as.character(test$patient),
      donor = test$donor,
      time = as.character(test$time),
      egfr = test$egfr,
      prediction = as.numeric(prediction),
      stringsAsFactors = FALSE,
      check.names = FALSE
    )
  )

  write.table(
    metric_row,
    metrics_path,
    sep = ",",
    row.names = FALSE,
    col.names = metrics_header,
    append = !metrics_header,
    qmethod = "double"
  )
  write.table(
    prediction_rows,
    predictions_path,
    sep = ",",
    row.names = FALSE,
    col.names = predictions_header,
    append = !predictions_header,
    qmethod = "double"
  )
  metrics_header <- FALSE
  predictions_header <- FALSE

  if (i == 1 || i %% 25 == 0 || i == nrow(specs)) {
    message(sprintf("Completed %d/%d fits (%s)", i, nrow(specs), spec$fit_id))
  }
  rm(block, train, test, fit, prediction, error, metric_row, prediction_rows)
  if (i %% 25 == 0) gc(verbose = FALSE)
}

message(sprintf("Completed all %d lme4 fits.", nrow(specs)))
