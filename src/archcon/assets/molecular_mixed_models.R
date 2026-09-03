args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) {
  stop("Usage: molecular_mixed_models.R DESIGN SPECS METRICS_OUT PREDICTIONS_OUT")
}
if (!requireNamespace("lme4", quietly = TRUE)) {
  stop("R package 'lme4' is required. Install it before running this benchmark.")
}

design <- read.csv(args[[1]], stringsAsFactors = FALSE, check.names = FALSE)
specs <- read.csv(args[[2]], stringsAsFactors = FALSE, check.names = FALSE)
design$time <- factor(design$time, levels = c("7d", "3m", "6m", "12m"))
design$patient <- factor(design$patient)

metrics <- list()
predictions <- list()

for (i in seq_len(nrow(specs))) {
  spec <- specs[i, ]
  if ("fit_id" %in% names(specs) && "fit_id" %in% names(design)) {
    block <- design[design$fit_id == spec$fit_id, ]
  } else {
    block <- design[
      design$model_id == spec$model_id &
        design[["repeat"]] == spec[["repeat"]] &
        design$fold == spec$fold,
    ]
  }
  train <- block[block$partition == "train", ]
  test <- block[block$partition == "test", ]
  n_features <- as.integer(spec$n_features)
  n_main_features <- as.integer(spec$n_main_features)
  n_time_interaction_features <- as.integer(spec$n_time_interaction_features)
  if (n_features != n_main_features + n_time_interaction_features) {
    stop(paste("Inconsistent feature counts for", spec$model_id))
  }
  fixed_terms <- "time"
  if (n_main_features > 0) {
    fixed_terms <- c(fixed_terms, paste0("x", seq_len(n_main_features)))
  }
  if (n_time_interaction_features > 0) {
    interaction_indices <- n_main_features + seq_len(n_time_interaction_features)
    fixed_terms <- c(fixed_terms, paste0("x", interaction_indices, " * time"))
  }
  formula_text <- paste("egfr ~", paste(fixed_terms, collapse = " + "), "+ (1 | patient)")
  fit <- lme4::lmer(
    stats::as.formula(formula_text),
    data = train,
    REML = TRUE,
    control = lme4::lmerControl(
      check.conv.singular = lme4:::.makeCC(action = "ignore", tol = 1e-4)
    )
  )
  prediction <- stats::predict(fit, newdata = test, re.form = NA, allow.new.levels = TRUE)
  error <- test$egfr - prediction
  convergence_messages <- fit@optinfo$conv$lme4$messages
  if (is.null(convergence_messages)) convergence_messages <- ""
  metadata <- data.frame(
    fit_id = if ("fit_id" %in% names(specs)) spec$fit_id else "",
    stage = if ("stage" %in% names(specs)) spec$stage else "outer_evaluation",
    model_id = spec$model_id,
    model_label = spec$model_label,
    candidate_id = if ("candidate_id" %in% names(specs)) spec$candidate_id else "",
    `repeat` = spec[["repeat"]],
    fold = spec$fold,
    inner_fold = if ("inner_fold" %in% names(specs)) spec$inner_fold else -1,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  metrics[[length(metrics) + 1]] <- cbind(metadata, data.frame(
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
  ))
  predictions[[length(predictions) + 1]] <- cbind(
    metadata[rep(1, nrow(test)), , drop = FALSE],
    data.frame(
    patient = as.character(test$patient),
    donor = test$donor,
    time = as.character(test$time),
    egfr = test$egfr,
    prediction = as.numeric(prediction),
    stringsAsFactors = FALSE,
    check.names = FALSE
  ))
}

write.csv(do.call(rbind, metrics), args[[3]], row.names = FALSE)
write.csv(do.call(rbind, predictions), args[[4]], row.names = FALSE)
