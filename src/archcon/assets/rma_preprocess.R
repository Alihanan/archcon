args <- commandArgs(trailingOnly=TRUE)

if (length(args) < 3) {
  stop("Usage: rma_preprocess.R <expression.csv> <raw_qc.csv> <CEL...>")
}

out_expression <- args[[1]]
out_raw_qc <- args[[2]]
cel_files <- args[3:length(args)]

suppressPackageStartupMessages(library(affy))

cel_data <- ReadAffy(filenames=cel_files)

# A deterministic subset of raw probe intensities is enough for browser QC.
raw_matrix <- exprs(cel_data)
n_keep <- min(5000L, nrow(raw_matrix))
raw_idx <- unique(as.integer(round(seq(1, nrow(raw_matrix), length.out=n_keep))))
raw_subset <- raw_matrix[raw_idx, , drop=FALSE]
colnames(raw_subset) <- basename(sampleNames(cel_data))
write.csv(raw_subset, file=out_raw_qc, row.names=FALSE)

eset <- rma(cel_data)

sample_names <- basename(rownames(pData(eset)))
sample_ID <- sapply(sample_names, function(x) {
  parts <- strsplit(x, "_", fixed=TRUE)[[1]]
  if (length(parts) >= 2) paste(parts[1:2], collapse="_") else x
})

# Preserve the thesis behavior: keep the first occurrence of a duplicate ID.
keep <- !duplicated(sample_ID)
eset <- eset[, keep]
sample_ID <- sample_ID[keep]

expr_t <- t(exprs(eset))
rownames(expr_t) <- sample_ID

write.csv(expr_t, file=out_expression, row.names=TRUE)
cat(sprintf("Saved %d samples x %d probes to %s\n", nrow(expr_t), ncol(expr_t), out_expression))
