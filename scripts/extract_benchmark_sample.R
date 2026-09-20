suppressPackageStartupMessages({
  library(jsonlite)
})

cat("=== Extracting Benchmark Sample (including ORDON*) ===\n")

# 1. Sample from D0840/docs
d0840_path <- "C:/Users/franc/Documents/Datasets/D0840/docs"
cat("Reading D0840/docs...\n")
d0840 <- readRDS(d0840_path)

# Filter non-empty RECTXT
valid_idx <- which(!is.na(d0840$RECTXT) & nchar(d0840$RECTXT) >= 50)
cat(sprintf(
  "Valid D0840 docs (len >= 50): %d / %d\n",
  length(valid_idx),
  nrow(d0840)
))

# Stratified sample of 1,000 docs from D0840 across top RECTYPEs
set.seed(42)
d0840_valid <- d0840[valid_idx, c("RECTXT", "RECTYPE", "SEJUM", "SEJUF")]
d0840_valid$doc_id <- paste0("D0840_", valid_idx)
d0840_valid$dataset <- "D0840"

# Group by RECTYPE and sample up to 50 docs per type to get wide variety including ORDON*
types <- split(d0840_valid, d0840_valid$RECTYPE)
sampled_list <- lapply(types, function(sub) {
  n <- min(nrow(sub), 30)
  sub[sample(nrow(sub), n), ]
})
d0840_sample <- do.call(rbind, sampled_list)

# If total > 1000, take 1000; if < 1000, fill randomly
if (nrow(d0840_sample) > 1000) {
  # ensure ORDON* types are well represented
  d0840_sample <- d0840_sample[sample(nrow(d0840_sample), 1000), ]
}
cat(sprintf(
  "Sampled D0840 docs: %d across %d types\n",
  nrow(d0840_sample),
  length(unique(d0840_sample$RECTYPE))
))

# 2. Sample from denut.rds
denut_path <- "C:/Users/franc/Documents/Datasets/denut.rds"
cat("Reading denut.rds...\n")
denut <- readRDS(denut_path)

denut_docs_list <- list()
for (i in seq_along(denut)) {
  b <- denut[[i]]
  doceds <- b[["sources"]][["doceds"]]
  if (is.data.frame(doceds) && nrow(doceds) > 0) {
    sub_docs <- doceds[, intersect(
      c("RECTXT", "RECTYPE", "SEJUM", "SEJUF", "ELTID", "EVTID"),
      names(doceds)
    )]
    valid_sub <- which(!is.na(sub_docs$RECTXT) & nchar(sub_docs$RECTXT) >= 50)
    if (length(valid_sub) > 0) {
      sub_docs <- sub_docs[valid_sub, ]
      sub_docs$doc_id <- paste0(
        "DENUT_",
        if ("ELTID" %in% names(sub_docs)) {
          sub_docs$ELTID
        } else {
          seq_len(nrow(sub_docs))
        }
      )
      sub_docs$dataset <- "DENUT"
      denut_docs_list[[length(denut_docs_list) + 1]] <- sub_docs[, c(
        "doc_id",
        "dataset",
        "RECTYPE",
        "SEJUM",
        "SEJUF",
        "RECTXT"
      )]
    }
  }
}
denut_all <- do.call(rbind, denut_docs_list)
cat(sprintf("Total valid docs in denut.rds: %d\n", nrow(denut_all)))

set.seed(42)
n_denut_sample <- min(nrow(denut_all), 300)
denut_sample <- denut_all[sample(nrow(denut_all), n_denut_sample), ]
cat(sprintf(
  "Sampled denut docs: %d across %d types\n",
  nrow(denut_sample),
  length(unique(denut_sample$RECTYPE))
))

# Combine both
final_sample <- rbind(
  d0840_sample[, c("doc_id", "dataset", "RECTYPE", "SEJUM", "SEJUF", "RECTXT")],
  denut_sample[, c("doc_id", "dataset", "RECTYPE", "SEJUM", "SEJUF", "RECTXT")]
)

names(final_sample) <- tolower(names(final_sample))
cat(sprintf("Combined benchmark sample: %d documents\n", nrow(final_sample)))

out_path <- "data/processed/benchmark_sample.jsonl"
dir.create(dirname(out_path), showWarnings = FALSE, recursive = TRUE)

con <- file(out_path, "w", encoding = "UTF-8")
for (i in seq_len(nrow(final_sample))) {
  row <- as.list(final_sample[i, ])
  writeLines(toJSON(row, auto_unbox = TRUE), con)
}
close(con)

cat(sprintf("Successfully written to: %s\n", out_path))
