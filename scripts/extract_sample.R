#!/usr/bin/env Rscript
# Stratified extraction script for redsan-doc-trimmer
# Excludes ORDON* (prescriptions) and BT* (transport vouchers)

suppressPackageStartupMessages({
  library(jsonlite)
})

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default_val) {
  idx <- which(args == flag)
  if (length(idx) > 0 && idx < length(args)) {
    return(args[idx + 1])
  }
  return(default_val)
}

rds_path <- get_arg(
  "--rds",
  "C:/Users/franc/Documents/Datasets/D0740 - dmo nutrition/docs_merged_00_25"
)
out_path <- get_arg("--output", "data/raw/corpus_sample.jsonl")
target_n <- as.integer(get_arg("--n", "2000"))
seed_val <- as.integer(get_arg("--seed", "42"))

cat(sprintf("=== redsan-doc-trimmer: Corpus Extraction ===\n"))
cat(sprintf("Input RDS: %s\n", rds_path))
cat(sprintf("Output:    %s\n", out_path))
cat(sprintf("Target N:  %d documents\n", target_n))
cat(sprintf("Seed:      %d\n\n", seed_val))

if (!file.exists(rds_path)) {
  stop(sprintf("Input file does not exist: %s", rds_path))
}

# Ensure output directory exists
out_dir <- dirname(out_path)
if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
}

cat("Loading RDS dataset...\n")
df <- readRDS(rds_path)
total_raw <- nrow(df)
cat(sprintf("Loaded %d documents from RDS.\n", total_raw))

# 1. Apply exclusions
cat("\nApplying exclusion filters (ORDON* and BT*)...\n")
rectype_clean <- ifelse(is.na(df$RECTYPE), "UNKNOWN", as.character(df$RECTYPE))
is_ordon <- grepl("^ORDON", rectype_clean, ignore.case = TRUE)
is_bt <- grepl("(^BT|TRANSPORT|BON)", rectype_clean, ignore.case = TRUE)
has_valid_text <- !is.na(df$RECTXT) & nchar(trimws(df$RECTXT)) >= 20

keep_mask <- !is_ordon & !is_bt & has_valid_text
filtered_df <- df[keep_mask, ]
n_filtered <- nrow(filtered_df)
cat(sprintf(
  "Remaining clinical documents after filter: %d (excluded %d ORDON/BT/empty records)\n",
  n_filtered,
  total_raw - n_filtered
))

# 2. Stratification variables
set.seed(seed_val)

# Era stratification
years <- as.integer(substr(as.character(filtered_df$RECDATE), 1, 4))
years[is.na(years)] <- 2015
era <- ifelse(
  years < 2010,
  "2000-2009",
  ifelse(years <= 2017, "2010-2017", "2018-2025")
)

# RECTYPE grouping: keep top types, group very rare into OTHER
top_types <- names(head(
  sort(table(filtered_df$RECTYPE), decreasing = TRUE),
  12
))
rectype_strat <- ifelse(
  filtered_df$RECTYPE %in% top_types,
  as.character(filtered_df$RECTYPE),
  "OTHER_NARRATIVE"
)

# Specialty grouping: top 10 specialties, group rest into OTHER
top_sejum <- names(head(sort(table(filtered_df$SEJUM), decreasing = TRUE), 10))
sejum_strat <- ifelse(
  filtered_df$SEJUM %in% top_sejum,
  as.character(filtered_df$SEJUM),
  "OTHER_SPECIALTY"
)

# Combined stratum
stratum <- paste(rectype_strat, era, sejum_strat, sep = "__")
strata_counts <- table(stratum)

# Proportional allocation with minimum representation
allocations <- pmax(
  1,
  round(target_n * (as.numeric(strata_counts) / n_filtered))
)
# Normalize to target_n
scale_factor <- target_n / sum(allocations)
allocations <- pmax(1, round(allocations * scale_factor))

# Sample indices per stratum
cat("Sampling across strata...\n")
sampled_indices <- unlist(lapply(seq_along(strata_counts), function(i) {
  strat_name <- names(strata_counts)[i]
  pool <- which(stratum == strat_name)
  n_to_take <- min(allocations[i], length(pool))
  sample(pool, size = n_to_take, replace = FALSE)
}))

# If slightly under/over target due to rounding, adjust
if (length(sampled_indices) > target_n) {
  sampled_indices <- sample(sampled_indices, size = target_n, replace = FALSE)
} else if (length(sampled_indices) < target_n) {
  remaining <- setdiff(seq_len(n_filtered), sampled_indices)
  n_extra <- min(target_n - length(sampled_indices), length(remaining))
  if (n_extra > 0) {
    sampled_indices <- c(
      sampled_indices,
      sample(remaining, size = n_extra, replace = FALSE)
    )
  }
}

sample_df <- filtered_df[sampled_indices, ]
cat(sprintf("Selected sample size: %d documents\n\n", nrow(sample_df)))

# 3. Print Stratification Breakdown
cat("=== Sample Breakdown by Document Type (RECTYPE) ===\n")
print(sort(table(sample_df$RECTYPE, useNA = "always"), decreasing = TRUE))

cat("\n=== Sample Breakdown by Era ===\n")
sample_years <- as.integer(substr(as.character(sample_df$RECDATE), 1, 4))
sample_era <- ifelse(
  sample_years < 2010,
  "2000-2009",
  ifelse(sample_years <= 2017, "2010-2017", "2018-2025")
)
print(table(sample_era))

cat("\n=== Sample Breakdown by Specialty (SEJUM, top 10) ===\n")
print(head(
  sort(table(sample_df$SEJUM, useNA = "always"), decreasing = TRUE),
  10
))

# 4. Identify anchor section columns (clinical content verification)
sec_cols <- c(
  "RECTXT_TRUE_CONCLUSIONDIAGNOSTIC",
  "RECTXT_TRUE_EXAMENCLINIQUE",
  "RECTXT_TRUE_RESULTATS",
  "RECTXT_TRUE_MOTIFINDICATION",
  "RECTXT_TRUE_ANTECEDENTS",
  "RECTXT_TRUE_ANAMNESE",
  "RECTXT_TRUE_HISTOIRERECENTE",
  "RECTXT_TRUE_TRAITEMENTDESORTIE",
  "RECTXT_TRUE_CONSIGNESDESORTIE"
)
available_sec_cols <- intersect(sec_cols, names(sample_df))

# 5. Export to JSONL
cat(sprintf("\nWriting %d records to %s...\n", nrow(sample_df), out_path))
con <- file(out_path, open = "wt", encoding = "UTF-8")

for (i in seq_len(nrow(sample_df))) {
  row_evt <- if (!is.na(sample_df$EVTID[i])) {
    as.character(sample_df$EVTID[i])
  } else {
    ""
  }
  row_elt <- if (!is.na(sample_df$ELTID[i])) {
    as.character(sample_df$ELTID[i])
  } else {
    ""
  }
  doc_id <- if (nchar(row_evt) > 0 && nchar(row_elt) > 0) {
    paste0("doc_", row_evt, "_", row_elt)
  } else {
    paste0("doc_sample_", i)
  }

  raw_txt <- as.character(sample_df$RECTXT[i])

  # Collect non-NA anchor sections
  anchors <- list()
  for (col in available_sec_cols) {
    val <- sample_df[[col]][i]
    if (!is.na(val) && nchar(trimws(as.character(val))) > 0) {
      clean_name <- sub("^RECTXT_TRUE_", "", col)
      anchors[[clean_name]] <- as.character(val)
    }
  }

  rec <- list(
    doc_id = doc_id,
    patient_id = if (!is.na(sample_df$PATID[i])) {
      as.character(sample_df$PATID[i])
    } else {
      NULL
    },
    rectype = if (!is.na(sample_df$RECTYPE[i])) {
      as.character(sample_df$RECTYPE[i])
    } else {
      "UNKNOWN"
    },
    recdate = if (!is.na(sample_df$RECDATE[i])) {
      as.character(sample_df$RECDATE[i])
    } else {
      NULL
    },
    sejum = if (!is.na(sample_df$SEJUM[i])) {
      as.character(sample_df$SEJUM[i])
    } else {
      NULL
    },
    char_length = nchar(raw_txt),
    line_count = length(strsplit(raw_txt, "\r?\n")[[1]]),
    rectxt = raw_txt,
    clinical_anchors = anchors
  )

  json_line <- toJSON(rec, auto_unbox = TRUE, null = "null")
  writeLines(json_line, con, useBytes = TRUE)
}

close(con)

file_size_mb <- file.info(out_path)$size / (1024 * 1024)
cat(sprintf("Extraction complete! File size: %.2f MB\n", file_size_mb))
