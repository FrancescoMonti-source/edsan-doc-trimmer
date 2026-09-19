#!/usr/bin/env Rscript
# Comprehensive Stratified Extraction Script for redsan-doc-trimmer
# Stratifies across Year (2000-2025), RECTYPE (500+ types), and SEJUF (398 UFs)
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
target_n <- as.integer(get_arg("--n", "10000"))
seed_val <- as.integer(get_arg("--seed", "42"))

cat(sprintf("=== redsan-doc-trimmer: Comprehensive Corpus Extraction ===\n"))
cat(sprintf("Input RDS: %s\n", rds_path))
cat(sprintf("Output:    %s\n", out_path))
cat(sprintf("Target N:  %d documents\n", target_n))
cat(sprintf("Seed:      %d\n\n", seed_val))

if (!file.exists(rds_path)) {
  stop(sprintf("Input file does not exist: %s", rds_path))
}

out_dir <- dirname(out_path)
if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
}

cat("Loading RDS dataset...\n")
df <- readRDS(rds_path)
total_raw <- nrow(df)
cat(sprintf("Loaded %d documents from RDS.\n", total_raw))

# 1. Apply exclusions (prescriptions and transport vouchers)
cat("\nApplying exclusion filters (ORDON* and BT*)...\n")
rectype_clean <- ifelse(is.na(df$RECTYPE), "UNKNOWN", as.character(df$RECTYPE))
is_ordon <- grepl("^ORDON", rectype_clean, ignore.case = TRUE)
is_bt <- grepl("(^BT|TRANSPORT|BON)", rectype_clean, ignore.case = TRUE)
has_valid_text <- !is.na(df$RECTXT) & nchar(trimws(df$RECTXT)) >= 20

keep_mask <- !is_ordon & !is_bt & has_valid_text
filtered_df <- df[keep_mask, ]
n_filtered <- nrow(filtered_df)
cat(sprintf(
  "Remaining clinical pool after filter: %d documents (excluded %d records)\n",
  n_filtered,
  total_raw - n_filtered
))

set.seed(seed_val)

# 2. Stratification variables
years <- as.integer(substr(as.character(filtered_df$RECDATE), 1, 4))
years[is.na(years)] <- 2015
years_str <- as.character(years)

# RECTYPE: Keep top 30 types, group remainder into OTHER_NARRATIVE
top_types <- names(head(
  sort(table(filtered_df$RECTYPE), decreasing = TRUE),
  30
))
rectype_strat <- ifelse(
  filtered_df$RECTYPE %in% top_types,
  as.character(filtered_df$RECTYPE),
  "OTHER_NARRATIVE"
)

# SEJUF (Unités Fonctionnelles): Keep all with >= 10 occurrences, group ultra-rare
uf_clean <- ifelse(
  is.na(filtered_df$SEJUF),
  "UNKNOWN_UF",
  as.character(filtered_df$SEJUF)
)
top_ufs <- names(which(table(uf_clean) >= 10))
uf_strat <- ifelse(uf_clean %in% top_ufs, uf_clean, "OTHER_UF")

# Service (SEJUM)
sejum_clean <- ifelse(
  is.na(filtered_df$SEJUM),
  "UNKNOWN_UM",
  as.character(filtered_df$SEJUM)
)

# Combined stratum: Year x RECTYPE x UF
stratum <- paste(years_str, rectype_strat, uf_strat, sep = "___")
strata_counts <- table(stratum)
cat(sprintf(
  "Total multi-dimensional strata (Year x RECTYPE x UF): %d\n",
  length(strata_counts)
))

# Step A: Guaranteed representation for every unique SEJUF (minimum 3 docs per UF where available)
cat(
  "Guaranteeing baseline coverage across all Unités Fonctionnelles (SEJUF)...\n"
)
unique_ufs <- unique(uf_clean)
uf_baseline_indices <- unlist(lapply(unique_ufs, function(u) {
  idx <- which(uf_clean == u)
  sample(idx, size = min(3, length(idx)), replace = FALSE)
}))
cat(sprintf(
  "Baseline UF guarantee: %d documents covering %d unique UFs.\n",
  length(uf_baseline_indices),
  length(unique_ufs)
))

# Step B: Proportional allocation for remaining quota
remaining_target <- max(0, target_n - length(uf_baseline_indices))
allocations <- pmax(
  1,
  round(remaining_target * (as.numeric(strata_counts) / n_filtered))
)
scale_factor <- remaining_target / sum(allocations)
allocations <- pmax(1, round(allocations * scale_factor))

cat("Sampling proportional allocation across strata...\n")
prop_indices <- unlist(lapply(seq_along(strata_counts), function(i) {
  strat_name <- names(strata_counts)[i]
  pool <- which(stratum == strat_name)
  n_to_take <- min(allocations[i], length(pool))
  sample(pool, size = n_to_take, replace = FALSE)
}))

# Combine and deduplicate
sampled_indices <- unique(c(uf_baseline_indices, prop_indices))

# Trim or fill to exact target_n
if (length(sampled_indices) > target_n) {
  # Always preserve the UF baseline, trim from proportional
  excess <- length(sampled_indices) - target_n
  trimmable <- setdiff(sampled_indices, uf_baseline_indices)
  to_drop <- sample(
    trimmable,
    size = min(excess, length(trimmable)),
    replace = FALSE
  )
  sampled_indices <- setdiff(sampled_indices, to_drop)
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
cat(sprintf(
  "\n=== Selected Final Sample: %d documents ===\n\n",
  nrow(sample_df)
))

# 3. Stratification Breakdown & Diversity Audit
sample_years <- as.integer(substr(as.character(sample_df$RECDATE), 1, 4))
sample_ufs <- as.character(sample_df$SEJUF)
sample_ums <- as.character(sample_df$SEJUM)
sample_types <- as.character(sample_df$RECTYPE)

cat(sprintf("Diversity in Sample:\n"))
cat(sprintf(
  "  - Unique Years:               %d (Range: %d to %d)\n",
  length(unique(sample_years)),
  min(sample_years, na.rm = TRUE),
  max(sample_years, na.rm = TRUE)
))
cat(sprintf(
  "  - Unique Unités Fonctionnelles (SEJUF): %d / %d (%.1f%% of all hospital UFs)\n",
  length(unique(sample_ufs)),
  length(unique_ufs),
  100 * length(unique(sample_ufs)) / length(unique_ufs)
))
cat(sprintf(
  "  - Unique Medical Services (SEJUM):     %d / %d\n",
  length(unique(sample_ums)),
  length(unique(sejum_clean))
))
cat(sprintf(
  "  - Unique Document Types (RECTYPE):     %d\n\n",
  length(unique(sample_types))
))

cat("=== Document Count by Year in Sample ===\n")
print(table(sample_years))

cat("\n=== Top 15 Document Types in Sample ===\n")
print(head(sort(table(sample_types), decreasing = TRUE), 15))

cat("\n=== Top 15 Medical Services (SEJUM) in Sample ===\n")
print(head(sort(table(sample_ums), decreasing = TRUE), 15))

# 4. Export Clean JSONL (without corrupted legacy columns)
cat(sprintf("\nWriting %d clean records to %s...\n", nrow(sample_df), out_path))
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
    sejuf = if (!is.na(sample_df$SEJUF[i])) {
      as.character(sample_df$SEJUF[i])
    } else {
      NULL
    },
    char_length = nchar(raw_txt),
    line_count = length(strsplit(raw_txt, "\r?\n")[[1]]),
    rectxt = raw_txt
  )

  json_line <- toJSON(rec, auto_unbox = TRUE, null = "null")
  writeLines(json_line, con, useBytes = TRUE)
}

close(con)

file_size_mb <- file.info(out_path)$size / (1024 * 1024)
cat(sprintf("Extraction complete! File size: %.2f MB\n", file_size_mb))
