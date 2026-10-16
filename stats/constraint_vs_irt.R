#!/usr/bin/env Rscript
# Compare direct LLM concept ratings (constraint task) with the IRT thetas.
#
#   Rscript stats/constraint_vs_irt.R data/results/constraint_s7.csv [data/results/constraint_s10.csv ...]
#       [--rds stats/plt_irt.rds] [--out stats/output/constraint_vs_irt]
#
# Each input is a jsonl_batch_runner output built from
# src/select_classic_cases.py (needs `id`, `case`, `constraint_prediction`;
# with SC the mean of `constraint_SC*` is used instead of the majority vote).
# Prints, per input file: overall Spearman vs theta_all / theta_agg / theta_no_sov,
# case-level mean ranking agreement, within-case Spearman (n >= 3), and the
# rows where the LLM rank and the IRT rank disagree most. Writes
# <out>_<stem>_cases.csv and <out>_<stem>_rows.csv.
suppressPackageStartupMessages(library(dplyr))

args <- commandArgs(trailingOnly = TRUE)
get_opt <- function(flag, default) {
  i <- match(flag, args); if (is.na(i)) return(default); args[i + 1]
}
rds_path <- get_opt("--rds", "stats/plt_irt.rds")
out_stem <- get_opt("--out", "stats/output/constraint_vs_irt")
inputs <- args[!args %in% c("--rds", rds_path, "--out", out_stem)]
if (length(inputs) == 0) stop("pass at least one runner output CSV")

irt <- readRDS(rds_path) %>%
  select(id, any_of(c("theta_all", "theta_agg", "theta_no_sov"))) %>%
  distinct(id, .keep_all = TRUE)
theta_cols <- setdiff(names(irt), "id")
dir.create(dirname(out_stem), showWarnings = FALSE, recursive = TRUE)

for (f in inputs) {
  stem <- tools::file_path_sans_ext(basename(f))
  d <- read.csv(f, stringsAsFactors = FALSE)
  # With self-consistency the runner's constraint_prediction is a MAJORITY VOTE;
  # for an ordinal rating the mean of the SC slots is the better point estimate,
  # so use it when constraint_SC* columns exist (and report the slot SD).
  sc_cols <- grep("^constraint_SC[0-9]+$", names(d), value = TRUE)
  if (length(sc_cols) > 0) {
    sc_mat <- sapply(d[sc_cols], function(x) suppressWarnings(as.numeric(x)))
    d$constraint <- rowMeans(sc_mat, na.rm = TRUE)
    d$constraint_sd <- apply(sc_mat, 1, sd, na.rm = TRUE)
    cat("  (constraint = mean of", length(sc_cols), "SC slots)\n")
  } else {
    d$constraint <- suppressWarnings(as.numeric(d$constraint_prediction))
    d$constraint_sd <- NA_real_
  }
  d <- d %>% filter(!is.na(constraint)) %>% inner_join(irt, by = "id")
  cat("\n==========", stem, ":", nrow(d), "scored rows,", n_distinct(d$case), "cases ==========\n")

  cat("\n-- Row-level Spearman (LLM constraint vs theta) --\n")
  for (tc in theta_cols) {
    r <- cor.test(d$constraint, d[[tc]], method = "spearman", exact = FALSE)
    cat(sprintf("  %-13s rho = %5.3f  (p = %.3g)\n", tc, r$estimate, r$p.value))
  }

  cases <- d %>%
    group_by(case) %>%
    summarise(n = n(), llm_mean = mean(constraint), llm_sd = sd(constraint),
              across(all_of(theta_cols), ~ mean(.x, na.rm = TRUE)), .groups = "drop") %>%
    mutate(llm_rank = rank(-llm_mean), irt_rank = rank(-.data[[theta_cols[1]]]),
           rank_diff = llm_rank - irt_rank) %>%
    arrange(llm_rank)
  cat("\n-- Case means (ranked by LLM) --\n")
  print(as.data.frame(cases), digits = 3, row.names = FALSE)
  cat("\n-- Case-level Spearman of means --\n")
  for (tc in theta_cols)
    cat(sprintf("  %-13s rho = %5.3f\n", tc, cor(cases$llm_mean, cases[[tc]], method = "spearman")))

  cat("\n-- Within-case Spearman (cases with n >= 3; NA = LLM gave every leader the same score) --\n")
  within <- d %>% group_by(case) %>% filter(n() >= 3) %>%
    summarise(n = n(), rho = suppressWarnings(cor(constraint, .data[[theta_cols[1]]], method = "spearman")),
              .groups = "drop")
  print(as.data.frame(within), digits = 3, row.names = FALSE)

  rows <- d %>%
    mutate(llm_rank = rank(-constraint), irt_rank = rank(-.data[[theta_cols[1]]]),
           rank_diff = llm_rank - irt_rank) %>%
    select(id, case, polity_name, leader_name, leader_first_year, leader_last_year,
           constraint, constraint_sd, all_of(theta_cols), llm_rank, irt_rank, rank_diff) %>%
    arrange(desc(abs(rank_diff)))
  cat("\n-- Largest disagreements (LLM rank - IRT rank; positive = LLM says LESS constrained than IRT) --\n")
  print(head(as.data.frame(rows), 12), digits = 3, row.names = FALSE)

  write.csv(cases, paste0(out_stem, "_", stem, "_cases.csv"), row.names = FALSE)
  write.csv(rows, paste0(out_stem, "_", stem, "_rows.csv"), row.names = FALSE)
  cat("\nWrote", paste0(out_stem, "_", stem, "_{cases,rows}.csv"), "\n")
}
