#!/usr/bin/env Rscript
# Export the famous-rulers test set (stats/famous_rulers.R) as a pipeline
# input CSV for the direct-constraint pilot.
#
#   Rscript stats/export_famous_rulers.R                 # -> data/temp/famous_rulers_sample.csv
#   Rscript stats/export_famous_rulers.R --se            # -> data/temp/famous_rulers_theta_se.csv
#
# Default: match each ruler to ONE row of stats/plt_irt.rds (longest tenure
# among duplicate spells, as in classic_cases_plot.R section C), join the
# matched ids back to data/plt_constraints.csv so every pipeline column is
# present, and add `label`, `period`, `n_rows`, `theta_all`, `theta_agg`,
# `theta_no_sov`, `original_row_idx` (position in plt_constraints.csv).
# Rows come out in famous_rulers order. Build with
#   python src/build_batch_jsonl.py --task constraint --input data/temp/famous_rulers_sample.csv ...
# and run with `--dataset data/temp/famous_rulers_sample.csv` (positional merge,
# 50-row output; no --original-idx-col needed).
#
# --se: refit the one-factor IRT model to obtain EAP posterior SEs (slow, a
# few minutes) and write id + theta_all_se for the matched rulers only.
suppressPackageStartupMessages(library(dplyr))
source("stats/famous_rulers.R")

args <- commandArgs(trailingOnly = TRUE)
want_se <- "--se" %in% args

plt_irt <- readRDS("stats/plt_irt.rds")
if (want_se) plt_irt <- add_theta_se(plt_irt) else plt_irt$theta_all_se <- NA_real_

rulers <- bind_rows(lapply(seq_len(nrow(famous_rulers)),
                           function(i) match_ruler(famous_rulers[i, ], plt_irt)))
stopifnot(nrow(rulers) == nrow(famous_rulers), !anyDuplicated(rulers$id))

if (want_se) {
  out <- rulers %>% select(id, label, theta_all = theta, theta_all_se = se)
  write.csv(out, "data/temp/famous_rulers_theta_se.csv", row.names = FALSE)
  cat(nrow(out), "rows ->", "data/temp/famous_rulers_theta_se.csv\n")
  quit(status = 0)
}

raw <- read.csv("data/plt_constraints.csv", stringsAsFactors = FALSE)
raw$original_row_idx <- seq_len(nrow(raw)) - 1L
extra <- plt_irt %>% select(id, theta_all, theta_agg, theta_no_sov)
out <- rulers %>%
  select(id, label, period, n_rows) %>%
  inner_join(raw, by = "id") %>%
  inner_join(extra, by = "id")
stopifnot(nrow(out) == nrow(rulers))
out <- out[match(rulers$id, out$id), ]
write.csv(out, "data/temp/famous_rulers_sample.csv", row.names = FALSE)
print(as.data.frame(out %>% select(label, polity_name, leader_name, leader_first_year, leader_last_year, theta_all)),
      digits = 3, row.names = FALSE)
cat("\n", nrow(out), "rows ->", "data/temp/famous_rulers_sample.csv\n")
