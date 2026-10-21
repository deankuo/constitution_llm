#!/usr/bin/env Rscript
# Direct LLM constraint ratings vs IRT theta for the famous-rulers test set.
#
#   Rscript stats/constraint_rulers_plot.R [--results data/results/constraint_rulers]
#          [--sample data/temp/famous_rulers_sample.csv] [--se data/temp/famous_rulers_theta_se.csv]
#          [--out stats/figures]
#
# Reads every constraint_rulers_s{7,10}_{bare,detailed}.csv the runner wrote
# (missing variants are skipped). LLM score = mean of the constraint_SC* slots
# (fallback: constraint_prediction). Writes
#   <out>/constraint_rulers_scatter.png  theta (x, ±1.96 SE) vs LLM mean (y, ±1 SD over slots), one panel per prompt
#   <out>/constraint_rulers_rank.png     rulers ordered by IRT theta; IRT rank vs LLM rank per prompt
#   stats/output/constraint_rulers_summary.csv   rho per prompt
#   stats/output/constraint_rulers_rows.csv      per-ruler scores and ranks, all prompts
suppressPackageStartupMessages({ library(dplyr); library(tidyr); library(ggplot2); library(ggrepel) })

args <- commandArgs(trailingOnly = TRUE)
get_opt <- function(flag, default) { i <- match(flag, args); if (is.na(i)) default else args[i + 1] }
res_dir <- get_opt("--results", "data/results/constraint_rulers")
sample_f <- get_opt("--sample", "data/temp/famous_rulers_sample.csv")
se_f     <- get_opt("--se", "data/temp/famous_rulers_theta_se.csv")
out_dir  <- get_opt("--out", "stats/figures")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
dir.create("stats/output", showWarnings = FALSE, recursive = TRUE)

period_levels <- c("Ancient", "Medieval", "Early modern", "Modern")
period_cols <- c(Ancient = "#B5651D", Medieval = "#2E7D6B", `Early modern` = "#3F5FA8", Modern = "#8E3B8E")
prompt_levels <- c("1-7, bare", "1-7, detailed", "0-10, bare", "0-10, detailed")

rulers <- read.csv(sample_f, stringsAsFactors = FALSE) %>%
  select(id, label, period, polity_name, leader_name, leader_first_year, leader_last_year, theta_all)
if (file.exists(se_f)) {
  rulers <- rulers %>% left_join(read.csv(se_f) %>% select(id, theta_all_se), by = "id")
} else rulers$theta_all_se <- NA_real_
rulers <- rulers %>%
  mutate(period = factor(period, levels = period_levels),
         irt_rank = rank(-theta_all),
         yrs = paste0(ifelse(leader_first_year < 0, paste0(abs(leader_first_year), " BC"), leader_first_year), "–",
                      ifelse(leader_last_year < 0, paste0(abs(leader_last_year), " BC"), leader_last_year)))

read_variant <- function(scale, variant) {
  f <- file.path(res_dir, sprintf("constraint_rulers_s%s_%s.csv", scale, variant))
  if (!file.exists(f)) { message("missing: ", f); return(NULL) }
  d <- read.csv(f, stringsAsFactors = FALSE)
  sc_cols <- grep("^constraint_SC[0-9]+$", names(d), value = TRUE)
  if (length(sc_cols)) {
    m <- sapply(d[sc_cols], function(x) suppressWarnings(as.numeric(x)))
    d$llm <- rowMeans(m, na.rm = TRUE); d$llm_sd <- apply(m, 1, sd, na.rm = TRUE)
    d$n_votes <- rowSums(!is.na(m))
  } else {
    d$llm <- suppressWarnings(as.numeric(d$constraint_prediction)); d$llm_sd <- NA_real_; d$n_votes <- 1L
  }
  lo <- if (scale == "7") 1 else 0; hi <- as.numeric(scale)
  d %>% transmute(id, prompt = sprintf("%d-%d, %s", lo, hi, variant), scale_lo = lo, scale_hi = hi,
                  llm, llm_sd, n_votes, llm_norm = (llm - lo) / (hi - lo))
}
llm <- bind_rows(lapply(c("7", "10"), function(s) lapply(c("bare", "detailed"), function(v) read_variant(s, v))))
if (nrow(llm) == 0) stop("no result files found in ", res_dir)
llm$prompt <- factor(llm$prompt, levels = prompt_levels)

d <- rulers %>% inner_join(llm, by = "id") %>% filter(!is.na(llm)) %>%
  group_by(prompt) %>% mutate(llm_rank = rank(-llm), rank_diff = llm_rank - irt_rank) %>% ungroup()

summ <- d %>% group_by(prompt) %>%
  summarise(n = n(),
            rho = cor(llm, theta_all, method = "spearman"),
            rho_p = cor.test(llm, theta_all, method = "spearman", exact = FALSE)$p.value,
            pearson = cor(llm, theta_all),
            mean_slot_sd = mean(llm_sd, na.rm = TRUE),
            share_unanimous = mean(llm_sd == 0, na.rm = TRUE),
            n_distinct_scores = n_distinct(llm),
            .groups = "drop") %>%
  mutate(label = sprintf("rho = %.2f  (n = %d)", rho, n))
cat("\n-- Spearman: LLM mean score vs theta_all, by prompt --\n")
print(as.data.frame(summ), digits = 3, row.names = FALSE)
write.csv(summ, "stats/output/constraint_rulers_summary.csv", row.names = FALSE)

wide <- d %>% select(label, period, polity_name, leader_name, yrs, theta_all, theta_all_se, irt_rank, prompt, llm, llm_sd, llm_rank) %>%
  pivot_wider(names_from = prompt, values_from = c(llm, llm_sd, llm_rank), names_sep = " | ") %>%
  arrange(irt_rank)
write.csv(wide, "stats/output/constraint_rulers_rows.csv", row.names = FALSE)

cat("\n-- Largest disagreements per prompt (LLM rank - IRT rank; positive = LLM says LESS constrained than IRT) --\n")
d %>% group_by(prompt) %>% slice_max(abs(rank_diff), n = 6, with_ties = FALSE) %>%
  select(prompt, label, yrs, theta_all, llm, irt_rank, llm_rank, rank_diff) %>%
  as.data.frame() %>% print(digits = 3, row.names = FALSE)

# -- Figure 1: scatter, one panel per prompt ---------------------------------
lab_pts <- d %>% group_by(prompt) %>% slice_max(abs(rank_diff), n = 7, with_ties = FALSE) %>% ungroup()
p_sc <- ggplot(d, aes(theta_all, llm)) +
  geom_errorbar(aes(xmin = theta_all - 1.96 * theta_all_se, xmax = theta_all + 1.96 * theta_all_se),
                orientation = "y", width = 0, colour = "grey80", linewidth = 0.4, na.rm = TRUE) +
  geom_errorbar(aes(ymin = llm - llm_sd, ymax = llm + llm_sd), width = 0, colour = "grey80", linewidth = 0.4, na.rm = TRUE) +
  geom_smooth(method = "lm", formula = y ~ x, se = FALSE, colour = "grey45", linewidth = 0.5, linetype = "dashed") +
  geom_point(aes(colour = period), size = 2.2) +
  geom_text_repel(data = lab_pts, aes(label = label), size = 2.4, colour = "grey25",
                  max.overlaps = Inf, box.padding = 0.35, min.segment.length = 0,
                  segment.colour = "grey70", seed = 42) +
  geom_text(data = summ, aes(x = -Inf, y = Inf, label = label), hjust = -0.1, vjust = 1.6,
            size = 3, colour = "grey20", inherit.aes = FALSE) +
  facet_wrap(~ prompt, scales = "free_y", ncol = 2) +
  scale_colour_manual(values = period_cols, name = NULL) +
  labs(x = expression(paste("IRT executive constraints (", theta, ", EAP ± 1.96 SE)")),
       y = "LLM direct constraint rating (mean of 3 samples ± 1 SD)",
       caption = paste0("50 famous rulers, one row each (longest tenure among duplicate spells). ",
                        "Labels: the seven largest rank disagreements in each panel. Dashed: OLS fit.")) +
  theme_minimal(base_size = 10) +
  theme(legend.position = "top", panel.grid.minor = element_blank(), strip.text = element_text(face = "bold"))
ggsave(file.path(out_dir, "constraint_rulers_scatter.png"), p_sc, width = 9, height = 8, dpi = 300)

# -- Figure 2: rank dumbbell, rulers ordered by IRT theta ---------------------
rk <- d %>% mutate(tick = paste0(label, "  (", yrs, ")"), tick = reorder(tick, -irt_rank))
p_rk <- ggplot(rk, aes(y = tick)) +
  geom_segment(aes(x = irt_rank, xend = llm_rank, yend = tick, colour = period), linewidth = 0.6, alpha = 0.6) +
  geom_point(aes(x = irt_rank), shape = 21, fill = "white", colour = "grey40", size = 1.8, stroke = 0.5) +
  geom_point(aes(x = llm_rank, colour = period), size = 2) +
  facet_wrap(~ prompt, ncol = 4) +
  scale_colour_manual(values = period_cols, name = NULL) +
  scale_x_reverse(breaks = c(1, 10, 20, 30, 40, 50)) +
  labs(x = "Rank among the 50 rulers (1 = most constrained, on the right; constraint falls to the left)", y = NULL,
       caption = "Hollow: IRT rank (rows ordered by it). Filled: LLM rank. Segment length = disagreement.") +
  theme_minimal(base_size = 9) +
  theme(legend.position = "top", panel.grid.minor = element_blank(), panel.grid.major.y = element_blank(),
        axis.text.y = element_text(size = 6.5), strip.text = element_text(face = "bold"))
ggsave(file.path(out_dir, "constraint_rulers_rank.png"), p_rk, width = 12, height = 9, dpi = 300)

cat("\nWrote", file.path(out_dir, "constraint_rulers_{scatter,rank}.png"), "and stats/output/constraint_rulers_{summary,rows}.csv\n")
