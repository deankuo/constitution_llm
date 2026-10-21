#!/usr/bin/env Rscript
# Direct LLM constraint ratings vs IRT theta for the sample96 test set.
#
#   Rscript stats/constraint_sample96_plot.R [--results data/results/constraint_sample96]
#          [--input data/sample96/sample96_constraint_input.csv] [--n-show 50]
#          [--tiers A] [--out stats/figures]
#
# Reads every constraint_sample96_s{7,10}_{bare,detailed}.csv the runner wrote
# (missing variants are skipped). LLM score = mean of the constraint_SC* slots
# (fallback: constraint_prediction). Correlation and consistency use ALL matched
# rows (90); the rank figure shows --n-show rows (default 50) drawn from the
# --tiers crosswalk tiers (default A = confident matches), evenly spaced along
# the IRT ranking so the whole theta range is covered. Writes
#   <out>/constraint_sample96_scatter.png   theta (x) vs LLM mean (y, ±1 SD over slots), one panel per prompt, all rows
#   <out>/constraint_sample96_rank.png      n-show leaders ordered by IRT theta; IRT rank vs LLM rank per prompt
#   <out>/constraint_sample96_prompts.png   bare vs detailed mean score per leader (between-prompt consistency)
#   stats/output/constraint_sample96_summary.csv      rho / consistency per prompt × tier subset
#   stats/output/constraint_sample96_consistency.csv  between-prompt agreement
#   stats/output/constraint_sample96_rows.csv         per-leader scores and ranks, all prompts
suppressPackageStartupMessages({ library(dplyr); library(tidyr); library(ggplot2); library(ggrepel) })

args <- commandArgs(trailingOnly = TRUE)
get_opt <- function(flag, default) { i <- match(flag, args); if (is.na(i)) default else args[i + 1] }
res_dir <- get_opt("--results", "data/results/constraint_sample96")
input_f <- get_opt("--input", "data/sample96/sample96_constraint_input.csv")
n_show  <- as.integer(get_opt("--n-show", "50"))
tiers   <- strsplit(get_opt("--tiers", "A"), ",")[[1]]
out_dir <- get_opt("--out", "stats/figures")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
dir.create("stats/output", showWarnings = FALSE, recursive = TRUE)

period_levels <- c("Ancient", "Medieval", "Early modern", "Modern")
period_cols <- c(Ancient = "#B5651D", Medieval = "#2E7D6B", `Early modern` = "#3F5FA8", Modern = "#8E3B8E")
prompt_levels <- c("1-7, bare", "1-7, detailed", "0-10, bare", "0-10, detailed")
fmt_yr <- function(y) ifelse(y < 0, paste0(abs(y), " BC"), as.character(y))

leaders <- read.csv(input_f, stringsAsFactors = FALSE) %>%
  select(id, sample_polity_name, sample_leader_name, match_tier, polity_name, leader_name,
         leader_first_year, leader_last_year, theta_all, theta_agg, theta_no_sov) %>%
  mutate(period = factor(cut(leader_first_year, c(-Inf, 500, 1500, 1800, Inf), labels = period_levels, right = FALSE),
                         levels = period_levels),
         label = sample_leader_name,
         yrs = paste0(fmt_yr(leader_first_year), "–", fmt_yr(leader_last_year)),
         irt_rank = rank(-theta_all))

read_variant <- function(scale, variant) {
  f <- file.path(res_dir, sprintf("constraint_sample96_s%s_%s.csv", scale, variant))
  if (!file.exists(f)) { message("missing: ", f); return(NULL) }
  d <- read.csv(f, stringsAsFactors = FALSE)
  sc_cols <- grep("^constraint_SC[0-9]+$", names(d), value = TRUE)
  if (length(sc_cols)) {
    m <- sapply(d[sc_cols], function(x) suppressWarnings(as.numeric(x)))
    d$llm <- rowMeans(m, na.rm = TRUE); d$llm_sd <- apply(m, 1, sd, na.rm = TRUE)
    d$llm_range <- apply(m, 1, function(v) diff(range(v, na.rm = TRUE)))
    d$n_votes <- rowSums(!is.na(m))
  } else {
    d$llm <- suppressWarnings(as.numeric(d$constraint_prediction)); d$llm_sd <- NA_real_
    d$llm_range <- NA_real_; d$n_votes <- 1L
  }
  lo <- if (scale == "7") 1 else 0; hi <- as.numeric(scale)
  d %>% transmute(id, prompt = sprintf("%d-%d, %s", lo, hi, variant), scale = scale, variant = variant,
                  scale_lo = lo, scale_hi = hi, llm, llm_sd, llm_range, n_votes,
                  llm_norm = (llm - lo) / (hi - lo))
}
llm <- bind_rows(lapply(c("7", "10"), function(s) lapply(c("bare", "detailed"), function(v) read_variant(s, v))))
if (nrow(llm) == 0) stop("no result files found in ", res_dir)
llm$prompt <- factor(llm$prompt, levels = prompt_levels)

d <- leaders %>% inner_join(llm, by = "id") %>% filter(!is.na(llm)) %>%
  group_by(prompt) %>% mutate(llm_rank = rank(-llm), rank_diff = llm_rank - irt_rank) %>% ungroup()

# -- Correlation + within-prompt consistency, per prompt × tier subset -------
subsets <- list(all = c("A", "B", "C"), `A+B` = c("A", "B"), A = "A")
summarise_subset <- function(dd, name) {
  dd %>% group_by(prompt) %>%
    summarise(subset = name, n = n(),
              rho_all = cor(llm, theta_all, method = "spearman"),
              rho_all_p = cor.test(llm, theta_all, method = "spearman", exact = FALSE)$p.value,
              rho_agg = cor(llm, theta_agg, method = "spearman"),
              rho_no_sov = cor(llm, theta_no_sov, method = "spearman"),
              pearson = cor(llm, theta_all),
              mean_slot_sd = mean(llm_sd, na.rm = TRUE),
              share_unanimous = mean(llm_range == 0, na.rm = TRUE),
              share_within_1 = mean(llm_range <= 1, na.rm = TRUE),
              mean_votes = mean(n_votes),
              n_distinct_scores = n_distinct(llm),
              .groups = "drop")
}
summ <- bind_rows(lapply(names(subsets), function(nm) summarise_subset(filter(d, match_tier %in% subsets[[nm]]), nm))) %>%
  mutate(subset = factor(subset, levels = names(subsets))) %>% arrange(prompt, subset)
cat("\n-- Spearman: LLM mean score vs theta, by prompt and crosswalk tier subset --\n")
print(as.data.frame(summ %>% select(prompt, subset, n, rho_all, rho_all_p, rho_agg, rho_no_sov, pearson)), digits = 3, row.names = FALSE)
cat("\n-- Within-prompt consistency over the SC slots --\n")
print(as.data.frame(summ %>% filter(subset == "all") %>%
                      select(prompt, n, mean_votes, mean_slot_sd, share_unanimous, share_within_1, n_distinct_scores)),
      digits = 3, row.names = FALSE)
write.csv(summ, "stats/output/constraint_sample96_summary.csv", row.names = FALSE)

# -- Between-prompt consistency (bare vs detailed, same scale) ---------------
pair <- d %>% select(id, label, period, match_tier, theta_all, irt_rank, scale, variant, llm, llm_rank) %>%
  pivot_wider(names_from = variant, values_from = c(llm, llm_rank)) %>%
  filter(!is.na(llm_bare), !is.na(llm_detailed))
cons <- pair %>% group_by(scale) %>%
  summarise(n = n(),
            rho_bare_detailed = cor(llm_bare, llm_detailed, method = "spearman"),
            pearson_bare_detailed = cor(llm_bare, llm_detailed),
            mean_abs_diff = mean(abs(llm_bare - llm_detailed)),
            mean_diff_detailed_minus_bare = mean(llm_detailed - llm_bare),
            share_identical = mean(llm_bare == llm_detailed),
            share_within_1 = mean(abs(llm_bare - llm_detailed) <= 1),
            mean_abs_rank_diff = mean(abs(llm_rank_bare - llm_rank_detailed)),
            .groups = "drop")
if (nrow(cons)) {
  cat("\n-- Between-prompt consistency: bare vs detailed (mean score per leader) --\n")
  print(as.data.frame(cons), digits = 3, row.names = FALSE)
  write.csv(cons, "stats/output/constraint_sample96_consistency.csv", row.names = FALSE)
}

wide <- d %>% select(label, sample_polity_name, match_tier, period, polity_name, leader_name, yrs,
                     theta_all, irt_rank, prompt, llm, llm_sd, llm_rank) %>%
  pivot_wider(names_from = prompt, values_from = c(llm, llm_sd, llm_rank), names_sep = " | ") %>%
  arrange(irt_rank)
write.csv(wide, "stats/output/constraint_sample96_rows.csv", row.names = FALSE)

cat("\n-- Largest disagreements per prompt (LLM rank - IRT rank; positive = LLM says LESS constrained than IRT) --\n")
d %>% group_by(prompt) %>% slice_max(abs(rank_diff), n = 6, with_ties = FALSE) %>%
  select(prompt, label, sample_polity_name, yrs, match_tier, theta_all, llm, irt_rank, llm_rank, rank_diff) %>%
  as.data.frame() %>% print(digits = 3, row.names = FALSE)

# -- Figure 1: scatter, all rows, one panel per prompt -----------------------
summ_all <- summ %>% filter(subset == "all") %>% mutate(lab = sprintf("rho = %.2f  (n = %d)", rho_all, n))
lab_pts <- d %>% group_by(prompt) %>% slice_max(abs(rank_diff), n = 7, with_ties = FALSE) %>% ungroup()
p_sc <- ggplot(d, aes(theta_all, llm)) +
  geom_errorbar(aes(ymin = llm - llm_sd, ymax = llm + llm_sd), width = 0, colour = "grey80", linewidth = 0.4, na.rm = TRUE) +
  geom_smooth(method = "lm", formula = y ~ x, se = FALSE, colour = "grey45", linewidth = 0.5, linetype = "dashed") +
  geom_point(aes(colour = period, shape = match_tier), size = 2.2) +
  geom_text_repel(data = lab_pts, aes(label = label), size = 2.4, colour = "grey25",
                  max.overlaps = Inf, box.padding = 0.35, min.segment.length = 0,
                  segment.colour = "grey70", seed = 42) +
  geom_text(data = summ_all, aes(x = -Inf, y = Inf, label = lab), hjust = -0.1, vjust = 1.6,
            size = 3, colour = "grey20", inherit.aes = FALSE) +
  facet_wrap(~ prompt, scales = "free_y", ncol = 2) +
  scale_colour_manual(values = period_cols, name = NULL) +
  scale_shape_manual(values = c(A = 16, B = 17, C = 4), name = "crosswalk tier") +
  labs(x = expression(paste("IRT executive constraints (", theta, ", EAP)")),
       y = "LLM direct constraint rating (mean of SC slots ± 1 SD)",
       caption = paste0("sample96 leaders matched to the IRT dataset (", n_distinct(d$id), " rows; tier A = confident match, ",
                        "B = small year/name drift, C = doubtful).\nLabels: the seven largest rank disagreements per panel. Dashed: OLS fit.")) +
  theme_minimal(base_size = 10) +
  theme(legend.position = "top", legend.box = "vertical", legend.spacing.y = unit(0, "pt"),
        plot.caption = element_text(hjust = 0),
        panel.grid.minor = element_blank(), strip.text = element_text(face = "bold"))
ggsave(file.path(out_dir, "constraint_sample96_scatter.png"), p_sc, width = 9, height = 5 + 3.5 * ceiling(n_distinct(d$prompt) / 2), dpi = 300)

# -- Figure 2: rank dumbbell, n_show leaders evenly spaced along IRT theta ---
pool <- leaders %>% filter(match_tier %in% tiers, id %in% d$id) %>% arrange(desc(theta_all))
n_show <- min(n_show, nrow(pool))
show_ids <- pool$id[unique(round(seq(1, nrow(pool), length.out = n_show)))]
rk <- d %>% filter(id %in% show_ids) %>%
  group_by(prompt) %>% mutate(irt_rank_s = rank(-theta_all), llm_rank_s = rank(-llm)) %>% ungroup() %>%
  mutate(tick = paste0(label, " · ", sample_polity_name, "  (", yrs, ")"), tick = reorder(tick, -irt_rank_s))
rho_show <- rk %>% group_by(prompt) %>% summarise(lab = sprintf("rho = %.2f", cor(llm, theta_all, method = "spearman")), .groups = "drop")
brks <- unique(c(1, seq(10, n_show, by = 10), n_show))
p_rk <- ggplot(rk, aes(y = tick)) +
  geom_segment(aes(x = irt_rank_s, xend = llm_rank_s, yend = tick, colour = period), linewidth = 0.6, alpha = 0.6) +
  geom_point(aes(x = irt_rank_s), shape = 21, fill = "white", colour = "grey40", size = 1.8, stroke = 0.5) +
  geom_point(aes(x = llm_rank_s, colour = period), size = 2) +
  geom_text(data = rho_show, aes(x = Inf, y = Inf, label = lab), hjust = -0.1, vjust = 1.8,  # x reversed: Inf = left edge
            size = 3, colour = "grey20", inherit.aes = FALSE) +
  facet_wrap(~ prompt, ncol = n_distinct(rk$prompt)) +
  scale_colour_manual(values = period_cols, name = NULL) +
  scale_x_reverse(breaks = brks) +
  labs(x = sprintf("Rank among the %d leaders shown (1 = most constrained, on the right; constraint falls to the left)", n_show), y = NULL,
       caption = sprintf(paste0("%d of %d matched sample96 leaders (crosswalk tier %s), evenly spaced along the IRT ranking. ",
                                "Hollow: IRT rank (rows ordered by it). Filled: LLM rank. Segment length = disagreement."),
                         n_show, n_distinct(d$id), paste(tiers, collapse = "/"))) +
  theme_minimal(base_size = 9) +
  theme(legend.position = "top", panel.grid.minor = element_blank(), panel.grid.major.y = element_blank(),
        axis.text.y = element_text(size = 6.5), strip.text = element_text(face = "bold"))
ggsave(file.path(out_dir, "constraint_sample96_rank.png"), p_rk, width = 4 + 3.5 * n_distinct(rk$prompt), height = 2 + 0.16 * n_show, dpi = 300)

# -- Figure 3: bare vs detailed per leader (between-prompt consistency) ------
if (nrow(pair)) {
  pair <- pair %>% mutate(scale_lab = ifelse(scale == "7", "1-7 scale", "0-10 scale"),
                          diff = llm_detailed - llm_bare)
  lab_pair <- pair %>% group_by(scale) %>% slice_max(abs(diff), n = 8, with_ties = FALSE) %>% ungroup()
  cons_lab <- cons %>% mutate(scale_lab = ifelse(scale == "7", "1-7 scale", "0-10 scale"),
                              lab = sprintf("rho = %.2f   mean |diff| = %.2f   within 1 pt: %.0f%%",
                                            rho_bare_detailed, mean_abs_diff, 100 * share_within_1))
  p_pr <- ggplot(pair, aes(llm_bare, llm_detailed)) +
    geom_abline(slope = 1, intercept = 0, colour = "grey70", linetype = "dashed", linewidth = 0.5) +
    geom_count(aes(colour = period), alpha = 0.8) +
    geom_text_repel(data = lab_pair, aes(label = label), size = 2.4, colour = "grey25",
                    max.overlaps = Inf, box.padding = 0.35, min.segment.length = 0, segment.colour = "grey70", seed = 42) +
    geom_text(data = cons_lab, aes(x = -Inf, y = Inf, label = lab), hjust = -0.05, vjust = 1.6,
              size = 3, colour = "grey20", inherit.aes = FALSE) +
    facet_wrap(~ scale_lab, scales = "free") +
    scale_colour_manual(values = period_cols, name = NULL) +
    scale_size_area(max_size = 6, name = "leaders") +
    labs(x = "bare prompt (mean of SC slots)", y = "detailed prompt (mean of SC slots)",
         caption = "Each point is one leader; point size = number of leaders at that exact pair of scores.\nDashed: identity. Labels: the eight largest bare–detailed gaps.") +
    theme_minimal(base_size = 10) +
    theme(legend.position = "top", plot.caption = element_text(hjust = 0),
          panel.grid.minor = element_blank(), strip.text = element_text(face = "bold"))
  ggsave(file.path(out_dir, "constraint_sample96_prompts.png"), p_pr, width = 4.5 + 4.5 * n_distinct(pair$scale), height = 5.5, dpi = 300)
}

cat("\nWrote", file.path(out_dir, "constraint_sample96_{scatter,rank,prompts}.png"),
    "and stats/output/constraint_sample96_{summary,consistency,rows}.csv\n")
