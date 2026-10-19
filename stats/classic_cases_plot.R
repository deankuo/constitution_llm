# ---------------------------------------------------------------------------
# Classic cases on the IRT scale (figure for the draft)
#
# theta = theta_agg in plt_irt.rds: unidimensional mirt on items_B (same item
# set as mod_B, the `theta` used in the "Facial Validity" chunk of IRT.Rmd).
# Rows are matched to cases with add_case() (overlap rule, name variants +
# year window). Unit = leader-spell row, as in the IRT.Rmd case table.
#
# Run from stats/:  Rscript classic_cases_plot.R
# ---------------------------------------------------------------------------
suppressPackageStartupMessages({
    library(dplyr)
    library(ggplot2)
    library(ggrepel)
})
source("stats/classic_cases.R")

theta_col <- "theta_all"   # swap to "theta_all" / "theta_no_sov" for robustness
out_dir   <- "stats/figures"
dir.create(out_dir, showWarnings = FALSE)

plt_irt <- readRDS("stats/plt_irt.rds") %>% mutate(theta = .data[[theta_col]])

# period of each case, in classic_case_spec order
case_period <- setNames(
    rep(c("Ancient", "Medieval", "Early modern", "Modern"), c(8, 6, 5, 5)),
    classic_case_levels)
period_levels <- c("Ancient", "Medieval", "Early modern", "Modern")

case_pts <- plt_irt %>%
    add_case() %>%
    filter(!is.na(theta)) %>%
    mutate(period = factor(case_period[as.character(case)], levels = period_levels))

case_ctr <- case_pts %>%
    group_by(case, period) %>%
    summarise(n     = n(),
              mu    = mean(theta),
              se    = sd(theta) / sqrt(n),
              mid   = (first(case_start) + first(case_end)) / 2,
              .groups = "drop") %>%
    mutate(lo = mu - 1.96 * se, hi = mu + 1.96 * se,
           case = reorder(case, mu))

period_cols <- c(Ancient = "#B5651D", Medieval = "#2E7D6B",
                 `Early modern` = "#3F5FA8", Modern = "#8E3B8E")

# A. Ranked dot plot: mean theta +/- 95% CI, leader rows behind ----------------
p_rank <- ggplot(case_ctr, aes(mu, case, colour = period)) +
    geom_vline(xintercept = median(plt_irt$theta, na.rm = TRUE),
               linetype = "dashed", colour = "grey60") +
    geom_jitter(data = mutate(case_pts, case = factor(case, levels(case_ctr$case))),
                aes(theta, case), height = 0.15, alpha = 0.15, size = 0.6,
                show.legend = FALSE) +
    geom_errorbarh(aes(xmin = lo, xmax = hi), height = 0.25) +
    geom_point(size = 2.6) +
    geom_text(aes(x = max(case_pts$theta) + 0.15, label = paste0("n=", n)),
              hjust = 0, size = 2.6, colour = "grey40") +
    scale_colour_manual(values = period_cols, name = NULL) +
    scale_x_continuous(expand = expansion(mult = c(0.02, 0.12))) +
    labs(x = expression(paste("Executive constraints (", theta, ")")), y = NULL,
         caption = "Points: leader-level EAP scores. Bars: mean ± 95% CI. Dashed line: full-sample median.") +
    theme_minimal(base_size = 10) +
    theme(legend.position = "top", panel.grid.minor = element_blank(),
          panel.grid.major.y = element_blank())

ggsave(file.path(out_dir, "classic_cases_theta_rank.png"), p_rank, width = 7, height = 6.5, dpi = 300)

# B. Over time: case means at the window midpoint, labelled -------------------
p_time <- ggplot(plt_irt, aes(leader_first_year, theta)) +
    geom_point(alpha = 0.02, size = 0.4, colour = "grey60") +
    geom_smooth(method = "gam", formula = y ~ s(x, bs = "cs"),
                colour = "grey30", linewidth = 0.6, se = FALSE) +
    geom_errorbar(data = case_ctr, aes(x = mid, ymin = lo, ymax = hi, colour = period),
                  inherit.aes = FALSE, width = 0) +
    geom_point(data = case_ctr, aes(mid, mu, fill = period), shape = 21,
               size = 2.6, colour = "black", stroke = 0.3, inherit.aes = FALSE) +
    geom_text_repel(data = case_ctr, aes(mid, mu, label = case, colour = period),
                    inherit.aes = FALSE, size = 2.7, max.overlaps = Inf,
                    box.padding = 0.45, min.segment.length = 0,
                    segment.colour = "grey70", seed = 42, show.legend = FALSE) +
    scale_colour_manual(values = period_cols, name = NULL, aesthetics = c("colour", "fill")) +
    coord_cartesian(xlim = c(-800, 2025)) +
    labs(x = "Year", y = expression(paste("Executive constraints (", theta, ")")),
         caption = "Grey: all leaders + GAM trend. Coloured: classic-case mean ± 95% CI at the case-window midpoint.") +
    theme_minimal(base_size = 10) +
    theme(legend.position = "top", panel.grid.minor = element_blank())

ggsave(file.path(out_dir, "classic_cases_theta_time.png"), p_time, width = 9, height = 5.5, dpi = 300)

case_ctr %>% arrange(desc(mu)) %>% print(n = Inf)

# =============================================================================
# C. Famous rulers: leader-level ranking with 95% CIs ------------------------
#
# Unit = ONE leader, not a polity-period. Each ruler is located by polity
# regex + name regex + a year window that must overlap the tenure; when the
# long-format data has several rows for the same ruler (duplicate spells,
# name variants) the row with the LONGEST tenure is kept, ties broken by id.
# The matched rows are printed so the choice can be audited.
#
# CI = theta +/- 1.96 * EAP posterior SE of that single row (measurement
# uncertainty), NOT a sampling CI across rows. IRT.Rmd now stores the SE as
# `theta_all_se`; if the rds predates that, the SAME one-factor model
# (items_C of IRT.Rmd) is refit here to obtain it.
# =============================================================================
source("stats/famous_rulers.R")
plt_irt <- add_theta_se(plt_irt)


rulers <- bind_rows(lapply(seq_len(nrow(famous_rulers)),
                           function(i) match_ruler(famous_rulers[i, ], plt_irt))) %>%
    mutate(period = factor(period, levels = period_levels),
           lo  = theta - 1.96 * se,
           hi  = theta + 1.96 * se,
           yrs = paste0(ifelse(leader_first_year < 0, paste0(abs(leader_first_year), " BC"), leader_first_year),
                        "–",
                        ifelse(leader_last_year < 0, paste0(abs(leader_last_year), " BC"), leader_last_year)),
           tick = paste0(label, "  (", yrs, ")"),
           tick = reorder(tick, theta))

cat("\n-- Famous rulers: matched rows (n_rows = candidate rows before taking the longest tenure) --\n")
rulers %>%
    arrange(desc(theta)) %>%
    select(label, polity_name, leader_name, leader_first_year, leader_last_year, n_rows, theta, se) %>%
    as.data.frame() %>% print(digits = 3, row.names = FALSE)

# -- ranked dot plot ----------------------------------------------------------
theta_med <- median(plt_irt$theta, na.rm = TRUE)
p_rulers <- ggplot(rulers, aes(theta, tick, colour = period)) +
    geom_vline(xintercept = theta_med, linetype = "dashed", colour = "grey60") +
    geom_errorbarh(aes(xmin = lo, xmax = hi), height = 0.3, linewidth = 0.5) +
    geom_point(size = 2.4) +
    scale_colour_manual(values = period_cols, name = NULL) +
    scale_x_continuous(expand = expansion(mult = 0.04)) +
    labs(x = expression(paste("Executive constraints (", theta, ")")), y = NULL,
         caption = paste0("One row per ruler (longest tenure among duplicate spells). ",
                          "Bars: EAP estimate ± 1.96 posterior SE. Dashed line: full-sample median.")) +
    theme_minimal(base_size = 10) +
    theme(legend.position = "top", panel.grid.minor = element_blank(),
          panel.grid.major.y = element_blank(),
          axis.text.y = element_text(size = 7.5))

ggsave(file.path(out_dir, "famous_rulers_theta_rank.png"), p_rulers,
       width = 7.5, height = 2 + 0.2 * nrow(rulers), dpi = 300)
