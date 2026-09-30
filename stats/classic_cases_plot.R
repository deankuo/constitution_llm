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
source("classic_cases.R")

theta_col <- "theta_agg"   # swap to "theta_all" / "theta_no_sov" for robustness
out_dir   <- "figures"
dir.create(out_dir, showWarnings = FALSE)

plt_irt <- readRDS("plt_irt.rds") %>% mutate(theta = .data[[theta_col]])

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

ggsave(file.path(out_dir, "classic_cases_theta_rank.pdf"), p_rank, width = 7, height = 6.5)
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

ggsave(file.path(out_dir, "classic_cases_theta_time.pdf"), p_time, width = 9, height = 5.5)
ggsave(file.path(out_dir, "classic_cases_theta_time.png"), p_time, width = 9, height = 5.5, dpi = 300)

case_ctr %>% arrange(desc(mu)) %>% print(n = Inf)
