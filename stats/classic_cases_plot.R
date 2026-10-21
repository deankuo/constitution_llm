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
    geom_errorbar(aes(xmin = lo, xmax = hi), height = 0.25) +
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

# B. Over time: famous leaders on the theta scale, tenure as a segment ---------
#
# Background = rug of ALL leader rows (x: first year of tenure, y: theta) plus
# the GAM trend. Foreground = a hand-picked subset of stats/famous_rulers.R
# (`time_rulers`, edit to taste): point at the tenure midpoint, horizontal
# segment = tenure, label = name (tenure). Chosen to span the whole period
# with both high- and low-theta leaders in every era. No era colours.
source("stats/famous_rulers.R")
time_rulers <- c(
    # Ancient
    "Hammurabi", "Ramses II", "Cyrus the Great", "Pericles", "Alexander the Great",
    "Qin Shi Huang", "Julius Caesar", "Augustus",
    # Medieval
    "Justinian I", "Tang Taizong", "Charlemagne", "Harun al-Rashid",
    "William the Conqueror", "Genghis Khan", "Frederick II (HRE)",
    # Early modern
    "Suleiman I", "Akbar", "Elizabeth I", "Louis XIV", "Peter the Great",
    "Frederick the Great", "George Washington",
    # Modern
    "Abraham Lincoln", "Meiji (Mutsuhito)", "Stalin", "Hitler", "F. D. Roosevelt",
    "Mao Zedong", "Margaret Thatcher", "Nelson Mandela", "Vladimir Putin")
tmp_irt <- plt_irt
if (!"theta_all_se" %in% names(tmp_irt)) tmp_irt$theta_all_se <- NA_real_
fmt_year <- function(y) ifelse(y < 0, paste0(abs(y), " BC"), as.character(y))
time_pts <- bind_rows(lapply(which(famous_rulers$label %in% time_rulers),
                             function(i) match_ruler(famous_rulers[i, ], tmp_irt))) %>%
    mutate(mid = (leader_first_year + leader_last_year) / 2,
           lab = paste0(label, " (", fmt_year(leader_first_year), "–", fmt_year(leader_last_year), ")"))
cat("\n-- Theta-over-time figure: leaders shown --\n")
time_pts %>% arrange(leader_first_year) %>%
    select(label, polity_name, leader_first_year, leader_last_year, theta) %>%
    as.data.frame() %>% print(digits = 3, row.names = FALSE)

# x_lo <- 200 * floor(min(time_pts$leader_first_year) / 200)
x_lo <- -2700
x_breaks <- seq(x_lo, 2000, by = 200)
p_time <- ggplot(plt_irt, aes(leader_first_year, theta)) +
    geom_smooth(method = "gam", formula = y ~ s(x, bs = "cs"),
                colour = "grey55", linewidth = 0.6, se = FALSE, na.rm = TRUE) +
    geom_segment(data = time_pts, aes(x = leader_first_year, xend = leader_last_year, y = theta, yend = theta),
                 inherit.aes = FALSE, colour = "#1F3A5F", linewidth = 1.6, lineend = "round", alpha = 0.9) +
    geom_point(data = time_pts, aes(mid, theta), inherit.aes = FALSE,
               shape = 21, size = 2.4, fill = "white", colour = "#1F3A5F", stroke = 0.8) +
    geom_text_repel(data = time_pts, aes(mid, theta, label = lab), inherit.aes = FALSE,
                    size = 2.5, colour = "grey15", max.overlaps = Inf, box.padding = 0.55,
                    point.padding = 0.3, min.segment.length = 0, segment.colour = "grey65",
                    segment.size = 0.3, force = 2, seed = 42) +
    scale_x_continuous(breaks = x_breaks, labels = fmt_year, expand = expansion(mult = 0.01)) +
    coord_cartesian(xlim = c(x_lo, 2025), ylim = c(-1.7, 2.1), clip = "off") +
    labs(x = NULL, y = expression(paste("Executive constraints (", theta, ")"))) +
    theme_minimal(base_size = 10) +
    theme(legend.position = "none", panel.grid.minor = element_blank(),
          axis.text.x = element_blank())

# Lower panel: how the leader rows are distributed over time (50-year bins).
# Counts range from a few dozen per bin before 500 BC to >10,000 per bin after
# 1500, so the y-axis is log10 — otherwise antiquity is invisible.
p_dist <- ggplot(plt_irt, aes(leader_first_year)) +
    geom_histogram(binwidth = 50, boundary = 0, fill = "grey55", colour = "white", linewidth = 0.2, na.rm = TRUE) +
    scale_x_continuous(breaks = x_breaks, labels = fmt_year, expand = expansion(mult = 0.01)) +
    # scale_y_log10(breaks = c(10, 100, 1000, 10000), labels = c("10", "100", "1k", "10k"),
    #               expand = expansion(mult = c(0, 0.05))) +
    coord_cartesian(xlim = c(x_lo, 2025)) +
    labs(x = NULL, y = "leader rows\nper 50 yrs (log)") +
    theme_minimal(base_size = 10) +
    theme(panel.grid.minor = element_blank(), axis.text.x = element_text(size = 8),
          axis.title.y = element_text(size = 8))

suppressPackageStartupMessages(library(patchwork))
p_time_full <- (p_time / p_dist) +
    plot_layout(heights = c(4, 1.1)) +
    plot_annotation(
        caption = paste0("Top: segments = tenure of ", nrow(time_pts), " well-known leaders, point at the tenure midpoint, ",
                         "height = that leader's theta; label = name (tenure). Grey curve: GAM trend of theta on first year of tenure.\n",
                         "Bottom: number of leader rows in the IRT data by first year of tenure, 50-year bins, log scale."),
        theme = theme(plot.caption = element_text(hjust = 0, size = 8)))

ggsave(file.path(out_dir, "classic_cases_theta_time.png"), p_time_full, width = 14, height = 8, dpi = 300)

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
