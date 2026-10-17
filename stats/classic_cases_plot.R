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
famous_rulers <- tibble::tribble(
    ~label,                    ~polity,                              ~name,                              ~start, ~end,  ~period,
    # ---- Ancient ----
    "Hammurabi",               "^Babylon$",                          "Hammurab",                         -1800, -1600, "Ancient",
    "Ramses II",               "^Egypt$",                            "Ramses II",                        -1300, -1200, "Ancient",
    "Cyrus the Great",         "Achaemenid",                         "Cyrus II",                          -600,  -500, "Ancient",
    "Pericles",                "^Athens$",                           "Pericles",                          -470,  -420, "Ancient",
    "Alexander the Great",     "Macedon",                            "Alexander III",                     -340,  -320, "Ancient",
    "Ashoka",                  "Maurya",                             "Asoka|Ashoka",                      -280,  -220, "Ancient",
    "Qin Shi Huang",           "^Qin Dynasty$",                      "Ying Zheng",                        -250,  -205, "Ancient",
    "Julius Caesar",           "^Roman Republic$",                   "Caesar",                             -60,   -40, "Ancient",
    "Augustus",                "^Roman Empire$",                     "Augustus",                           -30,    20, "Ancient",
    # ---- Medieval ----
    "Justinian I",             "Byzantine",                          "Justinian I",                        520,   570, "Medieval",
    "Tang Taizong",            "^Tang Dynasty$",                     "Li Shimin",                          620,   655, "Medieval",
    "Charlemagne",             "^Franks$",                           "Charles I",                          760,   820, "Medieval",
    "Harun al-Rashid",         "Abbasid",                            "Harun",                              780,   810, "Medieval",
    "William the Conqueror",   "^England$",                          "William I",                         1060,  1090, "Medieval",
    "Saladin",                 "^Egypt",                             "Saladin",                           1170,  1195, "Medieval",
    "Genghis Khan",            "^Mongolia$",                         "Genghis",                           1200,  1230, "Medieval",
    "Frederick II (HRE)",      "^Holy Roman Empire$",                "Frederick II",                      1210,  1255, "Medieval",
    "Mansa Musa",              "^Mali$",                             "Musa I",                            1300,  1340, "Medieval",
    # ---- Early modern ----
    "Mehmed II",               "^Ottoman",                           "Mehmed II",                         1451,  1485, "Early modern",
    "Suleiman I",              "^Ottoman",                           "Suleiman I|Suleyman I",             1515,  1570, "Early modern",
    "Ivan IV (the Terrible)",  "^Russia$",                           "Ivan IV",                           1530,  1590, "Early modern",
    "Akbar",                   "Mughal",                             "Akbar",                             1555,  1610, "Early modern",
    "Elizabeth I",             "^England$",                          "^Elizabeth$|Elizabeth I",           1555,  1605, "Early modern",
    "Louis XIV",               "^France$",                           "Louis XIV",                         1640,  1720, "Early modern",
    "Peter the Great",         "^Russia$",                           "Peter the Great",                   1680,  1730, "Early modern",
    "Frederick the Great",     "^Prussia$",                          "Frederick II",                      1735,  1790, "Early modern",
    "George Washington",       "United States",                      "George Washington",                 1785,  1800, "Early modern",
    # ---- Modern ----
    "Abraham Lincoln",         "United States",                      "Lincoln",                           1855,  1870, "Modern",
    "Meiji (Mutsuhito)",       "^Japan$",                            "Mutsuhito|^Meiji$",                 1865,  1915, "Modern",
    "Lenin",                   "Soviet|^Russia$",                    "Lenin|Ulyanov",                     1915,  1925, "Modern",
    "Stalin",                  "^Soviet Union$",                     "Stalin",                            1924,  1953, "Modern",
    "Hitler",                  "^Germany$",                          "Hitler",                            1933,  1945, "Modern",
    "Mussolini",               "^Italy$",                            "Mussolini",                         1922,  1943, "Modern",
    "F. D. Roosevelt",         "United States",                      "Franklin.*Roosevelt",               1933,  1945, "Modern",
    "Franco",                  "^Spain$",                            "Franco",                            1939,  1975, "Modern",
    "Mao Zedong",              "China",                              "^Mao",                              1949,  1976, "Modern",
    "Kim Il-sung",             "North Korea",                        "Kim Il Sung",                       1948,  1994, "Modern",
    "Margaret Thatcher",       "United Kingdom",                     "Thatcher",                          1979,  1990, "Modern",
    "Nelson Mandela",          "South Africa",                       "Mandela",                           1994,  1999, "Modern",
    "Vladimir Putin",          "^Russia$",                           "Putin",                             1999,  2008, "Modern",
    # Russia has no Putin row after 2008; the Medvedev tandem (Putin as PM) is
    # the closest later window the data offers
    "Medvedev (Putin tandem)", "^Russia$",                           "Medvedev",                          2008,  2012, "Modern",
    # ---- Modern democracies (comparison set) ----
    "Clement Attlee",          "United Kingdom",                     "Attlee",                            1945,  1951, "Modern",
    "Konrad Adenauer",         "^Germany$",                          "Adenauer",                          1949,  1963, "Modern",
    "Jawaharlal Nehru",        "^India$",                            "Nehru",                             1946,  1964, "Modern",
    "Olof Palme",              "^Sweden$",                           "Palme",                             1969,  1976, "Modern",
    "Francois Mitterrand",     "^France$",                           "Mitterrand",                        1981,  1995, "Modern",
    "Helmut Kohl",             "^Germany$",                          "Kohl",                              1982,  1998, "Modern",
    "Lee Teng-hui",            "^Taiwan$",                           "Lee Tenghui",                       1988,  2000, "Modern",
    "Lula da Silva",           "^Brazil$",                           "Lula",                              2003,  2011, "Modern",
    "Barack Obama",            "United States",                      "Obama",                             2009,  2017, "Modern"
)

# -- theta SE: stored by IRT.Rmd, or refit the same one-factor model ----------
if (!"theta_all_se" %in% names(plt_irt)) {
    message("plt_irt.rds has no `theta_all_se`; refitting the one-factor model (items_C) for EAP SEs")
    suppressPackageStartupMessages(library(mirt))
    items_C <- c("constitutionalism_prediction", "federalism_prediction",
                 "collegiality_prediction", "petition_prediction", "representation",
                 "symbolism_recode", "entry_4_prediction", "exit_4_prediction",
                 "checks_prediction", "tenure_3_prediction_w3")
    dat_C <- as.data.frame(plt_irt[items_C])
    for (v in items_C) {                       # recode to 0..K-1 as in IRT.Rmd
        lv <- sort(unique(na.omit(dat_C[[v]])))
        dat_C[[v]] <- match(dat_C[[v]], lv) - 1L
    }
    type_C <- vapply(dat_C, function(x) if (length(unique(na.omit(x))) <= 2) "2PL" else "graded",
                     character(1))
    set.seed(42)
    m_se <- mirt(dat_C, 1, itemtype = type_C, method = "EM",
                 technical = list(NCYCLES = 2000), verbose = FALSE)
    fs <- fscores(m_se, method = "EAP", full.scores.SE = TRUE)
    message(sprintf("  refit vs stored theta_all: r = %.4f (should be ~1)",
                    cor(fs[, "F1"], plt_irt$theta_all, use = "complete.obs")))
    plt_irt$theta_all_se <- fs[, "SE_F1"]
}

# -- match each ruler to one row ----------------------------------------------
match_ruler <- function(spec, df) {
    hits <- df %>%
        filter(grepl(spec$polity, polity_name, ignore.case = TRUE),
               grepl(spec$name,   leader_name, ignore.case = TRUE),
               leader_last_year  >= spec$start,
               leader_first_year <= spec$end,
               !is.na(theta_all)) %>%
        mutate(span = leader_last_year - leader_first_year) %>%
        arrange(desc(span), id)
    if (nrow(hits) == 0) {
        warning(sprintf("no row for '%s' (%s / %s)", spec$label, spec$polity, spec$name))
        return(NULL)
    }
    hits %>% slice(1) %>%
        transmute(label = spec$label, period = spec$period, n_rows = nrow(hits),
                  id, polity_name, leader_name, leader_first_year, leader_last_year,
                  theta = theta_all, se = theta_all_se)
}

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
