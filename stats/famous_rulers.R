# ---------------------------------------------------------------------------
# Famous rulers: leader-level test set shared by classic_cases_plot.R (section
# C) and export_famous_rulers.R (the direct-constraint pilot sample).
#
# Unit = ONE leader, not a polity-period. Each ruler is located by polity
# regex + name regex + a year window that must overlap the tenure; when the
# long-format data has several rows for the same ruler (duplicate spells,
# name variants) the row with the LONGEST tenure is kept, ties broken by id.
#
# `add_theta_se(plt_irt)` returns the data with `theta_all_se` (EAP posterior
# SE): kept from IRT.Rmd when stored, otherwise the SAME one-factor model
# (items_C of IRT.Rmd) is refit here to obtain it.
# ---------------------------------------------------------------------------
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
add_theta_se <- function(plt_irt) {
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
    plt_irt
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
