# ---------------------------------------------------------------------------
# Classic cases in data/plt_constraints.csv
#
# NOTE 1: plt_constraints.csv is LONG format -- one row per paramount-overlord /
#         office spell within a tenure. A leader can appear several times
#         (e.g. Stalin 4x, Gizurr Thorvaldsson 5x). That is not a bug.
# NOTE 2: several historical polities are split over more than one
#         `polity_name` string (id ranges interleave), so some cases need a
#         vector of names, not one string.
# NOTE 3: a polity_name alone is NOT enough for period-specific cases -- e.g.
#         "Roman Republic" also contains the 1849 Mazzini republic, "Venice"
#         runs to 1918, "Iceland" to 2017. Always combine with the year window.
# ---------------------------------------------------------------------------

## 1. Simple named vector: case label -> polity_name (";"-separated if multiple)
classic_cases <- c(
  # 古代
  "Athens (democracy)"        = "Athens",
  "Roman Republic"            = "Roman Republic",
  "Sparta"                    = "Kingdom Of Sparta",
  "Qin"                       = "Qin Dynasty",
  "Achaemenid Persia"         = "Achaemenid Empire",
  # 中世紀
  "Venice"                    = "Venice",
  "Icelandic Commonwealth"    = "Iceland",
  "Holy Roman Empire"         = "Holy Roman Empire",
  "Abbasid Caliphate"         = "The Abbasid Caliphate",
  "Heian Japan"               = "Japan",
  # 近代早期
  "Dutch Republic"            = "Netherlands",
  "Poland-Lithuania"          = "Poland-Lithuania",
  "France (Louis XIV)"        = "France",
  "Tokugawa Japan"            = "Japan",
  "Ottoman Empire"            = "Ottoman",
  # 現代
  "United Kingdom (1900)"     = "United Kingdom",
  "United States (1900)"      = "United States Of America",
  "Soviet Union (Stalin)"     = "Soviet Union",
  "Nazi Germany"              = "Germany",
  "Sweden (contemporary)"     = "Sweden"
)

## 2. Full specification: every polity_name variant + the year window
classic_case_spec <- list(
  # ---- 古代 -------------------------------------------------------------
  "Athens (democracy)"     = list(polity = c("Athens", "Athenai Attika"),
                                  start = -508, end = -322),
  "Roman Republic"         = list(polity = "Roman Republic",
                                  start = -509, end = -27),
  "Sparta"                 = list(polity = "Kingdom Of Sparta",
                                  start = -785, end = -192),
  "Qin"                    = list(polity = c("Qin Dynasty", "Before Qin"),
                                  start = -350, end = -206),
  "Achaemenid Persia"      = list(polity = c("Achaemenid Empire", "Achaemenid Kingdom"),
                                  start = -559, end = -330),
  # ---- 中世紀 -----------------------------------------------------------
  "Venice"                 = list(polity = "Venice",
                                  start = 697, end = 1797),
  "Icelandic Commonwealth" = list(polity = "Iceland",
                                  start = 930, end = 1262),
  "Holy Roman Empire"      = list(polity = c("Holy Roman Empire", "Holy_Roman_Empire",
                                             "Germano Roman Empire"),
                                  start = 800, end = 1806),
  "Abbasid Caliphate"      = list(polity = c("The Abbasid Caliphate", "The Caliphate"),
                                  start = 750, end = 1258),
  "Heian Japan"            = list(polity = "Japan",
                                  start = 794, end = 1185),
  # ---- 近代早期 ---------------------------------------------------------
  "Dutch Republic"         = list(polity = c("Netherlands", "The Netherlands",
                                             "United Provinces"),
                                  start = 1581, end = 1795),
  "Poland-Lithuania"       = list(polity = c("Poland-Lithuania", "Poland", "Lithuania"),
                                  start = 1569, end = 1795),
  "France (Louis XIV)"     = list(polity = "France",
                                  start = 1643, end = 1715),
  "Tokugawa Japan"         = list(polity = c("Japan", "Tokugawa"),
                                  start = 1603, end = 1868),
  "Ottoman Empire"         = list(polity = c("Ottoman", "Ottoman Empire"),
                                  start = 1300, end = 1922),
  # ---- 現代 -------------------------------------------------------------
  "United Kingdom (1900)"  = list(polity = c("United Kingdom", "Great Britain", "England"),
                                  start = 1885, end = 1914),
  "United States (1900)"   = list(polity = "United States Of America",
                                  start = 1885, end = 1914),
  "Soviet Union (Stalin)"  = list(polity = c("Soviet Union", "Soviet Russian Republic"),
                                  start = 1924, end = 1953),
  "Nazi Germany"           = list(polity = c("Germany", "German_Reich"),
                                  start = 1933, end = 1945),
  "Sweden (contemporary)"  = list(polity = "Sweden",
                                  start = 1990, end = 2020)
)

## 3. Helper: pull the rows for one case (year windows OVERLAP the tenure)
get_case <- function(df, case, spec = classic_case_spec) {
  s <- spec[[case]]
  stopifnot(!is.null(s))
  df[df$polity_name %in% s$polity &
     df$leader_last_year  >= s$start &
     df$leader_first_year <= s$end, ]
}

## 4. Sanity check -- every name must exist in the data
# d <- read.csv("data/plt_constraints.csv")
# all_names <- unlist(lapply(classic_case_spec, `[[`, "polity"))
# setdiff(all_names, unique(d$polity_name))   # must be character(0)


# ---------------------------------------------------------------------------
# 5. dplyr interface -- use this in IRT.Rmd instead of `polity_name %in% ...`
#
#    Why not just a `key_cases` character vector: group_by(polity_name) would
#    average "Athens" over -1201..1821 and "France" over 481..2017, and would
#    split the Ottoman / Dutch / UK / Nazi / HRE cases across two or three
#    polity_name strings. `add_case()` collapses the name variants AND applies
#    the year window, then you group by `case`.
# ---------------------------------------------------------------------------

classic_case_tbl <- local({
  do.call(rbind, lapply(names(classic_case_spec), function(k) {
    s <- classic_case_spec[[k]]
    data.frame(case = k, polity_name = s$polity,
               case_start = s$start, case_end = s$end,
               stringsAsFactors = FALSE)
  }))
})

# case ordering for printing: 古代 -> 現代
classic_case_levels <- names(classic_case_spec)

#' Keep only rows belonging to a classic case, adding a `case` column.
#' Rows outside every case window are dropped.
#'
#' @param match "overlap"  keep a tenure if it overlaps the window at all
#'                         (default; a tenure straddling the boundary is kept
#'                         whole, so e.g. the Tokugawa case reaches past 1868)
#'              "midpoint" keep a tenure only if its midpoint falls inside the
#'                         window (tighter; use for period-purity)
add_case <- function(df, tbl = classic_case_tbl, match = c("overlap", "midpoint")) {
  match <- base::match.arg(match)
  out <- dplyr::inner_join(df, tbl, by = "polity_name",
                           relationship = "many-to-many")
  out <- if (match == "overlap") {
    dplyr::filter(out,
                  .data$leader_last_year  >= .data$case_start,
                  .data$leader_first_year <= .data$case_end)
  } else {
    dplyr::filter(out,
                  (.data$leader_first_year + .data$leader_last_year) / 2 >= .data$case_start,
                  (.data$leader_first_year + .data$leader_last_year) / 2 <= .data$case_end)
  }
  dplyr::mutate(out, case = factor(.data$case, levels = classic_case_levels))
}
