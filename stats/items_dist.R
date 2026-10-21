# Distribution table of LLM-coded indicators -> stats/output/items_dist.tex
#
# Panel A: indicators exactly as coded by the LLM, from data/plt_constraints.csv
#          (full leader sample).
# Panel B: derived / recoded items used in the IRT (constitutionalism,
#          representation, symbolism reorder, entry_4, exit_4, checks_3,
#          tenure_3) taken from stats/plt_irt.rds, i.e. the IRT analysis
#          sample (leaders with non-missing first/last year), so the numbers
#          are exactly those the models see.
suppressPackageStartupMessages({library(dplyr); library(readr)})

raw <- read_csv("data/plt_constraints.csv", show_col_types = FALSE)
irt <- readRDS("stats/plt_irt.rds")
n_raw <- nrow(raw); n_irt <- nrow(irt)
cat(sprintf("raw N = %d   IRT sample N = %d\n", n_raw, n_irt))

raw <- raw %>% mutate(
    checks_sum = checks_local_prediction + checks_military_prediction +
        checks_judiciary_prediction + checks_council_prediction +
        checks_bureaucracy_prediction + checks_aristocracy_prediction +
        checks_assembly_prediction + checks_bourgeoisie_prediction +
        checks_clergy_prediction,
    elections_gated = if_else(assembly_prediction == 2, elections_prediction, NA_real_),
    tenure = leader_last_year - leader_first_year + 1
)
irt <- irt %>% mutate(checks_3 = case_when(checks_prediction == 0 ~ 0,
                                           checks_prediction <= 4 ~ 1,
                                           checks_prediction > 4 ~ 2))

# one row: proportions for categories 0..3, pooled ">=4", NA share, mean
row <- function(x, var, cats, levels = NULL, plus4 = FALSE, interval = FALSE) {
    # default: one level per "/"-separated category label
    if (is.null(levels)) levels <- seq_len(length(strsplit(cats, " / ")[[1]])) - 1
    f3 <- function(p) ifelse(is.na(p), "", formatC(p, format = "f", digits = 3))
    n <- length(x)
    if (interval) {
        cells <- rep("", 6)
    } else {
        p <- sapply(levels, function(l) sum(x == l, na.rm = TRUE) / n)
        p <- c(p, rep(NA, 4 - length(p)))
        p4 <- if (plus4) sum(x >= 4, na.rm = TRUE) / n else NA
        pna <- mean(is.na(x)); if (pna == 0) pna <- NA
        cells <- f3(c(p, p4, pna))
    }
    mu <- formatC(mean(x, na.rm = TRUE), format = "f", digits = if (interval) 2 else 3)
    sprintf("\\textit{%s} & %s & %s & %s \\\\", var, cats, paste(cells, collapse = " & "), mu)
}

panel_a <- c(
    row(raw$sovereign_prediction,    "Sovereignty",   "Semi / Full"),
    row(raw$constitution_prediction, "Constitution",  "No / Code of law / Constitution"),
    row(raw$federalism_prediction,   "Federalism",    "No / Yes"),
    row(raw$checks_sum,              "Checks",        "Number of checking actors (0--9)", levels = 0:3, plus4 = TRUE),
    row(raw$collegiality_prediction, "Collegiality",  "No / Yes"),
    row(raw$petition_prediction,     "Petitions",     "No / Yes"),
    row(raw$assembly_prediction,     "Assembly",      "No / Council / Legislature / Popular assembly"),
    row(raw$elections_gated,         "Legislative elections", "No / Non-competitive / Competitive"),
    row(raw$symbolism_prediction,    "Symbolism",     "Plain / Decorated / Deified / Ceremonial"),
    row(raw$tenure,                  "Tenure",        "Years in office (interval)", interval = TRUE)
)
# raw entry/exit have 11 and 15 substantive codes + 99; reported only via Panel B
na_entry <- mean(raw$entry_prediction == 99); na_exit <- mean(raw$exit_prediction == 99)

panel_b <- c(
    row(irt$constitutionalism_prediction, "Constitutionalism",
        "No / Constitution bypassed / Code of law / Constitution"),
    row(irt$checks_3,                     "Checks (3 levels)", "No (0) / Partial (1--4) / Full (5--9)"),
    row(irt$representation,               "Representation",
        "No / Council or non-competitive legislature / Competitive legislature / Popular assembly"),
    row(irt$symbolism_recode,             "Symbolism (reordered)", "Deified / Decorated / Plain / Ceremonial"),
    row(irt$entry_4_prediction,           "Entry",  "Irregular / Hereditary / Appointment / Election"),
    row(irt$exit_4_prediction,            "Exit",   "Irregular / Natural / Voluntary / Institutionalized"),
    row(irt$tenure_3_prediction_w3,       "Tenure (3 levels)", "$>$15 / 9--15 / $\\leq$8 years (leading window)", levels = 0:2)
)

fmt_n <- function(n) format(n, big.mark = ",")
note <- paste0(
    "\\textit{Note:} Columns 0--3 (and $\\geq$4 for the Checks count) report the proportion of ",
    "observations coded in each category; NA is the proportion missing; $\\mu$ is the mean category ",
    "value over non-missing observations. Unit of analysis: polity-leader. ",
    "Panel A reports the indicators exactly as coded by the LLM (\\textit{N} = ", fmt_n(n_raw), "). ",
    "Checks is the number of the nine actors (local, military, clergy, aristocracy, bourgeoisie, ",
    "bureaucracy, judiciary, assembly, council) coded as exercising an effective check. ",
    "Legislative elections is coded only when Assembly = Legislature and is NA otherwise. ",
    "Tenure is the number of years in office; $\\mu$ is the mean in years. ",
    "Panel B reports the recoded items that enter the IRT models, computed on the IRT analysis ",
    "sample (\\textit{N} = ", fmt_n(n_irt), "; leaders with missing first or last year dropped). ",
    "Constitutionalism combines Constitution with the bypass classifier: polities coded as having a ",
    "constitution that was bypassed in practice are placed between no written law and a code of law. ",
    "Checks (3 levels) collapses the count. Representation combines Assembly and Legislative ",
    "elections: councils and legislatures without competitive elections share one category. ",
    "Symbolism (reordered) rearranges the categories so that higher values indicate more constraint. ",
    "Entry collapses the original 11 entry codes (irregular = force or foreign imposition; appointment ",
    "= ruling party, royal council, military, head of state or head of government; election = ",
    "legislature or popular election); codes `other' and `unknown' are NA ",
    "(`unknown' alone: ", formatC(na_entry, format = "f", digits = 3), "). ",
    "Exit collapses the original 15 exit codes (irregular = deposition, assassination, suicide or ",
    "death on campaign or in battle; natural = natural causes or ill health; voluntary = retirement; ",
    "institutionalized = other regular exit or regular transition); `still in office' and `unknown' ",
    "are NA (`unknown' alone: ", formatC(na_exit, format = "f", digits = 3), "). ",
    "Tenure (3 levels) uses a leading three-leader window: the longest tenure among the leader and ",
    "the two leaders who follow within the same polity spell (spells split at gaps of more than ",
    "500 years), coded 2 if at most 8 years, 1 if 9--15, and 0 if more than 15."
)

tex <- c(
    "\\begin{table}[htbp]",
    "\\centering",
    "\\caption{Distribution of LLM-coded indicators}",
    "\\label{tab:coding}",
    "\\resizebox{\\textwidth}{!}{%",
    "\\scriptsize",
    "\\begin{tabular}{@{}llccccccc@{}}",
    "\\toprule",
    "Variable & Categories & 0 & 1 & 2 & 3 & $\\geq$4 & NA & $\\mu$ \\\\",
    "\\midrule",
    "\\multicolumn{9}{@{}l}{\\textit{Panel A: Indicators as coded by the LLM}} \\\\[2pt]",
    panel_a,
    "\\midrule[\\heavyrulewidth]",
    "\\multicolumn{9}{@{}l}{\\textit{Panel B: Recoded items used in the IRT models}} \\\\[2pt]",
    panel_b,
    "\\bottomrule",
    "\\end{tabular}%",
    "}",
    "\\vspace{2mm}",
    "\\begin{minipage}{\\textwidth}",
    "\\footnotesize",
    note,
    "\\end{minipage}",
    "\\end{table}"
)
writeLines(tex, "stats/output/items_dist.tex")
cat("wrote stats/output/items_dist.tex\n")
