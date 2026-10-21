# Correlation table for the Version-C IRT items (items_C of IRT.Rmd).
# Reads the scored dataset written by IRT.Rmd (stats/plt_irt.rds) so the
# recodes (constitutionalism, representation, symbolism_recode, entry_4,
# exit_4, checks sum, tenure_3_w3) are exactly those used in the IRT fits.
#
# Output:
#   stats/output/items_C_corr_pearson.csv     Pearson, pairwise-complete
#   stats/output/items_C_corr_polychoric.csv  polychoric (psych), pairwise
#   stats/output/items_C_corr_n.csv           pairwise N behind each cell
suppressPackageStartupMessages(library(psych))

plt_irt <- readRDS("stats/plt_irt.rds")

items_C <- c(
    "constitutionalism_prediction",
    "federalism_prediction",
    "collegiality_prediction",
    "petition_prediction",
    "representation",
    "symbolism_recode",
    "entry_4_prediction",
    "exit_4_prediction",
    "checks_prediction",
    "tenure_3_prediction_w3"
)
short <- c("constit", "federal", "colleg", "petition", "represent", "symbol",
           "entry4", "exit4", "checks", "tenure3")

dat <- as.data.frame(plt_irt[items_C])
names(dat) <- short

cat(sprintf("rows: %d   complete cases: %d\n", nrow(dat), sum(complete.cases(dat))))

r_pear <- cor(dat, use = "pairwise.complete.obs")
n_pair <- crossprod(!is.na(as.matrix(dat)))
r_poly <- polychoric(dat, max.cat = 12, correct = 0.5)$rho

write_tbl <- function(m, path, digits = 3) {
    out <- data.frame(item = rownames(m), round(m, digits), check.names = FALSE)
    write.csv(out, path, row.names = FALSE)
}
write_tbl(r_pear, "stats/output/items_C_corr_pearson.csv")
write_tbl(r_poly, "stats/output/items_C_corr_polychoric.csv")
write_tbl(n_pair, "stats/output/items_C_corr_n.csv", digits = 0)

cat("\n== Pearson (pairwise complete) ==\n")
print(round(r_pear, 2))
cat("\n== Polychoric ==\n")
print(round(r_poly, 2))

# ---------------------------------------------------------------------------
# LaTeX table (lower triangle, polychoric) -> stats/output/items_C_corr.tex
# ---------------------------------------------------------------------------
labels <- c("Constitutionalism", "Federalism", "Collegiality", "Petitions",
            "Representation", "Symbolism", "Entry", "Exit", "Checks", "Tenure")
stopifnot(length(labels) == ncol(r_poly))

fmt_cell <- function(x) {
    s <- formatC(x, format = "f", digits = 3)
    ifelse(x < 0, s, paste0("\\phantom{-}", s))
}

k <- ncol(r_poly)
off <- n_pair[lower.tri(n_pair)]
rows <- vapply(seq_len(k), function(i) {
    cells <- rep("", k)
    cells[seq_len(i)] <- fmt_cell(r_poly[i, seq_len(i)])
    sprintf("%-26s & %s \\\\", sprintf("%d. %s", i, labels[i]),
            paste(formatC(cells, width = -17), collapse = " & "))
}, character(1))

note <- paste0(
    "\\textit{Note:} Entries are polychoric correlations, not Pearson correlations. ",
    "All items are binary or ordinal (higher values denote more constraint on the executive); ",
    "the polychoric coefficient estimates the correlation between the latent continuous ",
    "variables assumed to underlie each pair of ordinal items, which the Pearson coefficient ",
    "computed on integer scores attenuates. Estimates use pairwise-complete observations ",
    sprintf("($N$ = %s--%s per cell) and a 0.5 continuity correction for empty cells ",
            format(min(off), big.mark = ","), format(max(off), big.mark = ",")),
    "(\\texttt{psych::polychoric}). ",
    "Checks is the count (0--9) of actors coded as exercising an effective check. ",
    "Tenure is a three-category item based on a leading three-leader window: for each leader, ",
    "the longest tenure among that leader and the two leaders who follow within the same ",
    "polity spell (spells are split at gaps of more than 500 years). The item equals 2 if ",
    "this maximum is at most 8 years, 1 if it is 9--15 years, and 0 if it exceeds 15 years, ",
    "so higher values indicate more frequent executive turnover."
)

tex <- c(
    "\\begin{table}[H]",
    "\\centering",
    "\\caption{Polychoric Correlations Among IRT Items}",
    "\\label{tab:corr-table}",
    "\\resizebox{\\textwidth}{!}{%",
    "\\footnotesize",
    sprintf("\\begin{tabular}{@{}l*{%d}{c}@{}}", k),
    "\\toprule",
    sprintf("%-26s & %s \\\\", "", paste(formatC(as.character(seq_len(k)), width = -17),
                                         collapse = " & ")),
    "\\midrule",
    rows,
    "\\bottomrule",
    "\\end{tabular}%",
    "}",
    "\\begin{minipage}{\\textwidth}",
    "\\vspace{4pt}\\footnotesize",
    note,
    "\\end{minipage}",
    "\\end{table}"
)
writeLines(tex, "stats/output/items_C_corr.tex")
cat("\nwrote stats/output/items_C_corr.tex\n")
