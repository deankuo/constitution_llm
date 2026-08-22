#===============================================================================
# 2026.08.22 This script is for IRT models on plt constraints dataset
#===============================================================================

# Setup
suppressPackageStartupMessages({
    library(haven)
    library(ggplot2)
    library(tidyverse)
})


# Load datasets
plt_gemini <- read_csv("./data/plt_constraints.csv", show_col_types = F)
problems(plt_gemini)
