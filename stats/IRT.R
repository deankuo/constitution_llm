#===============================================================================
# 2026.08.22 This script is for IRT models on plt constraints dataset
#===============================================================================

# Setup
suppressPackageStartupMessages({
    library(haven)
    library(ggplot2)
    library(tidyverse)
    library(mirt)
})

# mirt 平行運算（依你的機器調整核心數）
mirtCluster(4)


# Load datasets
plt_gemini <- read_csv("./data/plt_constraints.csv", show_col_types = F)
plt_llama <- read_csv("./data/Llama/plt_constraints_llama8b.csv", show_col_types = F)
plt_qwen <- read_csv("./data/Qwen/plt_constraints_qwen32b.csv", show_col_types = F)


# data aggregation
plt_gemini <- plt_gemini %>%
    mutate(
        entry_3_prediction = case_when(
            entry_prediction %in% c(0, 1) ~ 0,
            entry_prediction == 4 ~ 1,
            entry_prediction %in% c(2, 3, 5, 7, 8) ~ 2,
            entry_prediction %in% c(6, 9) ~ 3
        ),
        exit_3_prediction = case_when(
            exit_prediction %in% c(3, 4, 7, 8, 9, 10, 11, 12, 13) ~ 0,
            exit_prediction %in% c(5, 6) ~ 1,
            exit_prediction == 0 ~ 2,
            exit_prediction %in% c(1, 2) ~ 3
        ),
        checks_prediction = checks_local_prediction + checks_military_prediction +
                     checks_judiciary_prediction + checks_council_prediction +
                     checks_bureaucracy_prediction + checks_aristocracy_prediction +
                     checks_assembly_prediction + checks_bourgeoisie_prediction +
                     checks_clergy_prediction,
        across(c(exit_prediction, entry_prediction), ~na_if(., 99)))

#
plt_gemini %>%
    group_by(entry_3_prediction) %>%
    summarise(n = n(),
              rest = mean(sovereign_prediction + constitution_prediction +
                          federalism_prediction + checks_prediction +
                          collegiality_prediction + petition_prediction + assembly_prediction +
                          elections_prediction,
                          na.rm = TRUE))


#===============================================================================
# IRT: item sets
#===============================================================================

# Version A: 全部 19 題各自進模型
items_A <- c(
    "constitution_prediction",
    "sovereign_prediction",
    "federalism_prediction",
    "collegiality_prediction",
    "petition_prediction",
    "assembly_prediction",
    "elections_prediction",
    "symbolism_prediction",
    "entry_3_prediction",
    "exit_3_prediction",
    "checks_local_prediction",
    "checks_military_prediction",
    "checks_judiciary_prediction",
    "checks_council_prediction",
    "checks_bureaucracy_prediction",
    "checks_aristocracy_prediction",
    "checks_assembly_prediction",
    "checks_bourgeoisie_prediction",
    "checks_clergy_prediction"
)

# Version B: 9 個 checks 換成加總後的 checks_prediction，共 11 題
items_B <- c(
    "constitution_prediction",
    "sovereign_prediction",
    "federalism_prediction",
    "collegiality_prediction",
    "petition_prediction",
    "assembly_prediction",
    "elections_prediction",
    "symbolism_prediction",
    "entry_3_prediction",
    "exit_3_prediction",
    "checks_prediction"
)

# checks_prediction 高分格極稀疏（8 = 189 筆 0.14%、9 = 33 筆 0.02%），
# 門檻會估不穩甚至讓 EM 不收斂。預設把 7/8/9 併成 "7+"（合計 1,274 筆）。
# 只影響估計穩定性，不改變「checks 越多 → 約束越強」這個實質假設。
# 想跑原始 0-9 就把這個改成 FALSE。
COLLAPSE_CHECKS_TOP <- TRUE

plt_gemini <- plt_gemini %>%
    mutate(checks_prediction = if (COLLAPSE_CHECKS_TOP) {
        pmin(checks_prediction, 7)
    } else {
        checks_prediction
    })


#-------------------------------------------------------------------------------
# helper: 建 item matrix，並自動判斷每題的 itemtype
#   2 類 -> "2PL"；3 類以上 -> "graded" (Samejima GRM)
#   mirt 需要類別為連續整數，這裡把每題重新編成 0..K-1（順序保留）
#-------------------------------------------------------------------------------
make_irt_data <- function(df, items) {
    dat <- df %>% select(all_of(items)) %>% as.data.frame()

    for (v in items) {
        x <- dat[[v]]
        lv <- sort(unique(x[!is.na(x)]))
        # 印出來確認沒有意外的類別（例如殘留的 99）
        cat(sprintf("%-32s n_cat=%d  levels=%s  NA=%d (%.1f%%)\n",
                    v, length(lv), paste(lv, collapse = ","),
                    sum(is.na(x)), 100 * mean(is.na(x))))
        dat[[v]] <- match(x, lv) - 1L   # 重編成 0..K-1
    }
    dat
}

get_itemtype <- function(dat) {
    vapply(dat, function(x) {
        k <- length(unique(x[!is.na(x)]))
        if (k <= 2) "2PL" else "graded"
    }, character(1))
}

cat("\n===== Version A items =====\n")
dat_A <- make_irt_data(plt_gemini, items_A)
type_A <- get_itemtype(dat_A)

cat("\n===== Version B items =====\n")
dat_B <- make_irt_data(plt_gemini, items_B)
type_B <- get_itemtype(dat_B)

print(type_A)
print(type_B)

#-------------------------------------------------------------------------------
# 單維 IRT（unidimensional）
#   theta = 每個 leader-spell 的潛在「約束程度」
#   a_j   = 每個指標的 discrimination，就是你要的 weighting
#   N = 135,672，EM 會跑一陣子
#-------------------------------------------------------------------------------
set.seed(42)

mod_A <- mirt(dat_A, model = 1, itemtype = type_A,
              method = "EM", SE = TRUE,
              technical = list(NCYCLES = 2000))

mod_B <- mirt(dat_B, model = 1, itemtype = type_B,
              method = "EM", SE = TRUE,
              technical = list(NCYCLES = 2000))

saveRDS(mod_A, "./stats/mod_A_unidim.rds")
saveRDS(mod_B, "./stats/mod_B_unidim.rds")


#===============================================================================
# Evaluation
#   下面每一段都可以獨立跑，用來自己確認結果
#===============================================================================

#-------------------------------------------------------------------------------
# E1. 收斂了沒
#-------------------------------------------------------------------------------
extract.mirt(mod_A, "converged")
extract.mirt(mod_A, "iterations")
extract.mirt(mod_B, "converged")
extract.mirt(mod_B, "iterations")


#-------------------------------------------------------------------------------
# E2. Discriminations = 每個指標的 weighting  ★ 主要結果
#     IRTpars = TRUE 時，graded 題會給 a 和 b1..bK（門檻）
#-------------------------------------------------------------------------------
coef_A <- coef(mod_A, simplify = TRUE, IRTpars = TRUE)$items
coef_B <- coef(mod_B, simplify = TRUE, IRTpars = TRUE)$items

round(coef_A, 3)
round(coef_B, 3)

# 依 discrimination 排序，方便看誰權重高
coef_A %>%
    as.data.frame() %>%
    rownames_to_column("item") %>%
    select(item, a) %>%
    arrange(desc(a)) %>%
    print()


#-------------------------------------------------------------------------------
# E3. 標準化載荷 (F1) 與 communality (h2)
#     載荷是 discrimination 的標準化版本，比較好跨題比較
#-------------------------------------------------------------------------------
summary(mod_A)
summary(mod_B)


#-------------------------------------------------------------------------------
# E4. GRM 門檻是否遞增有序  ★ 檢驗 entry_3 / exit_3 / symbolism 的類別排序
#     若 b1 < b2 < b3 不成立（disordered thresholds），
#     代表該題的類別順序跟資料不一致 —— 這是重新考慮分類方式的訊號
#-------------------------------------------------------------------------------
graded_items <- names(type_A)[type_A == "graded"]

# FIX: discrimination 為負時，正確的門檻方向是「遞減」而非「遞增」。
#      (b_k = -d_k / a，a < 0 會把順序翻過來。已用模擬驗證。)
for (it in graded_items) {
    a_it <- coef_A[it, "a"]
    b <- coef_A[it, grep("^b[0-9]", colnames(coef_A))]
    b <- b[!is.na(b)]
    ok <- if (a_it >= 0) all(diff(b) > 0) else all(diff(b) < 0)
    cat(sprintf("%-32s a = %6.3f   b = %-28s ordered? %s\n",
                it, a_it, paste(round(b, 3), collapse = "  "), ok))
}


#-------------------------------------------------------------------------------
# E5. Item fit (S-X2)
#     注意：N = 135,672，S-X2 的檢定力極大，幾乎所有題目 p < .001。
#     不要看 p 值，看 RMSEA.S_X2（一般 < .05 算可接受）
#-------------------------------------------------------------------------------
fit_A <- itemfit(mod_A, fit_stats = "S_X2", na.rm = T)
fit_A %>% arrange(desc(RMSEA.S_X2)) %>% print()

fit_B <- itemfit(mod_B, fit_stats = "S_X2", na.rm = T)
fit_B %>% arrange(desc(RMSEA.S_X2)) %>% print()


#-------------------------------------------------------------------------------
# E6. Q3 局部相依 (local dependence)  ★ 檢驗 assembly / checks_assembly / elections
#     單維模型假設「給定 theta 後題目間無關」。
#     |Q3| > 0.2 通常視為違反。列出最嚴重的幾對。
#-------------------------------------------------------------------------------
q3_A <- residuals(mod_A, type = "Q3", suppress = 0.2)
print(q3_A)

# 完整矩陣 -> 轉成 pair 清單排序
q3_full <- residuals(mod_A, type = "Q3", verbose = FALSE)
q3_pairs <- as.data.frame(as.table(q3_full)) %>%
    filter(as.character(Var1) < as.character(Var2)) %>%
    rename(item1 = Var1, item2 = Var2, Q3 = Freq) %>%
    arrange(desc(abs(Q3)))
head(q3_pairs, 20)


#-------------------------------------------------------------------------------
# E7. 潛在分數 theta（= 綜合約束指標）與信度
#-------------------------------------------------------------------------------
# FIX: 原本物件也叫 theta_A，mutate() 建立 theta_A 欄位後會遮蔽掉外面的矩陣，
#      導致第二個引數 theta_A[, "SE_F1"] 變成對 vector 取二維索引而報錯。
#      改用 fs_A / fs_B 命名避開。
fs_A <- fscores(mod_A, method = "EAP", full.scores = TRUE, full.scores.SE = TRUE)
fs_B <- fscores(mod_B, method = "EAP", full.scores = TRUE, full.scores.SE = TRUE)

# empirical reliability
empirical_rxx(fs_A)
empirical_rxx(fs_B)

plt_gemini <- plt_gemini %>%
    mutate(theta_A = fs_A[, "F1"], theta_A_se = fs_A[, "SE_F1"],
           theta_B = fs_B[, "F1"], theta_B_se = fs_B[, "SE_F1"])


#-------------------------------------------------------------------------------
# E8. Version A vs Version B 比較
#     等權加總 (checks_prediction) vs 讓 IRT 各自估權重，差多少
#-------------------------------------------------------------------------------
cor(plt_gemini$theta_A, plt_gemini$theta_B, use = "complete.obs")
cor(plt_gemini$theta_A, plt_gemini$theta_B, use = "complete.obs", method = "spearman")

# 也跟最單純的等權加總分數比
plt_gemini <- plt_gemini %>%
    mutate(simple_sum = rowSums(across(all_of(items_A)), na.rm = TRUE))

cor(plt_gemini$theta_A, plt_gemini$simple_sum, use = "complete.obs")

ggplot(plt_gemini, aes(x = theta_A, y = theta_B)) +
    geom_hex(bins = 60) +
    scale_fill_viridis_c(trans = "log10") +
    labs(x = "theta (Version A: 19 items)",
         y = "theta (Version B: checks summed)") +
    theme_minimal()


#-------------------------------------------------------------------------------
# E9. Face validity：theta 依時代與若干已知政體
#-------------------------------------------------------------------------------
plt_gemini %>%
    mutate(era = cut(leader_first_year,
                     c(-3000, 0, 1000, 1500, 1800, 1900, 2030),
                     labels = c("pre-0", "0-1000", "1000-1500",
                                "1500-1800", "1800-1900", "1900+"))) %>%
    group_by(era) %>%
    summarise(n = n(), theta = mean(theta_A, na.rm = TRUE)) %>%
    print(n = Inf)

plt_gemini %>%
    group_by(polity_name) %>%
    filter(n() >= 30) %>%
    summarise(n = n(), theta = mean(theta_A, na.rm = TRUE)) %>%
    arrange(desc(theta)) %>%
    print(n = 20)


mirtCluster(remove = TRUE)

plt_gemini %>% group_by(polity_name) %>% filter(n() >= 30) %>%
    summarise(n = n(), theta = mean(theta_A, na.rm = TRUE)) %>%
    arrange(theta) %>% print(n = 20)

# 以及幾個定錨政體
plt_gemini %>%
    filter(polity_name %in% c("Roman Republic","Athens","Venice","England",
                              "Russia","Prussia","Ottoman Empire","China",
                              "Soviet Union","Germany","Japan","France")) %>%
    group_by(polity_name) %>%
    summarise(n = n(), theta = mean(theta_A, na.rm = TRUE)) %>%
    arrange(desc(theta)) %>% print(n = Inf)


#===============================================================================
# PART 2 — bifactor
#
#   為什麼要做：單維模型的三個獨立證據都指向同一個病根
#     (1) 參數  身分團體 block 的 lambda 被壓到 0.25-0.39
#               (雙因子解裡 checks_aristocracy 其實是 0.79)
#               checks_assembly h2 = 0.970 -> 這一題幾乎「就是」theta
#     (2) Q3    checks_aristocracy x checks_council = 0.499  (第二維度留在殘差)
#               checks_assembly    x elections      = -0.375 (過度解釋的特徵)
#     (3) 效度  Pakistan > Switzerland；Soviet Union > Venice
#
#   bifactor 不是刪題的替代品，而是刪題的「取代方案」：
#   specific factor 把 block 內部共享的變異吸走，general factor 的載荷才乾淨。
#   19 題全部保留。
#===============================================================================

#-------------------------------------------------------------------------------
# P2-0. elections 的結構性 0 -> NA
#   assembly != 2 的 97,480 列 (71.9%) 其 elections = 0 是 not-applicable，
#   不是「選舉分數最低」。pipeline 本來就是 pass-through 0、不呼叫 LLM。
#   缺失與否完全由 assembly_prediction 決定，而 assembly 本身在模型裡且已觀測
#   -> MAR，邊際概似有效，不會有偏誤。
#-------------------------------------------------------------------------------
plt_gemini <- plt_gemini %>%
    mutate(elections_prediction = if_else(assembly_prediction == 2,
                                          elections_prediction,
                                          NA_real_))

cat("\n===== Version A items (elections 修正後) =====\n")
dat_A2  <- make_irt_data(plt_gemini, items_A)
type_A2 <- get_itemtype(dat_A2)

cat("\n===== Version B items (elections 修正後) =====\n")
dat_B2  <- make_irt_data(plt_gemini, items_B)
type_B2 <- get_itemtype(dat_B2)


#-------------------------------------------------------------------------------
# P2-1a. 結構探索 —— 讓資料決定分組，而不是我先指定
#
#   為什麼不用 mirt(dat, 2, rotate="bifactorQ")：
#     (1) full-information EM 要在 61x61 = 3,721 個 quadrature 點上積分，
#         實測 500 cycles 仍未收斂 (simpleWarning: EM cycles terminated)
#     (2) bifactor 旋轉需要 >= 3 個因子 (1 general + >=2 specific)。
#         model=2 得到的是「兩個對等因子」，不是 bifactor
#         (證據：SS loadings 5.059 vs 5.778，份量相當；general 應該遠大於 specific)
#   有序資料的維度診斷，慣例就是在 polychoric 相關矩陣上做 EFA。
#   實測全樣本 135,672 x 18：polychoric 0.4 秒 + fa 0.3 秒。
#
#   elections 必須排除：NA 化之後它只在 assembly==2 上有值，
#   而那裡 assembly 是常數 -> polychoric(assembly, elections) 無定義。
#   它本來就屬於 factor 1（已知的建構假象），分組不是要檢驗的對象。
#-------------------------------------------------------------------------------
suppressPackageStartupMessages(library(psych))

items_efa <- setdiff(items_A, "elections_prediction")
X_efa <- as.data.frame(plt_gemini[, items_efa])

pc  <- psych::polychoric(X_efa)$rho          # 注意 warning：矩陣可能非正定而被 smoothing
fa4 <- psych::fa(pc, nfactors = 4, rotate = "bifactor", fm = "ml",
                 n.obs = nrow(X_efa))

L_efa <- round(unclass(fa4$loadings)[, 1:4], 2)
colnames(L_efa) <- c("g", "s1", "s2", "s3")
print(L_efa)

# 單一分數站不站得住 —— 這兩個數字直接進論文
om <- psych::omega(pc, nfactors = 3, fm = "ml", n.obs = nrow(X_efa), plot = FALSE)
cat(sprintf("omega_h = %.3f   omega_total = %.3f   ECV = %.3f\n",
            om$omega_h, om$omega.tot,
            sum(om$schmid$sl[, "g"]^2) / sum(om$schmid$sl[, 1:4]^2)))
# 參考值：omega_h > 0.70 支持用單一分數；ECV > 0.70 才支持「單維」。
# 實測 omega_h = 0.733、ECV = 0.530 -> 單一分數可用，但資料確實多維 = bifactor 的存在理由。


#-------------------------------------------------------------------------------
# P2-1b. split-half —— 打破循環論證
#
#   從同一批資料推出分組、再在同一批資料上估參數並報 g-loadings 當權重，
#   是 double-dipping（跟先前用 rest 挑 entry/exit 編碼是同一個毛病）。
#   EFA 只要 0.7 秒，所以 split-half 幾乎零成本：
#     A 半決定分組  ->  B 半估最終模型。
#-------------------------------------------------------------------------------
set.seed(20260823)
half <- sample(c(TRUE, FALSE), nrow(plt_gemini), replace = TRUE)

fa_half <- function(idx) {
    psych::fa(psych::polychoric(as.data.frame(plt_gemini[idx, items_efa]))$rho,
              nfactors = 4, rotate = "bifactor", fm = "ml", n.obs = sum(idx))
}
fa_A <- fa_half(half)
fa_B <- fa_half(!half)
LA <- unclass(fa_A$loadings)[, 1:4]
LB <- unclass(fa_B$loadings)[, 1:4]

# (i) 先確認因子有沒有對上（label switching / 因子本身不穩）
#     對角線 |r| 高 = 該因子在兩半都復現；低 = 那個因子是噪音，不要用。
cat("\n>> 因子對應矩陣（列 = A 半，欄 = B 半）\n")
print(round(cor(LA, LB), 2))

# 實測結果（seed 20260823）：
#   factor 1 (g)            r = 0.97   -> 穩定
#   factor 2 (身分團體型)    r = 0.97   -> 穩定  ★ 採用
#   factor 3 (主權)          r = 0.50   -> 不穩  ★ 不採用
#   factor 4                r = -0.89  -> 量級穩、符號相反（因子符號本來就任意）
#
#   -> 只有 g 和身分團體型可以拿來指定 specific factor。
#      sovereign / entry_3 / exit_3 那個「主權因子」在探索式解裡看得到，
#      但過不了 split-half，所以維持 NA（只掛 general），不另立 specific factor。

# (ii) 逐題比對 s1。判準只對「主要載荷題」有意義 ——
#      接近 0 的載荷本來就被噪音主導，用 sign 比對會產生假警報。
split_cmp <- data.frame(
    item  = sub("_prediction", "", items_efa),
    A     = round(LA[, 2], 2),
    B     = round(LB[, 2], 2)
) %>%
    mutate(primary = pmin(abs(A), abs(B)) > 0.30,
           stable  = primary & sign(A) == sign(B) & abs(A - B) < 0.15)

print(split_cmp, row.names = FALSE)
cat("身分團體型的主要載荷題:", sum(split_cmp$primary),
    " 其中兩半穩定:", sum(split_cmp$stable), "\n")

# 實測：aristocracy .77/.81、council .80/.69、clergy .64/.59、symbolism .51/.58、
#       local .46/.44、military .35/.32、federalism .25/.26  —— 七題全部復現。
#       symbolism 的載荷高於 federalism 和 military，歸入這一群有實證支持。


#-------------------------------------------------------------------------------
# P2-1c. specific factor 指派
#
#   來源分兩類，地位不同：
#
#   factor 1 = assembly 複合體   【指定，不是發現】
#       assembly / elections / checks_assembly。
#       這不是關於政治約束的理論主張，而是「這三題在建構上就重複」：
#       assembly x checks_assembly r = 0.77（assembly∈{0,1} -> checks_assembly=0 佔 99.7%）；
#       elections 由 pipeline 定義上被 assembly==2 gated。
#       探索式解看不出來 —— checks_assembly 在 EFA 裡 g = 0.97、h2 = 0.99，
#       綁架問題在任何自動找出來的結構裡都存在。所以必須指定。
#
#   factor 2 = 身分團體型       【資料推出，兩個獨立方法一致】
#       polychoric s1: aristocracy .88 / council .77 / clergy .68 / local .55
#                      symbolism .49 / military .35 / federalism .31
#       mirt full-info F1: aristocracy .93 / council .81 / clergy .75 / local .65
#                      federalism .63 / military .49 / symbolism .36
#       -> symbolism 是新增的（我原本放 NA），兩個方法都說它屬於這一群。
#          只改 specific factor 指派，類別編碼不動（留給老師評估）。
#
#   （原本想立的 factor 3 = 主權，已被 split-half 否決：因子對應 r = 0.50。
#     sovereign / entry_3 / exit_3 改為只掛 general。這是實測結果，不是理論偏好。）
#
#   NA = 制度型：它們「就是」general factor
#       constitution .70 / collegiality .67 / petition .83 /
#       judiciary .86 / bureaucracy .59 / bourgeoisie .73  —— 主要載荷都在 g。
#       原本我硬給了一個 s3，資料裡那個 s3 只有 bureaucracy(.73) 撐著，是空殼。
#       bifactor 的結構是「主導的那一群成為 g，偏離的幾群才拿 specific factor」。
#-------------------------------------------------------------------------------
spec_map <- c(
    # 1 —— 指定（建構假象）
    assembly_prediction            = 1,
    elections_prediction           = 1,
    checks_assembly_prediction     = 1,

    # 2 —— 資料推出（身分團體型）
    checks_aristocracy_prediction  = 2,
    checks_council_prediction      = 2,
    checks_clergy_prediction       = 2,
    checks_local_prediction        = 2,
    checks_military_prediction     = 2,
    federalism_prediction          = 2,
    symbolism_prediction           = 2,

    # NA —— 只掛 general factor
    #   探索式解裡 sovereign(s2 = .97) / entry_3 / exit_3 看似自成一個「主權因子」，
    #   但 split-half 的因子對應只有 r = 0.50 -> 不穩，不採用。
    #   寧可讓它們只掛 general，也不要立一個復現不了的 specific factor。
    sovereign_prediction           = NA,
    entry_3_prediction             = NA,
    exit_3_prediction              = NA,

    #   制度型：它們「就是」general factor
    constitution_prediction        = NA,
    collegiality_prediction        = NA,
    petition_prediction            = NA,
    checks_judiciary_prediction    = NA,
    checks_bureaucracy_prediction  = NA,
    checks_bourgeoisie_prediction  = NA
)

spec_A <- unname(spec_map[items_A])

# 跑之前肉眼確認這張表
data.frame(item = items_A, specific = spec_A, itemtype = unname(type_A2))

stopifnot(length(spec_A) == length(items_A),
          !any(is.na(match(items_A, names(spec_map)))))


#-------------------------------------------------------------------------------
# P2-1d. 模糊指派的敏感性測試
#
#   checks_bourgeoisie 是唯一兩個方法真的不一致的題目：
#       mirt        F1 = 0.634  >  F2 = 0.486   （偏身分團體）
#       polychoric  g  = 0.73,     s1 = 0.23    （偏 general）
#   federalism / checks_military 在 polychoric 上也偏弱（s1 = .31 / .35）。
#   分別換組重估，看 g-loadings 排序與 ECV / omega_h 穩不穩。
#-------------------------------------------------------------------------------
spec_alt1 <- spec_A; spec_alt1[items_A == "checks_bourgeoisie_prediction"] <- 2
spec_alt2 <- spec_A; spec_alt2[items_A %in% c("federalism_prediction",
                                              "checks_military_prediction")] <- NA

#-------------------------------------------------------------------------------
# P2-2. 估計
#   SE = FALSE：19 題 x 135,672 列的 bifactor 已經不便宜，先看點估計。
#   確認結果合理之後再開 SE = TRUE 重跑一次。
#-------------------------------------------------------------------------------
set.seed(20260823)

# 先在「elections 已修正」的資料上重跑一次單維，作為乾淨的對照組。
# 沒有這一步的話，mod_A -> mod_bf 的差異會同時混入
# (a) elections 結構性 0 改 NA 和 (b) bifactor 兩個變動，無法歸因。
mod_A2 <- mirt(dat_A2, model = 1, itemtype = type_A2,
               method = "EM", SE = FALSE, technical = list(NCYCLES = 2000))
saveRDS(mod_A2, "./stats/mod_A2_unidim.rds")

mod_bf <- bfactor(dat_A2, model = spec_A, itemtype = type_A2,
                  SE = FALSE, technical = list(NCYCLES = 4000))

saveRDS(mod_bf, "./stats/mod_bf.rds")

extract.mirt(mod_bf, "converged")
extract.mirt(mod_bf, "iterations")


#-------------------------------------------------------------------------------
# P2-3. 標準化載荷
#   多維標準化：lambda_jk = a_jk / sqrt(1.702^2 + sum_k a_jk^2)
#   (單維的 lambda = a / sqrt(a^2 + 1.702^2) 是它的特例)
#   也印 summary() 交叉核對。
#-------------------------------------------------------------------------------
std_loadings <- function(mod) {
    it   <- coef(mod, simplify = TRUE)$items
    acol <- grep("^a[0-9]+$", colnames(it), value = TRUE)
    A    <- as.matrix(it[, acol, drop = FALSE])
    A[is.na(A)] <- 0
    den  <- sqrt(1.702^2 + rowSums(A^2))
    L    <- A / den
    colnames(L) <- acol
    L
}

L_bf <- std_loadings(mod_bf)
round(L_bf, 3)

summary(mod_bf)   # 交叉核對用


#-------------------------------------------------------------------------------
# P2-4. omega-h 與 ECV  ★ 「能不能用單一分數」的正式證據
#
#   ECV   = 一般因子解釋掉的共同變異佔比。 > 0.70 一般視為支持單一分數。
#   omega_h = 總分變異中歸因於一般因子的比例。 > 0.70 同上。
#   (omega_h 遠低於 omega_total 表示總分其實在測多個東西)
#-------------------------------------------------------------------------------
bifactor_indices <- function(L) {
    lg <- L[, 1]                       # general factor loadings
    ls <- L[, -1, drop = FALSE]        # specific factor loadings
    ls[is.na(ls)] <- 0

    ECV <- sum(lg^2) / (sum(lg^2) + sum(ls^2))

    u <- 1 - lg^2 - rowSums(ls^2)      # uniqueness
    num_g <- sum(lg)^2
    num_s <- sum(colSums(ls)^2)
    omega_total <- (num_g + num_s) / (num_g + num_s + sum(u))
    omega_h     <-  num_g             / (num_g + num_s + sum(u))

    list(ECV = ECV, omega_h = omega_h, omega_total = omega_total,
         omega_h_ratio = omega_h / omega_total)
}

idx <- bifactor_indices(L_bf)
str(idx)


#-------------------------------------------------------------------------------
# P2-5. 單維 vs bifactor：general factor 載荷怎麼變
#   重點看身分團體那六題有沒有從 0.25-0.39 回到接近雙因子解的水準
#-------------------------------------------------------------------------------
# unidim_raw = 原始 mod_A（elections 帶結構性 0）
# unidim_fix = mod_A2（elections 已 NA，仍是單維）
# bifactor_g = 加上 testlet 之後的 general factor
# 比 unidim_fix -> bifactor_g 才是 bifactor 本身的效果
load_cmp <- data.frame(
    item       = items_A,
    specific   = spec_A,
    unidim_raw = summary(mod_A,  verbose = FALSE)$rotF[, 1],
    unidim_fix = summary(mod_A2, verbose = FALSE)$rotF[, 1],
    bifactor_g = L_bf[, 1],
    bifactor_s = apply(L_bf[, -1, drop = FALSE], 1,
                       function(r) { r[is.na(r)] <- 0; r[which.max(abs(r))] })
) %>%
    mutate(across(where(is.numeric), ~round(.x, 3)),
           delta = round(bifactor_g - unidim_fix, 3)) %>%
    arrange(desc(bifactor_g))

print(load_cmp, row.names = FALSE)


#-------------------------------------------------------------------------------
# P2-6. bifactor 的 theta_g，以及與單維 theta 的比較
#-------------------------------------------------------------------------------
fs_bf <- fscores(mod_bf, method = "EAP", full.scores = TRUE, full.scores.SE = TRUE)

plt_gemini <- plt_gemini %>%
    mutate(theta_g    = fs_bf[, 1],
           theta_g_se = fs_bf[, grep("^SE_", colnames(fs_bf))[1]])

cor(plt_gemini$theta_A, plt_gemini$theta_g, use = "complete.obs")
cor(plt_gemini$theta_A, plt_gemini$theta_g, use = "complete.obs", method = "spearman")


#-------------------------------------------------------------------------------
# P2-7. Version B 也跑一次 bifactor
#   九個 checks 已合併，所以只剩 assembly 複合體需要 testlet。
#   bifactor 後再比 theta_A vs theta_B 才有意義 —— 單維那次的 r = 0.983
#   是「theta 兩邊都被 assembly 複合體主導」的結果，不具資訊量。
#-------------------------------------------------------------------------------
# 與 Version A 同構：1 = assembly 複合體（指定）、NA = general。九個 checks 已被加總，所以身分團體 block 在 B 裡「結構上不存在」
# —— 這本身就是 A/B 對照要呈現的東西。
spec_map_B <- c(
    assembly_prediction     = 1,
    elections_prediction    = 1,

    sovereign_prediction    = NA,   # 同 A：主權因子過不了 split-half
    entry_3_prediction      = NA,
    exit_3_prediction       = NA,

    constitution_prediction = NA,
    collegiality_prediction = NA,
    petition_prediction     = NA,
    checks_prediction       = NA,
    federalism_prediction   = NA,
    symbolism_prediction    = NA
)
spec_B <- unname(spec_map_B[items_B])
data.frame(item = items_B, specific = spec_B)

mod_bf_B <- bfactor(dat_B2, model = spec_B, itemtype = type_B2,
                    SE = FALSE, technical = list(NCYCLES = 4000))
saveRDS(mod_bf_B, "./stats/mod_bf_B.rds")

L_bf_B <- std_loadings(mod_bf_B)
round(L_bf_B, 3)
str(bifactor_indices(L_bf_B))

fs_bf_B <- fscores(mod_bf_B, method = "EAP", full.scores = TRUE)
plt_gemini <- plt_gemini %>% mutate(theta_g_B = fs_bf_B[, 1])

# ★ bifactor 之後的 A vs B —— 這一個數字才有意義
cor(plt_gemini$theta_g, plt_gemini$theta_g_B, use = "complete.obs")


#===============================================================================
# PART 3 — 追加診斷
#===============================================================================

#-------------------------------------------------------------------------------
# E8b. 修正 simple_sum
#   原本 rowSums(..., na.rm = TRUE) 把 entry_3 / exit_3 的 NA (15.8% / 17.6%)
#   當成 0，那些列被系統性壓低。改成不補值，只在完整列上比較。
#-------------------------------------------------------------------------------
plt_gemini <- plt_gemini %>%
    mutate(simple_sum_strict = rowSums(across(all_of(items_A)), na.rm = FALSE))

cor(plt_gemini$theta_A,  plt_gemini$simple_sum_strict, use = "complete.obs")
cor(plt_gemini$theta_g,  plt_gemini$simple_sum_strict, use = "complete.obs")
sum(!is.na(plt_gemini$simple_sum_strict))   # 完整列剩多少


#-------------------------------------------------------------------------------
# E10. all-zero 診斷  ★ theta 低端有多少是「不知道」而不是「確定沒有」
#
#   bottom-20 政體有 60-95% 的列是 15 個制度指標全 0，
#   定錨政體 (Roman Republic / Athens / Venice / Soviet Union...) 只有 0-4%。
#   IRT 分不出「真的是 0」和「不知道，預設填 0」——兩者都是完美 Guttman 低 theta。
#-------------------------------------------------------------------------------
inst_items <- c("constitution_prediction", "federalism_prediction",
                "collegiality_prediction", "petition_prediction",
                "assembly_prediction",
                "checks_local_prediction", "checks_military_prediction",
                "checks_judiciary_prediction", "checks_council_prediction",
                "checks_bureaucracy_prediction", "checks_aristocracy_prediction",
                "checks_assembly_prediction", "checks_bourgeoisie_prediction",
                "checks_clergy_prediction")

conf_cols <- names(plt_gemini)[grepl("_confidence$", names(plt_gemini))]
conf_cols <- setdiff(conf_cols, "elections_confidence")   # 71.9% NA，會扭曲平均

plt_gemini <- plt_gemini %>%
    mutate(all_zero  = rowSums(across(all_of(inst_items)), na.rm = TRUE) == 0,
           conf_mean = rowMeans(across(all_of(conf_cols)), na.rm = TRUE))

polity_diag <- plt_gemini %>%
    group_by(polity_name) %>%
    filter(n() >= 30) %>%
    summarise(n         = n(),
              theta_g   = mean(theta_g,   na.rm = TRUE),
              all_zero  = mean(all_zero,  na.rm = TRUE),
              conf      = mean(conf_mean, na.rm = TRUE),
              med_year  = median(leader_first_year, na.rm = TRUE),
              .groups = "drop")

# 全 0 比例與 theta 的關係：若強負相關，theta 低端部分在測「資訊不足」
cor(polity_diag$theta_g, polity_diag$all_zero)
cor(polity_diag$conf,    polity_diag$all_zero)

polity_diag %>% arrange(desc(all_zero)) %>% print(n = 20)

ggplot(polity_diag, aes(x = all_zero, y = theta_g)) +
    geom_point(aes(size = n), alpha = 0.25) +
    geom_smooth(method = "loess", se = FALSE) +
    labs(x = "該政體『15 個制度指標全 0』的列比例",
         y = expression(theta[g]),
         title = "theta 的低端有多少是資訊不足造成的？") +
    theme_minimal()

# 敏感性分析：排除全 0 比例 > 0.5 的政體，重看載荷是否穩定
keep_polities <- polity_diag %>% filter(all_zero <= 0.5) %>% pull(polity_name)
cat("排除後保留政體數:", length(keep_polities), "/", nrow(polity_diag), "\n")

# 注意：make_irt_data() 會依「子樣本裡實際出現的 levels」重新編碼 0..K-1。
# 若某個類別在子樣本中消失，該題的門檻數會變，item parameters 不能跟全樣本逐一對照；
# 這裡看的是「載荷的相對排序與 ECV / omega_h 穩不穩」，不是逐個參數比大小。
dat_sens  <- make_irt_data(filter(plt_gemini, polity_name %in% keep_polities), items_A)
mod_sens  <- bfactor(dat_sens, model = spec_A, itemtype = get_itemtype(dat_sens),
                     SE = FALSE, technical = list(NCYCLES = 4000))
round(std_loadings(mod_sens), 3)
str(bifactor_indices(std_loadings(mod_sens)))


#-------------------------------------------------------------------------------
# E11. 定錨政體效度表  ★ bifactor 前後對照
#
#   單維模型的失敗：
#     Soviet Union 0.771 > Venice 0.625     <- 應該要反過來
#     Pakistan     1.58  > Switzerland 1.58 <- Pakistan 排第 9
#   bifactor 之後，威尼斯 / 鄂圖曼的約束 (身分團體 block) 權重應該回升，
#   蘇聯 / 巴基斯坦的形式制度優勢應該被 assembly testlet 吸走一部分。
#-------------------------------------------------------------------------------
anchors <- c("Roman Republic", "Athens", "Venice", "England", "Poland-Lithuania",
             "Switzerland", "Zurich", "Germany", "France", "Prussia",
             "Soviet Union", "Pakistan", "Russia", "Japan",
             "Ottoman Empire", "China")

anchor_tbl <- plt_gemini %>%
    filter(polity_name %in% anchors) %>%
    group_by(polity_name) %>%
    summarise(n        = n(),
              unidim   = mean(theta_A,  na.rm = TRUE),
              bifactor = mean(theta_g,  na.rm = TRUE),
              all_zero = mean(all_zero, na.rm = TRUE),
              .groups = "drop") %>%
    mutate(rank_uni = rank(-unidim),
           rank_bf  = rank(-bifactor),
           moved    = rank_uni - rank_bf) %>%
    arrange(desc(bifactor))

print(anchor_tbl, n = Inf)

# 兩個關鍵的成對比較 —— bifactor 應該把順序修正過來
anchor_tbl %>% filter(polity_name %in% c("Venice", "Soviet Union"))
anchor_tbl %>% filter(polity_name %in% c("Switzerland", "Pakistan"))

# bifactor 之後的 top / bottom 20
polity_diag %>% arrange(desc(theta_g)) %>% print(n = 20)
polity_diag %>% arrange(theta_g)       %>% print(n = 20)


#-------------------------------------------------------------------------------
# E12. polity_name 編碼修復
#   1,467 列、36 個政體有 mojibake (Graubã¼Nden / Sã£O Tomã© And Prã­Ncipe ...)
#   發表前要修。專案裡另有 utils/encoding_fix.py。
#-------------------------------------------------------------------------------
bad_encoding <- plt_gemini %>%
    filter(grepl("Ã|Â|ã|Å|â€", polity_name)) %>%
    distinct(polity_name)
print(bad_encoding, n = Inf)

# UTF-8 被當成 Latin-1 讀取的典型修法（修完務必肉眼檢查）
# plt_gemini <- plt_gemini %>%
#     mutate(polity_name = iconv(polity_name, from = "UTF-8", to = "latin1") %>%
#                          iconv(from = "UTF-8", to = "UTF-8"))
