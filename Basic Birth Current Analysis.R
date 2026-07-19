# =============================================================================
# DOES PERSONALITY BETTER REFLECT BIRTH OR CURRENT LOCATION?
# =============================================================================
# Analysis only: this script creates tables/CSV files and no figures.
#
# Analysis 1 (all participants):
#   Personality -> birthResidenceType versus
#   Personality -> currentResidenceType2
#
# Analysis 2 (movers only):
#   The same paired comparison among people whose two locations differ.
#
# Analysis 3 (stayer distinctiveness):
#   a) all stayers versus all movers;
#   b) within each birthplace, stayed versus moved away;
#   c) within each current location, stayer versus moved in.
#
# In every outer fold, personality scores are first imputed and then residualized
# for age and gender using the OUTER TRAINING DATA ONLY. The fitted adjustment is
# applied to the outer test data, preventing information leakage.
#
# cv.glmnet selects lambda internally within each outer training set. Alpha is
# fixed at 0.5 to keep this basic analysis computationally manageable.

library(tidyverse)
library(glmnet)

set.seed(42)

# ---- User settings ----------------------------------------------------------

PROJECT_DIR <- "/data/scripts/Ling/Proj5BirthCurrentPrediction/"
OUTER_FOLDS <- 5
OUTER_REPEATS <- 5       # use 10 for the final analysis if runtime permits
INNER_FOLDS <- 5
ELASTIC_NET_ALPHA <- 0.5 # 0 = ridge, 1 = lasso
RUN_ITEM_MODELS <- TRUE  # FALSE gives a much faster domain-only run

setwd(PROJECT_DIR)

# ---- Data -------------------------------------------------------------------

data <- read.csv("NP_resi.csv") %>%
  rename_with(~ sub("\\.x$", "", .x)) %>%
  select(
    -neuroticism23R, -neuroticism24, -neuroticism31, -neuroticism32R,
    -neuroticism41, -neuroticism42, -neuroticism43R, -neuroticism52,
    -others12, -others14
  )

# Predictor positions follow the original data layout used in the project.
domain_predictors <- names(data)[4:8]
item_predictors <- names(data)[30:217]
location_levels <- c("Village", "Town", "City", "Abroad")

required_columns <- c(
  "birthResidenceType", "currentResidenceType2", "ageAtAgreement", "gender"
)
missing_columns <- setdiff(required_columns, names(data))
if (length(missing_columns)) {
  stop("Missing required columns: ", paste(missing_columns, collapse = ", "))
}

normalize_location <- function(x) {
  x <- str_to_title(str_trim(as.character(x)))
  # Rename the source-data value once, then use "Abroad" throughout.
  recode(x, "Overseas" = "Abroad")
}

data <- data %>%
  mutate(
    birth_location = normalize_location(birthResidenceType),
    current_location = normalize_location(currentResidenceType2),
    gender = factor(gender)
  ) %>%
  filter(
    birth_location %in% location_levels,
    current_location %in% location_levels
  ) %>%
  drop_na(ageAtAgreement, gender) %>%
  mutate(
    birth_location = factor(birth_location, levels = location_levels),
    current_location = factor(current_location, levels = location_levels),
    mobility = factor(
      if_else(birth_location == current_location, "Stayer", "Mover"),
      levels = c("Mover", "Stayer")
    )
  )

if (nrow(data) == 0) {
  stop("No valid rows remain; inspect the values in the two residence columns")
}

# Missing-data policy:
# - age, gender, birth/current residence, and Big Five domains must be complete;
# - missing personality-item responses are median-imputed separately inside each
#   outer training fold. Test-fold values never contribute to an imputation.
complete_required <- c(
  "ageAtAgreement", "gender", "birth_location", "current_location",
  domain_predictors
)
required_missing <- colSums(is.na(data[, complete_required, drop = FALSE]))
if (any(required_missing > 0)) {
  stop(
    "Unexpected missing values in variables expected to be complete: ",
    paste(names(required_missing)[required_missing > 0], collapse = ", ")
  )
}

item_missingness <- tibble(
  Item = item_predictors,
  Missing_N = colSums(is.na(data[, item_predictors, drop = FALSE])),
  Missing_Percent = 100 * Missing_N / nrow(data)
) %>%
  arrange(desc(Missing_N), Item)

message("Missing personality-item responses: ", sum(item_missingness$Missing_N))
message("Items containing at least one missing response: ",
        sum(item_missingness$Missing_N > 0))
write.csv(item_missingness, "basic_item_missingness.csv", row.names = FALSE)

cohort_summary <- list(
  sample = tibble(
    Full_N = nrow(data),
    Movers = sum(data$mobility == "Mover"),
    Stayers = sum(data$mobility == "Stayer")
  ),
  transition_table = as.data.frame.matrix(
    table(data$birth_location, data$current_location)
  ) %>% rownames_to_column("Birth")
)

print(cohort_summary$sample)
print(cohort_summary$transition_table)
write.csv(cohort_summary$sample, "basic_cohort_summary.csv", row.names = FALSE)
write.csv(cohort_summary$transition_table, "basic_transition_table.csv", row.names = FALSE)

# ---- Resampling -------------------------------------------------------------

# Joint stratification makes birth and current proportions similar in every
# paired test fold. Random offsets avoid putting all tiny strata in fold 1.
make_joint_folds <- function(birth, current, k, repeats, seed = 42,
                             row_ids = seq_along(birth), max_attempts = 1000,
                             min_train_per_class = INNER_FOLDS) {
  strata <- interaction(birth, current, drop = TRUE)
  birth <- droplevels(factor(birth))
  current <- droplevels(factor(current))

  if (any(table(birth) < 2) || any(table(current) < 2)) {
    stop(
      "Every birth and current class needs at least two observations for outer CV. ",
      "Birth counts: ", paste(names(table(birth)), table(birth), collapse = ", "),
      "; current counts: ",
      paste(names(table(current)), table(current), collapse = ", ")
    )
  }

  output <- vector("list", k * repeats)
  position <- 1L

  for (repeat_id in seq_len(repeats)) {
    valid_assignment <- FALSE

    for (attempt in seq_len(max_attempts)) {
      set.seed(seed + repeat_id * 10000 + attempt)
      fold_id <- integer(length(strata))

      for (stratum in levels(strata)) {
        rows <- sample(which(strata == stratum))
        offset <- sample.int(k, 1) - 1L
        fold_id[rows] <- ((seq_along(rows) - 1L + offset) %% k) + 1L
      }

      # Each outer training set must contain every class occurring in this
      # analysis sample. Rare transition strata are allowed, but the marginal
      # birth/current classes cannot disappear from training.
      valid_assignment <- all(vapply(seq_len(k), function(fold) {
        train <- fold_id != fold
        all(table(birth[train]) >= min_train_per_class) &&
          all(table(current[train]) >= min_train_per_class)
      }, logical(1)))

      if (valid_assignment) break
    }

    if (!valid_assignment) {
      stop("Could not construct valid stratified folds after ", max_attempts,
           " attempts; reduce OUTER_FOLDS or combine very rare classes")
    }

    for (fold in seq_len(k)) {
      output[[position]] <- list(
        repeat_id = repeat_id,
        fold = fold,
        train = row_ids[fold_id != fold],
        test = row_ids[fold_id == fold]
      )
      position <- position + 1L
    }
  }

  output
}

paired_folds <- make_joint_folds(
  data$birth_location,
  data$current_location,
  OUTER_FOLDS,
  OUTER_REPEATS
)

# Movers need their own stratified folds. Subsetting the full-sample folds after
# their creation can accidentally remove an entire location class from a mover
# training fold. These folds remain perfectly paired between the mover birth
# and current models.
mover_rows <- which(data$mobility == "Mover")
mover_folds <- make_joint_folds(
  data$birth_location[mover_rows],
  data$current_location[mover_rows],
  OUTER_FOLDS,
  OUTER_REPEATS,
  seed = 142,
  row_ids = mover_rows
)

# ---- Fold-specific preprocessing -------------------------------------------

residualize_in_outer_fold <- function(df, train_rows, test_rows, predictors) {
  train_x <- as.matrix(df[train_rows, predictors, drop = FALSE])
  test_x <- as.matrix(df[test_rows, predictors, drop = FALSE])
  storage.mode(train_x) <- "double"
  storage.mode(test_x) <- "double"

  # Median imputation is robust for the ordinal/Likert personality items. Every
  # median is estimated only from the current outer training fold. The domains
  # are complete, so this same code leaves them unchanged.
  medians <- apply(train_x, 2, median, na.rm = TRUE)
  if (any(!is.finite(medians))) {
    stop("A predictor is entirely missing in an outer training fold")
  }
  for (column in seq_len(ncol(train_x))) {
    train_x[is.na(train_x[, column]), column] <- medians[column]
    test_x[is.na(test_x[, column]), column] <- medians[column]
  }

  covariate_train <- model.matrix(
    ~ ageAtAgreement + gender,
    data = df[train_rows, , drop = FALSE]
  )
  covariate_test <- model.matrix(
    ~ ageAtAgreement + gender,
    data = df[test_rows, , drop = FALSE]
  )

  # One multivariate least-squares fit residualizes every personality score.
  adjustment <- lm.fit(x = covariate_train, y = train_x)$coefficients
  adjustment[!is.finite(adjustment)] <- 0

  train_residuals <- train_x - covariate_train %*% adjustment
  test_residuals <- test_x - covariate_test %*% adjustment

  # Standardize using outer-training residuals so extracted coefficients describe
  # a one-SD increase in an age/gender-residualized personality predictor.
  training_means <- colMeans(train_residuals)
  training_sds <- apply(train_residuals, 2, sd)
  if (any(!is.finite(training_sds) | training_sds == 0)) {
    stop("A residualized predictor has zero or undefined training-fold variance")
  }

  list(
    train = sweep(
      sweep(train_residuals, 2, training_means, "-"),
      2, training_sds, "/"
    ),
    test = sweep(
      sweep(test_residuals, 2, training_means, "-"),
      2, training_sds, "/"
    )
  )
}

# ---- Metrics ---------------------------------------------------------------

multiclass_metrics <- function(actual, probabilities, baseline) {
  actual <- factor(actual, levels = location_levels)
  observed <- model.matrix(~ actual - 1)
  colnames(observed) <- location_levels
  probabilities <- probabilities[, location_levels, drop = FALSE]

  baseline_matrix <- matrix(
    rep(as.numeric(baseline), each = nrow(observed)),
    nrow = nrow(observed),
    dimnames = list(NULL, location_levels)
  )
  model_brier <- mean(rowSums((probabilities - observed)^2))
  baseline_brier <- mean(rowSums((baseline_matrix - observed)^2))
  actual_index <- match(as.character(actual), location_levels)
  predicted <- location_levels[max.col(probabilities, ties.method = "first")]
  recall <- vapply(location_levels, function(class_name) {
    rows <- actual == class_name
    if (!any(rows)) return(NA_real_)
    mean(predicted[rows] == class_name)
  }, numeric(1))

  c(
    BSS = 1 - model_brier / baseline_brier,
    Brier = model_brier,
    LogLoss = -mean(log(pmax(
      probabilities[cbind(seq_along(actual), actual_index)], 1e-15
    ))),
    Accuracy = mean(predicted == as.character(actual)),
    BalancedAccuracy = mean(recall, na.rm = TRUE)
  )
}

# Decompose the joint multinomial predictions into one-versus-rest BSS values.
# No separate binary models are fitted: each score uses the probability produced
# by the four-class model and the matching outer-training class prevalence.
settlement_bss <- function(actual, probabilities, baseline) {
  actual <- factor(actual, levels = location_levels)
  probabilities <- probabilities[, location_levels, drop = FALSE]

  bind_rows(lapply(location_levels, function(settlement) {
    observed <- as.numeric(actual == settlement)
    model_brier <- mean((probabilities[, settlement] - observed)^2)
    baseline_probability <- as.numeric(baseline[settlement])
    baseline_brier <- mean((baseline_probability - observed)^2)

    tibble(
      Settlement = settlement,
      BSS = 1 - model_brier / baseline_brier,
      Brier = model_brier,
      Baseline_Brier = baseline_brier,
      Training_Prevalence = baseline_probability,
      Test_Prevalence = mean(observed)
    )
  }))
}

binary_metrics <- function(actual, probability, baseline) {
  actual <- as.numeric(actual == "Stayer")
  predicted <- if_else(probability >= 0.5, 1, 0)
  model_brier <- mean((probability - actual)^2)
  baseline_brier <- mean((baseline - actual)^2)
  sensitivity <- if (any(actual == 1)) mean(predicted[actual == 1] == 1) else NA_real_
  specificity <- if (any(actual == 0)) mean(predicted[actual == 0] == 0) else NA_real_

  c(
    BSS = 1 - model_brier / baseline_brier,
    Brier = model_brier,
    LogLoss = -mean(actual * log(pmax(probability, 1e-15)) +
      (1 - actual) * log(pmax(1 - probability, 1e-15))),
    Accuracy = mean(predicted == actual),
    BalancedAccuracy = mean(c(sensitivity, specificity), na.rm = TRUE)
  )
}

# ---- Model fitting ----------------------------------------------------------

extract_multinomial_coefficients <- function(fit) {
  coefficient_list <- coef(fit, s = "lambda.1se")
  bind_rows(lapply(names(coefficient_list), function(class_name) {
    coefficient_matrix <- as.matrix(coefficient_list[[class_name]])
    tibble(
      Predictor = rownames(coefficient_matrix),
      Class = class_name,
      Coefficient = as.numeric(coefficient_matrix[, 1])
    ) %>%
      filter(Predictor != "(Intercept)")
  }))
}

extract_binary_coefficients <- function(fit) {
  coefficient_matrix <- as.matrix(coef(fit, s = "lambda.1se"))
  tibble(
    Predictor = rownames(coefficient_matrix),
    Class = "Stayer vs Mover",
    Coefficient = as.numeric(coefficient_matrix[, 1])
  ) %>%
    filter(Predictor != "(Intercept)")
}

make_inner_foldid <- function(y, k, seed) {
  y <- droplevels(factor(y))
  if (any(table(y) < k)) {
    stop(
      "An outer training class has fewer observations than INNER_FOLDS. Counts: ",
      paste(names(table(y)), table(y), collapse = ", ")
    )
  }

  set.seed(seed)
  foldid <- integer(length(y))
  for (class_name in levels(y)) {
    rows <- sample(which(y == class_name))
    offset <- sample.int(k, 1) - 1L
    foldid[rows] <- ((seq_along(rows) - 1L + offset) %% k) + 1L
  }
  foldid
}

fit_multinomial <- function(train_x, test_x, train_y, test_y, seed) {
  train_y <- factor(train_y, levels = location_levels)
  test_y <- factor(test_y, levels = location_levels)
  if (any(table(train_y) == 0)) stop("An outer training fold is missing a location class")

  baseline <- prop.table(table(train_y))
  inner_foldid <- make_inner_foldid(train_y, INNER_FOLDS, seed)
  set.seed(seed)
  fit <- cv.glmnet(
    train_x, train_y,
    family = "multinomial",
    type.multinomial = "grouped",
    alpha = ELASTIC_NET_ALPHA,
    foldid = inner_foldid,
    type.measure = "deviance",
    standardize = FALSE
  )

  probability <- predict(
    fit, newx = test_x, s = "lambda.1se", type = "response"
  )[, , 1]

  list(
    metrics = multiclass_metrics(test_y, probability, baseline),
    settlement_bss = settlement_bss(test_y, probability, baseline),
    lambda = fit$lambda.1se,
    coefficients = extract_multinomial_coefficients(fit)
  )
}

fit_binary <- function(train_x, test_x, train_y, test_y, seed) {
  train_y <- factor(train_y, levels = c("Mover", "Stayer"))
  test_y <- factor(test_y, levels = c("Mover", "Stayer"))
  if (any(table(train_y) == 0)) stop("An outer training fold is missing a mobility class")

  baseline <- mean(train_y == "Stayer")
  inner_foldid <- make_inner_foldid(train_y, INNER_FOLDS, seed)
  set.seed(seed)
  fit <- cv.glmnet(
    train_x, train_y,
    family = "binomial",
    alpha = ELASTIC_NET_ALPHA,
    foldid = inner_foldid,
    type.measure = "deviance",
    standardize = FALSE
  )
  probability <- as.numeric(predict(
    fit, newx = test_x, s = "lambda.1se", type = "response"
  ))

  list(
    metrics = binary_metrics(test_y, probability, baseline),
    lambda = fit$lambda.1se,
    coefficients = extract_binary_coefficients(fit)
  )
}

# ---- Birth versus current comparison ---------------------------------------

run_birth_current <- function(df, predictors, model_name, movers_only = FALSE) {
  folds_to_use <- if (movers_only) mover_folds else paired_folds
  results <- vector("list", length(folds_to_use))
  coefficient_results <- vector("list", length(folds_to_use))
  settlement_results <- vector("list", length(folds_to_use))

  for (i in seq_along(folds_to_use)) {
    split <- folds_to_use[[i]]
    train_rows <- split$train
    test_rows <- split$test

    x <- residualize_in_outer_fold(df, train_rows, test_rows, predictors)
    birth <- fit_multinomial(
      x$train, x$test, df$birth_location[train_rows],
      df$birth_location[test_rows], seed = 1000 + i
    )
    current <- fit_multinomial(
      x$train, x$test, df$current_location[train_rows],
      df$current_location[test_rows], seed = 2000 + i
    )

    results[[i]] <- bind_rows(
      as.data.frame(as.list(birth$metrics)) %>%
        mutate(Outcome = "Birth", Lambda = birth$lambda),
      as.data.frame(as.list(current$metrics)) %>%
        mutate(Outcome = "Current", Lambda = current$lambda)
    ) %>%
      mutate(
        Model = model_name,
        Sample = if_else(movers_only, "Movers only", "All participants"),
        Repeat = split$repeat_id,
        Fold = split$fold,
        TrainN = length(train_rows),
        TestN = length(test_rows)
      )

    coefficient_results[[i]] <- bind_rows(
      birth$coefficients %>% mutate(Outcome = "Birth"),
      current$coefficients %>% mutate(Outcome = "Current")
    ) %>%
      mutate(
        Model = model_name,
        Sample = if_else(movers_only, "Movers only", "All participants"),
        Repeat = split$repeat_id,
        Fold = split$fold
      )

    settlement_results[[i]] <- bind_rows(
      birth$settlement_bss %>% mutate(Outcome = "Birth"),
      current$settlement_bss %>% mutate(Outcome = "Current")
    ) %>%
      mutate(
        Model = model_name,
        Sample = if_else(movers_only, "Movers only", "All participants"),
        Repeat = split$repeat_id,
        Fold = split$fold,
        TestN = length(test_rows)
      )

    message(model_name, " [", ifelse(movers_only, "movers", "all"),
            "]: outer split ", i, "/", length(folds_to_use))
  }

  list(
    metrics = bind_rows(results),
    coefficients = bind_rows(coefficient_results),
    settlement_bss = bind_rows(settlement_results)
  )
}

# ---- Stayer distinctiveness -------------------------------------------------

run_stayer_analysis <- function(df, predictors, model_name, comparison,
                                location = NA_character_) {
  eligible <- switch(
    comparison,
    Overall = rep(TRUE, nrow(df)),
    `Stayed vs moved away` = df$birth_location == location,
    `Stayer vs moved in` = df$current_location == location,
    stop("Unknown stayer comparison")
  )
  results <- vector("list", length(paired_folds))
  coefficient_results <- vector("list", length(paired_folds))

  for (i in seq_along(paired_folds)) {
    split <- paired_folds[[i]]
    train_rows <- split$train[eligible[split$train]]
    test_rows <- split$test[eligible[split$test]]
    train_y <- df$mobility[train_rows]
    test_y <- df$mobility[test_rows]

    # Very small location-specific folds cannot support a valid binary model.
    train_class_counts <- table(factor(train_y, levels = c("Mover", "Stayer")))
    if (any(train_class_counts < INNER_FOLDS) || length(test_rows) == 0) {
      message("Skipping small stayer split: ", comparison, " / ",
              ifelse(is.na(location), "All", location), " / split ", i)
      results[[i]] <- NULL
      next
    }

    x <- residualize_in_outer_fold(df, train_rows, test_rows, predictors)
    fit <- fit_binary(x$train, x$test, train_y, test_y, seed = 3000 + i)
    results[[i]] <- as.data.frame(as.list(fit$metrics)) %>%
      mutate(
        Model = model_name,
        Comparison = comparison,
        Location = if_else(is.na(location), "All", location),
        Repeat = split$repeat_id,
        Fold = split$fold,
        TrainN = length(train_rows),
        TestN = length(test_rows),
        Lambda = fit$lambda
      )

    coefficient_results[[i]] <- fit$coefficients %>%
      mutate(
        Model = model_name,
        Comparison = comparison,
        Location = if_else(is.na(location), "All", location),
        Repeat = split$repeat_id,
        Fold = split$fold
      )
  }

  list(
    metrics = bind_rows(results),
    coefficients = bind_rows(coefficient_results)
  )
}

run_all_stayer_analyses <- function(df, predictors, model_name) {
  output <- list(
    run_stayer_analysis(df, predictors, model_name, "Overall")
  )
  for (location in location_levels) {
    output <- append(output, list(
      run_stayer_analysis(
        df, predictors, model_name, "Stayed vs moved away", location
      ),
      run_stayer_analysis(
        df, predictors, model_name, "Stayer vs moved in", location
      )
    ))
  }
  list(
    metrics = bind_rows(lapply(output, `[[`, "metrics")),
    coefficients = bind_rows(lapply(output, `[[`, "coefficients"))
  )
}

# ---- Corrected resampling inference ----------------------------------------

corrected_birth_current_test <- function(results, k) {
  results %>%
    select(Model, Sample, Repeat, Fold, Outcome, BSS) %>%
    pivot_wider(names_from = Outcome, values_from = BSS) %>%
    mutate(Difference = Current - Birth) %>%
    group_by(Model, Sample) %>%
    summarize(
      Birth_BSS = mean(Birth),
      Current_BSS = mean(Current),
      Current_minus_Birth = mean(Difference),
      Corrected_SE = sqrt((1 / n() + 1 / (k - 1)) * var(Difference)),
      df = n() - 1,
      t = Current_minus_Birth / Corrected_SE,
      p = 2 * pt(abs(t), df, lower.tail = FALSE),
      CI_Lower = Current_minus_Birth - qt(0.975, df) * Corrected_SE,
      CI_Upper = Current_minus_Birth + qt(0.975, df) * Corrected_SE,
      .groups = "drop"
    )
}

corrected_settlement_tests <- function(results, k) {
  tests <- results %>%
    select(Model, Sample, Settlement, Repeat, Fold, Outcome, BSS) %>%
    pivot_wider(names_from = Outcome, values_from = BSS) %>%
    mutate(Difference = Current - Birth) %>%
    group_by(Model, Sample, Settlement) %>%
    summarize(
      Birth_BSS = mean(Birth),
      Current_BSS = mean(Current),
      Current_minus_Birth = mean(Difference),
      Corrected_SE = sqrt((1 / n() + 1 / (k - 1)) * var(Difference)),
      df = n() - 1,
      t = Current_minus_Birth / Corrected_SE,
      p = 2 * pt(abs(t), df, lower.tail = FALSE),
      CI_Lower = Current_minus_Birth - qt(0.975, df) * Corrected_SE,
      CI_Upper = Current_minus_Birth + qt(0.975, df) * Corrected_SE,
      Resampling_Splits = n(),
      .groups = "drop"
    )

  # Four settlement tests form one multiplicity family within each predictor
  # model and analysis sample.
  tests %>%
    group_by(Model, Sample) %>%
    mutate(p_Holm = p.adjust(p, method = "holm")) %>%
    ungroup()
}

corrected_stayer_summary <- function(results, k) {
  results %>%
    group_by(Model, Comparison, Location) %>%
    summarize(
      Mean_BSS = mean(BSS),
      Mean_Balanced_Accuracy = mean(BalancedAccuracy),
      Corrected_SE = sqrt((1 / n() + 1 / (k - 1)) * var(BSS)),
      df = n() - 1,
      t = Mean_BSS / Corrected_SE,
      p = 2 * pt(abs(t), df, lower.tail = FALSE),
      CI_Lower = Mean_BSS - qt(0.975, df) * Corrected_SE,
      CI_Upper = Mean_BSS + qt(0.975, df) * Corrected_SE,
      Resampling_Splits = n(),
      .groups = "drop"
    )
}

# ---- Run and save analysis --------------------------------------------------

predictor_sets <- list("Big Five domains" = domain_predictors)
if (RUN_ITEM_MODELS) {
  predictor_sets[["Personality items"]] <- item_predictors
}

birth_current_runs <- unlist(lapply(names(predictor_sets), function(name) {
  predictors <- predictor_sets[[name]]
  list(
    run_birth_current(data, predictors, name, movers_only = FALSE),
    run_birth_current(data, predictors, name, movers_only = TRUE)
  )
}), recursive = FALSE)

stayer_runs <- lapply(names(predictor_sets), function(name) {
  run_all_stayer_analyses(data, predictor_sets[[name]], name)
})

birth_current_results <- bind_rows(lapply(birth_current_runs, `[[`, "metrics"))
birth_current_coefficients <- bind_rows(
  lapply(birth_current_runs, `[[`, "coefficients")
)
settlement_bss_results <- bind_rows(
  lapply(birth_current_runs, `[[`, "settlement_bss")
)
stayer_results <- bind_rows(lapply(stayer_runs, `[[`, "metrics"))
stayer_coefficients <- bind_rows(lapply(stayer_runs, `[[`, "coefficients"))

birth_current_summary <- birth_current_results %>%
  group_by(Model, Sample, Outcome) %>%
  summarize(
    across(c(BSS, Brier, LogLoss, Accuracy, BalancedAccuracy), mean),
    .groups = "drop"
  )

birth_current_test <- corrected_birth_current_test(
  birth_current_results, OUTER_FOLDS
)
settlement_bss_summary <- settlement_bss_results %>%
  group_by(Model, Sample, Outcome, Settlement) %>%
  summarize(
    BSS = mean(BSS),
    Brier = mean(Brier),
    Baseline_Brier = mean(Baseline_Brier),
    Training_Prevalence = mean(Training_Prevalence),
    Test_Prevalence = mean(Test_Prevalence),
    .groups = "drop"
  )
settlement_bss_tests <- corrected_settlement_tests(
  settlement_bss_results, OUTER_FOLDS
)
stayer_summary <- corrected_stayer_summary(stayer_results, OUTER_FOLDS)

summarize_coefficients <- function(coefficients, grouping_variables) {
  coefficients %>%
    group_by(across(all_of(c(grouping_variables, "Predictor", "Class")))) %>%
    summarize(
      Median_Coefficient = median(Coefficient),
      Mean_Coefficient = mean(Coefficient),
      Coefficient_SD = sd(Coefficient),
      Selection_Frequency = mean(abs(Coefficient) > 1e-12),
      Positive_Frequency = if_else(
        any(abs(Coefficient) > 1e-12),
        mean(Coefficient[abs(Coefficient) > 1e-12] > 0),
        NA_real_
      ),
      Sign_Consistency = if_else(
        any(abs(Coefficient) > 1e-12),
        max(
          mean(Coefficient[abs(Coefficient) > 1e-12] > 0),
          mean(Coefficient[abs(Coefficient) > 1e-12] < 0)
        ),
        NA_real_
      ),
      Resampling_Splits = n(),
      .groups = "drop"
    )
}

birth_current_coefficient_summary <- summarize_coefficients(
  birth_current_coefficients,
  c("Model", "Sample", "Outcome")
)
stayer_coefficient_summary <- summarize_coefficients(
  stayer_coefficients,
  c("Model", "Comparison", "Location")
)

print(birth_current_summary)
print(birth_current_test)
print(settlement_bss_summary)
print(settlement_bss_tests)
print(stayer_summary)
print(birth_current_coefficient_summary)
print(stayer_coefficient_summary)

write.csv(
  birth_current_results, "basic_birth_current_fold_results.csv", row.names = FALSE
)
write.csv(
  birth_current_summary, "basic_birth_current_summary.csv", row.names = FALSE
)
write.csv(
  birth_current_test, "basic_birth_current_corrected_test.csv", row.names = FALSE
)
write.csv(
  settlement_bss_results,
  "basic_settlement_bss_fold_results.csv",
  row.names = FALSE
)
write.csv(
  settlement_bss_summary,
  "basic_settlement_bss_summary.csv",
  row.names = FALSE
)
write.csv(
  settlement_bss_tests,
  "basic_settlement_bss_corrected_tests.csv",
  row.names = FALSE
)
write.csv(
  stayer_results, "basic_stayer_fold_results.csv", row.names = FALSE
)
write.csv(
  stayer_summary, "basic_stayer_summary.csv", row.names = FALSE
)
write.csv(
  birth_current_coefficients,
  "basic_birth_current_fold_coefficients.csv",
  row.names = FALSE
)
write.csv(
  birth_current_coefficient_summary,
  "basic_birth_current_coefficient_summary.csv",
  row.names = FALSE
)
write.csv(
  stayer_coefficients,
  "basic_stayer_fold_coefficients.csv",
  row.names = FALSE
)
write.csv(
  stayer_coefficient_summary,
  "basic_stayer_coefficient_summary.csv",
  row.names = FALSE
)

message("Analysis complete. Performance and coefficient CSV files were saved; no plots were created.")
