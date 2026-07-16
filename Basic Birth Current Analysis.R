# =============================================================================
# BASIC ANALYSIS: DOES PERSONALITY BETTER PREDICT BIRTH OR CURRENT LOCATION?
# =============================================================================
#
# This script deliberately uses the SAME four outcome categories, participants,
# predictors, and outer resampling splits for birth and current location.
# Therefore, Current - Birth performance is a meaningful paired comparison.
#
# Primary analysis: all participants.
# Secondary analysis: movers only (recommended for interpreting the contrast),
# because birth and current locations are identical by definition for stayers.
#
# Models: multinomial elastic net (alpha fixed at 0.5). cv.glmnet selects lambda
# inside each outer training set, so the outer test fold remains untouched.

library(tidyverse)
library(glmnet)

set.seed(42)

# ---- User settings ----------------------------------------------------------

PROJECT_DIR <- "/data/scripts/Ling/Proj5BirthCurrentPrediction/"
MIGRATION_FILE <- "/data/scripts/Ling/Proj3Migration/MigPrediction/migB5.csv"

OUTER_FOLDS <- 5
OUTER_REPEATS <- 5       # use 10 for the final analysis if runtime permits
INNER_FOLDS <- 5
ELASTIC_NET_ALPHA <- 0.5 # 0 = ridge, 1 = lasso, 0.5 = elastic net

RUN_ITEM_MODELS <- TRUE  # set FALSE for a quick domain-only run
ANALYZE_MOVERS_ONLY <- TRUE

setwd(PROJECT_DIR)

# ---- Data preparation -------------------------------------------------------

NP_resi <- read.csv("NP_resi.csv") %>%
  rename_with(~ sub("\\.x$", "", .x)) %>%
  select(
    -neuroticism23R, -neuroticism24, -neuroticism31, -neuroticism32R,
    -neuroticism41, -neuroticism42, -neuroticism43R, -neuroticism52,
    -others12, -others14
  )

migration <- read.csv(MIGRATION_FILE) %>%
  select(scode, migrator_type18)

data <- NP_resi %>%
  select(1:217) %>%
  left_join(migration, by = "scode") %>%
  mutate(
    # Handles both "Village stayer" and transitions such as
    # "Village to town" or "Village to village".
    birth_location = case_when(
      str_detect(migrator_type18, " stayer$") ~ str_remove(migrator_type18, " stayer$"),
      str_detect(migrator_type18, " to ") ~ str_extract(migrator_type18, "^[^ ]+"),
      TRUE ~ NA_character_
    ),
    current_location = case_when(
      str_detect(migrator_type18, " stayer$") ~ str_remove(migrator_type18, " stayer$"),
      str_detect(migrator_type18, " to ") ~ str_extract(migrator_type18, "(?<= to ).+$"),
      TRUE ~ NA_character_
    ),
    birth_location = str_to_title(birth_location),
    current_location = str_to_title(current_location),
    mover = birth_location != current_location
  )

location_levels <- c("Village", "Town", "City", "Abroad")

data <- data %>%
  filter(birth_location %in% location_levels, current_location %in% location_levels) %>%
  mutate(
    birth_location = factor(birth_location, levels = location_levels),
    current_location = factor(current_location, levels = location_levels)
  )

domain_predictors <- names(data)[4:8]
item_predictors <- names(data)[30:217]

# Use a single complete outcome cohort. Predictor missingness is imputed from
# each training fold below, without using test-fold information.
analysis_data <- data %>%
  drop_na(birth_location, current_location)

message("Full analysis cohort: ", nrow(analysis_data))
message("Movers: ", sum(analysis_data$mover), "; stayers: ", sum(!analysis_data$mover))
print(table(analysis_data$birth_location, analysis_data$current_location))

# ---- Paired, jointly stratified outer folds ---------------------------------

make_paired_folds <- function(birth, current, k = 5, repeats = 5, seed = 42) {
  strata <- interaction(birth, current, drop = TRUE)
  folds <- vector("list", k * repeats)
  position <- 1L

  for (repeat_id in seq_len(repeats)) {
    set.seed(seed + repeat_id)
    fold_id <- integer(length(strata))

    for (stratum in levels(strata)) {
      rows <- sample(which(strata == stratum))
      fold_id[rows] <- rep(seq_len(k), length.out = length(rows))
    }

    for (fold in seq_len(k)) {
      folds[[position]] <- list(
        repeat_id = repeat_id,
        fold = fold,
        train = which(fold_id != fold),
        test = which(fold_id == fold)
      )
      position <- position + 1L
    }
  }

  folds
}

paired_folds <- make_paired_folds(
  analysis_data$birth_location,
  analysis_data$current_location,
  k = OUTER_FOLDS,
  repeats = OUTER_REPEATS
)

# ---- Metrics and fitting ----------------------------------------------------

impute_from_training <- function(train_x, test_x) {
  train_x <- as.matrix(train_x)
  test_x <- as.matrix(test_x)

  medians <- apply(train_x, 2, median, na.rm = TRUE)
  medians[!is.finite(medians)] <- 0

  for (column in seq_len(ncol(train_x))) {
    train_x[!is.finite(train_x[, column]), column] <- medians[column]
    test_x[!is.finite(test_x[, column]), column] <- medians[column]
  }

  list(train = train_x, test = test_x)
}

multiclass_metrics <- function(actual, probabilities, baseline_probabilities) {
  actual <- factor(actual, levels = location_levels)
  observed <- model.matrix(~ actual - 1)
  colnames(observed) <- location_levels
  probabilities <- probabilities[, location_levels, drop = FALSE]

  model_brier <- mean(rowSums((probabilities - observed)^2))
  baseline_matrix <- matrix(
    rep(baseline_probabilities, each = nrow(observed)),
    nrow = nrow(observed),
    dimnames = list(NULL, location_levels)
  )
  baseline_brier <- mean(rowSums((baseline_matrix - observed)^2))

  actual_index <- match(as.character(actual), location_levels)
  predicted_class <- location_levels[max.col(probabilities, ties.method = "first")]
  recall <- vapply(location_levels, function(class_name) {
    rows <- actual == class_name
    if (!any(rows)) return(NA_real_)
    mean(predicted_class[rows] == class_name)
  }, numeric(1))

  c(
    BSS = 1 - model_brier / baseline_brier,
    Brier = model_brier,
    LogLoss = -mean(log(pmax(probabilities[cbind(seq_along(actual), actual_index)], 1e-15))),
    Accuracy = mean(predicted_class == as.character(actual)),
    BalancedAccuracy = mean(recall, na.rm = TRUE)
  )
}

fit_one_outcome <- function(train_x, test_x, train_y, test_y) {
  baseline <- prop.table(table(factor(train_y, levels = location_levels)))

  set.seed(42)
  fit <- cv.glmnet(
    x = train_x,
    y = factor(train_y, levels = location_levels),
    family = "multinomial",
    type.multinomial = "grouped",
    alpha = ELASTIC_NET_ALPHA,
    nfolds = INNER_FOLDS,
    type.measure = "deviance",
    standardize = TRUE,
    parallel = FALSE
  )

  probabilities <- predict(fit, newx = test_x, s = "lambda.1se", type = "response")[, , 1]
  probabilities <- probabilities[, location_levels, drop = FALSE]

  list(
    metrics = multiclass_metrics(test_y, probabilities, baseline),
    probabilities = probabilities,
    lambda = fit$lambda.1se
  )
}

run_paired_comparison <- function(df, predictors, model_name, movers_only = FALSE) {
  fold_results <- vector("list", length(paired_folds))

  for (i in seq_along(paired_folds)) {
    split <- paired_folds[[i]]
    train_rows <- split$train
    test_rows <- split$test

    # Movers-only is an evaluation of the same population definition in both
    # outcomes. Both training and testing are restricted to movers.
    if (movers_only) {
      train_rows <- train_rows[df$mover[train_rows]]
      test_rows <- test_rows[df$mover[test_rows]]
    }

    imputed <- impute_from_training(
      df[train_rows, predictors, drop = FALSE],
      df[test_rows, predictors, drop = FALSE]
    )

    birth_fit <- fit_one_outcome(
      imputed$train, imputed$test,
      df$birth_location[train_rows], df$birth_location[test_rows]
    )
    current_fit <- fit_one_outcome(
      imputed$train, imputed$test,
      df$current_location[train_rows], df$current_location[test_rows]
    )

    fold_results[[i]] <- bind_rows(
      as.data.frame(as.list(birth_fit$metrics)) %>% mutate(Outcome = "Birth", Lambda = birth_fit$lambda),
      as.data.frame(as.list(current_fit$metrics)) %>% mutate(Outcome = "Current", Lambda = current_fit$lambda)
    ) %>%
      mutate(
        Model = model_name,
        Sample = if_else(movers_only, "Movers only", "All participants"),
        Repeat = split$repeat_id,
        Fold = split$fold,
        TestN = length(test_rows)
      )

    message(model_name, " [", ifelse(movers_only, "movers", "all"), "]: fold ", i,
            "/", length(paired_folds))
  }

  bind_rows(fold_results)
}

# Nadeau-Bengio corrected paired test. With repeated k-fold CV, each paired
# observation is one Current-minus-Birth difference from the same outer test fold.
corrected_paired_test <- function(results, metric = "BSS", k = 5) {
  paired <- results %>%
    select(Model, Sample, Repeat, Fold, Outcome, all_of(metric)) %>%
    pivot_wider(names_from = Outcome, values_from = all_of(metric)) %>%
    mutate(Difference = Current - Birth)

  paired %>%
    group_by(Model, Sample) %>%
    summarize(
      Metric = metric,
      Birth = mean(Birth),
      Current = mean(Current),
      Current_minus_Birth = mean(Difference),
      Corrected_SE = sqrt((1 / n() + 1 / (k - 1)) * var(Difference)),
      df = n() - 1,
      t = Current_minus_Birth / Corrected_SE,
      p = 2 * pt(abs(t), df = df, lower.tail = FALSE),
      CI_Lower = Current_minus_Birth - qt(0.975, df) * Corrected_SE,
      CI_Upper = Current_minus_Birth + qt(0.975, df) * Corrected_SE,
      .groups = "drop"
    )
}

# ---- Run analyses -----------------------------------------------------------

all_results <- list(
  run_paired_comparison(analysis_data, domain_predictors, "Big Five domains")
)

if (ANALYZE_MOVERS_ONLY) {
  all_results <- append(all_results, list(
    run_paired_comparison(analysis_data, domain_predictors, "Big Five domains", movers_only = TRUE)
  ))
}

if (RUN_ITEM_MODELS) {
  all_results <- append(all_results, list(
    run_paired_comparison(analysis_data, item_predictors, "Personality items")
  ))

  if (ANALYZE_MOVERS_ONLY) {
    all_results <- append(all_results, list(
      run_paired_comparison(analysis_data, item_predictors, "Personality items", movers_only = TRUE)
    ))
  }
}

fold_metrics <- bind_rows(all_results)
bss_comparison <- corrected_paired_test(fold_metrics, metric = "BSS", k = OUTER_FOLDS)

metric_summary <- fold_metrics %>%
  group_by(Model, Sample, Outcome) %>%
  summarize(
    across(c(BSS, Brier, LogLoss, Accuracy, BalancedAccuracy), mean),
    .groups = "drop"
  )

print(metric_summary)
print(bss_comparison)

write.csv(fold_metrics, "basic_fold_metrics.csv", row.names = FALSE)
write.csv(metric_summary, "basic_metric_summary.csv", row.names = FALSE)
write.csv(bss_comparison, "basic_birth_current_bss_comparison.csv", row.names = FALSE)

# Positive Current_minus_Birth means personality predicts current location better.
# Negative Current_minus_Birth means personality predicts birth location better.
p <- ggplot(metric_summary, aes(x = Outcome, y = BSS, fill = Outcome)) +
  geom_col(width = 0.65) +
  facet_grid(Sample ~ Model) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey35") +
  scale_fill_manual(values = c(Birth = "#0072B2", Current = "#D55E00")) +
  labs(
    title = "Does Personality Better Predict Birth or Current Location?",
    subtitle = "Paired out-of-sample multiclass Brier skill scores",
    x = NULL,
    y = "Brier Skill Score"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none", strip.text = element_text(face = "bold"))

ggsave("basic_birth_vs_current_bss.png", p, width = 9, height = 6, dpi = 300)

message("Saved fold metrics, summaries, corrected comparisons, and BSS plot.")
