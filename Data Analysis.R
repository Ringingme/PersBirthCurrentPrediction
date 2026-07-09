setwd("/data/scripts/Ling/Proj5birthCurrentPrediction/")


# Packages
library(mlr3)
library(mlr3learners)
library(mlr3pipelines)
library(mlr3measures)
library(mlr3tuning)
library(tidyverse)
library(paradox)


# ============================================================================
# DATA PREPARATION (Do not change)
# ============================================================================
NP_resi = read.csv("NP_resi.csv") |> rename_with(~ sub("\\.x$", "", .x))|> 
  select(-neuroticism23R, -neuroticism24, -neuroticism31, -neuroticism32R, -neuroticism41, -neuroticism42, -neuroticism43R, -neuroticism52, -others12, -others14)
data <- NP_resi  |> select(1:217) 
data$gender = as.factor(data$gender)

## Separate movers and stayers (used for evaluation only)
data_movers <- data |> filter(currentResidenceType2 != birthResidenceType)
data_stayers <- data |> filter(currentResidenceType2 == birthResidenceType)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

# 1. Calculate Brier Skill Score (BSS)
# BSS = 1 - (MSE / reference_MSE)
# where reference_MSE is from a featureless model
calculate_bss <- function(actual, predicted_probs, positive_class) {
  # Ensure actual is binary with positive_class as 1
  actual_binary <- as.numeric(actual == positive_class)
  
  # BSE (Brier Squared Error)
  mse <- mean((predicted_probs - actual_binary)^2)
  
  # Reference: always predict base rate
  base_rate <- mean(actual_binary)
  reference_mse <- base_rate * (1 - base_rate)^2 + (1 - base_rate) * (0 - base_rate)^2
  
  # BSS
  bss <- 1 - (mse / reference_mse)
  return(bss)
}

# 2. Nadeau & Bengio (2003) Significance Test
# Tests if two paired models are significantly different
# Input: two vectors of multiclass BSS scores from same folds
# For multiclass, we average across classes, then compare the fold-wise averages
nadeau_bengio_test <- function(bss_matrix1, bss_matrix2) {
  # bss_matrix1 and bss_matrix2 should have shape (n_folds, n_classes)
  # Calculate mean BSS per fold (average across classes)
  mean_per_fold_1 <- rowMeans(bss_matrix1)
  mean_per_fold_2 <- rowMeans(bss_matrix2)
  
  n <- length(mean_per_fold_1)
  # For repeated 5-fold CV: 50 folds total = 5 folds * 10 repeats
  p <- 10  # number of repetitions
  
  # Mean difference
  mean_diff <- mean(mean_per_fold_1 - mean_per_fold_2)
  
  # Variance of differences
  var_diff <- var(mean_per_fold_1 - mean_per_fold_2)
  
  # Standard error (Nadeau & Bengio correction)
  # SE = sqrt((1/n + 1/(5*p)) * var_diff)
  se <- sqrt((1/n + 1/(5*p)) * var_diff)
  
  # t-statistic
  t_stat <- mean_diff / se
  
  # p-value (two-tailed)
  p_value <- 2 * pt(abs(t_stat), df = n - 1, lower.tail = FALSE)
  
  return(list(
    mean_diff = mean_diff,
    se = se,
    t_stat = t_stat,
    p_value = p_value,
    n_folds = n,
    mean_per_fold_1 = mean_per_fold_1,
    mean_per_fold_2 = mean_per_fold_2
  ))
}

# 3. Evaluate model on subset (movers or stayers)
# subset_mask: logical vector indicating which samples belong to the subset
evaluate_on_subset <- function(predictions, true_outcomes, subset_mask, target_classes) {
  subset_pred <- predictions[subset_mask, ]
  subset_true <- true_outcomes[subset_mask]
  
  bss_per_class <- sapply(target_classes, function(class) {
    calculate_bss(subset_true, subset_pred[, class], class)
  })
  
  return(bss_per_class)
}


# ============================================================================
# UNIFIED MODEL WITH REPEATED STRATIFIED CV (5-fold, 10 repeats)
# ============================================================================
# NOTE: Hyperparameter tuning (nested CV) is NOT implemented yet.
# This performs repeated stratified cross-validation.
# To implement hyperparameter tuning, you would need an inner resampling loop.

unified_model <- function(df, pred_cols, outcome_var, residualize = FALSE, label, resampling = NULL) {
  
  message(paste0("\n=== Running unified model: ", label, " ==="))
  
  # 1. Prepare data for analysis
  if (residualize) {
    analysis_data <- df %>%
      select(all_of(c(pred_cols, "gender", "ageAtAgreement", outcome_var, 
                      "currentResidenceType2", "birthResidenceType"))) %>%
      drop_na(!!sym(outcome_var), gender, ageAtAgreement)
    
    # Residualize predictors within the analysis dataset
    for (col in pred_cols) {
      residuals_vec <- residuals(lm(as.formula(paste(col, "~ gender + ageAtAgreement")), data = analysis_data))
      analysis_data[[col]] <- residuals_vec
    }
  } else {
    analysis_data <- df %>%
      select(all_of(c(pred_cols, outcome_var, 
                      "currentResidenceType2", "birthResidenceType"))) %>%
      drop_na(!!sym(outcome_var))
  }
  
  # Keep columns needed for mover/stayer identification
  full_data <- analysis_data
  n_samples <- nrow(full_data)
  
  # Get outcome classes
  target_classes <- sort(unique(full_data[[outcome_var]]))
  
  # 2. Create task for classification
  task_data <- analysis_data %>% 
    select(all_of(c(pred_cols, outcome_var))) %>%
    rename(target = !!sym(outcome_var))
  
  task <- as_task_classif(
    task_data,
    target = "target",
    id = label
  )
  
  # 3. Stratified repeated CV: 5-fold, 10 repeats (50 test sets total)
  if (is.null(resampling)) {
    resampling <- rsmp("repeated_cv", folds = 5, repeats = 10)
  }
  
  # Ensure stratification on outcome classes
  task$col_roles$stratum <- "target"
  
  # Instantiate the resampling object on the task
  resampling$instantiate(task)
  
  # 4. Define learner (multinomial regression)
  # NOTE: classif.multinom is used without hyperparameter tuning
  graph <- po("imputemean") %>>% 
    po("imputeoor") %>>%
    po("encode", method = "treatment") %>>%
    lrn("classif.multinom", predict_type = "prob")
  
  learner <- as_learner(graph)
  learner$id <- paste0("MultinomReg_", label)
  
  # 5. Run repeated stratified CV
  all_fold_results <- list()
  
  for (i in 1:resampling$iters) {
    if (i %% 5 == 0) message(paste0("  Completed fold ", i, " of ", resampling$iters))
    
    train_idx <- resampling$train_set(i)
    test_idx <- resampling$test_set(i)
    
    task_train <- task$clone()$filter(train_idx)
    task_test <- task$clone()$filter(test_idx)
    
    # Train on training set
    learner$train(task_train)
    
    # Get predictions on test set
    pred <- learner$predict(task_test)
    
    # Extract true outcomes and predicted probabilities
    true_outcomes <- task_test$data()$target
    pred_probs <- pred$prob
    
    # Calculate BSS for each class
    bss_per_class <- sapply(target_classes, function(class) {
      calculate_bss(true_outcomes, pred_probs[, class], class)
    })
    
    # Store results
    all_fold_results[[i]] <- list(
      fold_id = i,
      train_idx = train_idx,
      test_idx = test_idx,
      true_outcomes = true_outcomes,
      pred_probs = pred_probs,
      bss_per_class = bss_per_class,
      mean_bss = mean(bss_per_class)
    )
  }
  
  # 6. Aggregate results across all folds
  all_bss_values <- do.call(rbind, lapply(all_fold_results, function(x) x$bss_per_class))
  mean_bss_per_class <- colMeans(all_bss_values)
  overall_mean_bss <- mean(all_bss_values)
  
  # 7. Combine all predictions and outcomes for subset evaluation
  all_true <- unlist(lapply(all_fold_results, function(x) x$true_outcomes))
  all_pred <- do.call(rbind, lapply(all_fold_results, function(x) x$pred_probs))
  all_test_idx <- unlist(lapply(all_fold_results, function(x) x$test_idx))
  
  # 8. Identify movers and stayers from the FULL dataset using original indices
  is_mover <- full_data$currentResidenceType2[all_test_idx] != full_data$birthResidenceType[all_test_idx]
  mover_mask <- is_mover
  stayer_mask <- !is_mover
  
  # 9. Evaluate on subsets using logical masks
  bss_movers <- evaluate_on_subset(all_pred, all_true, mover_mask, target_classes)
  bss_stayers <- evaluate_on_subset(all_pred, all_true, stayer_mask, target_classes)
  
  # 10. Compute fold-wise BSS separately for movers and stayers (for per-class tests)
  bss_movers_per_fold <- matrix(NA, nrow = length(all_fold_results), ncol = length(target_classes))
  bss_stayers_per_fold <- matrix(NA, nrow = length(all_fold_results), ncol = length(target_classes))
  colnames(bss_movers_per_fold) <- target_classes
  colnames(bss_stayers_per_fold) <- target_classes
  
  for (fold_idx in 1:length(all_fold_results)) {
    fold_pred <- all_fold_results[[fold_idx]]$pred_probs
    fold_true <- all_fold_results[[fold_idx]]$true_outcomes
    fold_test_idx <- all_fold_results[[fold_idx]]$test_idx
    
    # Identify movers/stayers in this specific fold
    fold_is_mover <- full_data$currentResidenceType2[fold_test_idx] != full_data$birthResidenceType[fold_test_idx]
    fold_mover_mask <- fold_is_mover
    fold_stayer_mask <- !fold_is_mover
    
    # Compute BSS per class for movers and stayers in this fold
    for (class_idx in 1:length(target_classes)) {
      class_val <- target_classes[class_idx]
      
      if (sum(fold_mover_mask) > 0) {
        bss_movers_per_fold[fold_idx, class_idx] <- calculate_bss(
          fold_true[fold_mover_mask],
          fold_pred[fold_mover_mask, class_idx],
          class_val
        )
      }
      
      if (sum(fold_stayer_mask) > 0) {
        bss_stayers_per_fold[fold_idx, class_idx] <- calculate_bss(
          fold_true[fold_stayer_mask],
          fold_pred[fold_stayer_mask, class_idx],
          class_val
        )
      }
    }
  }
  
  return(list(
    label = label,
    mean_bss_per_class = mean_bss_per_class,
    overall_mean_bss = overall_mean_bss,
    all_bss_values = all_bss_values,
    bss_movers = bss_movers,
    bss_stayers = bss_stayers,
    bss_movers_per_fold = bss_movers_per_fold,
    bss_stayers_per_fold = bss_stayers_per_fold,
    all_fold_results = all_fold_results,
    target_classes = target_classes,
    all_pred_probs = all_pred,
    all_true_outcomes = all_true,
    mover_mask = mover_mask,
    stayer_mask = stayer_mask,
    resampling = resampling
  ))
}


# ============================================================================
# RUN MODELS
# ============================================================================
# Create a single resampling object to ensure all models use identical folds
set.seed(42)
shared_resampling <- rsmp("repeated_cv", folds = 5, repeats = 10)

# Model A1: Predict CURRENT residence (items - raw)
current_items_raw <- unified_model(
  df = data,
  pred_cols = colnames(data)[30:217],
  outcome_var = "currentResidenceType2",
  residualize = FALSE,
  label = "current_items_raw",
  resampling = shared_resampling
)
saveRDS(current_items_raw, file = "current_items_raw.rds")

# Model A2: Predict BIRTH residence (items - raw)
birth_items_raw <- unified_model(
  df = data,
  pred_cols = colnames(data)[30:217],
  outcome_var = "birthResidenceType",
  residualize = FALSE,
  label = "birth_items_raw",
  resampling = shared_resampling
)
saveRDS(birth_items_raw, file = "birth_items_raw.rds")

# Current residence (domains - raw)
current_domains_raw <- unified_model(
  df = data,
  pred_cols = colnames(data)[4:8],
  outcome_var = "currentResidenceType2",
  residualize = FALSE,
  label = "current_domains_raw",
  resampling = shared_resampling
)
saveRDS(current_domains_raw, file = "current_domains_raw.rds")

# Birth residence (domains - raw)
birth_domains_raw <- unified_model(
  df = data,
  pred_cols = colnames(data)[4:8],
  outcome_var = "birthResidenceType",
  residualize = FALSE,
  label = "birth_domains_raw",
  resampling = shared_resampling
)
saveRDS(birth_domains_raw, file = "birth_domains_raw.rds")

# Current residence (items - residualized)
current_items_resid <- unified_model(
  df = data,
  pred_cols = colnames(data)[30:217],
  outcome_var = "currentResidenceType2",
  residualize = TRUE,
  label = "current_items_resid",
  resampling = shared_resampling
)
saveRDS(current_items_resid, file = "current_items_resid.rds")

# Birth residence (items - residualized)
birth_items_resid <- unified_model(
  df = data,
  pred_cols = colnames(data)[30:217],
  outcome_var = "birthResidenceType",
  residualize = TRUE,
  label = "birth_items_resid",
  resampling = shared_resampling
)
saveRDS(birth_items_resid, file = "birth_items_resid.rds")

# Current residence (domains - residualized)
current_domains_resid <- unified_model(
  df = data,
  pred_cols = colnames(data)[4:8],
  outcome_var = "currentResidenceType2",
  residualize = TRUE,
  label = "current_domains_resid",
  resampling = shared_resampling
)
saveRDS(current_domains_resid, file = "current_domains_resid.rds")

# Birth residence (domains - residualized)
birth_domains_resid <- unified_model(
  df = data,
  pred_cols = colnames(data)[4:8],
  outcome_var = "birthResidenceType",
  residualize = TRUE,
  label = "birth_domains_resid",
  resampling = shared_resampling
)
saveRDS(birth_domains_resid, file = "birth_domains_resid.rds")


# ============================================================================
# SIGNIFICANCE TESTING (Nadeau & Bengio, 2003)
# ============================================================================
# Note: All models use identical folds (shared_resampling), so comparisons are valid.

message("\n" %+% strrep("=", 90))
message("SIGNIFICANCE TESTS (Nadeau & Bengio, 2003)")
message("=" %+% strrep("=", 90))

# ============================================================================
# SECTION A: Current vs Birth Residence Prediction (Aggregate across classes)
# ============================================================================

message("\n" %+% strrep("-", 90))
message("SECTION A: Current vs Birth Residence (Multiclass Average)")
message(strrep("-", 90))

# Test A1: Current vs Birth for ITEMS (raw)
message("\nA1: Current vs Birth prediction (ITEMS - RAW)")
test_A1 <- nadeau_bengio_test(current_items_raw$all_bss_values, birth_items_raw$all_bss_values)
message(paste0("  Mean BSS diff (Current - Birth): ", round(test_A1$mean_diff, 4)))
message(paste0("  t-stat: ", round(test_A1$t_stat, 4), " | p-value: ", round(test_A1$p_value, 4)))
if (test_A1$p_value < 0.05) message("  *** SIGNIFICANT ***")

# Test A2: Current vs Birth for DOMAINS (raw)
message("\nA2: Current vs Birth prediction (DOMAINS - RAW)")
test_A2 <- nadeau_bengio_test(current_domains_raw$all_bss_values, birth_domains_raw$all_bss_values)
message(paste0("  Mean BSS diff (Current - Birth): ", round(test_A2$mean_diff, 4)))
message(paste0("  t-stat: ", round(test_A2$t_stat, 4), " | p-value: ", round(test_A2$p_value, 4)))
if (test_A2$p_value < 0.05) message("  *** SIGNIFICANT ***")

# Test A3: Current vs Birth for ITEMS (residualized)
message("\nA3: Current vs Birth prediction (ITEMS - RESIDUALIZED)")
test_A3 <- nadeau_bengio_test(current_items_resid$all_bss_values, birth_items_resid$all_bss_values)
message(paste0("  Mean BSS diff (Current - Birth): ", round(test_A3$mean_diff, 4)))
message(paste0("  t-stat: ", round(test_A3$t_stat, 4), " | p-value: ", round(test_A3$p_value, 4)))
if (test_A3$p_value < 0.05) message("  *** SIGNIFICANT ***")

# Test A4: Current vs Birth for DOMAINS (residualized)
message("\nA4: Current vs Birth prediction (DOMAINS - RESIDUALIZED)")
test_A4 <- nadeau_bengio_test(current_domains_resid$all_bss_values, birth_domains_resid$all_bss_values)
message(paste0("  Mean BSS diff (Current - Birth): ", round(test_A4$mean_diff, 4)))
message(paste0("  t-stat: ", round(test_A4$t_stat, 4), " | p-value: ", round(test_A4$p_value, 4)))
if (test_A4$p_value < 0.05) message("  *** SIGNIFICANT ***")

# ============================================================================
# SECTION B: Movers vs Stayers - CURRENT Residence (Per-Class)
# ============================================================================

message("\n" %+% strrep("-", 90))
message("SECTION B: Movers vs Stayers - CURRENT Residence (Per-Class)")
message(strrep("-", 90))

# B1: Items (raw)
message("\nB1: Current Residence - ITEMS (RAW)")
message("  Class-wise comparison (Movers BSS vs Stayers BSS):")
for (class_idx in 1:length(current_items_raw$target_classes)) {
  class_val <- current_items_raw$target_classes[class_idx]
  test_B1_class <- nadeau_bengio_test(
    matrix(current_items_raw$bss_movers_per_fold[, class_idx], ncol = 1),
    matrix(current_items_raw$bss_stayers_per_fold[, class_idx], ncol = 1)
  )
  message(paste0("    ", class_val, ": diff=", round(test_B1_class$mean_diff, 4), 
                 " | p=", round(test_B1_class$p_value, 4),
                 if_else(test_B1_class$p_value < 0.05, " *", "")))
}

# B2: Domains (raw)
message("\nB2: Current Residence - DOMAINS (RAW)")
message("  Class-wise comparison (Movers BSS vs Stayers BSS):")
for (class_idx in 1:length(current_domains_raw$target_classes)) {
  class_val <- current_domains_raw$target_classes[class_idx]
  test_B2_class <- nadeau_bengio_test(
    matrix(current_domains_raw$bss_movers_per_fold[, class_idx], ncol = 1),
    matrix(current_domains_raw$bss_stayers_per_fold[, class_idx], ncol = 1)
  )
  message(paste0("    ", class_val, ": diff=", round(test_B2_class$mean_diff, 4), 
                 " | p=", round(test_B2_class$p_value, 4),
                 if_else(test_B2_class$p_value < 0.05, " *", "")))
}

# B3: Items (residualized)
message("\nB3: Current Residence - ITEMS (RESIDUALIZED)")
message("  Class-wise comparison (Movers BSS vs Stayers BSS):")
for (class_idx in 1:length(current_items_resid$target_classes)) {
  class_val <- current_items_resid$target_classes[class_idx]
  test_B3_class <- nadeau_bengio_test(
    matrix(current_items_resid$bss_movers_per_fold[, class_idx], ncol = 1),
    matrix(current_items_resid$bss_stayers_per_fold[, class_idx], ncol = 1)
  )
  message(paste0("    ", class_val, ": diff=", round(test_B3_class$mean_diff, 4), 
                 " | p=", round(test_B3_class$p_value, 4),
                 if_else(test_B3_class$p_value < 0.05, " *", "")))
}

# B4: Domains (residualized)
message("\nB4: Current Residence - DOMAINS (RESIDUALIZED)")
message("  Class-wise comparison (Movers BSS vs Stayers BSS):")
for (class_idx in 1:length(current_domains_resid$target_classes)) {
  class_val <- current_domains_resid$target_classes[class_idx]
  test_B4_class <- nadeau_bengio_test(
    matrix(current_domains_resid$bss_movers_per_fold[, class_idx], ncol = 1),
    matrix(current_domains_resid$bss_stayers_per_fold[, class_idx], ncol = 1)
  )
  message(paste0("    ", class_val, ": diff=", round(test_B4_class$mean_diff, 4), 
                 " | p=", round(test_B4_class$p_value, 4),
                 if_else(test_B4_class$p_value < 0.05, " *", "")))
}

# ============================================================================
# SECTION C: Movers vs Stayers - BIRTH Residence (Per-Class)
# ============================================================================

message("\n" %+% strrep("-", 90))
message("SECTION C: Movers vs Stayers - BIRTH Residence (Per-Class)")
message(strrep("-", 90))

# C1: Items (raw)
message("\nC1: Birth Residence - ITEMS (RAW)")
message("  Class-wise comparison (Movers BSS vs Stayers BSS):")
for (class_idx in 1:length(birth_items_raw$target_classes)) {
  class_val <- birth_items_raw$target_classes[class_idx]
  test_C1_class <- nadeau_bengio_test(
    matrix(birth_items_raw$bss_movers_per_fold[, class_idx], ncol = 1),
    matrix(birth_items_raw$bss_stayers_per_fold[, class_idx], ncol = 1)
  )
  message(paste0("    ", class_val, ": diff=", round(test_C1_class$mean_diff, 4), 
                 " | p=", round(test_C1_class$p_value, 4),
                 if_else(test_C1_class$p_value < 0.05, " *", "")))
}

# C2: Domains (raw)
message("\nC2: Birth Residence - DOMAINS (RAW)")
message("  Class-wise comparison (Movers BSS vs Stayers BSS):")
for (class_idx in 1:length(birth_domains_raw$target_classes)) {
  class_val <- birth_domains_raw$target_classes[class_idx]
  test_C2_class <- nadeau_bengio_test(
    matrix(birth_domains_raw$bss_movers_per_fold[, class_idx], ncol = 1),
    matrix(birth_domains_raw$bss_stayers_per_fold[, class_idx], ncol = 1)
  )
  message(paste0("    ", class_val, ": diff=", round(test_C2_class$mean_diff, 4), 
                 " | p=", round(test_C2_class$p_value, 4),
                 if_else(test_C2_class$p_value < 0.05, " *", "")))
}

# C3: Items (residualized)
message("\nC3: Birth Residence - ITEMS (RESIDUALIZED)")
message("  Class-wise comparison (Movers BSS vs Stayers BSS):")
for (class_idx in 1:length(birth_items_resid$target_classes)) {
  class_val <- birth_items_resid$target_classes[class_idx]
  test_C3_class <- nadeau_bengio_test(
    matrix(birth_items_resid$bss_movers_per_fold[, class_idx], ncol = 1),
    matrix(birth_items_resid$bss_stayers_per_fold[, class_idx], ncol = 1)
  )
  message(paste0("    ", class_val, ": diff=", round(test_C3_class$mean_diff, 4), 
                 " | p=", round(test_C3_class$p_value, 4),
                 if_else(test_C3_class$p_value < 0.05, " *", "")))
}

# C4: Domains (residualized)
message("\nC4: Birth Residence - DOMAINS (RESIDUALIZED)")
message("  Class-wise comparison (Movers BSS vs Stayers BSS):")
for (class_idx in 1:length(birth_domains_resid$target_classes)) {
  class_val <- birth_domains_resid$target_classes[class_idx]
  test_C4_class <- nadeau_bengio_test(
    matrix(birth_domains_resid$bss_movers_per_fold[, class_idx], ncol = 1),
    matrix(birth_domains_resid$bss_stayers_per_fold[, class_idx], ncol = 1)
  )
  message(paste0("    ", class_val, ": diff=", round(test_C4_class$mean_diff, 4), 
                 " | p=", round(test_C4_class$p_value, 4),
                 if_else(test_C4_class$p_value < 0.05, " *", "")))
}

# ============================================================================
# SUMMARY: Descriptive Statistics
# ============================================================================

message("\n" %+% strrep("-", 90))
message("SECTION D: Descriptive Statistics (Mean BSS per Class)")
message(strrep("-", 90))

# Current Residence - Items (Raw)
message("\nD1: Current Residence - ITEMS (RAW)")
message("      Settlement Type | Movers Mean BSS | Stayers Mean BSS | Difference")
for (class_idx in 1:length(current_items_raw$target_classes)) {
  class_val <- current_items_raw$target_classes[class_idx]
  m_mean <- mean(current_items_raw$bss_movers_per_fold[, class_idx], na.rm = TRUE)
  s_mean <- mean(current_items_raw$bss_stayers_per_fold[, class_idx], na.rm = TRUE)
  message(sprintf("      %-15s | %15.4f | %16.4f | %10.4f", class_val, m_mean, s_mean, m_mean - s_mean))
}

# Birth Residence - Items (Raw)
message("\nD2: Birth Residence - ITEMS (RAW)")
message("      Settlement Type | Movers Mean BSS | Stayers Mean BSS | Difference")
for (class_idx in 1:length(birth_items_raw$target_classes)) {
  class_val <- birth_items_raw$target_classes[class_idx]
  m_mean <- mean(birth_items_raw$bss_movers_per_fold[, class_idx], na.rm = TRUE)
  s_mean <- mean(birth_items_raw$bss_stayers_per_fold[, class_idx], na.rm = TRUE)
  message(sprintf("      %-15s | %15.4f | %16.4f | %10.4f", class_val, m_mean, s_mean, m_mean - s_mean))
}

message("\n" %+% strrep("=", 90) %+% "\n")





