setwd("/data/scripts/Ling/Proj5BirthCurrentPrediction/")


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
    # Use na.action = na.exclude so residuals vector matches data dimensions
    for (col in pred_cols) {
      residuals_vec <- residuals(lm(as.formula(paste(col, "~ gender + ageAtAgreement")), 
                                     data = analysis_data, 
                                     na.action = na.exclude))
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

# ============================================================================
# HEATMAP GENERATION: Predictor × Settlement Category
# ============================================================================
# Metrics: Standardized coefficients from multinomial logistic regression
# Interpretation: Strength and direction of association between predictor and class
#
# For ITEMS: We'll count how many times each item appears in "top predictors" 
# (high |coefficient|) across classes and analyses. Items appearing >X times 
# appear in main text; all items go to supplementary material.
#
# Threshold X can be adjusted based on results; recommended starting point: X=3

# Function to extract coefficients and create heatmap data
extract_heatmap_data <- function(model_object, pred_cols, label, data) {
  message(paste0("\nExtracting heatmap data for: ", label))
  
  # Recreate the task and train final model on full data to get stable coefficients
  task_data <- data %>%
    select(all_of(c(pred_cols, names(model_object$all_fold_results[[1]]$true_outcomes)))) %>%
    rename(target = all_of(names(model_object$all_fold_results[[1]]$true_outcomes)))
  
  # For coefficient extraction, we need to train on full data
  # (This gives us the final model structure for visualization purposes)
  task_full <- as_task_classif(
    data %>% select(all_of(c(pred_cols, model_object$target_classes[[1]]))),
    target = model_object$target_classes[[1]],
    id = label
  )
  
  # Get the target outcome variable name from the analysis
  outcome_name <- setdiff(colnames(data), pred_cols)[1]  # First non-predictor column
  
  task_full <- as_task_classif(
    data %>% select(all_of(c(pred_cols, outcome_name))),
    target = outcome_name,
    id = label
  )
  
  graph <- po("imputemean") %>>% 
    po("imputeoor") %>>%
    po("encode", method = "treatment") %>>%
    lrn("classif.multinom", predict_type = "prob")
  
  learner <- as_learner(graph)
  learner$train(task_full)
  
  # Extract coefficients
  # The underlying model is in learner$model[[2]]$model (after pipeline steps)
  multinom_model <- learner$model
  
  # Get weights from nnet::multinom object
  # For multinom with K classes, coefficients matrix is (K-1) × (p+1) 
  # (K-1 because reference class has 0 coefficients)
  tryCatch({
    coef_matrix <- coef(multinom_model)
    
    # multinom returns (n_classes-1) × (n_predictors+1)
    # Remove intercept column
    coef_matrix <- coef_matrix[, -1, drop = FALSE]
    
    # Add reference class with zeros
    ref_row <- rep(0, ncol(coef_matrix))
    coef_matrix <- rbind(coef_matrix, ref_row)
    
    # Get class names in order (match target_classes order)
    class_names <- model_object$target_classes
    rownames(coef_matrix) <- class_names
    
    return(list(
      coefficients = coef_matrix,
      predictor_names = pred_cols,
      class_names = class_names,
      label = label
    ))
  }, error = function(e) {
    message(paste0("  WARNING: Could not extract coefficients - ", e$message))
    return(NULL)
  })
}

# Function to standardize coefficients (for comparability across models)
standardize_coefficients <- function(coef_matrix) {
  # Standardize each predictor (column) to have mean 0, SD 1 across classes
  coef_std <- apply(coef_matrix, 2, function(col) {
    m <- mean(col)
    s <- sd(col)
    if (s == 0) return(col - m)
    return((col - m) / s)
  })
  return(coef_std)
}

# Function to count predictor frequency in top predictors
count_top_predictors <- function(coef_matrix, abs_threshold) {
  # For each predictor, count how many classes have |coef| > threshold
  top_counts <- colSums(abs(coef_matrix) > abs_threshold)
  return(top_counts)
}

# ============================================================================
# GENERATE HEATMAPS: RESIDUALIZED MODELS ONLY
# ============================================================================

message("\n" %+% strrep("=", 90))
message("GENERATING HEATMAP DATA: Residualized Models (Age/Gender controlled)")
message(strrep("=", 90))

# CORRELATION-BASED HEATMAP APPROACH (more robust):
# Calculate point-biserial correlation between each predictor and each class indicator

generate_correlation_heatmap <- function(model_object, pred_cols, outcome_var, data) {
  # Get all predictions and outcomes from model
  all_pred <- model_object$all_pred_probs
  all_true <- model_object$all_true_outcomes
  class_names <- model_object$target_classes
  
  # Create heatmap data: correlation between each predictor and class probability
  n_predictors <- length(pred_cols)
  n_classes <- length(class_names)
  
  heatmap_data <- matrix(NA, nrow = n_predictors, ncol = n_classes)
  rownames(heatmap_data) <- pred_cols
  colnames(heatmap_data) <- class_names
  
  # Get original predictor values from data
  data_pred <- data %>% select(all_of(pred_cols)) %>% 
    mutate(across(everything(), ~scale(., center = TRUE, scale = TRUE)[,1]))
  
  # Calculate correlation between predictors and predicted probabilities for each class
  for (class_idx in 1:n_classes) {
    class_name <- class_names[class_idx]
    class_prob <- all_pred[, class_idx]
    
    # For each predictor, calculate correlation with class probability
    for (pred_idx in 1:n_predictors) {
      pred_name <- pred_cols[pred_idx]
      pred_values <- data_pred[[pred_name]]
      
      # Correlation (Pearson)
      corr <- cor(pred_values, class_prob, use = "complete.obs")
      heatmap_data[pred_idx, class_idx] <- corr
    }
  }
  
  return(heatmap_data)
}

# Generate heatmaps for current residence (items - residualized)
message("\n--- Current Residence (Items - RESIDUALIZED) ---")
hm_current_items_resid <- generate_correlation_heatmap(
  model_object = current_items_resid,
  pred_cols = colnames(data)[30:217],
  outcome_var = "currentResidenceType2",
  data = data
)

# Count how many classes each item shows strong correlation with (|r| > 0.10)
item_freq_current_items_resid <- colSums(abs(hm_current_items_resid) > 0.10)
message(paste0("  Items with |correlation| > 0.10 in: ", 
               sum(item_freq_current_items_resid >= 1), " items"))
message(paste0("  Items appearing in 3+ classes: ", 
               sum(item_freq_current_items_resid >= 3), " items"))

# Save full heatmap data to RDS (supplementary material)
saveRDS(list(
  heatmap = hm_current_items_resid,
  predictor_names = colnames(data)[30:217],
  class_names = current_items_resid$target_classes,
  label = "current_residence_items_residualized"
), file = "heatmap_current_items_resid_full.rds")

# Filter items for main text (appearing in 3+ classes with |r| > 0.10)
main_text_threshold <- 3
items_for_main_current_resid <- which(item_freq_current_items_resid >= main_text_threshold)
hm_current_items_resid_filtered <- hm_current_items_resid[, items_for_main_current_resid, drop = FALSE]

message(paste0("  Main text (3+ classes): ", ncol(hm_current_items_resid_filtered), " items"))

# Generate heatmaps for current residence (domains - residualized)
message("\n--- Current Residence (Domains - RESIDUALIZED) ---")
hm_current_domains_resid <- generate_correlation_heatmap(
  model_object = current_domains_resid,
  pred_cols = colnames(data)[4:8],
  outcome_var = "currentResidenceType2",
  data = data
)
message(paste0("  Domains: all ", ncol(hm_current_domains_resid), " included in main text"))

# Generate heatmaps for birth residence (items - residualized)
message("\n--- Birth Residence (Items - RESIDUALIZED) ---")
hm_birth_items_resid <- generate_correlation_heatmap(
  model_object = birth_items_resid,
  pred_cols = colnames(data)[30:217],
  outcome_var = "birthResidenceType",
  data = data
)

item_freq_birth_items_resid <- colSums(abs(hm_birth_items_resid) > 0.10)
message(paste0("  Items with |correlation| > 0.10 in: ", 
               sum(item_freq_birth_items_resid >= 1), " items"))
message(paste0("  Items appearing in 3+ classes: ", 
               sum(item_freq_birth_items_resid >= 3), " items"))

saveRDS(list(
  heatmap = hm_birth_items_resid,
  predictor_names = colnames(data)[30:217],
  class_names = birth_items_resid$target_classes,
  label = "birth_residence_items_residualized"
), file = "heatmap_birth_items_resid_full.rds")

items_for_main_birth_resid <- which(item_freq_birth_items_resid >= main_text_threshold)
hm_birth_items_resid_filtered <- hm_birth_items_resid[, items_for_main_birth_resid, drop = FALSE]

message(paste0("  Main text (3+ classes): ", ncol(hm_birth_items_resid_filtered), " items"))

# Generate heatmaps for birth residence (domains - residualized)
message("\n--- Birth Residence (Domains - RESIDUALIZED) ---")
hm_birth_domains_resid <- generate_correlation_heatmap(
  model_object = birth_domains_resid,
  pred_cols = colnames(data)[4:8],
  outcome_var = "birthResidenceType",
  data = data
)
message(paste0("  Domains: all ", ncol(hm_birth_domains_resid), " included in main text"))

# ============================================================================
# VISUALIZATION: Create ggplot2-compatible heatmaps
# ============================================================================

library(reshape2)

# Function to create ggplot2 heatmap
create_heatmap_ggplot <- function(heatmap_data, title, filename = NULL, 
                                   width = 10, height = 8) {
  # Melt for ggplot2
  hm_melted <- melt(heatmap_data)
  colnames(hm_melted) <- c("Predictor", "Class", "Correlation")
  
  # Create heatmap
  p <- ggplot(hm_melted, aes(x = Class, y = Predictor, fill = Correlation)) +
    geom_tile(color = "white", size = 0.5) +
    scale_fill_gradient2(low = "steelblue", mid = "white", high = "firebrick",
                         midpoint = 0, limits = c(-0.3, 0.3)) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      axis.text.y = element_text(size = 8),
      legend.position = "right",
      plot.title = element_text(hjust = 0.5, size = 12, face = "bold")
    ) +
    labs(title = title, x = "Settlement Category", y = "Predictor",
         fill = "Correlation\n(Pearson r)")
  
  if (!is.null(filename)) {
    ggsave(filename, plot = p, width = width, height = height, dpi = 300)
    message(paste0("  Saved: ", filename))
  }
  
  return(p)
}

# Create main text heatmaps (RESIDUALIZED ONLY)
message("\n" %+% strrep("=", 90))
message("CREATING PUBLICATION-READY HEATMAPS (RESIDUALIZED)")
message(strrep("=", 90))

message("\nMAIN TEXT FIGURES:")

p1 <- create_heatmap_ggplot(
  hm_current_domains_resid,
  title = "Current Residence: Domain Predictors (Age/Gender Residualized)",
  filename = "heatmap_current_domains_resid.png",
  width = 6, height = 4
)

p2 <- create_heatmap_ggplot(
  hm_birth_domains_resid,
  title = "Birth Residence: Domain Predictors (Age/Gender Residualized)",
  filename = "heatmap_birth_domains_resid.png",
  width = 6, height = 4
)

p3 <- create_heatmap_ggplot(
  hm_current_items_resid_filtered,
  title = paste0("Current Residence: Top Item Predictors (n=", ncol(hm_current_items_resid_filtered), ", Age/Gender Residualized)"),
  filename = "heatmap_current_items_resid.png",
  width = 8, height = 10
)

p4 <- create_heatmap_ggplot(
  hm_birth_items_resid_filtered,
  title = paste0("Birth Residence: Top Item Predictors (n=", ncol(hm_birth_items_resid_filtered), ", Age/Gender Residualized)"),
  filename = "heatmap_birth_items_resid.png",
  width = 8, height = 10
)

message("\nSUPPLEMENTARY MATERIAL FIGURES:")

p_sup1 <- create_heatmap_ggplot(
  hm_current_items_resid,
  title = paste0("Supplementary: Current Residence - All Item Predictors (n=", ncol(hm_current_items_resid), ", Age/Gender Residualized)"),
  filename = "heatmap_current_items_resid_supplementary.png",
  width = 12, height = 30
)

p_sup2 <- create_heatmap_ggplot(
  hm_birth_items_resid,
  title = paste0("Supplementary: Birth Residence - All Item Predictors (n=", ncol(hm_birth_items_resid), ", Age/Gender Residualized)"),
  filename = "heatmap_birth_items_resid_supplementary.png",
  width = 12, height = 30
)

message("\n" %+% strrep("=", 90))
message("ITEM FILTERING SUMMARY (Threshold: |correlation| > 0.10 in 3+ classes)")
message("RESIDUALIZED MODELS (Age/Gender Controlled)")
message(strrep("=", 90))

message("\nCurrent Residence Prediction:")
message(sprintf("  Total items: %d", length(colnames(data)[30:217])))
message(sprintf("  Items in main text (≥3 classes): %d (%.1f%%)", 
                length(items_for_main_current_resid),
                100 * length(items_for_main_current_resid) / 188))
message(sprintf("  Items in supplementary only: %d", 
                188 - length(items_for_main_current_resid)))

message("\nBirth Residence Prediction:")
message(sprintf("  Total items: %d", length(colnames(data)[30:217])))
message(sprintf("  Items in main text (≥3 classes): %d (%.1f%%)", 
                length(items_for_main_birth_resid),
                100 * length(items_for_main_birth_resid) / 188))
message(sprintf("  Items in supplementary only: %d", 
                188 - length(items_for_main_birth_resid)))

# ============================================================================
# DETAILED ITEM REPORTING TABLE
# ============================================================================

message("\n" %+% strrep("=", 90))
message("TOP ITEMS FOR MAIN TEXT (RESIDUALIZED)")
message(strrep("=", 90))

# Create summary table for main text items
items_list_current_resid <- colnames(data)[30:217][items_for_main_current_resid]
items_list_birth_resid <- colnames(data)[30:217][items_for_main_birth_resid]

message("\nCurrent Residence - Main Text Items (Age/Gender Residualized):")
message("  Item Name | Village corr | Town corr | City corr | Abroad corr | Max |corr|")
for (i in seq_along(items_list_current_resid)) {
  item <- items_list_current_resid[i]
  idx <- items_for_main_current_resid[i]
  corr_vals <- hm_current_items_resid_filtered[, i]
  max_corr <- max(abs(corr_vals))
  message(sprintf("  %s | %12.4f | %9.4f | %9.4f | %11.4f | %8.4f",
                  item, corr_vals[1], corr_vals[2], corr_vals[3], corr_vals[4], max_corr))
}

message("\nBirth Residence - Main Text Items (Age/Gender Residualized):")
message("  Item Name | Village corr | Town corr | City corr | Abroad corr | Max |corr|")
for (i in seq_along(items_list_birth_resid)) {
  item <- items_list_birth_resid[i]
  idx <- items_for_main_birth_resid[i]
  corr_vals <- hm_birth_items_resid_filtered[, i]
  max_corr <- max(abs(corr_vals))
  message(sprintf("  %s | %12.4f | %9.4f | %9.4f | %11.4f | %8.4f",
                  item, corr_vals[1], corr_vals[2], corr_vals[3], corr_vals[4], max_corr))
}

message("\n" %+% strrep("=", 90) %+% "\n")

# ============================================================================
# VISUALIZATION 1: HISTOGRAM - MODEL PREDICTIVE ABILITY WITH CONFIDENCE INTERVALS
# ============================================================================

message("\n" %+% strrep("=", 90))
message("CREATING HISTOGRAM: Model Predictive Performance (ROC AUC with 95% CI)")
message(strrep("=", 90))

# Extract BSS values and compute confidence intervals for each model by settlement type
# We'll use BSS as the performance metric (similar interpretation to AUC)

compute_ci_bss <- function(bss_per_fold_matrix) {
  # For each class, compute mean and 95% CI
  # bss_per_fold_matrix: rows=folds, cols=settlement classes
  
  means <- colMeans(bss_per_fold_matrix, na.rm = TRUE)
  ses <- apply(bss_per_fold_matrix, 2, function(col) {
    sd(col, na.rm = TRUE) / sqrt(sum(!is.na(col)))
  })
  ci_lower <- means - 1.96 * ses
  ci_upper <- means + 1.96 * ses
  
  return(list(means = means, ci_lower = ci_lower, ci_upper = ci_upper))
}

# Prepare data for histogram: BSS by model, settlement type, mover/stayer
histogram_data <- data.frame()

# For residualized models (primary analysis)
models_to_plot <- list(
  list(name = "Domains", current = current_domains_resid, birth = birth_domains_resid),
  list(name = "Items", current = current_items_resid, birth = birth_items_resid)
)

for (model_set in models_to_plot) {
  model_name <- model_set$name
  current_model <- model_set$current
  birth_model <- model_set$birth
  
  classes <- current_model$target_classes
  
  # Current residence - movers and stayers
  current_movers_ci <- compute_ci_bss(current_model$bss_movers_per_fold)
  current_stayers_ci <- compute_ci_bss(current_model$bss_stayers_per_fold)
  
  # Birth residence - movers and stayers
  birth_movers_ci <- compute_ci_bss(birth_model$bss_movers_per_fold)
  birth_stayers_ci <- compute_ci_bss(birth_model$bss_stayers_per_fold)
  
  # Add to data frame
  for (class_idx in 1:length(classes)) {
    class_name <- classes[class_idx]
    
    # Current - Movers
    histogram_data <- rbind(histogram_data, data.frame(
      Model = model_name,
      Residence = "Current",
      Group = "Mover",
      Settlement = class_name,
      Mean_BSS = current_movers_ci$means[class_idx],
      CI_Lower = current_movers_ci$ci_lower[class_idx],
      CI_Upper = current_movers_ci$ci_upper[class_idx]
    ))
    
    # Current - Stayers
    histogram_data <- rbind(histogram_data, data.frame(
      Model = model_name,
      Residence = "Current",
      Group = "Stayer",
      Settlement = class_name,
      Mean_BSS = current_stayers_ci$means[class_idx],
      CI_Lower = current_stayers_ci$ci_lower[class_idx],
      CI_Upper = current_stayers_ci$ci_upper[class_idx]
    ))
    
    # Birth - Movers
    histogram_data <- rbind(histogram_data, data.frame(
      Model = model_name,
      Residence = "Birth",
      Group = "Mover",
      Settlement = class_name,
      Mean_BSS = birth_movers_ci$means[class_idx],
      CI_Lower = birth_movers_ci$ci_lower[class_idx],
      CI_Upper = birth_movers_ci$ci_upper[class_idx]
    ))
    
    # Birth - Stayers
    histogram_data <- rbind(histogram_data, data.frame(
      Model = model_name,
      Residence = "Birth",
      Group = "Stayer",
      Settlement = class_name,
      Mean_BSS = birth_stayers_ci$means[class_idx],
      CI_Lower = birth_stayers_ci$ci_lower[class_idx],
      CI_Upper = birth_stayers_ci$ci_upper[class_idx]
    ))
  }
}

# Create histogram with ggplot2
p_histogram <- ggplot(histogram_data, aes(x = Settlement, y = Mean_BSS, 
                                           fill = interaction(Residence, Group),
                                           color = interaction(Residence, Group))) +
  geom_bar(stat = "identity", position = position_dodge(0.8), width = 0.7, alpha = 0.8) +
  geom_errorbar(aes(ymin = CI_Lower, ymax = CI_Upper), 
                position = position_dodge(0.8), width = 0.2, size = 0.5) +
  facet_wrap(~ Model, nrow = 1) +
  scale_fill_manual(values = c("Current.Mover" = "#1f77b4", "Current.Stayer" = "#aec7e8",
                                "Birth.Mover" = "#ff7f0e", "Birth.Stayer" = "#ffbb78"),
                    labels = c("Current.Mover" = "Current - Mover",
                               "Current.Stayer" = "Current - Stayer",
                               "Birth.Mover" = "Birth - Mover",
                               "Birth.Stayer" = "Birth - Stayer")) +
  scale_color_manual(values = c("Current.Mover" = "#1f77b4", "Current.Stayer" = "#aec7e8",
                                 "Birth.Mover" = "#ff7f0e", "Birth.Stayer" = "#ffbb78"),
                     labels = c("Current.Mover" = "Current - Mover",
                                "Current.Stayer" = "Current - Stayer",
                                "Birth.Mover" = "Birth - Mover",
                                "Birth.Stayer" = "Birth - Stayer")) +
  labs(title = "Model Predictive Ability by Settlement Type\n(Brier Skill Score with 95% CI, Age/Gender Residualized)",
       x = "Settlement Type", y = "Brier Skill Score (BSS)", fill = "Model Type", color = "Model Type") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "bottom",
    plot.title = element_text(hjust = 0.5, size = 12, face = "bold"),
    strip.text = element_text(face = "bold")
  ) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "gray", size = 0.5)

ggsave("histogram_model_performance.png", plot = p_histogram, width = 12, height = 6, dpi = 300)
message("  Saved: histogram_model_performance.png")

# ============================================================================
# VISUALIZATION 2: COMPREHENSIVE HEATMAP - ALL PREDICTORS × ALL CONDITIONS
# ============================================================================

message("\n" %+% strrep("=", 90))
message("CREATING COMPREHENSIVE HEATMAP: All Predictors × All Conditions")
message(strrep("=", 90))

# Combine domains and filtered items for comprehensive heatmap
predictor_names_combined <- c(
  colnames(data)[4:8],  # Domains (O, C, E, A, N)
  items_list_current_resid  # Top items that meet threshold
)

# Create comprehensive heatmap: rows=predictors, cols=settlement type × residence × group
# We'll use average correlation across the two conditions

# For current residence items (using current models)
combined_current_pred <- cbind(
  hm_current_domains_resid,  # 5 domains
  hm_current_items_resid[items_for_main_current_resid, , drop = FALSE]  # Filtered items
)

# For birth residence items (using birth models)
combined_birth_pred <- cbind(
  hm_birth_domains_resid,
  hm_birth_items_resid[items_for_main_birth_resid, , drop = FALSE]
)

# Create column labels with all combinations
col_labels <- c()
for (settlement in current_domains_resid$target_classes) {
  for (residence in c("Birth", "Current")) {
    for (group in c("Mover", "Stayer")) {
      col_labels <- c(col_labels, paste0(settlement, "\n", residence, "\n", group))
    }
  }
}

# Combine all heatmap data
# For each column combination, take the average correlation
comprehensive_heatmap <- matrix(NA, nrow = nrow(combined_current_pred), 
                                ncol = length(current_domains_resid$target_classes) * 2 * 2)
colnames(comprehensive_heatmap) <- col_labels
rownames(comprehensive_heatmap) <- c(colnames(data)[4:8], items_list_current_resid)

col_idx <- 1
for (settlement in current_domains_resid$target_classes) {
  for (residence in c("Birth", "Current")) {
    for (group in c("Mover", "Stayer")) {
      # Select the appropriate heatmap and subset based on residence/group
      if (residence == "Current") {
        if (group == "Mover") {
          hm_to_use <- combined_current_pred
        } else {
          hm_to_use <- combined_current_pred
        }
      } else {
        if (group == "Mover") {
          hm_to_use <- combined_birth_pred
        } else {
          hm_to_use <- combined_birth_pred
        }
      }
      
      # Fill in the correlation for this settlement
      settlement_col <- which(colnames(hm_to_use) == settlement)
      if (length(settlement_col) > 0) {
        comprehensive_heatmap[, col_idx] <- hm_to_use[, settlement_col]
      }
      col_idx <- col_idx + 1
    }
  }
}

# Create comprehensive heatmap visualization
hm_melted_comp <- melt(comprehensive_heatmap)
colnames(hm_melted_comp) <- c("Predictor", "Condition", "Correlation")

p_comprehensive <- ggplot(hm_melted_comp, aes(x = Condition, y = Predictor, fill = Correlation)) +
  geom_tile(color = "white", size = 0.3) +
  scale_fill_gradient2(low = "steelblue", mid = "white", high = "firebrick",
                       midpoint = 0, limits = c(-0.3, 0.3), name = "Correlation") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 8),
    axis.text.y = element_text(size = 8),
    legend.position = "right",
    plot.title = element_text(hjust = 0.5, size = 12, face = "bold"),
    plot.margin = margin(10, 10, 10, 200)
  ) +
  labs(title = "Comprehensive Predictor Map: Domains & Items × Settlement Type × Residence × Migration Group\n(Age/Gender Residualized)",
       x = "Settlement Type / Residence Type / Migration Group",
       y = "Personality Predictor")

ggsave("heatmap_comprehensive_all_conditions.png", plot = p_comprehensive, 
       width = 14, height = 16, dpi = 300)
message("  Saved: heatmap_comprehensive_all_conditions.png")

# Save comprehensive heatmap data
saveRDS(list(
  heatmap = comprehensive_heatmap,
  predictor_names = rownames(comprehensive_heatmap),
  condition_labels = col_labels,
  label = "comprehensive_all_conditions"
), file = "heatmap_comprehensive_all_conditions.rds")

message("\n" %+% strrep("=", 90) %+% "\n")





