setwd("/data/scripts/Ling/Proj5birthCurrentPrediction/")


# Packages
library(mlr3)
library(mlr3learners)
library(mlr3pipelines)
library(mlr3measures)
library(tidyverse)


# Data preparation
NP_resi = read.csv("NP_resi.csv") |> rename_with(~ sub("\\.x$", "", .x))|> 
  select(-neuroticism23R, -neuroticism24, -neuroticism31, -neuroticism32R, -neuroticism41, -neuroticism42, -neuroticism43R, -neuroticism52, -others12, -others14) # exclude items not on the personality domains
data <- NP_resi  |> select(1:217) 
data$gender = as.factor(data$gender)

## Filter for movers and stayers to avoid overlaps in the outcome categorization
data_movers <- data |> filter(currentResidenceType2 != birthResidenceType)
data_stayers <- data |> filter(currentResidenceType2 == birthResidenceType)


# MODEL 1: Elastic net binary logistic regression model: original item scores
model1 <- function(df, pred_cols, outcome_var, label) {
  
  # 1. Identify the distinct categories
  categories <- sort(unique(na.omit(df[[outcome_var]])))
  
  all_results <- list()
  
  for (target in categories) {
    message(paste0("Running model for category: ", target))
    
    # 2. Transform the data into Binary (Target vs. Rest)
    # We select ONLY predictors and the outcome
    temp_data <- df %>%
      select(all_of(c(pred_cols, outcome_var))) %>%
      drop_na(!!sym(outcome_var)) %>%
      mutate(
        binary_outcome = factor(ifelse(!!sym(outcome_var) == target, target, "Rest"),
                                levels = c(target, "Rest")) # Target is positive class
      ) %>%
      select(-!!sym(outcome_var))
    
    # 3. Create the Task with Stratification
    task <- as_task_classif(temp_data, target = "binary_outcome", id = paste0(label, "_", target))
    task$col_roles$stratum <- "binary_outcome" 
    
    # 4. Define the Pipeline (Impute -> Encode -> Elastic Net)
    graph <- po("imputemean") %>>% 
      po("imputeoor") %>>% 
      po("encode", method = "treatment") %>>% 
      lrn("classif.glmnet", alpha = 0.5, predict_type = "prob")
    
    glmn_lrn <- as_learner(graph)
    glmn_lrn$id <- paste0("ElasticNet_", target)
    
    # Define Featureless Baseline for this specific target
    baseline_lrn <- lrn("classif.featureless", predict_type = "prob")
    baseline_lrn$id <- paste0("Featureless_", target)
    
    # 5. Benchmarking (Stratified CV)
    resampling <- rsmp("cv", folds = 5)
    measures <- list(
      msr("classif.auc"),
      msr("classif.prauc"),
      msr("classif.bbrier") # This is your MSER
    )
    
    design <- benchmark_grid(
      tasks = task,
      learners = list(glmn_lrn, baseline_lrn),
      resamplings = resampling
    )
    
    bmr <- benchmark(design)
    
    # 6. Get Coefficients from Full Dataset (No CV)
    glmn_lrn$train(task)
    
    # Extract coefficients for the specific target model
    raw_fit <- glmn_lrn$model$classif.glmnet$model
    coefs <- as.matrix(coef(raw_fit, s = min(raw_fit$lambda)))
    
    coef_df <- tibble(
      variable = rownames(coefs),
      coefficient = as.numeric(coefs),
      target_class = target
    )
    
    # 7. Collect Results
    all_results[[target]] <- list(
      metrics = bmr$aggregate(measures),
      coefficients = coef_df,
      prevalence = mean(temp_data$binary_outcome == target)
    )
  }
  
  return(all_results)
}


# Run model1 on domains and items

birth_item <- model1(
   df = data_movers,
   pred_cols = colnames(data)[30:217],
   outcome_var = "birthResidenceType",
   label = "birth_items"
 )
saveRDS(birth_item, file = "birth_item.rds")

current_item <- model1(
  df = data_movers,
  pred_cols = colnames(data)[30:217],
  outcome_var = "currentResidenceType2",
  label = "current_items"
)
saveRDS(current_item, file = "current_item.rds")

birth_domain <- model1(
  df = data_movers,
  pred_cols = colnames(data)[4:8],
  outcome_var = "birthResidenceType",
  label = "birth_domains"
)
saveRDS(birth_domain, file = "birth_domain.rds")

current_domain <- model1(
  df = data_movers,
  pred_cols = colnames(data)[4:8],
  outcome_var = "currentResidenceType2",
  label = "current_domains"
)
saveRDS(current_domain, file = "current_domain.rds")

stayer_item <- model1(
  df = data_stayers,
  pred_cols = colnames(data)[30:217],
  outcome_var = "currentResidenceType2",
  label = "stayer_items"
  )
saveRDS(stayer_item, file = "stayer_item.rds")

  
stayer_domain <- model1(
  df = data_stayers,
  pred_cols = colnames(data)[4:8],
  outcome_var = "currentResidenceType2",
  label = "stayer_domains"
)
saveRDS(stayer_domain, file = "stayer_domain.rds")



# MODEL 2: Residual gender and age from the models
model2 <- function(df, pred_cols, outcome_var, label) {
  
  categories <- sort(unique(na.omit(df[[outcome_var]])))
  all_results <- list()
  
  for (target in categories) {
    message(paste0("Running residualized model for category: ", target))
    
    # 1. Prepare Data
    temp_data <- df %>%
      select(all_of(c(pred_cols, "gender", "ageAtAgreement", outcome_var))) %>%
      drop_na(!!sym(outcome_var), gender, ageAtAgreement) %>%
      mutate(
        binary_outcome = factor(ifelse(!!sym(outcome_var) == target, target, "Rest"),
                                levels = c(target, "Rest"))
      ) %>%
      select(-!!sym(outcome_var))
    
    # 2. Create the Task
    task <- as_task_classif(temp_data, target = "binary_outcome", id = paste0(label, "_", target))
    task$col_roles$stratum <- "binary_outcome"
    
    # 3. Create RHS-only formulas for po("mutate")
    # Format: list(col_name = ~ as.numeric(residuals(lm(col_name ~ gender + ageAtAgreement))))
    mutation_list <- list()
    for (col in pred_cols) {
      # The "~" at the start makes it a RHS formula with no Left Hand Side
      mutation_list[[col]] <- as.formula(
        paste0("~ as.numeric(residuals(lm(", col, " ~ gender + ageAtAgreement)))")
      )
    }
    
    # 4. Define the Pipeline
    graph <- po("imputemean") %>>% 
      po("imputeoor") %>>% 
      po("mutate", mutation = mutation_list) %>>%
      po("select", selector = selector_name(pred_cols)) %>>% 
      po("encode", method = "treatment") %>>% 
      lrn("classif.glmnet", alpha = 0.5, predict_type = "prob")
    
    glmn_lrn <- as_learner(graph)
    glmn_lrn$id <- paste0("ElasticNet_Resid_", target)
    
    # Featureless Baseline
    baseline_lrn <- lrn("classif.featureless", predict_type = "prob")
    baseline_lrn$id <- paste0("Featureless_", target)
    
    # 5. Benchmarking
    resampling <- rsmp("cv", folds = 5)
    measures <- list(
      msr("classif.auc"), 
      msr("classif.prauc"), 
      msr("classif.bbrier")
    )
    
    design <- benchmark_grid(tasks = task, learners = list(glmn_lrn, baseline_lrn), resamplings = resampling)
    bmr <- benchmark(design)
    
    # 6. Final Coefficients (Full Dataset)
    glmn_lrn$train(task)
    raw_fit <- glmn_lrn$model$classif.glmnet$model
    coefs <- as.matrix(coef(raw_fit, s = min(raw_fit$lambda)))
    
    coef_df <- tibble(
      variable = rownames(coefs),
      coefficient = as.numeric(coefs),
      target_class = target
    )
    
    # 7. Calculate prevalence of the specific binary target
    prev_val <- mean(temp_data$binary_outcome == target)
    
    all_results[[target]] <- list(
      metrics = bmr$aggregate(measures),
      coefficients = coef_df,
      prevalence = prev_val
    )
  }
  
  return(all_results)
}

# Run model2 on domains and items

birth_item2 <- model2(
  df = data_movers,
  pred_cols = colnames(data)[30:217],
  outcome_var = "birthResidenceType",
  label = "birth_items2"
)
saveRDS(birth_item2, file = "birth_item2.rds")

current_item2 <- model2(
  df = data_movers,
  pred_cols = colnames(data)[30:217],
  outcome_var = "currentResidenceType2",
  label = "current_items2"
)
saveRDS(current_item2, file = "current_item2.rds")

birth_domain2 <- model2(
  df = data_movers,
  pred_cols = colnames(data)[4:8],
  outcome_var = "birthResidenceType",
  label = "birth_domains2"
)
saveRDS(birth_domain2, file = "birth_domain2.rds")

current_domain2 <- model2(
  df = data_movers,
  pred_cols = colnames(data)[4:8],
  outcome_var = "currentResidenceType2",
  label = "current_domains2"
)
saveRDS(current_domain2, file = "current_domain2.rds")

stayer_item2 <- model2(
  df = data_stayers,
  pred_cols = colnames(data)[30:217],
  outcome_var = "currentResidenceType2",
  label = "stayer_items2"
)
saveRDS(stayer_item2, file = "stayer_item2.rds")


stayer_domain2 <- model2(
  df = data_stayers,
  pred_cols = colnames(data)[4:8],
  outcome_var = "currentResidenceType2",
  label = "stayer_domains2"
)
saveRDS(stayer_domain2, file = "stayer_domain2.rds")





