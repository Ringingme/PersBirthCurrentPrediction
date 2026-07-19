# =============================================================================
# FIGURES FOR THE BIRTH VERSUS CURRENT RESIDENCE ANALYSIS
# =============================================================================
# Run "Basic Birth Current Analysis.R" first. This script reads its saved CSV
# files and creates figures only; it never refits a model.

library(tidyverse)

PROJECT_DIR <- "/data/scripts/Ling/Proj5BirthCurrentPrediction/"
OUTER_FOLDS <- 5
setwd(PROJECT_DIR)

# Adjustable heatmap thresholds.
ITEM_COEFFICIENT_THRESHOLD <- 0.01
ITEM_SELECTION_FREQUENCY_THRESHOLD <- 0.50
ITEM_MIN_OCCURRENCES <- 3
COEFFICIENT_COLOR_LIMIT <- 0.30

# ---- Read analysis outputs --------------------------------------------------

settlement_bss_results <- read.csv("basic_settlement_bss_fold_results.csv")
settlement_bss_tests <- read.csv("basic_settlement_bss_corrected_tests.csv")
coefficient_summary <- read.csv("basic_birth_current_coefficient_summary.csv")

required_files_data <- list(
  settlement_bss_results = settlement_bss_results,
  settlement_bss_tests = settlement_bss_tests,
  coefficient_summary = coefficient_summary
)
if (any(vapply(required_files_data, nrow, integer(1)) == 0)) {
  stop("At least one required analysis output is empty")
}
required_settlement_columns <- c(
  "Model", "Sample", "Outcome", "Repeat", "Fold", "Settlement", "BSS"
)
missing_settlement_columns <- setdiff(
  required_settlement_columns, names(settlement_bss_results)
)
if (length(missing_settlement_columns)) {
  stop(
    "The settlement results are missing required columns: ",
    paste(missing_settlement_columns, collapse = ", ")
  )
}

location_levels <- c("Village", "Town", "City", "Abroad")

# =============================================================================
# FIGURE 1: OVERALL AND SETTLEMENT-SPECIFIC BSS
# =============================================================================

corrected_mean_ci <- function(scores, k = OUTER_FOLDS) {
  scores <- scores[is.finite(scores)]
  n <- length(scores)
  if (n < 2) {
    return(tibble(
      Mean_BSS = NA_real_, Corrected_SE = NA_real_,
      CI_Lower = NA_real_, CI_Upper = NA_real_
    ))
  }

  corrected_se <- sqrt((1 / n + 1 / (k - 1)) * var(scores))
  critical_value <- qt(0.975, df = n - 1)
  tibble(
    Mean_BSS = mean(scores),
    Corrected_SE = corrected_se,
    CI_Lower = mean(scores) - critical_value * corrected_se,
    CI_Upper = mean(scores) + critical_value * corrected_se
  )
}

# This uses the already saved one-versus-rest settlement BSS values. It is a
# macro-average of Village, Town, and City, not a refitted three-class BSS.
non_abroad_macro_bss <- settlement_bss_results %>%
  filter(Settlement != "Abroad") %>%
  group_by(Model, Sample, Outcome, Repeat, Fold) %>%
  summarize(BSS = mean(BSS), .groups = "drop") %>%
  mutate(Category = "Overall excluding Abroad")

corrected_macro_test <- function(results, k = OUTER_FOLDS) {
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

non_abroad_macro_test <- corrected_macro_test(non_abroad_macro_bss)
write.csv(
  non_abroad_macro_test,
  "figure_macro_bss_excluding_abroad_corrected_test.csv",
  row.names = FALSE
)

trained_bss <- bind_rows(
  non_abroad_macro_bss %>%
    select(Model, Sample, Outcome, Repeat, Fold, Category, BSS),
  settlement_bss_results %>%
    transmute(Model, Sample, Outcome, Repeat, Fold,
              Category = Settlement, BSS)
)

bss_figure_data <- trained_bss %>%
  group_by(Model, Sample, Outcome, Category) %>%
  group_modify(~ corrected_mean_ci(.x$BSS)) %>%
  ungroup() %>%
  mutate(
    Category = factor(
      Category,
      levels = c("Overall excluding Abroad", location_levels)
    ),
    Model = recode(
      Model,
      "Big Five domains" = "Domains",
      "Personality items" = "Items"
    ),
    Series = factor(
      paste(Model, Outcome),
      levels = c(
        "Domains Birth", "Domains Current",
        "Items Birth", "Items Current"
      )
    ),
    CI_Label = sprintf(
      "%.4f\n[%.4f, %.4f]",
      Mean_BSS, CI_Lower, CI_Upper
    ),
    CI_Label_Y = if_else(Mean_BSS >= 0, CI_Upper, CI_Lower),
    CI_Label_VJust = if_else(Mean_BSS >= 0, -0.15, 1.15)
  )

# Overall markers use the corrected paired p-value. Settlement markers use the
# Holm-adjusted value across the four settlement tests in each model/sample.
significance_markers <- bind_rows(
  non_abroad_macro_test %>%
    transmute(
      Model, Sample, Category = "Overall excluding Abroad",
      p_for_marker = p
    ),
  settlement_bss_tests %>%
    transmute(Model, Sample, Category = Settlement, p_for_marker = p_Holm)
) %>%
  filter(p_for_marker < 0.05) %>%
  mutate(
    Predictor_Model = recode(
      Model,
      "Big Five domains" = "Big Five domains",
      "Personality items" = "Personality items"
    ),
    Marker = case_when(
      p_for_marker < 0.001 ~ "***",
      p_for_marker < 0.01 ~ "**",
      TRUE ~ "*"
    ),
    Category = factor(
      Category,
      levels = c("Overall excluding Abroad", location_levels)
    )
  ) %>%
  left_join(
    bss_figure_data %>%
      filter(Model != "Featureless") %>%
      group_by(Sample, Category) %>%
      summarize(Top = max(CI_Upper, Mean_BSS, na.rm = TRUE), .groups = "drop"),
    by = c("Sample", "Category")
  ) %>%
  group_by(Sample, Category) %>%
  arrange(Model) %>%
  mutate(Marker_Order = row_number()) %>%
  ungroup()

bss_range <- diff(range(
  c(bss_figure_data$CI_Lower, bss_figure_data$CI_Upper),
  na.rm = TRUE
))
if (!is.finite(bss_range) || bss_range == 0) bss_range <- 0.02
significance_markers <- significance_markers %>%
  mutate(Y = Top + Marker_Order * 0.06 * bss_range)

write.csv(
  bss_figure_data,
  "figure_bss_overall_settlements_data.csv",
  row.names = FALSE
)

# Whole-sample and movers-only figures each combine domains and items. The
# panels use independent y scales so the smaller domain BSS values remain clear.
make_sample_bss_plot <- function(sample_name, title, filename) {
  plot_data <- bss_figure_data %>%
    filter(Sample == sample_name) %>%
    mutate(
      Predictor_Model = factor(
        recode(
          Model,
          "Domains" = "Big Five domains",
          "Items" = "Personality items"
        ),
        levels = c("Big Five domains", "Personality items")
      ),
      Display_Label = if_else(
        Model == "Items",
        sprintf("%.3f [%.3f, %.3f]", Mean_BSS, CI_Lower, CI_Upper),
        sprintf("%.4f [%.4f, %.4f]", Mean_BSS, CI_Lower, CI_Upper)
      ),
      Label_Y = if_else(Mean_BSS >= 0, CI_Upper, CI_Lower),
      Label_VJust = if_else(Mean_BSS >= 0, -0.15, 1.15)
    )

  panel_ranges <- plot_data %>%
    group_by(Predictor_Model) %>%
    summarize(
      Panel_Range = diff(range(c(CI_Lower, CI_Upper), na.rm = TRUE)),
      Panel_Range = if_else(
        is.finite(Panel_Range) & Panel_Range > 0,
        Panel_Range,
        0.01
      ),
      .groups = "drop"
    )

  marker_data <- significance_markers %>%
    filter(Sample == sample_name) %>%
    select(-Y, -Top, -Marker_Order) %>%
    left_join(
      plot_data %>%
        group_by(Predictor_Model, Sample, Category) %>%
        summarize(
          Top = max(CI_Upper, Mean_BSS, na.rm = TRUE),
          .groups = "drop"
        ),
      by = c("Predictor_Model", "Sample", "Category")
    ) %>%
    left_join(panel_ranges, by = "Predictor_Model") %>%
    mutate(Y = Top + 0.06 * Panel_Range)

  plot <- ggplot(
    plot_data,
    aes(x = Category, y = Mean_BSS, fill = Series)
  ) +
    geom_col(
      position = position_dodge(width = 0.82),
      width = 0.72,
      color = "grey25",
      linewidth = 0.18
    ) +
    geom_errorbar(
      aes(ymin = CI_Lower, ymax = CI_Upper),
      position = position_dodge(width = 0.82),
      width = 0.12,
      linewidth = 0.5
    ) +
    geom_text(
      aes(
        y = Label_Y,
        label = Display_Label,
        group = Series,
        vjust = Label_VJust
      ),
      position = position_dodge(width = 0.82),
      angle = 0,
      size = 2.4,
      show.legend = FALSE
    ) +
    geom_text(
      data = marker_data,
      aes(x = Category, y = Y, label = Marker),
      inherit.aes = FALSE,
      fontface = "bold",
      size = 4
    ) +
    facet_grid(
      rows = vars(Predictor_Model),
      scales = "free_y",
      space = "free_y"
    ) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey35") +
    scale_fill_manual(values = c(
      "Domains Birth" = "#9ECAE1",
      "Domains Current" = "#3182BD",
      "Items Birth" = "#FDD0A2",
      "Items Current" = "#E6550D"
    )) +
    labs(
      title = title,
      subtitle = paste0(
        "BSS = 0 is the featureless reference. ",
        "* p < .05, ** p < .01, *** p < .001; ",
        "settlement p-values are Holm-adjusted."
      ),
      x = NULL,
      y = "Brier Skill Score",
      fill = NULL
    ) +
    theme_minimal(base_size = 11) +
    theme(
      panel.grid.major.x = element_blank(),
      axis.text.x = element_text(angle = 25, hjust = 1),
      strip.text = element_text(face = "bold"),
      legend.position = "bottom",
      plot.title = element_text(face = "bold", hjust = 0.5),
      plot.subtitle = element_text(hjust = 0.5)
    )

  ggsave(filename, plot, width = 13, height = 9, dpi = 300)
  plot
}

p_bss_whole_sample <- make_sample_bss_plot(
  "All participants",
  "Personality Prediction of Birth and Current Residence: Whole Sample",
  "figure_bss_whole_sample.png"
)

p_bss_movers_only <- make_sample_bss_plot(
  "Movers only",
  "Personality Prediction of Birth and Current Residence: Movers Only",
  "figure_bss_movers_only.png"
)

# =============================================================================
# FIGURE 2: DOMAIN AND THRESHOLDED ITEM COEFFICIENT HEATMAP
# =============================================================================

coefficient_heatmap_data <- coefficient_summary %>%
  mutate(
    Predictor_Set = recode(
      Model,
      "Big Five domains" = "Domains",
      "Personality items" = "Items"
    ),
    Passes_Item_Threshold =
      abs(Median_Coefficient) >= ITEM_COEFFICIENT_THRESHOLD &
      Selection_Frequency >= ITEM_SELECTION_FREQUENCY_THRESHOLD
  )

selected_items_table <- coefficient_heatmap_data %>%
  filter(Predictor_Set == "Items") %>%
  group_by(Predictor) %>%
  summarize(
    Occurrences = sum(Passes_Item_Threshold),
    Maximum_Absolute_Coefficient = max(abs(Median_Coefficient)),
    .groups = "drop"
  ) %>%
  filter(Occurrences >= ITEM_MIN_OCCURRENCES) %>%
  arrange(desc(Occurrences), desc(Maximum_Absolute_Coefficient))

selected_items <- selected_items_table$Predictor
if (length(selected_items) == 0) {
  warning(
    "No items passed the heatmap thresholds; reduce one or more item thresholds"
  )
}

domain_predictors <- coefficient_heatmap_data %>%
  filter(Predictor_Set == "Domains") %>%
  distinct(Predictor) %>%
  pull(Predictor)

heatmap_predictor_order <- c(domain_predictors, selected_items)

coefficient_heatmap_filtered <- coefficient_heatmap_data %>%
  filter(Predictor_Set == "Domains" | Predictor %in% selected_items) %>%
  mutate(
    Plot_Coefficient = if_else(
      Predictor_Set == "Items" & !Passes_Item_Threshold,
      NA_real_,
      Median_Coefficient
    ),
    Predictor_Set = factor(Predictor_Set, levels = c("Domains", "Items")),
    Predictor = factor(Predictor, levels = rev(heatmap_predictor_order)),
    Class = factor(Class, levels = location_levels),
    Outcome = factor(Outcome, levels = c("Birth", "Current")),
    Condition = interaction(Outcome, Class, sep = "\n", lex.order = TRUE),
    Sample = factor(Sample, levels = c("All participants", "Movers only"))
  )

p_coefficients <- ggplot(
  coefficient_heatmap_filtered,
  aes(x = Condition, y = Predictor, fill = Plot_Coefficient)
) +
  geom_tile(color = "white", linewidth = 0.25) +
  facet_grid(
    rows = vars(Predictor_Set),
    cols = vars(Sample),
    scales = "free_y",
    space = "free_y",
    drop = FALSE
  ) +
  scale_fill_gradient2(
    low = "#3B4CC0",
    mid = "white",
    high = "#B40426",
    midpoint = 0,
    limits = c(-COEFFICIENT_COLOR_LIMIT, COEFFICIENT_COLOR_LIMIT),
    oob = scales::squish,
    na.value = "white",
    name = "Median\ncoefficient"
  ) +
  labs(
    title = "Personality Coefficients for Birth and Current Residence",
    subtitle = paste0(
      "Items: |median coefficient| >= ", ITEM_COEFFICIENT_THRESHOLD,
      ", selection frequency >= ", ITEM_SELECTION_FREQUENCY_THRESHOLD,
      ", in >= ", ITEM_MIN_OCCURRENCES, " cells"
    ),
    x = "Outcome and settlement type",
    y = NULL
  ) +
  theme_minimal(base_size = 9) +
  theme(
    panel.grid = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1),
    axis.text.y = element_text(size = 7),
    strip.text = element_text(face = "bold"),
    plot.title = element_text(face = "bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5),
    legend.position = "right"
  )

heatmap_height <- max(7, 4.5 + 0.20 * length(selected_items))
ggsave(
  "figure_coefficient_heatmap_domains_items.png",
  p_coefficients,
  width = 15,
  height = heatmap_height,
  dpi = 300,
  limitsize = FALSE
)
write.csv(
  coefficient_heatmap_filtered,
  "figure_coefficient_heatmap_domains_items_data.csv",
  row.names = FALSE
)
write.csv(
  selected_items_table,
  "figure_coefficient_heatmap_selected_items.csv",
  row.names = FALSE
)

message(
  "Saved two BSS figures and one coefficient heatmap."
)
