# =============================================================================
# FIGURES FOR THE BIRTH VERSUS CURRENT RESIDENCE ANALYSIS
# =============================================================================
# Run "Basic Birth Current Analysis.R" first. This script reads its saved CSV
# files and creates figures only; it never refits a model.

library(tidyverse)

PROJECT_DIR <- "/data/scripts/Ling/Proj5BirthCurrentPrediction/"
OUTER_FOLDS <- 5
setwd(PROJECT_DIR)

# Adjustable heatmap settings.
TOP_ITEMS_PER_CONDITION <- 20
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
required_coefficient_columns <- c(
  "Model", "Sample", "Outcome", "Predictor", "Class", "Median_Coefficient"
)
missing_coefficient_columns <- setdiff(
  required_coefficient_columns, names(coefficient_summary)
)
if (length(missing_coefficient_columns)) {
  stop(
    "The coefficient summary is missing required columns: ",
    paste(missing_coefficient_columns, collapse = ", ")
  )
}

location_levels <- c("Village", "Town", "City", "Abroad")
heatmap_condition_levels <- unlist(lapply(location_levels, function(location) {
  paste(location, c("Birth", "Current"), sep = "\n")
}))

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

# Whole-sample and movers-only figures place domain and item models together in
# one grouped bar panel for direct comparison within each settlement category.
make_sample_bss_plot <- function(sample_name, title, filename) {
  plot_data <- bss_figure_data %>%
    filter(Sample == sample_name) %>%
    mutate(
      Display_Label = if_else(
        Model == "Items",
        sprintf("%.3f\n[%.3f, %.3f]", Mean_BSS, CI_Lower, CI_Upper),
        sprintf("%.4f\n[%.4f, %.4f]", Mean_BSS, CI_Lower, CI_Upper)
      ),
      Category_X = as.numeric(Category)
    )

  plot_range <- diff(range(
    c(plot_data$CI_Lower, plot_data$CI_Upper),
    na.rm = TRUE
  ))
  if (!is.finite(plot_range) || plot_range == 0) plot_range <- 0.01
  label_offset <- 0.035 * plot_range

  plot_data <- plot_data %>%
    mutate(
      Label_Y = if_else(
        Mean_BSS >= 0,
        CI_Upper + label_offset,
        CI_Lower - label_offset
      ),
      Label_VJust = if_else(Mean_BSS >= 0, 0, 1)
    )

  marker_data <- significance_markers %>%
    filter(Sample == sample_name) %>%
    select(-Y, -Top, -Marker_Order) %>%
    mutate(
      Plot_Model = recode(
        Model,
        "Big Five domains" = "Domains",
        "Personality items" = "Items"
      )
    ) %>%
    left_join(
      plot_data %>%
        group_by(Model, Sample, Category) %>%
        summarize(
          Top = max(CI_Upper, Mean_BSS, na.rm = TRUE),
          .groups = "drop"
        ),
      by = c("Plot_Model" = "Model", "Sample", "Category")
    ) %>%
    mutate(
      # With four bars dodged across width 0.90, the pair midpoints are
      # 0.225 category units to the left/right of the category center.
      Marker_X = as.numeric(Category) + if_else(
        Plot_Model == "Domains", -0.225, 0.225
      ),
      Y = Top + 0.07 * plot_range
    )

  plot <- ggplot(
    plot_data,
    aes(x = Category_X, y = Mean_BSS, fill = Series)
  ) +
    geom_col(
      position = position_dodge(width = 0.90),
      width = 0.82,
      color = "grey25",
      linewidth = 0.18
    ) +
    geom_errorbar(
      aes(ymin = CI_Lower, ymax = CI_Upper),
      position = position_dodge(width = 0.90),
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
      position = position_dodge(width = 0.90),
      angle = 0,
      size = 2.2,
      lineheight = 0.9,
      show.legend = FALSE
    ) +
    geom_text(
      data = marker_data,
      aes(
        x = Marker_X, y = Y, label = Marker
      ),
      inherit.aes = FALSE,
      fontface = "bold",
      size = 4
    ) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey35") +
    scale_fill_manual(values = c(
      "Domains Birth" = "#8C8C8C",
      "Domains Current" = "#000000",
      "Items Birth" = "#7FB8E6",
      "Items Current" = "#0072CE"
    )) +
    scale_x_continuous(
      breaks = seq_along(levels(plot_data$Category)),
      labels = levels(plot_data$Category),
      expand = expansion(add = 0.55)
    ) +
    labs(
      title = title,
      x = NULL,
      y = "Brier Skill Score",
      fill = NULL
    ) +
    theme_minimal(base_size = 11) +
    theme(
      panel.grid.major.x = element_blank(),
      axis.text.x = element_text(angle = 0, hjust = 0.5),
      strip.text = element_text(face = "bold"),
      legend.position = "bottom",
      plot.title = element_text(face = "bold", hjust = 0.5)
    )

  ggsave(filename, plot, width = 16, height = 9, dpi = 300)
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
# TABLE: BIRTH-CURRENT COEFFICIENT CONSISTENCY
# =============================================================================

# Correlate matched birth and current coefficient profiles. Settlement-specific
# rows correlate predictors within one settlement; overall rows pool the matched
# predictor-settlement coefficients across all four settlements.
paired_coefficients <- coefficient_summary %>%
  select(Model, Sample, Predictor, Class, Outcome, Median_Coefficient) %>%
  group_by(Model, Sample, Predictor, Class, Outcome) %>%
  summarize(Coefficient = mean(Median_Coefficient), .groups = "drop") %>%
  pivot_wider(names_from = Outcome, values_from = Coefficient) %>%
  filter(is.finite(Birth), is.finite(Current))

safe_correlation <- function(birth, current, method) {
  complete <- is.finite(birth) & is.finite(current)
  birth <- birth[complete]
  current <- current[complete]

  if (length(birth) < 3 || sd(birth) == 0 || sd(current) == 0) {
    return(tibble(Estimate = NA_real_, p = NA_real_))
  }

  test <- suppressWarnings(cor.test(
    birth,
    current,
    method = method,
    exact = FALSE
  ))
  tibble(Estimate = unname(test$estimate), p = test$p.value)
}

summarize_coefficient_consistency <- function(data) {
  pearson <- safe_correlation(data$Birth, data$Current, "pearson")
  spearman <- safe_correlation(data$Birth, data$Current, "spearman")

  tibble(
    Matched_Coefficients = nrow(data),
    Pearson_r = pearson$Estimate,
    Pearson_p = pearson$p,
    Spearman_rho = spearman$Estimate,
    Spearman_p = spearman$p
  )
}

settlement_coefficient_consistency <- paired_coefficients %>%
  group_by(Model, Sample, Class) %>%
  group_modify(~ summarize_coefficient_consistency(.x)) %>%
  ungroup() %>%
  rename(Settlement = Class)

overall_coefficient_consistency <- paired_coefficients %>%
  group_by(Model, Sample) %>%
  group_modify(~ summarize_coefficient_consistency(.x)) %>%
  ungroup() %>%
  mutate(Settlement = "Overall pooled")

coefficient_consistency_table <- bind_rows(
  settlement_coefficient_consistency,
  overall_coefficient_consistency
) %>%
  group_by(Model, Sample) %>%
  mutate(
    Pearson_p_Holm = p.adjust(Pearson_p, method = "holm"),
    Spearman_p_Holm = p.adjust(Spearman_p, method = "holm")
  ) %>%
  ungroup() %>%
  mutate(
    Settlement = factor(
      Settlement,
      levels = c(location_levels, "Overall pooled")
    )
  ) %>%
  arrange(Model, Sample, Settlement)

print(coefficient_consistency_table)
write.csv(
  coefficient_consistency_table,
  "coefficient_birth_current_consistency.csv",
  row.names = FALSE
)

# =============================================================================
# FIGURE 2: DOMAIN AND CONDITION-SPECIFIC TOP-20 ITEM COEFFICIENT HEATMAP
# =============================================================================

coefficient_heatmap_data <- coefficient_summary %>%
  mutate(
    Predictor_Set = recode(
      Model,
      "Big Five domains" = "Domains",
      "Personality items" = "Items"
    ),
    Class = factor(Class, levels = location_levels),
    Outcome = factor(Outcome, levels = c("Birth", "Current")),
    Condition = factor(
      paste(Class, Outcome, sep = "\n"),
      levels = heatmap_condition_levels
    )
  )

# Rank items independently within every sample, outcome, and settlement cell.
# Therefore, rank 1 can refer to a different item in every heatmap column.
top_item_coefficients <- coefficient_heatmap_data %>%
  filter(Predictor_Set == "Items") %>%
  group_by(Sample, Outcome, Class) %>%
  arrange(desc(abs(Median_Coefficient)), Predictor, .by_group = TRUE) %>%
  slice_head(n = TOP_ITEMS_PER_CONDITION) %>%
  mutate(
    Item_Rank = row_number(),
    Heatmap_Row = sprintf("Item rank %02d", Item_Rank),
    Cell_Label = Predictor
  ) %>%
  ungroup()

make_coefficient_heatmap <- function(sample_name, title, filename) {
  domain_data <- coefficient_heatmap_data %>%
    filter(Predictor_Set == "Domains", Sample == sample_name) %>%
    mutate(
      Heatmap_Row = Predictor,
      Cell_Label = ""
    )

  item_data <- top_item_coefficients %>%
    filter(Sample == sample_name)

  domain_order <- domain_data %>%
    distinct(Predictor) %>%
    pull(Predictor)
  item_rank_order <- sprintf(
    "Item rank %02d",
    seq_len(TOP_ITEMS_PER_CONDITION)
  )
  heatmap_row_order <- c(domain_order, item_rank_order)

  plot_data <- bind_rows(domain_data, item_data) %>%
    mutate(
      Predictor_Set = factor(Predictor_Set, levels = c("Domains", "Items")),
      Heatmap_Row = factor(Heatmap_Row, levels = rev(heatmap_row_order)),
      Plot_Coefficient = Median_Coefficient,
      Text_Color = if_else(
        abs(Plot_Coefficient) >= 0.15,
        "white",
        "black"
      )
    )

  plot <- ggplot(
    plot_data,
    aes(x = Condition, y = Heatmap_Row, fill = Plot_Coefficient)
  ) +
    geom_tile(color = "white", linewidth = 0.25) +
    geom_vline(
      xintercept = c(2.5, 4.5, 6.5),
      color = "grey45",
      linewidth = 0.45
    ) +
    geom_text(
      data = plot_data %>% filter(Predictor_Set == "Items"),
      aes(label = Cell_Label, color = Text_Color),
      size = 2.15,
      fontface = "plain",
      show.legend = FALSE
    ) +
    facet_grid(
      rows = vars(Predictor_Set),
      scales = "free_y",
      space = "free_y",
      drop = TRUE
    ) +
    scale_fill_gradient2(
      low = "#3B4CC0",
      mid = "white",
      high = "#B40426",
      midpoint = 0,
      limits = c(-COEFFICIENT_COLOR_LIMIT, COEFFICIENT_COLOR_LIMIT),
      oob = scales::squish,
      name = "Median\ncoefficient"
    ) +
    scale_color_identity() +
    scale_y_discrete(
      labels = function(labels) {
        if_else(str_detect(labels, "^Item rank"), "", labels)
      }
    ) +
    labs(
      title = title,
      subtitle = paste0(
        "The ", TOP_ITEMS_PER_CONDITION,
        " largest absolute item coefficients are ranked separately ",
        "within every settlement and outcome."
      ),
      x = "Settlement and outcome",
      y = NULL
    ) +
    theme_minimal(base_size = 9) +
    theme(
      panel.grid = element_blank(),
      axis.text.x = element_text(angle = 0, hjust = 0.5),
      axis.text.y = element_text(size = 7),
      strip.text = element_text(face = "bold"),
      plot.title = element_text(face = "bold", hjust = 0.5),
      plot.subtitle = element_text(hjust = 0.5),
      legend.position = "right"
    )

  heatmap_height <- max(10, 5.5 + 0.32 * TOP_ITEMS_PER_CONDITION)
  ggsave(
    filename,
    plot,
    width = 16,
    height = heatmap_height,
    dpi = 300,
    limitsize = FALSE
  )

  list(plot = plot, data = plot_data)
}

coefficient_heatmap_whole <- make_coefficient_heatmap(
  "All participants",
  "Personality Coefficients: Whole Sample",
  "figure_coefficient_heatmap_whole_sample.png"
)

coefficient_heatmap_movers <- make_coefficient_heatmap(
  "Movers only",
  "Personality Coefficients: Movers Only",
  "figure_coefficient_heatmap_movers_only.png"
)

coefficient_heatmap_filtered <- bind_rows(
  coefficient_heatmap_whole$data,
  coefficient_heatmap_movers$data
)

write.csv(
  coefficient_heatmap_filtered,
  "figure_coefficient_heatmap_data.csv",
  row.names = FALSE
)
write.csv(
  top_item_coefficients,
  "figure_coefficient_heatmap_top20_items.csv",
  row.names = FALSE
)

message(
  paste0(
    "Saved separate whole-sample and movers-only BSS and coefficient figures, ",
    "plus the birth-current coefficient-consistency table."
  )
)
