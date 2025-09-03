library(effsize)

# merge_metrics_data <- function(base_string) {
#   path_bleu4 <- paste(base_string, "bleu-4.csv", sep = "_")
#   path_meteor <- paste(base_string, "meteor.csv", sep = "_")
#   path_rouge <- paste(base_string, "rouge-l.csv", sep = "_")
  
#   bleu4 <- read.csv(path_bleu4)
#   meteor <- read.csv(path_meteor)
#   rouge <- read.csv(path_rouge)
  
#   merged_data <- merge(bleu4, meteor, by = "Instance")
#   merged_data <- merge(merged_data, rouge, by = "Instance")
  
#   return(merged_data)
# }
merge_metrics_data <- function(folder) {
  base <- basename(folder)

  path_bleu <- file.path(folder, paste0(base, "_bleu-4.csv"))
  path_rouge <- file.path(folder, paste0(base, "_rouge-l.csv"))
  path_meteor <- file.path(folder, paste0(base, "_meteor.csv"))
  path_chrf <- file.path(folder, paste0(base, "_chrf.csv"))

  print(paste("Reading BLEU from:", path_bleu))
  print(paste("Reading ROUGE from:", path_rouge))
  print(paste("Reading METEOR from:", path_meteor))
  print(paste("Reading ChrF from:", path_chrf))

  bleu <- read.csv(path_bleu)
  rouge <- read.csv(path_rouge)
  meteor <- read.csv(path_meteor)
  chrf <- read.csv(path_chrf)

  merged <- merge(bleu, rouge, by = "Instance")
  merged <- merge(merged, meteor, by = "Instance")
  merged <- merge(merged, chrf, by = "Instance")

  colnames(merged) <- c("Instance", "BLEU", "ROUGE", "METEOR", "ChrF")
  return(merged)
}

args <- commandArgs(trailingOnly = TRUE)
print("Command-line arguments:")
print(args)

if (length(args) != 2) {
  stop("Please provide exactly two folder paths: BASE and COMPARISON.")
}

# data1 <- merge_metrics_data(args[1])
# data2 <- merge_metrics_data(args[2])
base_data <- merge_metrics_data(args[1])
comp_data <- merge_metrics_data(args[2])

p_values <- vector("numeric", length = 3)
cliffs_deltas <- vector("numeric", length = 3)

# print("===============BLEU-4===============")
# wilcox_test_result <- wilcox.test(data1$BLEU.4, data2$BLEU.4, paired = TRUE)
# p_values[1] <- wilcox_test_result$p.value
# print(wilcox_test_result)

# cliffs_delta_result <- cliff.delta(data1$BLEU.4, data2$BLEU.4, paired = TRUE)
# cliffs_deltas[1] <- cliffs_delta_result$estimate
# print(cliffs_delta_result)

# print("===============METEOR===============")
# wilcox_test_result <- wilcox.test(data1$METEOR, data2$METEOR, paired = TRUE)
# p_values[2] <- wilcox_test_result$p.value
# print(wilcox_test_result)

# cliffs_delta_result <- cliff.delta(data1$METEOR, data2$METEOR, paired = TRUE)
# cliffs_deltas[2] <- cliffs_delta_result$estimate
# print(cliffs_delta_result)

# print("===============ROUGE-L===============")
# wilcox_test_result <- wilcox.test(data1$ROUGE.L, data2$ROUGE.L, paired = TRUE)
# p_values[3] <- wilcox_test_result$p.value
# print(wilcox_test_result)

# cliffs_delta_result <- cliff.delta(data1$ROUGE.L, data2$ROUGE.L, paired = TRUE)
# cliffs_deltas[3] <- cliffs_delta_result$estimate
# print(cliffs_delta_result)

metrics <- c("BLEU", "ROUGE", "METEOR", "ChrF")
p_values <- numeric(length(metrics))
cliffs_deltas <- numeric(length(metrics))

for (i in seq_along(metrics)) {
  metric <- metrics[i]
  cat("\n=============== ", metric, " ===============\n")

  #w_test <- wilcox.test(base_data[[metric]], comp_data[[metric]], paired = TRUE)
  w_test <- wilcox.test(comp_data[[metric]], base_data[[metric]], paired = TRUE)
  p_values[i] <- w_test$p.value
  print(w_test)

  #cd_result <- cliff.delta(base_data[[metric]], comp_data[[metric]], paired = TRUE)
  cd_result <- cliff.delta(comp_data[[metric]], base_data[[metric]], paired = TRUE)
  cliffs_deltas[i] <- cd_result$estimate
  print(cd_result)
}

# adjusted_p_values <- p.adjust(p_values, method = "holm")
# print("Adjusted p-values for multiple comparisons:")
# print(adjusted_p_values)

# print("Cliff's deltas:")
# print(cliffs_deltas[1])
# print(cliffs_deltas[2])
# print(cliffs_deltas[3])

# results <- data.frame(
#   Metric = c("BLEU-4", "METEOR", "ROUGE-L"),
#   P_Value = p_values,
#   Adjusted_P_Value = adjusted_p_values,
#   Cliffs_Delta = cliffs_deltas
# )

# output_file <- paste("w_t_results", gsub("/", "_", args[1]), "vs", gsub("/", "_", args[2]), ".csv", sep = "_")
# write.csv(results, output_file, row.names = FALSE)
adjusted_p <- p.adjust(p_values, method = "holm")
cat("\n=== Adjusted P-values (Holm) ===\n")
print(adjusted_p)
cat("\n=== Cliff's Deltas ===\n")
print(cliffs_deltas)

results <- data.frame(
  Metric = metrics,
  P_Value = p_values,
  Adjusted_P_Value = adjusted_p,
  Cliffs_Delta = cliffs_deltas
)

output_file <- paste0("results_", basename(args[1]), "_vs_", basename(args[2]), ".csv")
write.csv(results, output_file, row.names = FALSE)
cat(paste0("\n[Saved] Statistical results written to ", output_file, "\n"))
