library(tidyverse)
library(rpart)
library(caret)
library(ggplot2)
library(randomForest)
library(pROC)

# Import data
url <- "https://raw.githubusercontent.com/ccscaiado/MLRepo/refs/heads/main/Assignment%202%20Datasets/Pokemon/pokemon.csv"
pokemon <- read.csv(url)

# Data type conversion
pokemon <- pokemon %>%
  mutate(
    is_legendary = as.factor(is_legendary),
    capture_rate = as.numeric(as.character(capture_rate)),
    type1 = as.factor(type1),
    generation = as.factor(generation),
    weight_kg = ifelse(is.na(weight_kg), median(weight_kg, na.rm = TRUE), weight_kg),
    height_m = ifelse(is.na(height_m), median(height_m, na.rm = TRUE), height_m),
    capture_rate = ifelse(is.na(capture_rate), median(capture_rate, na.rm = TRUE), capture_rate),
    percentage_male = ifelse(is.na(percentage_male), median(percentage_male, na.rm = TRUE), percentage_male)
  )

# Feature selection
pokemon_2 <- pokemon %>%
  select(-abilities, -classfication, -japanese_name, -name, -pokedex_number, -type2)

# Summary of numeric columns
summary(select(pokemon_2, where(is.numeric)))

# Type1 distribution plot
ggplot(pokemon_2, aes(x = type1)) +
  geom_bar(fill = "skyblue", color = "black") +
  labs(x = "Type1", y = "Count") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Prepare formula
selected_columns <- setdiff(colnames(pokemon_2), "type1")
form_cls <- as.formula(paste("type1 ~", paste(selected_columns, collapse = "+")))

# Split data for Decision Tree
set.seed(42)
trains <- createDataPartition(pokemon_2$type1, p = 0.8, list = FALSE)
train_data <- pokemon_2[trains,]
test_data <- pokemon_2[-trains,]

# Decision Tree model
fit_df_cls <- rpart(
  form_cls,
  data = train_data,
  method = "class",
  parms = list(split = "information"),
  control = rpart.control(minsplit = 10, cp = 0.005, maxdepth = 5)
)

# Prune tree
fit_df_cls_pruned <- prune(fit_df_cls, cp = 0.02)

# Variable importance
varimpdata <- data.frame(importance = fit_df_cls_pruned$variable.importance)
ggplot(varimpdata, aes(x = reorder(rownames(varimpdata), importance), y = importance)) +
  geom_col(fill = "steelblue") +
  labs(x = "Variables", y = "Importance") +
  coord_flip() +
  theme_minimal()

# Tree visualization
plot(fit_df_cls_pruned, uniform = TRUE, main = "Decision Tree")
text(fit_df_cls_pruned, use.n = TRUE, all = TRUE, cex = 0.7)

# Predict and evaluate (Decision Tree)
trainpredlab <- predict(fit_df_cls_pruned, newdata = train_data, type = "class")
testpredlab <- predict(fit_df_cls_pruned, newdata = test_data, type = "class")
train_cm <- confusionMatrix(trainpredlab, train_data$type1)
test_cm <- confusionMatrix(testpredlab, test_data$type1)

# Split data for Random Forest
set.seed(52)
trains_rf <- createDataPartition(pokemon_2$type1, p = 0.8, list = FALSE)
train_data_rf <- pokemon_2[trains_rf,]
test_data_rf <- pokemon_2[-trains_rf,]

# Random Forest model
fit_cls_rf <- randomForest(
  form_cls,
  data = train_data_rf,
  ntree = 200,
  mtry = 6,
  importance = TRUE
)

# Variable importance plot
varImpPlot(fit_cls_rf, main = "Variable Importance")

# Predict and evaluate (Random Forest)
trainpredlab_rf <- predict(fit_cls_rf, newdata = train_data_rf, type = "class")
testpredlab_rf <- predict(fit_cls_rf, newdata = test_data_rf, type = "class")
train_cm_rf <- confusionMatrix(trainpredlab_rf, train_data_rf$type1)
test_cm_rf <- confusionMatrix(testpredlab_rf, test_data_rf$type1)

# Model comparison results
results <- data.frame(
  Model = c("Decision Tree", "Random Forest"),
  Training_Accuracy = c(train_cm$overall["Accuracy"] * 100, train_cm_rf$overall["Accuracy"] * 100),
  Testing_Accuracy = c(test_cm$overall["Accuracy"] * 100, test_cm_rf$overall["Accuracy"] * 100),
  Training_Kappa = c(train_cm$overall["Kappa"], train_cm_rf$overall["Kappa"]),
  Testing_Kappa = c(test_cm$overall["Kappa"], test_cm_rf$overall["Kappa"])
)

results_long <- results %>%
  pivot_longer(cols = -Model, names_to = "Metric", values_to = "Value")

ggplot(results_long, aes(x = Metric, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.7), width = 0.6) +
  geom_text(aes(label = round(Value, 2)), position = position_dodge(width = 0.7), vjust = -0.5, size = 3.5) +
  scale_fill_brewer(palette = "Set1") +
  labs(title = "Performance Comparison of Decision Tree and Random Forest",
       x = "Metrics",
       y = "Values (%) or Kappa",
       fill = "Model") +
  theme_minimal()

