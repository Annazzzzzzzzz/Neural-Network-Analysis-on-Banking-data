wd='~/Desktop/NN'
setwd(wd)

library(keras)
library(dplyr)
library(tensorflow)
library(neuralnet)
library(MLmetrics)
library(pROC)
library(caret)
library(ggplot2)
library(reshape2)
library(mccr)
library(recipes)
library(themis)


##  0. importing data
data <- read.csv('data.csv')


## 1. simple processing
# for y: yes --> 1; no --> 0
label_col <- "y"
data[[label_col]] <- ifelse(tolower(data[[label_col]]) %in% c("yes"), 1, 0)

#delete non_numerical attributes(from pre-processing)
cols_to_drop <- c("job", "marital", "education", "contact", "month", "day_bin", "day")
data <-data %>% select(-all_of(cols_to_drop)) 

# splitting train & test set(from pre-processing)
set.seed(357)
train_index <- createDataPartition(data$y, p = 0.7, list = FALSE)
train0 <- data[train_index, ]
test  <- data[-train_index, ]

# splitting train & val
set.seed(357)
val_idx <- createDataPartition(train0$y, p = 0.8, list = FALSE)
train <- train0[val_idx, ]
val <- train0[-val_idx, ]
train$y <- as.factor(train$y)

# SMOTE(from pre-processing)
recipe_smote <- recipe(y ~ ., data = train) %>%
  step_smote(y, over_ratio = 1) # adjusting ratio to 1

train_smote <- prep(recipe_smote) %>% bake(new_data = NULL)
train_smote$y <- as.numeric(train_smote$y) - 1
train$y <- as.numeric(train$y) - 1

# splitting y and other variables
# without SMOTE oversampling
X_train_nosmote <- as.matrix(train %>% select(-y))
y_train_nosmote <- as.matrix(train$y)
# with SMOTE oversampling
X_train <- as.matrix(train_smote %>% select(-y))
y_train <- as.matrix(train_smote$y)
X_val <- as.matrix(val %>% select(-y))
y_val <- as.matrix(val$y)
X_test <- as.matrix(test %>% select(-y))
y_test <- as.matrix(test$y)


## 2. building Neural Network Model
# 2.1 building model using Keras(single testing)
# 2.1.1 without SMOTE
# building model
nnmodel0 <- keras_model_sequential()
nnmodel0 %>%
  layer_dense(units = 64, activation = 'relu', input_shape = ncol(X_train_nosmote)) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 16, activation = 'relu') %>%
  layer_dense(units = 1, activation = 'sigmoid')
summary(nnmodel0)

# mcc calculation function
mcc_metric <- custom_metric("mcc", function(y_true, y_pred) {
  y_pred_pos <- k_round(y_pred)
  y_true_pos <- k_round(y_true)
  
  tp <- k_sum(y_true_pos * y_pred_pos)
  tn <- k_sum((1 - y_true_pos) * (1 - y_pred_pos))
  fp <- k_sum((1 - y_true_pos) * y_pred_pos)
  fn <- k_sum(y_true_pos * (1 - y_pred_pos))
  
  numerator <- tp * tn - fp * fn
  denominator <- k_sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + k_epsilon()
  
  return(numerator / denominator)
})

# compile
focal_loss <- function(gamma = 2, alpha = 0.8) {
  function(y_true, y_pred) {
    epsilon <- k_epsilon()
    y_pred <- k_clip(y_pred, epsilon, 1.0 - epsilon)
    
    pt <- tf$where(k_equal(y_true, 1), y_pred, 1 - y_pred)
    alpha_t <- tf$where(k_equal(y_true, 1), alpha, 1 - alpha)
    
    loss <- -alpha_t * k_pow(1 - pt, gamma) * k_log(pt)
    return(k_mean(loss))
  }
}

nnmodel0 %>% compile(
  optimizer = optimizer_adam(learning_rate = 0.001),
  loss = focal_loss(gamma = 2, alpha = 0.8),
  #loss = focal_loss(gamma = 1.0, alpha = 0.5),
  #loss = "binary_crossentropy",
  metrics = list("AUC", "Precision", "Recall", mcc_metric)
)

# model fitting
history <- nnmodel0 %>% fit(
  X_train_nosmote, y_train_nosmote,
  epochs = 100,
  batch_size = 32,
  validation_data = list(X_val, y_val), 
  class_weight = list("0" = 1, "1" = 7.5),
  callbacks = list(
    callback_early_stopping(monitor = "val_mcc", patience = 10, mode = "max",
                            restore_best_weights = TRUE
    )
  )
)

# model prediction
pred_prob0 <- nnmodel0 %>% predict(X_test)
thresholds0 <- seq(0.01, 0.99, by = 0.05)
mcc_scores0 <- sapply(thresholds0, function(t) {
  preds0 <- ifelse(pred_prob0 > t, 1, 0)
  mccr(y_test, preds0)
})
best_t0 <- thresholds0[which.max(mcc_scores0)]
cat("Best MCC:", max(mcc_scores0), "at threshold =", best_t0, "\n")
# using optimal thresholds to predict
pred_class0 <- ifelse(pred_prob0 >= best_t0, 1, 0)

# model evaluation
# MCC
mcc_val0 <- mccr(y_test, pred_class0)
# AUC
roc_obj0 <- roc(y_test, pred_prob0)
auc_val0 <- auc(roc_obj0)
# confusion matrix
cm0 <- confusionMatrix(factor(pred_class0), factor(y_test))
# result print
cat("Test MCC:", mcc_val0, "\n") # Test MCC: 0.3188029 
cat("Test AUC:", auc_val0, "\n") # Test AUC: 0.7476313 
print(cm0)


# 2.1.2 with SMOTE
# building model
nnmodel1 <- keras_model_sequential()
nnmodel1 %>%
  layer_dense(units = 64, activation = 'relu', input_shape = ncol(X_train)) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 16, activation = 'relu') %>%
  layer_dense(units = 1, activation = 'sigmoid')
summary(nnmodel1)

# compile
nnmodel1 %>% compile(
  optimizer = optimizer_adam(learning_rate = 0.001),
  loss = focal_loss(gamma = 2, alpha = 0.8), # remains unchanged
  #loss = focal_loss(gamma = 1.0, alpha = 0.5),
  #loss = "binary_crossentropy",
  metrics = list("AUC", "Precision", "Recall", mcc_metric)
)

# model fitting
history <- nnmodel1 %>% fit(
    X_train, y_train,
    epochs = 100,
    batch_size = 32,
    validation_data = list(X_val, y_val), # validation set remains unchanged
    class_weight = list("0" = 1, "1" = 7.5),
    callbacks = list(
      callback_early_stopping(monitor = "val_mcc", patience = 10, mode = "max",
        restore_best_weights = TRUE
      )
    )
)

# model prediction
pred_prob <- nnmodel1 %>% predict(X_test)
thresholds <- seq(0.1, 0.9, by = 0.05)
mcc_scores <- sapply(thresholds, function(t) {
  preds <- ifelse(pred_prob > t, 1, 0)
  mccr(y_test, preds)
})
best_t <- thresholds[which.max(mcc_scores)]
cat("Best MCC:", max(mcc_scores), "at threshold =", best_t, "\n")
# using optimal thresholds to predict
pred_class <- ifelse(pred_prob >= best_t, 1, 0)

# training curve
df_hist <- data.frame(
  epoch = 1:length(history$metrics$loss),
  loss = history$metrics$loss,
  val_loss = history$metrics$val_loss
)
# visualization
ggplot(df_hist, aes(x = epoch)) +
  geom_line(aes(y = loss, color = "Train Loss")) +
  geom_line(aes(y = val_loss, color = "Val Loss")) +
  labs(title = "Training vs Validation Loss", y = "Loss") +
  theme_minimal()
plot(history)  # loss & accuracy curve

# model evaluation
# MCC
mcc_val <- mccr(y_test, pred_class)
# AUC
roc_obj <- roc(y_test, pred_prob)
auc_val <- auc(roc_obj)
# confusion matrix
cm <- confusionMatrix(factor(pred_class), factor(y_test))
# result
cat("Test MCC:", mcc_val, "\n") # Test MCC: 0.4728296 
cat("Test AUC:", auc_val, "\n") # Test AUC: 0.8871797 
print(cm)

# ROC curve
plot(roc_obj, col="#1f78b4", main=paste0("ROC Curve (AUC=", round(auc_val,3), ")"))
abline(a=0, b=1, lty=2, col="gray")

# 2.2 5-fold cross validation
set.seed(519)
folds <- createFolds(data$y, k = 5, list = TRUE, returnTrain = FALSE)
results <- data.frame(Fold=1:5, MCC=NA, AUC=NA)

for (i in 1:5) {
  cat("Fold", i, "\n")
  # splitting training set & validation set
  val_idx <- folds[[i]]
  train <- data[-val_idx, ]
  train$y <- as.factor(train$y)
  val <- data[val_idx, ]
  
  # SMOTE
  recipe_smote <- recipe(y ~ ., data = train) %>%
    step_smote(y, over_ratio = 1) # adjusting ratio to 1
  train <- prep(recipe_smote) %>% bake(new_data = NULL)
  train$y <- as.numeric(train$y) - 1

  # splitting y and other variables
  X_train <- as.matrix(train %>% select(-y))
  y_train <- as.matrix(train$y)
  X_val <- as.matrix(val %>% select(-y))
  y_val <- as.matrix(as.numeric(val$y))
  
  # building model
  nnmodel2 <- keras_model_sequential() %>%
    layer_dense(units = 64, activation = "relu", input_shape = ncol(X_train)) %>%
    layer_dropout(rate = 0.2) %>%
    layer_dense(units = 32, activation = "relu") %>%
    layer_dropout(rate = 0.2) %>%
    layer_dense(units = 16, activation = 'relu') %>%
    layer_dense(units = 1, activation = "sigmoid")
  
  # compile
  nnmodel2 %>% compile(
    optimizer = optimizer_adam(learning_rate = 0.001),
    loss = focal_loss(gamma = 2, alpha = 0.8),
    metrics = list("AUC", "Precision", "Recall", mcc_metric)
  )
  
  history <- nnmodel2 %>% fit(
    X_train, y_train,
    epochs = 50,
    batch_size = 32,
    verbose = 0,
    class_weight = list("0" = 1, "1" = 7.5)
  )
  
  # prediction on validation set
  pred_prob2 <- nnmodel2 %>% predict(X_val)
  
  thresholds2 <- seq(0.1, 0.9, by = 0.05)
  mcc_scores2 <- sapply(thresholds2, function(t) {
    preds2 <- ifelse(pred_prob2 > t, 1, 0)
    mccr(y_val, preds2)
  })
  best_t2 <- thresholds2[which.max(mcc_scores2)]
  # using optimal thresholds to predict
  pred_class2 <- ifelse(pred_prob2 >= best_t2, 1, 0)
  
  # evaluation
  mcc_val <- mccr(y_val, pred_class2)
  auc_val <- auc(roc(y_val, pred_prob2))
  
  results$MCC[i] <- mcc_val
  results$AUC[i] <- auc_val
}

print(results)
#   Fold       MCC       AUC
#1    1 0.4788025 0.9022898
#2    2 0.5115520 0.9085926
#3    3 0.5167670 0.9026427
#4    4 0.5053699 0.9054099
#5    5 0.4967705 0.9012658
cat("Mean MCC:", mean(results$MCC), "\n") # Mean MCC: 0.5018524
cat("Mean AUC:", mean(results$AUC), "\n") # Mean AUC: 0.9040402


## 3. Uncertainty Analysis
# segmenting into 3 conf. groups
test_conf <- test
# High > 0.6; 0.4 <= Mid <= 0.6; Low < 0.4
test_conf$conf_group <- ifelse(pred_prob > 0.6, "High", 
                          ifelse(pred_prob < 0.4, "Low", "Mid"))
# visualization
# boxplot
vars_to_plot <- c("balance", "duration", "campaign", "pdays")
test_long <- melt(test_conf, id.vars = "conf_group", measure.vars = vars_to_plot)
ggplot(test_long, aes(x = conf_group, y = value, fill = conf_group)) +
  geom_boxplot(alpha = 0.6) +
  facet_wrap(~ variable, scales = "free_y") +
  labs(title = "Distribution of Variables by Confidence Group",
       x = "Confidence Group", y = "Value") +
  theme_minimal()

# density plot
ggplot(test_long, aes(x = value, fill = conf_group)) +
  geom_density(alpha = 0.4) +
  facet_wrap(~ variable, scales = "free") +
  labs(title = "Density plot of Variables by Confidence Group",
       x = "Value", y = "Density")

# balance density
ggplot(test_conf, aes(x = balance, fill = conf_group)) +
  geom_density(alpha = 0.4) +
  labs(title = "Balance distribution across prediction confidence groups")
# duration density
ggplot(test_conf, aes(x = duration, fill = conf_group)) +
  geom_density(alpha = 0.4) +
  labs(title = "Duration distribution across prediction confidence groups")
# campaign density
ggplot(test_conf, aes(x = campaign, fill = conf_group)) +
  geom_density(alpha = 0.4) +
  labs(title = "Campaign distribution across prediction confidence groups")
# pdays density
ggplot(test_conf, aes(x = pdays, fill = conf_group)) +
  geom_density(alpha = 0.4) +
  labs(title = "Pdays distribution across prediction confidence groups")


## 4. False Positive/Negative Analysis
test_fp <- test
test_fp$pred_prob <- as.vector(pred_prob)   # nnmodel1 的预测概率
test_fp$pred_class <- as.vector(pred_class) # nnmodel1 的预测类别
test_fp$true_label <- y_test

test_fp$error_type <- ifelse(test_fp$true_label == 1 & test_fp$pred_class == 0, "FN",
                          ifelse(test_fp$true_label == 0 & test_fp$pred_class == 1, "FP",
                                 ifelse(test_fp$true_label == test_fp$pred_class, "Correct", "Other")))

# visualization
test_long_fp <- melt(test_fp, id.vars = "error_type", measure.vars = vars_to_plot)
ggplot(test_long_fp, aes(x = error_type, y = value, fill = error_type)) +
  geom_boxplot(alpha = 0.6) +
  facet_wrap(~ variable, scales = "free_y") +
  labs(title = "Distribution of Variables by Error Type",
       x = "Confidence Group", y = "Value") +
  theme_minimal()

# density plot
ggplot(test_long_fp, aes(x = value, fill = error_type)) +
  geom_density(alpha = 0.4) +
  facet_wrap(~ variable, scales = "free") +
  labs(title = "Density plot of Variables by Error Type",
       x = "Value", y = "Density")



