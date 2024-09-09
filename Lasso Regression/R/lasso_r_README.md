1. Install Required R Packages
Ensure that the necessary R packages are installed. You can install them by running the following commands in your R environment:

```
install.packages("caret")
install.packages("tictoc")
install.packages("tidyverse")
install.packages("RSNNS")
install.packages("lubridate")
install.packages("doParallel")
install.packages("hms")
install.packages("glmnet")
```
2. Load and Prepare Data
The script reads the dataset, filters based on a PearsonR threshold, and splits the data into training and testing sets.

Reading and Preparing Data:

```
df <- read_csv("/TrainData/")
df <- df[df$PearsonR >= 0.7,]
```
3. Split Data into Training and Testing Sets
Split the dataset into training (70%) and testing (30%) sets using the createDataPartition function:

```
set.seed(42)
trainIndex <- createDataPartition(df$epa_pm25, p = .7, list = FALSE, times = 1)

train <- df[trainIndex,]
test <- df[-trainIndex,]

train_x <- train[c("pm25_cf_1", "temperature", "humidity")]
test_x <- test[c("pm25_cf_1", "temperature", "humidity")]
```
4. Standardize the Data
To ensure consistent scaling across features, standardize the training and testing sets using the scale function:

```
train_x <- scale(train_x, center = TRUE, scale = TRUE)
test_x <- scale(test_x, center = attr(train_x, "scaled:center"), scale = attr(train_x, "scaled:scale"))

train_x <- train_x %>% as.data.frame() %>% mutate(epa_pm25 = train$epa_pm25)
test_x <- test_x %>% as.data.frame() %>% mutate(epa_pm25 = test$epa_pm25)
```
5. Train Linear Regression Model (with glmnet)
Train the Linear Regression model for 10 iterations, track the time taken for training and prediction, and collect evaluation metrics. The model uses a regularization grid with lambda = 0.05 and alpha = 0.2.

```
fitControl <- trainControl(method = "none")
LR_grid <- expand.grid(lambda = 0.05, alpha = 0.2)
test_results <- tibble()
time_trains <- NULL
time_pred <- NULL

for(i in c(1:10)){
  # Train the model and track time taken
  time_start <- Sys.time()
  LR_model_pm <- caret::train(epa_pm25 ~ ., data = train_x, preProcess = c("center", "scale"), method = "glmnet", tuneGrid = LR_grid, trControl = fitControl, metric = "Rsquared")
  time_end <- Sys.time()
  time_tr_1 <- as_hms(time_end - time_start)
  
  # Save the model
  write_rds(LR_model_pm, str_c("/anonymized_path/output_models/LR_n_model_01_05_70_30_", i))
  
  # Test the model
  time_start <- Sys.time()
  LR_n_pred_pm <- predict(LR_model_pm, test_x)
  time_end <- Sys.time()
  time_pred_1 <- as_hms(time_end - time_start)
  
  # Evaluate the model
  LR_n_metric_pm <- postResample(pred = LR_n_pred_pm, obs = test_x$epa_pm25)
  LR_n_mertric_df_pm <- data.frame(as.list(LR_n_metric_pm))
  
  # Append results
  test_results <- rbind(test_results, LR_n_mertric_df_pm)
  time_trains <- c(time_trains, as.character(time_tr_1))
  time_pred <- c(time_pred, as.character(time_pred_1))
  print(str_c("completed model ", i, " of 10"))
}
```
6. Aggregate Results
After training all models, calculate the average RMSE, R², and MAE, along with the average training and prediction times:

```
test_results <- test_results %>%
  mutate(model = c(1:nrow(test_results)), time_train = time_trains, time_pred = time_pred) %>%
  mutate(type = "single")

test_results_ag <- test_results %>%
  summarize(RMSE = mean(RMSE), Rsquared = mean(Rsquared), MAE = mean(MAE),
            time_pred = mean(time_pred), time_train = mean(time_train)) %>%
  mutate(model = NA, type = "agg")
```
7. Save Results
Save Results to CSV:

```
test_sum <- rbind(test_results, test_results_ag)
write_csv(test_sum, "/output/LR_n_pm_results_01_05_70_30.csv")
```
Save Results to Text File:

```
file.create("/output/LR_n_pm_results_01_05_70_30.txt")
file_con <- "/output/LR_n_pm_results_01_05_70_30.txt"
write_lines("LR_n results", file = file_con)

for(i in c(1:10)){
  write_lines(str_c("Model ", i, " : RMSE = ", test_results$RMSE[i], " R^2 = ", test_results$Rsquared[i]), file = file_con, append = TRUE)
}
write_lines(str_c("Aggregate Model : RMSE = ", test_results_ag$RMSE, " R^2 = ", test_results_ag$Rsquared), file = file_con, append = TRUE)
```
8. Running the Script
Once the setup is complete, run the script in your R environment. The results will be saved as both a CSV file and a text file with detailed model performance metrics for each model and the aggregated results.
