1. Install Required R Packages
Ensure that the necessary R packages are installed. You can install them by running the following commands in your R environment:

```
install.packages("caret")
install.packages("tictoc")
install.packages("tidyverse")
install.packages("kernlab")
install.packages("brnn")
install.packages("RSNNS")
install.packages("lubridate")
install.packages("doParallel")
install.packages("keras")
```
2. Load and Prepare Data
The script reads the dataset, filters based on a PearsonR threshold, and splits the data into training and testing sets.

Reading and Preparing Data:

```
df <- read_csv("/TrainData/")
df <- df[df$PearsonR >= 0.7,]
```
3. Split Data into Training and Testing Sets
Split the dataset into training (80%) and testing (20%) sets using the createDataPartition function:

```
set.seed(42)
trainIndex <- createDataPartition(df$epa_pm25, p = .8, list = FALSE, times = 1)

train <- df[trainIndex,]
test <- df[-trainIndex,]

train_x <- train[c("pm25_cf_1", "temperature", "humidity")]
test_x <- test[c("pm25_cf_1", "temperature", "humidity")]
```
4. Standardize the Data
To ensure the data is on the same scale, standardize the training and testing sets using the scale function:

```
train_x <- scale(train_x, center = TRUE, scale = TRUE)
test_x <- scale(test_x, center = attr(train_x, "scaled:center"), scale = attr(train_x, "scaled:scale"))

train_x <- train_x %>% as.data.frame() %>% mutate(epa_pm25 = train$epa_pm25)
test_x <- test_x %>% as.data.frame() %>% mutate(epa_pm25 = test$epa_pm25)
```
5. Train k-NN Model
Train the K-Nearest Neighbors model (kknn function) and track the time taken for training and prediction.

```
test_results <- tibble()
time_trains <- tibble()
time_pred <- tibble()

for(i in c(1:1)){
  # Train the model and track time taken
  time_start <- Sys.time()
  knear_model_pm <- train.kknn(epa_pm25 ~ ., data = train_x, kmax = 5)
  time_end <- Sys.time()
  time_tr_1 <- time_end - time_start
  
  # Test the model
  time_start <- Sys.time()
  knear_pred_pm <- predict(knear_model_pm, test_x)
  time_end <- Sys.time()
  time_pred_1 <- time_end - time_start
  
  # Evaluate the model
  knear_metric_pm <- postResample(pred = knear_pred_pm, obs = test_x$epa_pm25)
  knear_mertric_df_pm <- data.frame(as.list(knear_metric_pm))
  
  # Append results
  test_results <- rbind(test_results, knear_mertric_df_pm)
  
  # Append train and test timing
  time_trains <- c(time_trains, time_tr_1)
  time_pred <- c(time_pred, time_pred_1)
}
```
6. Aggregate Results
After training the model, calculate the average RMSE, R², and MAE.

```
time_results <- tibble(c(1:nrow(test_results)), time_trains, time_pred)
colnames(time_results) <- c("model_run", "time_train", "time_predict")

# Add timing columns to results
test_results <- test_results %>%
  mutate(model = c(1:nrow(test_results)), time_train = time_trains, time_pred = time_pred) %>%
  mutate(type = "single")

# Create aggregated results
test_results_ag <- test_results %>%
  summarize(RMSE = mean(RMSE), Rsquared = mean(Rsquared), MAE = mean(MAE)) %>%
  mutate(model = NA, time_train = NA, time_predict = NA, type = "agg")

# Bind aggregate and individual results
test_sum <- rbind(test_results, test_results_ag)
```
7. Save Results
Save Results to CSV:

```
write_csv(test_sum, "/output/knear_pm_results_12_4.csv")
```
Save Results to Text File:

```
file.create("/output/knear_pm_results_12_05.txt")
file_con <- "/output/knear_pm_results_12_05.txt"
write_lines("knear results", file = file_con)

for(i in c(1:10)){
  write_lines(str_c("Model ", i, " : RMSE = ", test_results$RMSE[i], " R^2 = ", test_results$Rsquared[i]), file = file_con, append = TRUE)
}
write_lines(str_c("Aggregate Model : RMSE = ", test_results_ag$RMSE, " R^2 = ", test_results_ag$Rsquared), file = file_con, append = TRUE)
```
8. Running the Script
Once the setup is complete, run the script in your R environment. The results will be saved as both a CSV file and a text file with detailed model performance metrics for each model and the aggregated results.

