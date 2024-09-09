1. Install Required R Packages
Ensure that you have the necessary R packages installed. You can install them by running the following commands in your R environment:

```
install.packages("caret")
install.packages("tictoc")
install.packages("tidyverse")
install.packages("kernlab")
install.packages("brnn")
install.packages("RSNNS")
install.packages("doParallel")
install.packages("rpart")
install.packages("lubridate")
install.packages("hms")
```
2. Load and Prepare Data
Your data is stored in multiple CSV files, and the script consolidates these files into one. The consolidated data is filtered based on the Pearson correlation coefficient (PearsonR).

Reading Data:
If your data is in multiple files, use the following approach to read and combine them:

```
train_data_fp <- "/TrainData"
td_files <- list.files(train_data_fp)

df <- str_c(train_data_fp,"/",td_files) %>% 
  lapply(read_csv) %>% 
  bind_rows()
```
You can also use the pre-consolidated CSV file:

```
df <- read_csv("/Users/ttrefoni/Documents/tt_pm25tuning/data/TrainData/single_trainpm.csv")
```
Filtering Data:
Filter the data to include only rows where PearsonR is 0.7 or higher:

```
df <- df[df$PearsonR >= 0.7,]
```
3. Split Data into Training and Testing Sets
The dataset is split into training (70%) and testing (30%) sets using the createDataPartition function:

```
set.seed(42)
trainIndex <- createDataPartition(df$epa_pm25, p = .7, list = FALSE, times = 1)
train <- df[trainIndex,]
test <- df[-trainIndex,]
```
Features and Labels: The features used for training (train_x and test_x) are PM2.5, temperature, and humidity, while the target variable is epa_pm25.
```
train_x <- train[c("pm25_cf_1", "temperature", "humidity")]
test_x <- test[c("pm25_cf_1", "temperature", "humidity")]
```
4. Standardize the Data
To ensure the data is on the same scale, standardize the training and testing sets based on the training data:

```
train_x <- scale(train_x, center = TRUE, scale = TRUE)
test_x <- scale(test_x, center = attr(train_x, "scaled:center"), scale = attr(train_x, "scaled:scale"))
```
5. Train Decision Tree Models
You train the decision tree model (rpart) for 10 iterations, while capturing the time taken for training and prediction. During each iteration, the model's performance metrics (RMSE and R²) are calculated and stored:

```
for(i in c(1:10)){
  time_start <- Sys.time()
  DT_reg_model_pm <- rpart(epa_pm25 ~ ., method = "anova", data = train_x)
  time_end <- Sys.time()
  time_tr_1 <- as_hms(time_end - time_start)

  time_start <- Sys.time()
  DT_reg_pred_pm <- predict(DT_reg_model_pm, test_x)
  time_end <- Sys.time()
  time_pred_1 <- as_hms(time_end - time_start)

  DT_reg_metric_pm <- postResample(pred = DT_reg_pred_pm, obs = test_x$epa_pm25)
  
  DT_reg_mertric_df_pm <- data.frame(as.list(DT_reg_metric_pm)) 
  test_results <- rbind(test_results, DT_reg_mertric_df_pm)
  
  time_trains <- c(time_trains, as.character(time_tr_1))
  time_pred <- c(time_pred, as.character(time_pred_1))
  print(str_c("completed model ", i, " of 10"))
}
```
6. Aggregate Results
After training all 10 models, calculate the aggregate RMSE, R², and MAE, along with the average time for training and prediction:

```
test_results_ag <- test_results %>%
  summarize(RMSE = mean(RMSE), Rsquared = mean(Rsquared), MAE = mean(MAE),
            time_pred = as_hms(mean(as_hms(time_pred))),
            time_train = as_hms(mean(as_hms(time_train)))) %>%
  mutate(model = NA, type = "agg")
```
7. Save Results
Save to CSV: The individual and aggregate results are combined and saved to a CSV file:

```
test_sum <- rbind(test_results, test_results_ag)
write_csv(test_sum, "/Users/ttrefoni/Documents/tt_pm25tuning/output/DT_reg_pm_results_01_08_70_30.csv")
```
Save to Text File: The results are also written to a text file, including individual model results and the aggregate model:

```
file.create("/Users/ttrefoni/Documents/tt_pm25tuning/output/DT_reg_pm_results_01_08_70_30.txt")
file_con <- "/Users/ttrefoni/Documents/tt_pm25tuning/output/DT_reg_pm_results_01_08_70_30.txt"
write_lines("DT_reg results", file = file_con)

for(i in c(1:(nrow(test_sum) - 1))){
  write_lines(str_c("Model ", i, " : RMSE = ", test_results$RMSE[i], " R^2 = ", test_results$Rsquared[i]), file = file_con, append = TRUE)
}
write_lines(str_c("Aggregate Model ", " : RMSE = ", test_results_ag$RMSE, " R^2 = ", test_results_ag$Rsquared), file = file_con, append = TRUE)
```
8. Running the Script
Once the setup is complete, run the script in your R environment. The output will be saved as both a CSV file and a text file with detailed model performance metrics.
