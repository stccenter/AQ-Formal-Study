1. Install Required R Packages
Ensure that the necessary R packages and TensorFlow dependencies are installed. You can install them by running the following commands in your R environment:

```
install.packages("caret")
install.packages("tictoc")
install.packages("tidyverse")
install.packages("keras")
install.packages("mlbench")
install.packages("dplyr")
install.packages("magrittr")
install.packages("neuralnet")
install.packages("nnet")
install.packages("tensorflow")
install.packages("reticulate")
```
For TensorFlow and Keras installation, run:

```
tensorflow::install_tensorflow()
keras::install_keras()
```
Make sure Python and the correct environment are set up using reticulate:

```
use_python("/anonymized_path/.virtualenvs/r-tensorflow/Scripts/python.exe")
```
2. Load and Prepare Data
The script reads the dataset, filters based on a PearsonR threshold, and splits the data into sequences for LSTM input.

Reading and Preparing Data:

```
df <- read_csv("/TrainData/")
df <- df[df$PearsonR >= 0.7,]

# Sort by datetime
df <- arrange(df, datetime)

X <- df[c("pm25_cf_1", "temperature", "humidity")]

# Scale/center data
X <- as_tibble(scale(X, center = TRUE, scale = TRUE))
Y <- df["epa_pm25"]
```
3. Create Time Sequences
This section creates sequences using the past 24 hours of data to predict the next hour's value.

```
create_sequences <- function(X, Y, time_steps = 24) {
  Xs <- ys <- list()
  
  for (i in 1:(nrow(X) - time_steps + 1)) {
    Xs <- c(Xs, list(X[i:(i + time_steps - 1), , drop = FALSE]))
    ys <- c(ys, Y[i + time_steps - 1,])
    
    if(i %% 100 == 0){
      print(str_c("completed ", i, " / ", (nrow(X) - time_steps + 1)))
    }
  }
  
  return(list(Xs = Xs, ys = ys))
}

time_steps <- 24
sequences <- create_sequences(X, Y, time_steps)

# Accessing sequences
X_seq <- sequences$Xs
y_seq <- unlist(sequences$ys)
4. Split Data into Training and Testing Sets
R
Copy code
set.seed(42)
index <- createDataPartition(y_seq, p = 0.8, list = FALSE)
X_train <- X_seq[index]
y_train <- y_seq[index]
X_test <- X_seq[-index]
y_test <- y_seq[-index]

# Import numpy module to ensure proper data type for model fitting
np <- import('numpy')

# Convert R arrays to numpy arrays
X_train_np <- np$array(X_train)
y_train_np <- np$array(y_train)
X_test_np <- np$array(X_test)
y_test_np <- np$array(y_test)
```
5. Create and Train the WLSTM Model
Define a function to create the WLSTM model with Keras and fit it multiple times.

```
create_wlstm_model <- function(input_shape) {
  model <- keras::keras_model_sequential()
  model %>%
    layer_lstm(units = 50, return_sequences = TRUE, input_shape = input_shape) %>%
    layer_lstm(units = 50) %>%
    layer_dense(units = 1) %>%
    compile(optimizer = 'adam', loss = 'mean_squared_error')
  
  return(model)
}

wlstm_model <- create_wlstm_model(c(time_steps, ncol(X_seq[[1]])))
```
6. Train the WLSTM Model Multiple Times
Train the WLSTM model 10 times, collect performance metrics, and track the time taken for training and prediction.

```
test_results <- tibble()
time_trains <- NULL
time_pred <- NULL
rmses <- NULL
r2s <- NULL

for (i in 1:10) {
  set.seed(42)  # Set seed for reproducibility
  
  # Train model and collect time
  time_start <- Sys.time()
  wlstm_model <- create_wlstm_model(c(time_steps, ncol(X_seq[[1]])))
  fit_result <- fit(object = wlstm_model, x = X_train_np, y = y_train_np, epochs = 10, batch_size = 32, verbose = 0)
  time_end <- Sys.time()
  time_tr_1 <- time_end - time_start
  
  # Evaluate the result and collect timing
  time_start <- Sys.time()
  eval_result <- evaluate_model(wlstm_model, X_test_np, y_test_np)
  time_end <- Sys.time()
  time_pred_1 <- time_end - time_start
  
  # Append metrics
  rmses <- c(rmses, eval_result$rmse)
  r2s <- c(r2s, eval_result$r2)
  
  # Append train and test timing
  time_trains <- c(time_trains, time_tr_1)
  time_pred <- c(time_pred, time_pred_1)
  print(str_c("completed model ", i, " of 10"))
}
```
7. Aggregate Results
Calculate the average RMSE and R² across all models.

```
time_results <- tibble(c(1:length(time_trains), time_trains, time_pred))
colnames(time_results) <- c("model_run", "time_train", "time_predict")

# Add timing columns to results
test_results <- tibble(model = c(1:length(r2s)), time_train = time_trains, time_pred = time_pred, Rsquared = r2s, RMSE = rmses) %>%
  mutate(type = "single")

# Create aggregated results
test_results_ag <- test_results %>%
  summarize(RMSE = mean(RMSE), Rsquared = mean(Rsquared), time_pred = mean(time_pred), time_train = mean(time_train)) %>%
  mutate(model = NA, type = "agg")

# Bind aggregate and individual results
test_sum <- rbind(test_results, test_results_ag)
```
8. Save Results
Save Results to CSV:

```
write_csv(test_sum, "/output/WLSTM_pm_results_01_09_80_20.csv")
```
Save Results to Text File:

```
file.create("/output/WLSTM_pm_results_01_09_80_20.txt")
file_con <- "/output/WLSTM_pm_results_01_09_80_20.txt"
write_lines("WLSTM results", file = file_con)

for(i in c(1:10)){
  write_lines(str_c("Model ", i, " : RMSE = ", test_results$RMSE[i], " R^2 = ", test_results$Rsquared[i]), file = file_con, append = TRUE)
}
write_lines(str_c("Aggregate Model : RMSE = ", test_results_ag$RMSE, " R^2 = ", test_results_ag$Rsquared), file = file_con, append = TRUE)
```

9. Running the Script
Once the setup is complete, run the script in your R environment. The results will be saved as both a CSV file and a text file with detailed model performance metrics for each model and the aggregated results.
