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

R
Copy code
tensorflow::install_tensorflow()
keras::install_keras()
Make sure Python and the correct environment are set up using reticulate:

```
use_python("C:/PATH/python.exe")
```
2. Load and Prepare Data
Your data is stored in a single CSV file, and the script reads the dataset, filters based on a PearsonR threshold, and sorts the data by datetime.

Reading and Preparing Data:

```
df <- read_csv("/TrainData/")
df <- df[df$PearsonR >= 0.7,]

# Sort by datetime and select the first 10,000 rows
df <- arrange(df, datetime)[1:10000,]

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
    
    if(i %% 10000 == 0){
      print(str_c("completed ", i, " / ", (nrow(X) - time_steps + 1)))
    }
  }
  
  return(list(Xs = Xs, ys = ys))
}

time_steps <- 24
sequences <- create_sequences(X, Y, time_steps)
X_seq <- sequences$Xs
y_seq <- unlist(sequences$ys)

rm(sequences)
gc()
```
4. Split Data into Training and Testing Sets
```
set.seed(42)
index <- createDataPartition(y_seq, p = 0.7, list = FALSE)
X_train <- X_seq[index]
y_train <- y_seq[index]
X_test <- X_seq[-index]
y_test <- y_seq[-index]
```
5. Convert Data to Numpy Arrays
To use with Keras, convert the data into numpy arrays.

```
np <- import('numpy')
X_train_np <- np$array(X_train)
y_train_np <- np$array(y_train)
X_test_np <- np$array(X_test)
y_test_np <- np$array(y_test)

gc()
```
6. Create and Train the RNN Model
Define a function to create the RNN model with Keras and fit it multiple times.

```
create_RNN_model <- function(input_shape) {
  model <- keras_model_sequential() %>%
    layer_simple_rnn(units = 50, input_shape = input_shape, return_sequences = TRUE) %>%
    layer_simple_rnn(units = 50, return_sequences = FALSE) %>%
    layer_dense(units = 1) %>%
    compile(optimizer = 'adam', loss = 'mean_squared_error')
  
  return(model)
}

RNN_model <- create_RNN_model(c(time_steps, ncol(X_seq[[1]])))

# Function to calculate RMSE and R²
evaluate_model <- function(model, X_test_np, y_test) {
  y_pred <- predict(model, X_test_np)
  rmse <- sqrt(mean((y_test - y_pred)^2))
  r2 <- cor(y_test, y_pred)^2
  return(list(rmse = rmse, r2 = r2))
}
```
7. Train the RNN Model Multiple Times
Train the model 10 times, collect performance metrics, and track time taken for training and prediction.

```
test_results <- tibble()
time_trains <- NULL
time_pred <- NULL
rmses <- NULL
r2s <- NULL

for (i in 1:10) {
  set.seed(42)
  
  # Train model
  time_start <- Sys.time()
  RNN_model <- create_RNN_model(c(time_steps, ncol(X_seq[[1]])))
  fit_result <- fit(RNN_model, X_train_np, y_train_np, epochs = 10, batch_size = 32, verbose = 0)
  time_end <- Sys.time()
  time_tr_1 <- time_end - time_start
  
  # Evaluate model
  time_start <- Sys.time()
  eval_result <- evaluate_model(RNN_model, X_test_np, y_test)
  time_end <- Sys.time()
  time_pred_1 <- time_end - time_start
  
  # Save metrics
  rmses <- c(rmses, eval_result$rmse)
  r2s <- c(r2s, eval_result$r2)
  
  time_trains <- c(time_trains, time_tr_1)
  time_pred <- c(time_pred, time_pred_1)
  
  print(str_c("Completed model ", i, " of 10"))
}
```
8. Aggregate and Save Results
Save Results to CSV:

```
test_results <- tibble(model = 1:length(r2s), time_train = time_trains, time_pred = time_pred, Rsquared = r2s, RMSE = rmses) %>%
  mutate(type = "single")

test_results_ag <- test_results %>%
  summarize(RMSE = mean(RMSE), Rsquared = mean(Rsquared), time_pred = mean(time_pred), time_train = mean(time_train)) %>%
  mutate(model = NA, type = "agg")

test_sum <- rbind(test_results, test_results_ag)
write_csv(test_sum, "C:/PATH/RNN_pm_results_01_09_70_30.csv")
```
Save Results to Text File:

```
file.create("C:/PATH/RNN_pm_results_01_09_70_30.txt")
file_con <- "C:/PATH/RNN_pm_results_01_09_70_30.txt"
write_lines("RNN results", file = file_con)

for(i in 1:10){
  write_lines(str_c("Model ", i, " : RMSE = ", test_results$RMSE[i], " R^2 = ", test_results$Rsquared[i]), file = file_con, append = TRUE)
}
write_lines(str_c("Aggregate Model : RMSE = ", test_results_ag$RMSE, " R^2 = ", test_results_ag$Rsquared), file = file_con, append = TRUE)
```
9. Running the Script
Once the setup is complete, run the script in your R environment. The results will be saved in both CSV and text formats, detailing the model performance metrics for each model and the aggregated results.
