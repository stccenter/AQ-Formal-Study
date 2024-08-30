# setup -------------------------------------------------------------------
library(caret)
library(tictoc)
library(tidyverse)
library(keras)
library(mlbench)
library(dplyr)
library(magrittr)
library(neuralnet)
library(nnet)
library(tensorflow)
library(reticulate)
# install_tensorflow()
#install_keras()
# read in data ------------------------------------------------------------

# 
# train_data_fp <- "/Users/ttrefoni/Documents/tt_pm25tuning/data/TrainData/TrainData"
# td_files <- list.files(train_data_fp)
# 
# df <- str_c(train_data_fp,"/",td_files) %>% 
#   lapply(read_csv) %>% 
#   bind_rows 

#write to one csv
# write_csv(df,"/Users/ttrefoni/Documents/tt_pm25tuning/data/TrainData/single_trainpm.csv")
df <- read_csv("/Users/ttrefoni/Documents/tt_pm25_tuning/data/TrainData/single_trainpm.csv")
# Filter data based on a fixed PearsonR threshold (example: 0.7)
df <- df[df$PearsonR >= 0.7,]

#sort by datetime
df <- arrange(df,datetime)[1:10000,]

X <- df[c("pm25_cf_1", "temperature", "humidity")]

#scale/center data
X <- as_tibble(scale(X,center=TRUE,scale=TRUE))
Y <- df["epa_pm25"]



# Creating sequences with the past 24 hours to predict the next hour
create_sequences <- function(X, Y, time_steps = 24) {
  Xs <- ys <- list()
  
  for (i in 1:(nrow(X) - time_steps + 1)) {
    Xs <- c(Xs, list(X[i:(i + time_steps - 1), , drop = FALSE]))
    ys <- c(ys, Y[i + time_steps - 1,])
    
    #print every 100 i
    if(i %% 10000 ==0 ){
      print(str_c("completed",i,"/",(nrow(X) - time_steps + 1)))
    }
  }
  
  return(list(Xs = Xs, ys = ys))
}

# Creating sequences with the past 24 hours to predict the next hour
time_steps <- 24
 sequences <- create_sequences(X, Y, time_steps)
# saveRDS(sequences,"C:/Users/ttrefoni/Documents/tt_pm25_tuning/data/rdata_objects/sequences_RNN")
# sequences <- read_rds("C:/Users/ttrefoni/Documents/tt_pm25_tuning/data/rdata_objects/sequences_pm25")
# Accessing sequences
X_seq <- sequences$Xs
y_seq <- unlist(sequences$ys)

rm(sequences)
gc()

# Splitting into training and testing
set.seed(42)
index <- createDataPartition(y_seq, p = 0.7, list = FALSE)
X_train <- X_seq[index]
y_train <- y_seq[index]
X_test <- X_seq[-index]
y_test <- y_seq[-index]

# Import numpy module to ensure proper data type for model fitting 
np <- import('numpy')

# Install  keras_self_attention
reticulate::py_install( "keras-self-attention")
keras <- import("keras")
seq_self_attention <- import("keras_self_attention")

# Convert R arrays to numpy arrays
X_train_np <- np$array(X_train)
y_train_np <- np$array(y_train)
X_test_np <- np$array(X_test)
y_test_np <- np$array(y_test)

#remove extra data 
gc()
# Function to create RNN model with attention
create_RNN_model <- function(input_shape) {

  model <- keras_model_sequential() %>%
    layer_simple_rnn(units = 50, input_shape = input_shape, return_sequences = TRUE) %>%
    layer_simple_rnn(units = 50, batch_input_shape = input_shape,return_sequences = FALSE) %>%
    layer_dense(units = 1) %>% 
    compile(optimizer = 'adam', loss = 'mean_squared_error')
  
  return(model)
}

#example 
RNN_model <- create_RNN_model(c(time_steps, ncol(X_seq[[1]])))


# Function to calculate RMSE and R^2
evaluate_model <- function(model, X_test_np, y_test) {
  y_pred <- predict(model, X_test_np)
  
  rmse <- sqrt(mean((y_test - y_pred)^2))
  r2 <- cor(y_test, y_pred)^2
  return(list(rmse = rmse, r2 = r2))
}

# Train the RNN model multiple times and collect metrics
test_results <- tibble()
time_trains <- NULL
time_pred<-NULL
rmses <- NULL
r2s <- NULL
use_python("C:/Users/ttrefoni/Documents/.virtualenvs/r-tensorflow/Scripts/python.exe")
for (i in 1:10) {
  set.seed(42)  # Set seed for reproducibility
  #train model and colect time 
  time_start <- Sys.time()
  
  RNN_model <- create_RNN_model(c(time_steps, ncol(X_seq[[1]])))
  fit_result <- fit(object = RNN_model, x = X_train_np, y = y_train_np, epochs = 10, batch_size = 32, verbose = 0)
  
  time_end<-Sys.time()
  time_tr_1<-time_end-time_start
  
  #write_rds(RNN_model,str_c("/Users/ttrefoni/Documents/tt_pm25tuning/output_models/RNN_model_01_09_70_30_",i))
  
  #evaluate the result and collect timing 
  time_start <- Sys.time()
  eval_result <- evaluate_model(RNN_model, X_test_np, y_test)
  time_end<-Sys.time()
  time_pred_1<-time_end-time_start
  
  rmses <- c(rmses, eval_result$rmse)
  r2s<- c(r2s, eval_result$r2)
  
  #append train and test timing 
  time_trains<-c(time_trains,time_tr_1)
  time_pred<-c(time_pred,time_pred_1)
  print(str_c("completed model ",i," of 10"))
}

#convert timing to df
time_results<-tibble(c(1:length(time_trains),time_trains,time_pred))
colnames(time_results)<-c("model_run","time_train","time_predict")

test_results <- tibble()
#add timing columns to results
test_results <-
  tibble(model=c(1:length(r2s)),time_train=time_trains,time_pred=time_pred,
         Rsquared=r2s,RMSE=rmses) %>%
  mutate(type="single")
#create aggregated results
test_results_ag <-
  test_results %>%
  summarize(RMSE=mean(RMSE),Rsquared=mean(Rsquared),
            time_pred=mean(time_pred),time_train=mean(time_train)) %>%
  mutate(model=NA,type="agg")

#bind aggregatge and individual results
test_sum <-rbind(test_results,test_results_ag)
write_csv(test_sum,"C:/Users/ttrefoni/Documents/tt_pm25_tuning/output/RNN_pm_results_01_09_70_30.csv")
# write to  file  ---------------------------------------------------------
file.create("C:/Users/ttrefoni/Documents/tt_pm25_tuning/output/RNN_pm_results_01_09_70_30.txt")
file_con <- "C:/Users/ttrefoni/Documents/tt_pm25_tuning/output/RNN_pm_results_01_09_70_30.txt"
write_lines("DNN results",file=file_con)

for(i in c(1:10)){
  write_lines(str_c("Model ",i," : RMSE = ",test_results$RMSE[i], " R^2 = ",test_results$Rsquared[i]), file=file_con,append=TRUE)
  print(i)
}
write_lines(str_c("Aggregate Model "," : RMSE = ",test_results_ag$RMSE, " R^2 = ",test_results_ag$Rsquared), file=file_con,append=TRUE)

