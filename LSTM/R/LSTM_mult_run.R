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
library(hms)
library(ggplot2)


#import numpy
# Import NumPy
np <- import("numpy")

# funcitons ---------------------------------------------------------------
# Function to create WLSTM model 
create_wlstm_model <- function(input_shape) {
  
  model <- keras::keras_model_sequential()
  
  model %>%
    layer_lstm(units = 50, return_sequences = TRUE, input_shape = input_shape) %>%
    #additional lstm layer
    layer_lstm(units = 50) %>%
    # Dense layer
    layer_dense(units = 1) %>%
    # Compile the model
    compile(optimizer = 'adam', loss = 'mean_squared_error')
  
  return(model)
  
}

# Function to calculate RMSE and R^2
evaluate_model <- function(model, X_test_np, y_test_np) {
  y_pred <- predict(model, X_test_np)
  
  rmse <- sqrt(mean((y_test - y_pred)^2))
  r2 <- cor(y_test, y_pred)^2
  return(list(rmse = rmse, r2 = r2))
}

#function to create sequences
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

#write/read to one csv
df <- read_csv("/Users/ttrefoni/Documents/tt_pm25_tuning/data/TrainData/single_trainpmv2.csv")

# Filter data based on a fixed PearsonR threshold (example: 0.7)
df <- df[df$PearsonR >= 0.7,]

#sort by datetime according to sensor pair groups 
df <- df %>% 
  group_by(file_id) %>% 
  arrange(datetime,.by_group=TRUE)

  
#extract predictive columns
X <- df[c("pm25_cf_1", "temperature", "humidity")] 

#scale/center data
X <- as_tibble(scale(X,center=TRUE,scale=TRUE)) 

Y <- df["epa_pm25"]

# X <- X[1:10000,]
# Y <- Y[1:10000,]
# Creating sequences with the past 24 hours to predict the next hour
time_steps <- 24
#try alternate create sequences
sequences <- create_sequences(X, Y, time_steps)
 saveRDS(sequences,"C:/Users/ttrefoni/Documents/tt_pm25_tuning/data/rdata_objects/sequences_pm25_group_sensor_01_20")
sequences <- read_rds("C:/Users/ttrefoni/Documents/tt_pm25_tuning/data/rdata_objects/sequences_pm25_group_sensor_01_20")

# Accessing sequences
X_seq <- sequences$Xs
y_seq <- unlist(sequences$ys)

 
#plot for interest 
# ggplot(X_seq[[100]],aes(x=datetime,y=pm25_cf_1))+
#   geom_line()

# Split into training and testing
set.seed(42)

# index <- sample(c(1:length(y_seq)), 0.8*length(y_seq), replace = FALSE, prob = NULL)
index <- createDataPartition(y_seq, p = 0.7, list = FALSE)

X_train <- X_seq[index]
y_train <- y_seq[index]
X_test <- X_seq[-index]
y_test <- y_seq[-index]


#convert sequences to 3d array
x_train_array <- array(unlist(X_train),dim=c(time_steps,ncol(X_train[[1]]),length(X_train))) %>% 
  aperm(c(3,1,2))
dim(x_train_array)
#I added this and it worked after but I didn't actually change any of the inputs 
x_train_np <- np$array(x_train_array)

#create 1d ouput array 
y_train_array <- array(y_train)
dim(y_train_array)
x_train_np <- np$array(x_train_array)

#create test arrays
X_test_array <- array(unlist(X_test),dim=c(time_steps,ncol(X_test[[1]]),length(X_test))) %>% 
  aperm(c(3,1,2))
y_test_array <- array(y_test)


# train WLSTM Model multiple times and collect metrics --------------------

# Train the WLSTM model multiple times and collect metrics
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

  #create model
  wlstm_model <- create_wlstm_model(c(time_steps, ncol(X_seq[[1]])))
  
  #fit to input data
  fit_result <- fit(object = wlstm_model, x = x_train_array, y = y_train_array, epochs = 10, batch_size = 32, verbose = 0)
  
  time_end<-Sys.time()
  time_tr_1<-as_hms(time_end-time_start)
  
  #write_rds(wlstm_model,str_c("/Users/ttrefoni/Documents/tt_pm25tuning/output_models/wlstm_model_01_09_70_30_",i))

  #evaluate the result and collect timing
  time_start <- Sys.time()
   eval_result <- evaluate_model(wlstm_model, X_test_array, y_test_array)
  time_end<-Sys.time()
  time_pred_1<-as_hms(time_end-time_start)
  
  rmses <- c(rmses, eval_result$rmse)
  r2s <- c(r2s, eval_result$r2)

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
write_csv(test_sum,"C:/Users/ttrefoni/Documents/tt_pm25_tuning/output/LSTM_pm_results_01_19_70_30.csv")
# write to  file  ---------------------------------------------------------
file.create("C:/Users/ttrefoni/Documents/tt_pm25_tuning/output/LTSM_pm_results_01_19_70_30.txt")
file_con <- "C:/Users/ttrefoni/Documents/tt_pm25_tuning/output/LTSM_pm_results_01_19_70_30.txt"
write_lines("DNN results",file=file_con)

 for(i in c(1:10)){
  write_lines(str_c("Model ",i," : RMSE = ",test_results$RMSE[i], " R^2 = ",test_results$Rsquared[i]), file=file_con,append=TRUE)
   print(i)
   }
write_lines(str_c("Aggregate Model "," : RMSE = ",test_results_ag$RMSE, " R^2 = ",test_results_ag$Rsquared), file=file_con,append=TRUE)

