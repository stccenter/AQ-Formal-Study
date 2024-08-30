# setup -------------------------------------------------------------------
library(caret)
library(tictoc)
library(tidyverse)
library(randomForest)
library(ranger)
library(kernlab)
library(brnn)
library(xgboost)
library(neuralnet)
library(NeuralNetTools)
library(nnet)
library(RSNNS)
library(lubridate)
library(doParallel)
library(hms)

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
df <- read_csv("/Users/ttrefoni/Documents/tt_pm25tuning/data/TrainData/single_trainpm.csv")
# Filter data based on a fixed PearsonR threshold (example: 0.7)
df <- df[df$PearsonR >= 0.7,]



#split into training and testing 
set.seed(42)
trainIndex <- createDataPartition(df$epa_pm25, p = .7, 
                                  list = FALSE, 
                                  times = 1)
head(trainIndex,20)

train <- df[trainIndex,]
test <- df[-trainIndex,]

train_x <- train[c("pm25_cf_1", "temperature", "humidity")]
test_x <-  test[c("pm25_cf_1", "temperature", "humidity")]

#scale/center train and test according to train 
train_x <- scale(train_x,center=TRUE,scale=TRUE) 
test_x <- scale(test_x, center=attr(train_x, "scaled:center"), scale=attr(train_x, "scaled:scale"))

train_x <- train_x%>% as.data.frame() %>% 
  mutate(epa_pm25=train$epa_pm25)
test_x <- test_x%>% as.data.frame() %>% 
  mutate(epa_pm25=test$epa_pm25)

#train multiple models with data set
xg_grid <- expand.grid(eta = 0.1,
                       max_depth = 6,
                       nrounds = 200,
                       gamma = 0,
                       colsample_bytree=.5, min_child_weight=0.5,subsample=0.1)
fitControl <- trainControl(method="none")

test_results <- tibble()

test_results <- tibble()
time_trains <- NULL
time_pred<-NULL
for(i in c(1:10)){
  #train the model and track time taken 
  time_start<-Sys.time()
  xg_model_pm <- caret::train(epa_pm25~.,data=train_x,preProcess=c("center","scale"), method="xgbTree",tuneGrid=xg_grid,trControl=fitControl)
  time_end<-Sys.time()
  time_tr_1<-as_hms(time_end-time_start)
  
  #save model
  write_rds(xg_model_pm,str_c("/Users/ttrefoni/Documents/tt_pm25tuning/output_models/xgb_n_model_01_05_70_30_",i))
  
  #test data set
  time_start<-Sys.time()
  xgb_n_pred_pm <- predict(xg_model_pm,test_x)
  time_end<-Sys.time()
  time_pred_1<-as_hms(time_end-time_start)
  
  xgb_n_metric_pm <- postResample(pred = xgb_n_pred_pm, obs = test_x$epa_pm25)
  
  xgb_n_mertric_df_pm <- data.frame(as.list(xgb_n_metric_pm)) 
  #bind to test results 
  test_results <- rbind(test_results,xgb_n_mertric_df_pm)
  
  #append train and test timing 
  time_trains<-c(time_trains,as.character(time_tr_1))
  time_pred<-c(time_pred,as.character(time_pred_1))
  print(str_c("completed model ",i," of 10"))
  
}

time_results<-tibble(c(1:nrow(test_results)),time_trains,time_pred)

colnames(time_results)<-c("model_run","time_train","time_predict")

#add timing columns to results
test_results <- test_results %>%
  mutate(model=c(1:nrow(test_results)),time_train=time_trains,time_pred=time_pred) %>%
  mutate(type="single")
#create aggregated results
test_results_ag <-
  test_results %>%
  summarize(RMSE=mean(RMSE),Rsquared=mean(Rsquared),MAE=mean(MAE),
            time_pred=mean(time_pred),time_train=mean(time_train)) %>%
  mutate(model=NA,type="agg")

#bind aggregatge and individual results
test_sum <-rbind(test_results,test_results_ag)
write_csv(test_sum,"/Users/ttrefoni/Documents/tt_pm25tuning/output/xgb_n_pm_results_01_05_70_30.csv")
# write to  file  ---------------------------------------------------------
file.create("/Users/ttrefoni/Documents/tt_pm25tuning/output/xgb_n_pm_results_01_05_70_30.txt")
file_con <- "/Users/ttrefoni/Documents/tt_pm25tuning/output/xgb_n_pm_results_01_05_70_30.txt"
write_lines("xgb_n results",file=file_con)

for(i in c(1:10)){
  write_lines(str_c("Model ",i," : RMSE = ",test_results$RMSE[i], " R^2 = ",test_results$Rsquared[i]), file=file_con,append=TRUE)
  print(i)
}
write_lines(str_c("Aggregate Model "," : RMSE = ",test_results_ag$RMSE, " R^2 = ",test_results_ag$Rsquared), file=file_con,append=TRUE)

