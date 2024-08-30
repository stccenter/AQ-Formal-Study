# setup -------------------------------------------------------------------
library(caret)
library(tictoc)
library(tidyverse)
library(kernlab)
library(brnn)
library(RSNNS)
library(doParallel)
library(rpart)
library(lubridate)
library(hms)

# read in data ------------------------------------------------------------

# 
# train_data_fp <- "/Users/CISC/Documents/tt_pm25tuning/data/TrainData/TrainData"
# td_files <- list.files(train_data_fp)
# 
# df <- str_c(train_data_fp,"/",td_files) %>% 
#   lapply(read_csv) %>% 
#   bind_rows 

#write to one csv
# write_csv(df,"/Users/CISC/Documents/tt_pm25tuning/data/TrainData/single_trainpm.csv")
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

test_results <- tibble()
time_trains <- NULL
time_pred<-NULL

for(i in c(1:10)){
  time_start<-Sys.time()
  DT_reg_model_pm <- rpart(epa_pm25 ~ .,  
               method = "anova", data = train_x )
  time_end<-Sys.time()
  time_tr_1<-as_hms(time_end-time_start)

  #test data set
  time_start<-Sys.time()
  DT_reg_pred_pm <- predict(DT_reg_model_pm,test_x)
  time_end<-Sys.time()
  time_pred_1<-as_hms(time_end-time_start)
  #calculate metrics
  DT_reg_metric_pm <- postResample(pred = DT_reg_pred_pm, obs = test_x$epa_pm25)
  
  DT_reg_mertric_df_pm <- data.frame(as.list(DT_reg_metric_pm)) 
  #bind to test results 
  test_results <- rbind(test_results,DT_reg_mertric_df_pm)
  
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
            time_pred=as_hms(mean(as_hms(time_pred))),time_train=as_hms(mean(as_hms(time_train)))) %>%
  mutate(model=NA,type="agg")

#bind aggregatge and individual results
test_sum <-rbind(test_results,test_results_ag)
write_csv(test_sum,"/Users/ttrefoni/Documents/tt_pm25tuning/output/DT_reg_pm_results_01_08_70_30.csv")
# write to  file  ---------------------------------------------------------
file.create("/Users/ttrefoni/Documents/tt_pm25tuning/output/DT_reg_pm_results_01_08_70_30.txt")
file_con <- "/Users/ttrefoni/Documents/tt_pm25tuning/output/DT_reg_pm_results_01_08_70_30.txt"
write_lines("DT_reg results",file=file_con)

for(i in c(1:(nrow(test_sum)-1))){
  write_lines(str_c("Model ",i," : RMSE = ",test_results$RMSE[i], " R^2 = ",test_results$Rsquared[i]), file=file_con,append=TRUE)
  print(i)
}
write_lines(str_c("Aggregate Model "," : RMSE = ",test_results_ag$RMSE, " R^2 = ",test_results_ag$Rsquared), file=file_con,append=TRUE)
