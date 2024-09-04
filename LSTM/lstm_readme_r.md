

# Overview
This R script implements a Long Short-Term Memory (LSTM) model using the keras package for time series prediction. The script includes data preprocessing, model training, evaluation, and saving results.

Instructions
1. Install Required Packages
Before running the script, you need to install all the necessary R packages. Open R or RStudio and run the following command to install the packages:
    
    ```install.packages(c("caret", "tictoc", "tidyverse", "keras", "mlbench", "dplyr", "magrittr", "neuralnet", "nnet",       "tensorflow", "reticulate", "hms", "ggplot2"))```

2. Change Data Path or Python Executable Path
    
    To run the script on your computer, you need to update the paths to match the location of your files and Python installation.
    
    * Data Path:
    Find line 67 in the script where the data file is loaded. Replace the file path with the path where your data file is located on your computer.

    Example: If your data file is located in "C:/mydata/single_trainpmv2.csv", change the line to:
        
        ```df <- read_csv("C:/mydata/single_trainpmv2.csv")```

    * Python Executable Path:
    If you are using TensorFlow with Python, find line 133 in the script and update the Python executable path to match where Python is installed on your system.
        
        Example: If Python is installed in "C:/Python/Scripts/python.exe", change the line to:

        ```use_python("C:/Python/Scripts/python.exe")```

    * Save and Read Sequence Data:
    Find lines 89 and 90 in the script where the sequence data is saved and read. Update these paths to where you want to save and read the sequence data on your computer.

    Example: If you want to save and read the sequence data from "C:/mydata/sequences_pm25_group_sensor_01_20", change the lines to:

    `saveRDS(sequences, "C:/mydata/sequences_pm25_group_sensor_01_20")```
   
    ```sequences <- read_rds("C:/mydata/sequences_pm25_group_sensor_01_20")```


4. Change Hyperparameter Settings
    If you want to adjust the model's hyperparameters, such as the number of epochs (iterations) or batch size, find the relevant lines in the script:

    
    Find line 143 and change the values for epochs and batch_size as needed.

    Example: To change to 20 epochs and a batch size of 64, modify the line:

    ```fit_result <- fit(object = wlstm_model, x = x_train_array, y = y_train_array, epochs = 20, batch_size = 64, verbose =0)```

5. How to Run the Script

        1. Open R or RStudio.
        2. Make sure you have installed all the required packages and updated the paths for your data file and Python executable.
        3. Run the entire script or execute it line by line to process your data, train the model, and evaluate the results.


6. Find Results and Change Output Path if Necessary
    * Results Location:
        
        By default, the script saves results as CSV and text files. The output files are saved to a specific location on your computer.

        Example paths used in the script:

            ```write_csv(test_sum, "C:/Users/ttrefoni/Documents/tt_pm25_tuning/output/LSTM_pm_results_01_19_70_30.csv")```

            ```file.create("C:/Users/ttrefoni/Documents/tt_pm25_tuning/output/LTSM_pm_results_01_19_70_30.txt")```

    * Change Output Path:
        
        To change where the results are saved, update the paths on lines 184, 186, and 187 to your desired directory.

        Example: If you want to save the results in "D:/myresults/", change the lines to:

            ```write_csv(test_sum, "new_output_directory/LSTM_pm_results.csv")```
      
            ```file.create("new_output_directory/LTSM_pm_results.txt")```

