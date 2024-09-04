# GMU Systematic AQ Study.

This repository is organized to accommodate various deep learning models and libraries, with a clear hierarchical structure. At the top level, folders are categorized by different neural network architectures, such as DNN, RNN, etc. Within each architecture folder, there are further subfolders that correspond to specific libraries or frameworks, such as Tensorflow or PyTorch. Each of these subfolders contains the code, scripts, and metrics.

1. **Organize and Locate Files**  
   a. All training data files are stored in the Training Data folder. Each dataset is in CSV format and includes necessary variables such as "temperature," "relative humidity," "pm25_cf_1," and "epa_pm25."  
   b. The repository is organized by neural network architecture folders at the top level (e.g., DNN/, RNN/). Inside each architecture folder, there are subfolders for different libraries or frameworks (e.g., TensorFlow/, PyTorch/), containing the relevant model code, scripts, and metrics.  
   c. Each library or framework subfolder contains a config folder where the configuration files for the models are stored. For example:  
      - TensorFlow: `DNN/TensorFlow/`  
      - RStudio: `DNN/RStudio/`  
      - PyTorch: `DNN/PyTorch/`  
      - Scikit-Learn: `DNN/Scikit-Learn/`  
      - XGBoost: `DNN/XGBoost/`

2. **Install the Required Software**  
   a. **TensorFlow:**  
      Download and install using pip:  
      ```
      pip install tensorflow tensorflow_decision_forests
      ```
   b. **RStudio:**  
      Download and install RStudio from RStudio's official website: [RStudio IDE](https://www.rstudio.com/categories/rstudio-ide/).  
      Install the required R packages (example):  
      ```
      install.packages(c("caret", "reticulate"))
      ```
   c. **PyTorch:**  
      Download and install using pip:  
      ```
      pip install torch
      ```
   d. **Scikit-Learn**  
      Download and install using pip:  
      ```
      pip install scikit-learn
      ```
   e. **XGBoost**  
      Download and install using pip:  
      ```
      pip install xgboost
      ```

3. **Locate Configuration Files**  
   Once all the necessary software is installed, locate the files for each package. These files include the preprocessing, configuration, and model run, but our focus here is the model parameters which may be altered in each of the individual scripts (see 1.c.).

4. **Follow Model README Steps**  
   Each model folder contains a README file with specific instructions on how to run the models. Follow these steps for each package:  
   - **Prepare the Data:**  
     Preprocess the data by selecting only sensor pairs with a Pearson correlation of 0.7 or higher.  
     Reduce the dataset to the variables: "temperature," "relative humidity," "pm25_cf_1," and "epa_pm25."  
     Split the data into training and testing sets (e.g., 80/20 or 70/30).  
   - **Set Hyperparameters:**  
     Use the default hyperparameters specified in the configuration files unless modifications are required.  
   - **Run the Models:**  
     Execute the scripts or commands outlined in the README files to train and evaluate the models. This could involve running a Python or R script depending on the model and framework.

5. **Integrate All Results**  
   After running the models, you should load each output into a "results.csv" file. Please include the R^2, RMSE, and elapsed time.  
   Later, we will use this "results.csv" file to visualize our data.

6. **Visualize Results**  
   To visualize the results, we will employ RStudio, specifically, R's package "ggplot2". Make sure to load in your consolidated "results.csv" file.

## TensorFlow

TensorFlow, coupled with its high-level API Keras, provides a robust environment for designing a diverse array of ML models. It is particularly effective for developing neural network models such as the ones employed in this study. TensorFlow uses 'tf.keras’ to implement regression models. RF and Lasso are built with extensions like TensorFlow Decision Forests (TF-DF) demonstrating its versatility across both deep learning and traditional ML domains.

## RStudio

RStudio facilitates ML through its integration with R and Python, offering access to various packages and frameworks. It utilizes the Caret package for training conventional ML models such as RF and XGBoost. For regression models like OLS and Lasso, RStudio leverages native R packages and Python integrations through Reticulate. Advanced DL models including LSTM and RNN are also supported using TensorFlow and Keras, providing a flexible and powerful toolset for both classical and modern ML approaches.

## PyTorch

PyTorch, known for its flexibility and powerful GPU acceleration, is used in the development of DL models. While it is not traditionally used for simple regression models, it is ideal for constructing complex neural network models. For some Regression modelling, external packages or custom implementations are necessary to bridge its capabilities to traditional statistical modeling tasks.

## Scikit-Learn

Scikit-Learn is a comprehensive library used extensively for data preparation, model training, and evaluation across a spectrum of ML tasks such as classification, regression, and clustering. It supports many algorithms included in this study and other advanced regression and AI/ML algorithms. This package excels due to its ease of use, efficiency, and broad applicability in tackling both simple and complex ML problems.

The data is first preprocessed to only sensor pairs with a Pearson correlation of 0.7 or higher, reduced to the variables "temperature", "relative humidity", "pm25_cf_1", and "epa_pm25" which correspond to temperature, relative humidity, PM2.5 values from Purple Air sensors where the calibration factor equals 1, and PM2.5 values from the co-located EPA sensor. This is then put into a random test-train split (80/20 vs 70/30). The hyperparameters are then set in accordance with the defaults of each package. These hyperparameters are then fed into a model approximately ten times, where we then find the average of the R^2, RMSE, and time elapsed. These results are then appended to a results.csv with package name, model name, R^2, RMSE, time elapsed, etc. for later visualization.

## XGBoost

XGBoost, standing for eXtreme Gradient Boosting, is a highly efficient implementation of gradient boosted decision trees designed for speed and performance. This standalone library excels in handling various types of predictive modeling tasks including regression, classification, and ranking. XGBoost can be used with several data science environments and programming languages including Python, R, and Julia, among others. XGBoost works well to build hybrid models as it integrates smoothly with both Scikit-Learn and TensorFlow via wrappers that allow its algorithms to be tuned and cross-validated in a consistent matter. It also functions well as a standalone model, using functions like XGBRegressor.

The data is first preprocessed to only sensor pairs with a Pearson correlation of 0.7 or higher, reduced to the variables "temperature", "relative humidity", "pm25_cf_1", and "epa_pm25" which correspond to temperature, relative humidity, PM2.5 values from Purple Air sensors where the calibration factor equals 1, and PM2.5 values from the co-located EPA sensor. This is then put into a random test-train split (80/20 vs 70/30). The hyperparameters are then set in accordance with the defaults of each package. These hyperparameters are then fed into a model approximately ten times, where we then find the average of the R^2, RMSE, and time elapsed. These results are then appended to a results.csv with package name, model name, R^2, RMSE, time elapsed, etc. for later visualization.
