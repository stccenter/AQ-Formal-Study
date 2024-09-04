# GMU Systematic AQ Study.

This repository is organized to accommodate various deep learning models and libraries, with a clear hierarchical structure. At the top level, folders are categorized by different neural network architectures, such as DNN, RNN, etc. Within each architecture folder, there are further subfolders that correspond to specific libraries or frameworks, such as Tensorflow or PyTorch. Each of these subfolders contains the code, scripts, and metrics.

Data is organized into the "Training Data" folder. 

## TensorFlow
TensorFlow, coupled with its high-level API Keras, provides a robust environment for designing a diverse array of ML models. It is particularly effective for developing neural network models such as the ones employed in this study. TensorFlow uses 'tf.keras’ to implement regression models. RF and Lasso are built with extensions like TensorFlow Decision Forests (TF-DF) demonstrating its versatility across both deep learning and traditional ML domains. 

## RStudio
RStudio facilitates ML through its integration with R and Python, offering access to various packages and frameworks. It utilizes the Caret package for training conventional ML models such as RF and XGBoost. For regression models like OLS and Lasso, RStudio leverages native R packages and Python integrations through Reticulate. Advanced DL models including LSTM and RNN are also supported using TensorFlow and Keras, providing a flexible and powerful toolset for both classical and modern ML approaches. 

## PyTorch
PyTorch, known for its flexibility and powerful GPU acceleration, is used in the development of DL models. While it is not traditionally used for simple regression models, it is ideal for constructing complex neural network models. For some Regression modelling, external packages or custom implementations are necessary to bridge its capabilities to traditional statistical modeling tasks. 

## Scikit-Learn
Scikit-Learn is a comprehensive library used extensively for data preparation, model training, and evaluation across a spectrum of ML tasks such as classification, regression, and clustering. It supports many algorithms included in this study and other advanced regression and AI/ML algorithms. This package excels due to its ease of use, efficiency, and broad applicability in tackling both simple and complex ML problems. 

## XGBoost
XGBoost, standing for eXtreme Gradient Boosting, is a highly efficient implementation of gradient boosted decision trees designed for speed and performance. This standalone library excels in handling various types of predictive modeling tasks including regression, classification, and ranking. XGBoost can be used with several data science environments and programming languages including Python, R, and Julia, among others. XGBoost works well to build hybrid models as it integrates smoothly with both Scikit-Learn and TensorFlow via wrappers that allow its algorithms to be tuned and cross-validated in a consistent matter. It also functions well as a standalone model, using functions like XGBRegressor.  

