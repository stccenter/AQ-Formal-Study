# LSTM with python-tensorflow
### Disclamer: (This README was written with Windows + VS Code in mind, if using mac or linux libraries and performance may differ)

## Install the packages:
tensorflow with cpu, table with necessary versions (python, tensorflow)
Using python with pip installed, run the following command:
`pip install notebook tensorflow scikit-learn numpy pandas scipy`


## Change data path:
Both files in the Python-Tensorflow directory will require you to alter the path to the training data that you're using.

### If using included data:
The path to the data included in this repository relative to the python code will be: `'../../Training Data'`
Simply go into both Jupyter Notebook files in the "Python-Tensorflow" directory and locate the training data directory constant, this will be the last line of code in the first cell in each file:
`TRAIN_DATA_DIRECTORY = 'path/to/your/train/data'`

Simpy change the path to the path included above

### If using your own:
In the event you are not using the data included in this repository by default, simply follow the steps above however you will replace the provided directory with your own directory.

## Change hyperparameter settings
To change the hyperparameters, navigate to: "Python-Tensorflow/create_python_LSTM.ipynb"
Once there scroll down to the second cell it will look like this:
```python
#define HPS
epoch=30
batchsize=32
units1=50
units2=50
lrate=0.001
```
Once you've found this, all you need to do is change the values to whatever your preference is.


## How to run the scripts?
To run the scripts you will navigate to the Jupyter notebooks starting with "Python-Tensorflow/gen_training_data_updt.ipynb"
Once there, you will navigate to the top right of your Jupyter notebook and select "Select Kernel"
![Select Kernel Screenshot](../Tutorial%20Screenshots/select_kernel.png)
You will then select the kernel that you've installed all libraries with as well as Jupyter notebooks.
Finally, you will press "Run All", also found in the Jupyter notebook.

After you have finished running the data generation you will go back and perform the same thing with "Python-Tensorflow/create_python_LSTM.ipynb"

## Where to find the results? 
After you have finished running the code, you will be able to find your results in the "Python-Tensorflow" directory, the will be in the form of a `.csv` and a `.keras` files.