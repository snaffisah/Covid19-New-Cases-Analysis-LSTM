# Covid19 New Cases Analysis LSTM

Creating a model prediction for Covid19 new cases (continuous data) in Malaysia using deep learning approach with LSTM neural network

### Description
Objective: Create a deep learning model using LSTM neural
network to predict new cases in Malaysia using the past 30 days
of number of cases

* Model training - Deep learning
* Method: Sequential, LSTM
* Module: Sklearn & Tensorflow

In this analysis, dataset used from https://github.com/MoH-Malaysia/covid19-public

### About The Dataset:
There are 2 dataset used in this analysis:-
1. cases_malaysia_train.csv (680 data entries with 31 column)
   25/1/2020-4/12/2021
2. cases_malaysia_test.csv (100 data entries with 31 column)
   5/12/2021-14/3/2022

To predict new cases, we only focus on 'cases_new' column. There are few missing data and symbol found and data cleaning process were applied.

### Deep learning model with LSTM layer
A sequential model was created with 2 LSTM layer, 2 Batch Normalization layer, 2 Dropout layer, 1 Dense layer:
<p align="center">
  <img src="https://github.com/snaffisah/Covid19-New-Cases-Analysis-LSTM/blob/main/Image/model%20architecture.JPG">
</p>

<p align="center">
  <img src="https://github.com/snaffisah/Covid19-New-Cases-Analysis-LSTM/blob/main/Image/model%20flow.JPG">
</p>
Batch normalization, do help reduce the error as it scaled all data into the same scale and get improvement with the training speed. 

For the dropout layer, it help in reducing the overfitting and generalization error.

Data were trained with 200 epoch:
<p align="center">
  <img src="https://github.com/snaffisah/Covid19-New-Cases-Analysis-LSTM/blob/main/Image/epoch.JPG">
</p>

### Result
<p align="center">
  <img src="https://github.com/snaffisah/Covid19-New-Cases-Analysis-LSTM/blob/main/Image/result.JPG">
</p>

By using the created model, able to achieve 0.2% for mean absolute percentage error. Its good enough to be used for future new cases prediction and goverment can take the necessary precaution steps to avoid it from spreading.

### How to run the pythons file:
1. Load the module 1st by running 'Covid19_newcase_analysis_module.py'
2. Run training file 'Covid19_newcase_analysis_train.py' 

Enjoy!

