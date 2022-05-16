# Credit_Risk_Classification
Challenge12

# Description

Credit risk poses a classification problem that’s inherently imbalanced. This is because healthy loans easily outnumber risky loans. In this Challenge, you’ll use various techniques to train and evaluate models with imbalanced classes. You’ll use a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers.

This challenge consists of the following subsections:

- Split the Data into Training and Testing Sets

- Create a Logistic Regression Model with the Original Data

- Predict a Logistic Regression Model with Resampled Training Data

- Write a Credit Risk Analysi Report

1. An overview of the analysis: Explain the purpose of this analysis.

2. The results: Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of both machine learning models.

3. A summary: Summarize the results from the machine learning models. Compare the two versions of the dataset predictions. Include your recommendation, if any, for the model to use the original vs. the resampled data. If you don’t recommend either model, justify your reasoning.

# Technologies

This project used JupyterLab and used following Anaconda packages:

- pandas - Data analysis toolkit for Python.

- numpy - Fundamental package for scientific computing with Python.

- sklearn.metrics - Quantify the quality of predictions.

- sklearn.linear_model - LinearRegression modeling.

- sklearn.train_test_split - Split arrays or matrices into random train and test subsets.

- imbalanced-learn - Python package offering re-sampling techniques

# Instructions:

## - Split the Data into Training and Testing Sets
Open the starter code notebook and then use it to complete the following steps.

##### Import the modules
import numpy as np

import pandas as pd

from pathlib import Path

from sklearn.metrics import balanced_accuracy_score

from sklearn.metrics import confusion_matrix

from imblearn.metrics import classification_report_imbalanced

import warnings
warnings.filterwarnings('ignore')

##### Read the lending_data.csv data from the Resources folder into a Pandas DataFrame.
lending_df = pd.read_csv(
    Path('./Resources/lending_data.csv'))
    
##### Review the DataFrame
lending_df

##### Create the labels set (y) from the “loan_status” column, and then create the features (X) DataFrame from the remaining columns.
##### Separate the y variable, the labels
y = lending_df["loan_status"]

##### Separate the X variable, the features
X = lending_df.drop(columns="loan_status")

##### Review the y variable Series
y.head()

##### Review the X variable DataFrame
X.head()

##### Note A value of 0 in the “loan_status” column means that the loan is healthy. A value of 1 means that the loan has a high risk of defaulting.
##### Check the balance of the labels variable (y) by using the value_counts function.
lending_df["loan_status"].value_counts()

##### Split the data into training and testing datasets by using train_test_split.
##### Import the train_test_learn module
from sklearn.model_selection import train_test_split

##### Split the data using train_test_split
##### Assign a random_state of 1 to the function
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=1)

## - Create a Logistic Regression Model with the Original Data
Employ your knowledge of logistic regression to complete the following steps:

#####  Fit a logistic regression model by using the training data (X_train and y_train).
#####  Import the LogisticRegression module from SKLearn
from sklearn.linear_model import LogisticRegression

#####  Instantiate the Logistic Regression model
#####  Assign a random_state parameter of 1 to the model
logistic_regression_model = LogisticRegression(random_state=1)

#####  Fit the model using training data
lr_model = logistic_regression_model.fit(X_train,y_train)

#####  Save the predictions on the testing data labels by using the testing feature data (X_test) and the fitted model.
#####  Make a prediction using the testing data
test_data = lr_model.predict(X_test)

#####  Evaluate the model’s performance by doing the following:
#####  Calculate the accuracy score of the model.
#####  Print the balanced_accuracy score of the model
balanced_accuracy_score(y_test, test_data)

#####  Generate a confusion matrix for the model
confusion_matrix(y_test, test_data)

#####  Print the classification report for the model
print(classification_report_imbalanced(y_test, test_data))

Answer the following question:

Question: How well does the logistic regression model predict both the 0 (healthy loan) and 1 (high-risk loan) labels?

Answer: The logistic regression model well fit the original data and also does well predict both the 0 (healthy loan) and 1 (high-risk loan) labels.

For the 0 (healthy loan) predictions, there is a 100% precision and 99% recall. For the 1 (high-risk loan) predictions, there is a 85% precison and 91% recall.

## - Predict a Logistic Regression Model with Resampled Training Data
Did you notice the small number of high-risk loan labels? Perhaps, a model that uses resampled data will perform better. You’ll thus resample the training data and then reevaluate the model. Specifically, you’ll use RandomOverSampler.

To do so, complete the following steps:

#####  Use the RandomOverSampler module from the imbalanced-learn library to resample the data. Be sure to confirm that the labels have an equal number of data points.
#####  Import the RandomOverSampler module form imbalanced-learn
from imblearn.over_sampling import RandomOverSampler

#####  Instantiate the random oversampler model
#####  Assign a random_state parameter of 1 to the model
random_oversampler = RandomOverSampler(random_state=1)

#####  Fit the original training data to the random_oversampler model
X_resampled, y_resampled = random_oversampler.fit_resample(X_train, y_train)

#####  Count the distinct values of the resampled labels data
y_resampled.value_counts()

#####  Use the LogisticRegression classifier and the resampled data to fit the model and make predictions.
#####  Instantiate the Logistic Regression model
#####  Assign a random_state parameter of 1 to the model
model = LogisticRegression(random_state=1)

#####  Fit the model using the resampled training data
lr_resampled = model.fit(X_resampled, y_resampled)

#####  Make a prediction using the testing data
lr_prediction = lr_resampled.predict(X_test)

#####  Evaluate the model’s performance by doing the following:
#####  Calculate the accuracy score of the model.
#####  Print the balanced_accuracy score of the model 
print(balanced_accuracy_score(y_test,lr_prediction))

#####  Generate a confusion matrix for the model
confusion_matrix(y_test,lr_prediction)

#####  Print the classification report for the model
print(classification_report_imbalanced(y_test, lr_prediction))

Answer the following question:

Question: How well does the logistic regression model, fit with oversampled data, predict both the 0 (healthy loan) and 1 (high-risk loan) labels?

Answer: The logistic regression model well fit the oversampled data and well predict both the 0 (healthy loan) and 1 (high-risk loan).

For the 0 (healthy loan) predictions, there is a 100% precision and 99% recall.

For the 1 (high-risk loan) predictions, there is a 84% precison and 99% recall.

When comparing oversampled data and original data, it seems like we lose 1% in 1 (high-risk loan) predictions, but we gain 8% in recall. And we can see that the data are classified and predict the high-risk loan more correctly when using the oversampled data.

# Credit Risk Analysis Report

## Overview of the Analysis 

In this analysis, we evaluate lending data from lending services company and using supervised machine learning techniques and along with financail Python programming skills. We use this data to train and evaluate models, and to determine the credit of borrowers. We will evaluate and trained the model using the original data and oversampled data. The goal of this analysis is to develop the best model to predict the most accurate loan and need to classifed as a "Healthy loan" or "High-risk loan". 

The dataset tgat we are analyzing contains 8 clolumns and 77,536 rows. The column features included: loan_size, interest_rate, borrower_income. debt_to_income, num_of_accounts, derogatory_marks, total_debt and loan_status. The loans status is identified as "0" or "1". "0" indicate a "Healthey loan" and "1" indicate a "High-risk loan". We used logistic Regression modeling technique to find the probability dataset outcome. We also used oversampling technique to select the minority variable and add them to the training set.

- Split the Data into Training and Testing Sets
    - Randomly split the data so we have a portion of the data that we can train the model to as well as keep a portion of the data that we can use to test our model, and compare results.
    
- Create a Logistic Regression Model with the Original Data
    - Create a model, Fit the model using the training data, and use the model to make Predictions on the test data.
    
- Predict a Logistic Regression Model with the Original Training Data
    - Use the model to make predictions on the test data and analyze results.
    
- Create a Logistic Regression Model with the Resampled Data
    - Create a model, Fit the model using the resampled training data, and use the model to make Predictions on the test data.
    
- Predict a Logistic Regression Model with the Resampled Training Data
    - Use the model to make predictions on the test data and analyze results.
    
- Analalys and Compare Results

## The Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of both machine learning models.

- Machine Learning Model 1 :
    - Balanced Accuracy scores : 95.2%
    - "0" Precision : 100%
    - "0" Recall    : 99%
    - "1" Precision : 85%
    - "1" Recall    : 91%
    
Model 1 Calssification reoprt as follows: 
![](image/https://github.com/Khaingz/Credit_Risk_Classification/blob/main/Model-1.png)

- Machine Learning Model 2 :
    - Balanced Accuracy scores : 99.4%
    - "0" Precision : 100%
    - "0" Recall    : 99%
    - "1" Precision : 84%
    - "1" Recall    : 99%
 
Model 1 Calssification reoprt as follows: 
![](image/https://github.com/Khaingz/Credit_Risk_Classification/blob/main/Model-2.png)


## Summary

When comparing the regaression model, fit with oversampled data and original data, we saw it was gained about 4% in balanced accuracy. Although accuracy is increased, it is moportant to look into recall and precision because accuracy does not always tell the full story. Although we lose 1% in 1 (high-risk loan) predictions, but we gain 8% in recall. And we can see that the data are classified and predict the high-risk loan more correctly when using the oversampled data.

In machine learning model 2, we can see the balanced accuracy score percentage is increased 4.2% . Also the second model is the best fit for this application as it does a better job to classified overall "1" values correctly which is measured by recall. This information could be very beneficial to the firm to make sure that high-risk loans are managed correctly.
