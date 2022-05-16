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

- Split the Data into Training and Testing Sets
Open the starter code notebook and then use it to complete the following steps.

Read the lending_data.csv data from the Resources folder into a Pandas DataFrame.

Create the labels set (y) from the “loan_status” column, and then create the features (X) DataFrame from the remaining columns.

Note A value of 0 in the “loan_status” column means that the loan is healthy. A value of 1 means that the loan has a high risk of defaulting.

Check the balance of the labels variable (y) by using the value_counts function.

Split the data into training and testing datasets by using train_test_split.

- Create a Logistic Regression Model with the Original Data
Employ your knowledge of logistic regression to complete the following steps:

Fit a logistic regression model by using the training data (X_train and y_train).

Save the predictions on the testing data labels by using the testing feature data (X_test) and the fitted model.

Evaluate the model’s performance by doing the following:

Calculate the accuracy score of the model.

Generate a confusion matrix.

Print the classification report.

Answer the following question: How well does the logistic regression model predict both the 0 (healthy loan) and 1 (high-risk loan) labels?

- Predict a Logistic Regression Model with Resampled Training Data
Did you notice the small number of high-risk loan labels? Perhaps, a model that uses resampled data will perform better. You’ll thus resample the training data and then reevaluate the model. Specifically, you’ll use RandomOverSampler.

To do so, complete the following steps:

Use the RandomOverSampler module from the imbalanced-learn library to resample the data. Be sure to confirm that the labels have an equal number of data points.

Use the LogisticRegression classifier and the resampled data to fit the model and make predictions.

Evaluate the model’s performance by doing the following:

Calculate the accuracy score of the model.

Generate a confusion matrix.

Print the classification report.

Answer the following question: How well does the logistic regression model, fit with oversampled data, predict both the 0 (healthy loan) and 1 (high-risk loan) labels?

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



- Machine Learning Model 2 :
    - Balanced Accuracy scores : 99.4%
    - "0" Precision : 100%
    - "0" Recall    : 99%
    - "1" Precision : 84%
    - "1" Recall    : 99%
 
Model 1 Calssification reoprt as follows:




## Summary

When comparing the regaression model, fit with oversampled data and original data, we saw it was gained about 4% in balanced accuracy. Although accuracy is increased, it is moportant to look into recall and precision because accuracy does not always tell the full story. Although we lose 1% in 1 (high-risk loan) predictions, but we gain 8% in recall. And we can see that the data are classified and predict the high-risk loan more correctly when using the oversampled data.

In machine learning model 2, we can see the balanced accuracy score percentage is increased 4.2% . Also the second model is the best fit for this application as it does a better job to classified overall "1" values correctly which is measured by recall. This information could be very beneficial to the firm to make sure that high-risk loans are managed correctly.

