# Predicting Vehicle Loan Default using AdaBoost
This project addresses the challenge of predicting vehicle loan default of Indian borrowers using data acquired in 2019 from the `LTFS Data Science FinHack (ML Hackathon)`. Check out the **[link](https://datahack.analyticsvidhya.com/contest/ltfs-datascience-finhack-an-online-hackathon/)** to the competition for a detailed dataset and problem description.

# Project Overview
The objective is to accurately predict the probability of loanee / borrower defaulting on a vehicle loan in the first EMI (Equated Monthly Installments) on the due date. Python with Jupyter notebooks are used for the implementation, relying heavily on scikit-learn library. AUC is utilized as the evaluation metric due to class imbalance. In the project pipelines are extensively used to avoid data leakage and simplify transformations.

AdaBoost is chosen because it's a more sophisticated model capable of capturing non-linear patterns and uses less RAM compared to RandomForest.

## Data Description
* `data_dictionary.csv` contains a brief description on each variable provided in the training and test set.
* `train.csv` contains the training data with details on loan as described in the last section
* `test.csv` contains details of all customers and loans for which the participants are to submit probability of default.

## Project Structure
* The key emphasis in the project is modularity, which is why multiple notebooks and Python files are created. The data is first explored and investigated in detail in `data_exploring.ipynb`, and using the insights gained, modelling is then performed in `modelling.ipynb`. Transformations done in `data_exploring.ipynb` are extracted into functions in `helpers.py` so they can be effectively used in a pipeline.
* `config.py` contains global constants, denoted with uppercase letters, used in all notebooks.
* `helpers.py` contains custom functions and classes used in all notebooks.

## Modelling
The key steps performed in modelling are following:
* Basic model training -> Evaluating performance of relatively basic AdaBoost models, one trained on a 50,000-sample subset and the other on the entire training dataset. The same test data is used for both models;
* Learning curves -> Creating two learning curves: one for the training set size and another for the `n_estimators` hyperparameter of AdaBoost;
* Tuning all hyperparameters except the split criterion for the AdaBoost estimator using BayesSearchCV;
* Tuning the 'criterion' hyperparameter for the base estimator of AdaBoost using GridSearchCV;
* Evaluating the model with the best hyperparameters using the default decision threshold on test data;
* Optimizing the decision threshold using the F1 metric;
* Evaluating the model with the best hyperparameters using the optimized decision threshold on the test data;
* Training the best model on the entire dataset;
* Evaluating the best model on the entire dataset;
* Creating predictions for `test.csv` and submitting results.

## Results

Below are the AUC results for CV and the optimized AdaBoost model:

| Dataset                   | AUC    |
|-------------------|----------------|
| CV  | 65.888% |
| Train Set  | 67.144% |
| **Test Set**   | **66.302%** |
| Entire Set | 67.069% |

This approach would approximately yield a ranking of 113th place on the Private Leaderboard.

## Presentation
The presentation in English is available as PDF in this repository: [link](https://raw.githubusercontent.com/aleksa-radojicic/lt_loan_default_prediction_adaboost/refs/heads/main/Presentation.pdf).