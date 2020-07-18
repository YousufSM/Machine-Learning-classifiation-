# Machine-Learning-classification-
Classification task using GLM, XGBoost and AutoML


As an assignment under the module of Applied Machine Learning, our task was to create 3 classification models and compete on a private Kaggle competition. The [data](https://www.kaggle.com/c/aml2020/data) provided had 97 masked variables.

I have used [H20](https://www.h2o.ai/) packages since this was the project requirement plus H20 provides seamless solutions to handle missing values, create variable interactions, provide grid search and [Bayesian Hyper Parameters](https://towardsdatascience.com/a-conceptual-explanation-of-Bayesian-model-based-hyperparameter-optimization-for-machine-learning-b8172278050f) using HeperOpt. H20 is also been preferred due to its capability of parallel processing on multicore devices. Additionally, it provides an easier implementation of XGBoost and AutoML to me.


Details of the models have been discussed below:

## 1. Logistic Regression using H2OGeneralizedLinearEstimator
Steps involved in the model are:
* Creating splines for most important numeric variables. Importance of variables has been identified through a random forest method.
* Treating high cardinality variables (categorical variables having >30 factors) using Target Encoders.
* Creating Interactions for most important categorical variables. Importance of the variables has been identified through binomial regression. Adding Interactions increased the Kagle score from 77% to 83%
* Running random grid search on **'alpha'** paramter for the H2OGeneralizedLinearEstimator(binomial) function.

The model scored an accuracy of 0.848 on Kaggle.

## 2. XGBoost using Random Grid Search and HyperOpt
Steps involved in the model are:
* Treating high cardinality variables (categorical variables having >30 factors) using Target Encoders. Because I did not want H20 to create one-hot encodings for these high cardinality variables.
* Creating Interactions for most important categorical variables. Importance of the variables has been identified through binomial regression.
* Running Random Grid Search and HyperOpt to get best hyperparameters. Eventually, Hyperopt returned a better result with less time. Hyper-parameters chosen for Grid search and HyperOpt were **'max_depth', 'sample_rate', 'min_rows', 'n_trees', 'learn_rate'**.

The model scored an accuracy of 0.865 on Kaggle

## 3. AutoML (Automated Machine Learning)
Steps involved in the model are:
* Creating splines for most important numeric variables. Importance of variables has been identified through a random forest method.
* Treating high cardinality variables (categorical variables having >30 factors) using Target Encoders.
* Creating Interactions for most important categorical variables. Importance of the variables has been identified through binomial regression.
* Running AutoML.

The model scored an accuracy of 0.860 on Kaggle

The [official document](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html) of H20 AutoML states "The H2O AutoML interface is designed to have as few parameters as possible so that all the user needs to do is point to their dataset, identify the response column and optionally specify a time constraint or limit on the number of total models trained." Therefore, there were not many parameters to optimise. few of the parameters that have been optimised for the algorithm are discussed in the notebook.

I have also tried AutoML using [TPOT](https://github.com/EpistasisLab/tpot) as well, but in my opinion, I found AutoML from H20 much easier due to its capabilities of handling NAs and its speed. But to explore the data preparation and feature engineering steps TPOT is a better tool.
