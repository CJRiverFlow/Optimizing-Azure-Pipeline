# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, I have built and optimized an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is later compared to an Azure AutoML run.

## Summary
The dataset used in this project contains Bank marketing campaign data, this information is based on phone calls.
We seek to predict if the client subscribed to a term deposit or not.  
Two models have been built, a custom scikit-learn logistic regression, using HyperDrive for hyperparameter tuning. The other model was
created with AutoML or Azure Automated Machine Learning where several algorithms are trained and evaluated.  
Comparing both pipelines we found that the model with best accuracy was VotingEnsemble with 0.9178 of accuracy from AutoML. 

## Scikit-learn Pipeline
As first step a compute instance is created in azure, using the jupyter access point two files are added: train.py and the project notebook. In the script file a Tabular Dataset is created with the Bank Marketing URL, this dataset is preprocessed by a data cleaning and formating function and later splitted in 70% for training and 30% for testing datasets, the algorithm used for the model training is Logistic Regression, this script receives the hyperparameters, score the model and save it on an output directory.

On the project notebook, a sklearn estimator is created, this uses the train script and a compute cluster we created previously. Hyperdrive is a azure package that contains modules and clases for hyperparameter tuning, in the configuration a random parameter sampling is selected to avoid bias due its advantages in the speed/accuracy tradeoff, with this sampler we defined a discrete search space for the model hyperparameters 'C' or Inverse of Regularization Strength and 'max_iter' or Maximum number of iterations to converge. Another important Hyperdrive parameter is the early stopping policy which helps to avoid overtraining and works when the run has slow perfomance, the policy selected in the project is Badit Policy which is based on slack factor/slack amount and evaluation interval. The Bandit terminates runs where the primary metric is not within the specified slack factor/slack amount compared to the best performing run.

## AutoML
Azure AutoML automates the process of training, evaluation and other iteratives steps in the machine learning modeling, this includes a complete pipeline that aims to offer high scale, efficiency and quality models. The AutoMLConfig class has been configured with the following parameters: 
1. experiment_timeout_minutes - 35 minutes, 
2. task - classification,
3. primary_metric - accuracy, 
4. training_data - training_data (Tabular Dataset preprocessed), 
5. compute_target - mycluster,
5. label_column_name - y, 
6. n_cross_validations - 10. 

From the Voting Emsemble model, which displayed better accuracy, some of the hyperparameters selected are:  
* n_estimators=25,
* n_jobs=1
* min_samples_leaf=0.01,
* min_samples_split=0.01,
* min_weight_fraction_leaf=0.0,

## Pipeline comparison
The best sklearn model created with hyperdrive has a accuracy value of 0.9125, with the AutoML run the best model was the VotingEnsemble with 0.9178, this difference is small but the AutoML got better perfomance. The main difference in the both models are the structure, the hyperdrive run consisted in one model while the AutoMl train and evaluates different models like LightGBM, XGBoost or RandomForest and also applies emsemble models that combines multiple learning algorithms to obtain better predictive performance, that is in our case the model with best accuracy.    

## Future work
* Compare models based on other metrics like Recall, F1 Score or ROC.
* Test other Early Termination Policy and compare results with Bandit selection.
* Compare different options for the parameter sampler in the hyperdrive configuration.
