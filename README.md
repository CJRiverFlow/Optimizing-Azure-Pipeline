# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, I have built and optimized an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is later compared to an Azure AutoML run.

## Summary
The dataset used in this project contains Bank marketing campaign data, this information is based on phone calls. This dataset have 32950 client registries with 20 data columns with information like age, job, loan, marital status, education, campaign, month, contact and other economic information, also it contains a target column with information if the client subscribed or not subscribed to a term deposit or not. We seek to predict this target column by applying different machine learning pipelines in azure.  
  
Two models have been built, a custom scikit-learn logistic regression, using HyperDrive for hyperparameter tuning. The other model was created with AutoML or Azure Automated Machine Learning where several algorithms are trained and evaluated.  
Comparing both pipelines we found that the model with best accuracy was VotingEnsemble with 0.9178 of accuracy from AutoML. 

## Scikit-learn Pipeline
As first step a compute instance is created in azure, using the jupyter access point two files are added: train.py and the project notebook. In the script file a Tabular Dataset is created with the Bank Marketing URL, this dataset is preprocessed by a data cleaning and formating function and later splitted in 70% for training and 30% for testing datasets, the algorithm used for the model training is Logistic Regression, this script receives the hyperparameters, score the model and save it on an output directory.

On the project notebook, a sklearn estimator is created, this uses the train script and a compute cluster we created previously. Hyperdrive is a azure package that contains modules and clases for hyperparameter tuning, in the configuration a random parameter sampling is selected to avoid bias due its advantages in the speed/accuracy tradeoff, with this sampler we defined a discrete search space for the model hyperparameters 'C' or Inverse of Regularization Strength and 'max_iter' or Maximum number of iterations to converge. Random sampling was selected in this project as this supports early termination of low perfomance runs and can be completed faster with a compute cluster, the Grid Sampling also support early termination but it is used for exhaustive or very longer runs, this aspect is similar to the Bayesian Sampling, so this two require higher budget and time to explore the hyperparameter space, also the Bayesian sampling requires a small number of concurrent runs in order to have a better convergence. 

An important Hyperdrive parameter is the early stopping policy which helps to avoid overtraining and works when the run has slow perfomance, the policy selected in the project is Badit Policy which is based on slack factor/slack amount and evaluation interval. The Bandit terminates runs where the primary metric is not within the specified slack factor/slack amount compared to the best performing run. The Bandit policy was selected in this project because it monitors the best model and kills the all the runs that under the settings, having a better control on the resources and time on each run, more than the Median Stopping policy which can lengthen them because it uses an average of the last measurements keeping some unnecesary runs for more time, also the Truncation Selection Policy only kills a percentage of the low perfomance runs keeping all the others. 

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
* Increase the discrete search space for the parameter sampler in the hyperdrive configuration, this will enhance the sklearn model with a wider range of option values for the "C" and "max_iter" hyperparameters of the logistic regression, allowing to train and then select the best model with higher accuracy.
* Create the Logistic Regression model by passing not only the two parameters used in this project, it is possible to set other hyperparameters for this class like: tolerance of stopping criteria, penalty, intercept_scaling, solver, n_jobs, l1_ratio. Adding this hyperparameters will allow us to have a more complete control on the training model and we would be able to increase the perfomance of the model.
* Increase the 'experiment_timeout_minutes' in the AutoML configuration, allowing the evaluation of more algorithms like ANNs, and the final ensemble model will have better perfomance. The AutoML config can include also other parameters like 'enable_stack_ensemble' that is another method to combine the predictions from multiple well-performing machine learning models, also the parameter 'enable_dnn' will allows us to include deep neural network models during selection (good to work with higher timeouts), adding this parameters will include new capabilities to the new AutoML run improving the final emsemble model accuracy.       
