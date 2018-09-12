BayesianOptimization
==================== 
CSE515T: Bayesian Methods in Machine Learning
Final Project: Bayesian Optimization with Support Vector Regression

Members: 

Seunghwan "Nigel" Kim

Jae Sang Ha


Investigating and Applying Bayesian Optimization on two datasets.

We propose to use the Wine Quality dataset found from UCI Machine Learning Repository(https://archive.ics.uci.edu/ml/datasets/Wine+Quality), which was originally created for research purpose[Cortez et al. 2009]. Our dataset have total of 4898 instances and 11 features. 11 input features are fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol, and one output is quality of wine represented as score from 0 to 10. 
	The dataset consists of two data -red wine and white wine- that has wine’s physicochemical and sensory characteristics. The output label class is ordered (0 to 10) but it is not balanced. From what we have observed from the class distribution, both wine datasets follow a behavior of a distribution similar to that of the normal distribution. Also, it is mentioned with the dataset that the outputs are centered around the mean of 5, and does not include class values smaller than 3 or bigger than 8. 
From this dataset and its output distribution behavior, we can carefully propose that this dataset can be treated in a Bayesian way, for we can set a prior on the score range(output). Also, this could be treated in a frequentist way by applying Naive Bayes Classifier to classify multiclass dataset. 

# Please see 515 Project Report.docx for a full final report on the project #


Reference
--------- 
P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

