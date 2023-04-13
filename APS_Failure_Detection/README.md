<h1><b> Predicting Air pressure system failures in Scania trucks </b></h1>

 
The dataset consists of data collected from heavy Scania trucks in everyday usage. 
The system in focus is the Air Pressure system (APS) which generates pressurized air that is utilized in various functions in a truck, such as braking and gear changes.
The datasets' positive class consists of component failures for a specific component of the APS system. 
The negative class consists of trucks with failures for components not related to the APS. 
The training set contains 60000 examples in total in which 59000 belong to the negative class and 1000 positive class. 
The test set contains 16000 examples. There are 171 attributes per record.  

Dataset - https://archive.ics.uci.edu/ml/datasets/APS+Failure+at+Scania+Trucks

<h4>
For the data cleaning part,  we addressed null values in the data cleaning process by replacing them with either the median or mode depending on the variable 
data type (int, float, object). Skewness values greater than 1 or less than -1 indicate highly skewed distributions (right/positive or left/negative skewness),
while values between 0.5 and 1 or -0.5 and -1 indicate moderately skewed distributions. A value between -0.5 and 0.5 suggests a fairly symmetrical distribution.
Since our data was mostly left-skewed, we chose to replace it with the median.

We also measured the correlation between two variables using a scale ranging from -1 to 1. A correlation of 1 represents a perfect positive linear relationship,
while a correlation of -1 indicates a perfect negative linear relationship, and a correlation of 0 signifies no linear relationship. In our case, 
we found both positive and negative linear relationships among the features.

For feature scaling it involves standardizing or normalizing the features in a dataset to ensure that they have similar statistical properties, 
such as mean and variance, and are on the same scale. This is important for several reasons.

In linear regression, feature scaling is important for improving the performance of the model. 
Algorithms such as gradient descent are sensitive to the scale of the features and can require a lot of computation to converge to a solution. 
Feature scaling can reduce the computational complexity of these algorithms by scaling the features to a similar range. 
Additionally, feature scaling can make it easier to interpret the results of the model, as the scale of the features will be 
similar and the coefficients of the model will be easier to compare.

In our case, we have used PCA for the dimensionality reduction of high-dimensional data by projecting it onto a lower-dimensional 
space while retaining as much variance as possible.

Feature scaling is not necessary for Naive Bayes because the algorithm is not sensitive to the scale of the features.
Naive Bayes assumes that the features are conditionally independent given the class variable, so the scale of the features does not affect
the performance of the algorithm.

In Random Forest, feature scaling is not necessary because the algorithm is not sensitive to the scale of the features. 
Random Forest is an ensemble algorithm that builds multiple decision trees and averages their predictions, so the scale of the features does not affect
the performance of the algorithm. 
However, feature scaling can still be applied if it improves the performance or interpretability of the model.

Grid search is a hyperparameter tuning technique used to find the optimal combination of hyperparameters for a machine learning model. 
Performance metrics are used to evaluate the performance of a model based on its predictions.

We have used grid search to evaluate a model's performance, by defining a set of hyperparameters and their corresponding values that we want to test.
Then, we have trained and evaluated the model using each combination of hyperparameters in the grid search.

Once the grid search is complete, we can compare the performance of the model using different performance metrics 
such as accuracy, precision, recall, F1 score, etc and choose for the best model.
These metrics provide insights into how well the model is performing and how well it can generalize to new data.

The random forest model exhibited the best overall performance in our case. Therefore, we will be utilizing it to predict the classes for the testing data.

</h4>
