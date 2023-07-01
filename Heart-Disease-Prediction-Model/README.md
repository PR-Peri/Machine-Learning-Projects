# Classification of Heart Disease using Machine Learning Techniques


  The most complicated and complex task in the field of medical is the prediction of heart disease becuase Heart is considered to be the most vital organ of the human body. In general, the objective of my project is to classify and predict heart disease accurately. If the data at hand is used to develop screening and diagnostic models, it will not only reduce the strain on medical personnel but also aid early detection and prompt treatment for patients thereby drastically enhancing the health system. Furthermore, it can also aid in devising a monitory and preventive program for those who might be vulnerable that cause them to suffer from heart disease, based on their medical or family history. In this era of Data Science, ML algorithms are constantly being used, across various fields, to gain meaningful insights and leverage the information mined to make decisions.

  As for this project, it includes analysis of the heart disease based on the patient's heart dataset with proper data cleaning and data processing. Then, a model will be implemented, next it will be trained and predictions will be made with four different algorithms, such as Logistic Regression, Naive Bayes, Random Forest and Artificial Neural Networks. After implmenting the models. then they will be  used to evaluate the performance of all 4 algorithms and then we can finally make comparision, to find for the most accurate technique so that it can be implemented in the future. 

  The heart disease prediction can be performed by following the procedure which is similar to the diagram below which specifies the research methodology for building a classification model required for the prediction of the heart diseases in patients. Initally, Data is cleaned through the cycles, for example, replacement of missing values, smoothing the noisy information, and settling the irregularities in the information and uneccesary columns has been dropped. The model forms a fundamental procedure for carrying out the heart disease prediction using machine learning techniques. In order to make predictions, a classifier needs to be trained with the records and then produce a classification model which is fed with a new unknown record and the prediction is made. The research methodology of this project  includes the Performance Evaluation of all  our classification algorithms.

  In the percentage split, the training and testing data is split up in percentage of data such as 80% and 20% where the 80% is used for training and 20% is used for testing. The performance of the classification models derived by the ML algorithm is measured using the confusion matrix and classficiation report. The confusion matrix is a contingency table that displays the number of instances assigned to each class that  allows us to calculate the classification accuracy. Not only that, I have also included precision,recall and ROC score for this project .
  
  
-------------------------------------
IMPLEMENTATION METHODS
-------------------------------------
Logistic Regression: At the training stage, Logistic Regression algorithm estimates the coefficient values by using stochastic gradient descent. The model can be trained for a fixed or as much as no of max_iter by using gradient descent. Coefficients values are updated until the model predicts the correct class label for each training data. 

Random Forest: Random Forest is an ensemble classification method which is based on the Decision Tree algorithm. This algorithm takes a portion of the dataset and then builds a tree,
repeat this step for creating a forest by combining the generated trees. At the test stage, each tree predicts a class label for each test data and majority values of the class label are assigned to the test data.  

Naive Bayes At the training stage, it calculates the mean and standard deviation of each attribute. This mean and standard deviation will be used to calculate the probabilities for the test data. For this reason, some attributes values are too big or too small from the mean. When testing the data pattern, it contains those attributes values,that affects the classifier performance and sometimes gives wrong output labels.

Neural networks, it is trained using gradient descent where the estimate of the error used to update the weights is calculated based on a subset of the training dataset. The number of examples from the training dataset used in the error gradient is called the batch size and its an important hyperparameter that influences the dynamics of the algorithm. Batch size controls the accuracy of the estimated error of the gradient when training neural networks.


------------------------------
RESULTS
------------------------------
From the output, we can see that LR outperforms for Hungary and Statlog dataset whereas ANN outperforms for Cleveland Dataset. The least accuracy from the overall diagram is given by NB for Hungary dataset and after further analysis, we have discovered that NB did not perform well due to a higher bias but lower variance compared to LR. If the data set follows the bias then NB will be a better classifier. Both NB and LR are linear classifiers, LR makes a prediction for the probability using a direct functional form whereas NB figures out how the data was generated given the results. that is why LR perofrms better than NB
 


