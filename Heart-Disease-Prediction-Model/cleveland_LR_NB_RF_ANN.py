#!/usr/bin/env python
# coding: utf-8

# In[1]:


# for basic operations
import numpy as np
import pandas as pd
import pandas_profiling
import warnings 
warnings.filterwarnings(action= 'ignore')
# for data visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# for advanced visualizations 
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode(connected = True)
from bubbly.bubbly import bubbleplot

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.naive_bayes import GaussianNB
import eli5 
from eli5.sklearn import PermutationImportance

#for SHAP values
import shap 
from pdpbox import pdp, info_plots #for partial plots

# for model explanation
import shap
pd.set_option('mode.chained_assignment', None)


# In[2]:


# reading the data
data2 = pd.read_csv('heart.csv')
# getting the shape
data2.shape


# In[3]:


data2.head(3)


# In[4]:


# describing the data and getting some informations
# data.info
data2.describe()


# In[5]:


data2.info()


# In[6]:


init_notebook_mode()


# In[7]:


data2.sort_values(by=['age'], inplace=True)


# In[8]:


import warnings
warnings.filterwarnings('ignore')

figure = bubbleplot(dataset = data2, x_column = 'trestbps', y_column = 'chol', 
    bubble_column = 'sex', time_column = 'age', size_column = 'oldpeak', color_column = 'sex', 
    x_title = "Resting Blood Pressure", y_title = "Cholestrol", title = 'BP vs Chol. vs Age vs Sex vs Heart Rate',
    x_logscale = False, scale_bubble = 4, height = 650)

# py.iplot(figure, config={'scrollzoom': True})
py.iplot(figure, config={'scrollzoom': False})


# In[9]:


# making a heat map
from pylab import savefig
plt.rcParams['figure.figsize'] = (25, 20)
plt.style.use('ggplot')
sns.heatmap(data2.corr(), annot = True,linewidths=2, linecolor='green', cmap = 'Spectral',annot_kws={"fontsize":20})
sns.set(font_scale=4)
plt.title('Heatmap based on Heart Dataset', fontsize = 40)
# plt.savefig('heatmap3.png')
plt.show()


# In[10]:


# checking the distribution of age among the patients
import warnings
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (25, 10)
sns.distplot(data2['age'], color = 'BROWN')
plt.title('Distribution of Age', fontsize = 30)
# plt.savefig('Distribution3.png')
plt.show()


# In[11]:


# plotting a donut chart for visualizing each of the recruitment channel's share

size = data2['sex'].value_counts()
colors = ['orange', 'green']
labels = "Male", "Female"
explode = [0, 0.01]

my_circle = plt.Circle((0, 0), 0.7, color = 'white')
plt.rcParams['figure.figsize'] = (10, 15)
plt.pie(size, colors = colors, labels = labels, shadow = True, explode = explode, autopct = '%.2f%%',textprops={'fontsize': 22})
plt.title('Visualizing based on Gender [Cleveland Dataset]', fontsize = 20)
p = plt.gcf()
p.gca().add_artist(my_circle)
plt.legend()
# plt.savefig('Gender3.png')
plt.show()


# In[12]:


# plotting the target attribute

plt.rcParams['figure.figsize'] = (18, 8)
plt.style.use('seaborn-talk')
sns.countplot(data2['target'], palette = 'winter')
plt.title('Countplot of Target', fontsize = 15)
# plt.savefig('Target3.png')

plt.show()


# In[13]:


plt.rcParams['figure.figsize'] = (12, 9)
sns.violinplot(data2['target'], data2['chol'], palette = 'rocket')
plt.title('Relationship between Cholestrol and Target', fontsize = 20, fontweight = 30)
# plt.savefig('Chol_target3.png')
plt.show()


# In[14]:


data2.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',
       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']

data2.columns
# data2.columns =['age','sex','cp','trestbps','chol','fbs','restecg','thalach',
#                 'exang','oldpeak','slope','ca','thal','target']


# In[15]:


data2.drop(columns=['num_major_vessels'],inplace=True)


# In[16]:


data2['sex'][data2['sex'] == 0] = 'female'
data2['sex'][data2['sex'] == 1] = 'male'

data2['chest_pain_type'][data2['chest_pain_type'] == 1] = 'typical angina'
data2['chest_pain_type'][data2['chest_pain_type'] == 2] = 'atypical angina'
data2['chest_pain_type'][data2['chest_pain_type'] == 3] = 'non-anginal pain'
data2['chest_pain_type'][data2['chest_pain_type'] == 4] = 'asymptomatic'

data2['fasting_blood_sugar'][data2['fasting_blood_sugar'] == 0] = 'lower than 120mg/ml'
data2['fasting_blood_sugar'][data2['fasting_blood_sugar'] == 1] = 'greater than 120mg/ml'

data2['rest_ecg'][data2['rest_ecg'] == 0] = 'normal'
data2['rest_ecg'][data2['rest_ecg'] == 1] = 'ST-T wave abnormality'
data2['rest_ecg'][data2['rest_ecg'] == 2] = 'left ventricular hypertrophy'

data2['exercise_induced_angina'][data2['exercise_induced_angina'] == 0] = 'no'
data2['exercise_induced_angina'][data2['exercise_induced_angina'] == 1] = 'yes'

data2['st_slope'][data2['st_slope'] == 1] = 'upsloping'
data2['st_slope'][data2['st_slope'] == 2] = 'flat'
data2['st_slope'][data2['st_slope'] == 3] = 'downsloping'

data2['thalassemia'][data2['thalassemia'] == 3] = 'normal'
data2['thalassemia'][data2['thalassemia'] == 6] = 'fixed defect'
data2['thalassemia'][data2['thalassemia'] == 7] = 'reversable defect'


# In[18]:


data2.head(5)


# In[19]:


y = data2['target']

data2 = data2.drop('target', axis = 1)

print("Shape of y:", y.shape)


# In[20]:


# one hot encoding of the data
# drop_first = True, means dropping the first categories from each of the attribues 
# for ex gender having gender_male and gender-female would be male having values 1 and 0
data2 = pd.get_dummies(data2, drop_first=True)


# In[21]:


# checking the dataset after encoding
data2.head()


# In[22]:


x = data2
# checking the shapes of x and y
print("Shape of x:", x.shape)
print("Shape of y:", y.shape)


# In[23]:


# splitting the sets into training and test sets

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# getting the shapes
print("Shape of x_train :", x_train.shape)
print("Shape of x_test :", x_test.shape)
print("Shape of y_train :", y_train.shape)
print("Shape of y_test :", y_test.shape)


# **!LOGISTIC REGRESSION MODEL IMPLEMENTATION!** 

# In[24]:


from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn import metrics

logreg = LogisticRegression(solver='lbfgs',max_iter=1500)
logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)
 # evaluating the model
accLR = accuracy_score(y_test,y_pred)
print("Accuracy : ", round(accLR*100,2),"%")
# print("Training Accuracy :", logreg.score(x_train, y_train))
# print("Testing Accuracy  :", logreg.score(x_test, y_test))
print("")

# cofusion matrix
cmLR = confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize'] = (5, 3)
sns.heatmap(cmLR, annot = True, annot_kws = {'size':15}, cmap = 'gist_heat').set(title="Cleveland Dataset: Logistic Regression")

# classification report
crLR = classification_report(y_test, y_pred)
print(crLR)


# In[25]:


from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score 

recallLR = recall_score(y_test, y_pred)
print("Recall is: ",recallLR)

precisionLR = precision_score(y_test, y_pred)
print("Precision is: ",precisionLR)

f1_metricLR = f1_score(y_test, y_pred, average = "macro")
print('F1-Score :', f1_metricLR)


# In[26]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_test, logreg.predict(x_test))
fprLR, tprLR, thresholds = roc_curve(y_test, logreg.predict_proba(x_test)[:,1])

plt.figure()
plt.plot(fprLR, tprLR, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (LR)',fontweight=30)
plt.legend(loc="lower right")
plt.rcParams['figure.figsize'] = (5, 4)
# plt.savefig('Log_ROC')
plt.show()
 


# In[27]:


# let's check the auc score
from sklearn.metrics import auc
aucLR = auc(fprLR, tprLR)
print("AUC Score :", round(aucLR*100,2),"%")


# **!NAIVE BAYES MODEL IMPLEMENTATION!**

# In[28]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
gnb = GaussianNB()
gnb.fit(x_train, y_train.ravel())
y_pred = gnb.predict(x_test)

# evaluating the model
# print("Training Accuracy :", gnb.score(x_train, y_train))
# print("Testing Accuracy  :", gnb.score(x_test, y_test))
accNB = accuracy_score(y_test,y_pred)
print("Accuracy : ", round(accNB*100,2),"%")

# cofusion matrix
cmNB = confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize'] = (5, 3)
sns.heatmap(cmNB, annot = True, annot_kws = {'size':15}, cmap = 'gist_heat').set(title="Cleveland Dataset: Naive Bayes")

# classification report
crNB = classification_report(y_test, y_pred)
print(crNB)


# In[29]:


from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score 

recallNB = recall_score(y_test, y_pred)
print("Recall is: ",recallNB)

precisionNB = precision_score(y_test, y_pred)
print("Precision is: ",precisionNB)

f1_metricNB = f1_score(y_test, y_pred, average = "macro")
print('F1-Score :', f1_metricNB)


# In[30]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve

fprNB, tprNB, thresholdsNB = roc_curve(y_test, y_pred)

plt.plot([0,1],[0,1],'k--') #plot the diagonal line
plt.plot(fprNB, tprNB, label='NB') #plot the ROC curve
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (NB)',fontweight=30)
plt.legend(loc="lower right")
plt.rcParams['figure.figsize'] = (5, 4)
plt.show()
 


# In[31]:


# let's check the auc score

from sklearn.metrics import auc
aucNB = auc(fprNB, tprNB)
print("AUC Score :", aucNB)


# In[32]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

model = RandomForestClassifier(n_estimators = 50, max_depth = 5)
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
y_pred_quant = model.predict_proba(x_test)[:, 1]
y_pred = model.predict(x_test)

# # evaluating the model
accRF = accuracy_score(y_test,y_pred)
print("Accuracy : ",round(accRF*100,2),"%")
# print("Training Accuracy :", model.score(x_train, y_train))
# print("Testing Accuracy  :", model.score(x_test, y_test))
print("")

# cofusion matrix
cmRF = confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize'] = (5, 3)
sns.heatmap(cmRF, annot = True, annot_kws = {'size':15}, cmap = 'gist_heat').set(title="Cleveland Dataset: Random Forest")

from sklearn.metrics import accuracy_score

# classification report
crRF = classification_report(y_test, y_pred)
print(crRF)


# In[33]:


from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

recallRF = recall_score(y_test, y_pred)
print("Recall is: ",recallRF)

precisionRF = precision_score(y_test, y_pred)
print("Precision is: ",precisionRF)

f1_metricRF = f1_score(y_test, y_pred, average = "macro")
print('F1-Score :', f1_metricRF)


# In[34]:


from sklearn.metrics import roc_curve

fprRF,tprRF, thresholdsRF = roc_curve(y_test, y_pred_quant)

fig, ax = plt.subplots()
ax.plot(fprRF, tprRF)
ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="-", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (RF)',fontweight=30)
# plt.legend(loc="lower right")
plt.rcParams['figure.figsize'] = (5, 4)
plt.show()


# In[35]:


# let's check the auc score

from sklearn.metrics import auc
aucRF = auc(fprRF, tprRF)
print("AUC Score :", aucRF)


# In[36]:


data3=pd.read_csv('heart.csv')
data3.head()


# In[37]:


data3.info()


# In[38]:


chest_pain=pd.get_dummies(data3['cp'],prefix='cp',drop_first=True)
data3=pd.concat([data3,chest_pain],axis=1)

data3.drop(['cp'],axis=1,inplace=True)

sp=pd.get_dummies(data3['slope'],prefix='slope')
th=pd.get_dummies(data3['thal'],prefix='thal')

rest_ecg=pd.get_dummies(data3['restecg'],prefix='restecg')
frames=[data3,sp,th,rest_ecg]

data3=pd.concat(frames,axis=1)

data3.drop(['slope','thal','restecg'],axis=1,inplace=True)


# In[39]:


data3.head(5)


# In[40]:


X = data3.drop(['target'], axis = 1)
y = data3.target.values


# In[41]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[42]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train.shape)
print(X_test.shape)


# In[43]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import keras
from keras.models import Sequential
from keras.layers import Dense
import warnings


# In[44]:


classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(11, kernel_initializer="uniform", activation = 'relu', input_dim = 22))

# Adding the second hidden layer
classifier.add(Dense(11, kernel_initializer="uniform", activation = 'relu'))

# Adding the output layer
classifier.add(Dense(1, kernel_initializer="uniform", activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[45]:


classifier.fit(X_train, y_train, batch_size = 5, epochs = 1000)


# In[46]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)


# In[47]:


import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

cmANN = confusion_matrix(y_test, y_pred.round())
sns.heatmap(cmANN,annot=True,cmap="Blues",fmt="d",cbar=False)
#accuracy score


from sklearn.metrics import accuracy_score
acANN=accuracy_score(y_test, y_pred.round())
print('Accuracy: ',acANN)
print("")

cmANN = confusion_matrix(y_test, y_pred.round())
plt.rcParams['figure.figsize'] = (5, 3)
sns.heatmap(cmANN, annot = True, annot_kws = {'size':15}, cmap = 'gist_heat').set(title="Cleveland Dataset: Artificial Neural Networks")

crANN = classification_report(y_test, y_pred.round())
print(crANN)


# In[65]:


recallANN = recall_score(y_test, y_pred.round())
print("Recall is: ",recallANN)

precisionANN = precision_score(y_test, y_pred.round())
print("Precision is: ",precisionANN)

f1_metricANN = f1_score(y_test, y_pred.round(), average = "macro")
print('F1-Score :', f1_metricANN)


# In[49]:


#### from sklearn.metrics import roc_curve, auc
fprANN, tprANN, thresholdsANN = roc_curve(y_test, y_pred.round())

plt.figure()
plt.plot(fprANN, tprANN, label='Artificial Neural Network (area = %0.2f)' )
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic',fontweight=30)
plt.legend(loc="lower right")
plt.rcParams['figure.figsize'] = (5, 4)
# plt.savefig('Log_ROC')
plt.show()


# In[50]:


from sklearn.metrics import auc
aucANN = auc(fprANN, tprANN)

print("AUC Score :", aucANN)


# In[51]:


print("--------------------------------------------------")
print("      RESULTS FOR LOGISTIC REGRESSION             ")
print("--------------------------------------------------")
print(" Accuracy  : ", round(accLR*100,2),"%")
print(" AUC Score : ", round(aucLR*100,2),"%")
print(" Precision : ", round(precisionLR*100,2),"%")
print(" Recall    : ", round(recallLR*100,2),"%")
print(' F1-Score  : ', round(f1_metricLR*100,2),"%")
print("--------------------------------------------------")
print(" ")


print("--------------------------------------------------")
print("            RESULTS FOR NAIVE BAYES               ")
print("--------------------------------------------------")
print(" Accuracy  : ", round(accNB*100,2),"%")
print(" AUC Score : ", round(aucNB*100,2),"%")
print(" Precision : ", round(precisionNB*100,2),"%")
print(" Recall    : ", round(recallNB*100,2),"%")
print(' F1-Score  : ', round(f1_metricNB*100,2),"%")
print("--------------------------------------------------")
print(" ")



print("--------------------------------------------------")
print("           RESULTS FOR RANDOM FOREST              ")
print("--------------------------------------------------")
print(" Accuracy  : ", round(accRF*100,2),"%")
print(" AUC Score : ", round(aucRF*100,2),"%")
print(" Precision : ", round(precisionRF*100,2),"%")
print(" Recall    : ", round(recallRF*100,2),"%")
print(' F1-Score  : ', round(f1_metricRF*100,2),"%")
print("--------------------------------------------------")
print(" ")



print("--------------------------------------------------")
print("   RESULTS FOR ARTIFICIAL NEURAL NETWORK    ")
print("--------------------------------------------------")
print(' Accuracy  : ', round(acANN*100,2),"%")
print(" AUC Score : ", round(aucANN*100,2),"%")
print(" Precision : ", round(precisionANN*100,2),"%")
print(" Recall    : ", round(recallANN*100,2),"%")
print(' F1-Score  : ', round(f1_metricANN*100,2),"%")
print("--------------------------------------------------")


# In[52]:


plt.plot(fprNB, tprNB, color='orange', label='NB') 
plt.plot(fprRF,tprRF, color='blue', label='RF')  
plt.plot(fprLR, tprLR, color='red', label='LR')
plt.plot(fprANN, tprANN, color='green', label='ANN')

plt.rcParams['figure.figsize'] = (7,6 )
plt.plot([0, 1], [0, 1], color='green', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()


# In[53]:


# importing ML Explanability Libraries
#for purmutation importance
perm = PermutationImportance(logreg, random_state = 0).fit(x_test, y_test)
eli5.show_weights(perm, feature_names = x_test.columns.tolist())


base_features = data2.columns.values.tolist()
feat_name = 'st_depression'
pdp_dist = pdp.pdp_isolate(model=logreg, dataset=x_test, model_features = base_features, feature = feat_name)
pdp.pdp_plot(pdp_dist, feat_name)
plt.show()

base_features = data2.columns.values.tolist()

feat_name = 'chest_pain_type_atypical angina'
pdp_dist = pdp.pdp_isolate(model = model, dataset = x_test, model_features = base_features, feature = feat_name)

pdp.pdp_plot(pdp_dist, feat_name)
plt.show()

explainer=shap.LinearExplainer(logreg,x_test)
shap_values = explainer.shap_values(x_test)
shap.summary_plot(shap_values, x_test, feature_names=x_test.columns)


def patient_analysis(logreg, patient):
    explainer = shap.LinearExplainer(logreg,x_train)
    shap_values = explainer.shap_values(patient)
    shap.initjs()
    return shap.force_plot(explainer.expected_value, shap_values, patient)

# let's do some real time prediction for patients
patients = x_test.iloc[1,:].astype(float)
patient_analysis(logreg, patients)

# let's do some real time prediction for patients
patients = x_test.iloc[2,:].astype(float)
patient_analysis(logreg, patients)



shap.force_plot(explainer.expected_value, shap_values, x_train)

