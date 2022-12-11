import streamlit as st
import pandas as pd
import seaborn as sns
import altair as alt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

dfa = pd.read_csv("dfa.csv")
#################################################################################
#  1 plot
st.title("Countplot of SmokerStatus ") 
b=sns.countplot(x='SmokerStatus', data = dfa)
for p in b.patches:
        b.annotate("%.0f" % p.get_height(), (p.get_x() + 
    p.get_width() / 2., p.get_height()), 
        ha='center', va='center', rotation=0, 
    xytext=(0, 18), textcoords='offset points')
st.pyplot()
#################################################################################
# # 2 plot
a=pd.crosstab(dfa.SmokerStatus,dfa.Customer_Needs_1).plot(kind='barh')

st.title('Stacked Bar Chart of SmokerStatus against Customer_Needs_1')
st.pyplot()
#################################################################################
## 3 plot
a=pd.crosstab(dfa.Occupation,dfa.PurchasedPlan2).plot(kind='barh')
st.title('Stacked Bar Chart of Occupation against PurchasedPlan2')
st.pyplot()
#################################################################################
# # 4 plot
a=pd.crosstab(dfa.PurchasedPlan2,dfa.age_bins).plot(kind='barh')
st.title('Stacked Bar Chart of PurchasedPlan2 against Age_Bins')
st.pyplot()
#################################################################################
# 5 plot
table=pd.crosstab(dfa.Occupation, dfa.Customer_Needs_1)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
st.title('Stacked Bar Chart of Occupation against Customer_Needs_1')
st.pyplot()
#################################################################################
# 6 plot
boruta_score=pd.read_csv("boruta_score.csv")
sns_boruta_plot = sns.catplot(x="Score", y="Features", data = boruta_score[0:30], kind = "bar", height=5, aspect=2, palette='coolwarm')
st.title("Boruta Top 30 Features") 
st.pyplot()
#################################################################################
# 7 plot
rfe_score=pd.read_csv("rfe_score.csv")
sns_rfe_plot = sns.catplot(x="Score", y="Features", data = rfe_score[0:30], kind = "bar", 
               height=5, aspect=2, palette='coolwarm')
st.title("RFE Top 30 Features") 
st.pyplot()
#################################################################################
# 8 plot
cluster=pd.read_csv("cluster.csv")

X = cluster.drop('MalaysiaPR',axis=1)
y = cluster['MalaysiaPR']

kmeans5 = KMeans(n_clusters=5)
y_kmeans5 = kmeans5.fit_predict(X)
# print(y_kmeans5)
kmeans5.cluster_centers_

Error =[]
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i).fit(X)
    kmeans.fit(X)
    Error.append(kmeans.inertia_)

plt.plot(range(1, 11), Error)
plt.title('Elbow method')
plt.xlabel('No of clusters')
plt.ylabel('Error')
st.pyplot()