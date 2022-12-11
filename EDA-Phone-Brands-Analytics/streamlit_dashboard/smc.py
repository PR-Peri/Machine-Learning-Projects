import streamlit as st
import pandas as pd
import seaborn as sns
import altair as alt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


dfa = pd.read_csv("samsung.csv")
dfb = pd.read_csv("motorola.csv")
dfc = pd.read_csv("blue.csv")
dfd = pd.read_csv("yellow.csv")


#################################################################################
# # 1 plot
a=sns.countplot(dfa.Sentiment_type)
st.title('Sentiment analysis for Samsung Support')
st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)

#################################################################################
# # 2 plot
a=sns.countplot(dfb.Sentiment_type)
st.title('Sentiment analysis for Motorola Support')
st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)

#################################################################################
# # 3 plot
a =dfc.plot(x="User",y="Negative",kind='bar',color="Blue")
st.title('Score_Table 2')
st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)

#################################################################################
# # 4 plot
a =dfd.plot(x="User",y="Positive",kind='bar',color="Yellow")
st.title('Score_Table 3')
st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)

#################################################################################


