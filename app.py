#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import streamlit as st


# In[2]:


data=pd.read_csv("HR Employee data.csv")


# In[3]:


data.head(10)


# In[4]:


data.isnull().sum()


# In[5]:


data.info()


# # Remove Unecessary Columns

# In[6]:


# Unique identifiers (no predictive value)
data.drop(['EmployeeID', 'EmployeeNumber'], axis=1, inplace=True)

# Constant columns (no variation, so no predictive power)
data.drop(['EmployeeCount', 'StandardHours', 'Over18'], axis=1, inplace=True)

# Post-attrition information — known only AFTER the employee leaves (leads to data leakage)
data.drop(['LeavingYear', 'Reason', 'RelievingStatus'], axis=1, inplace=True)


# # Encode Categorical Value

# In[7]:


categorical_cols = data.select_dtypes(include='object').columns.tolist()
categorical_cols


# In[8]:


categorical_cols.remove("Attrition")


# In[9]:


le=LabelEncoder()
for col in categorical_cols:
    data[col]=le.fit_transform(data[col])


# In[10]:


st.title("Employee Attrition Prediction- Random Forest")


# In[11]:


data["Attrition"]=data["Attrition"].map({"Yes":1,"No":0})


# In[12]:


data.Attrition


# # Split Feature & Target

# In[44]:


X=data.drop(["Attrition"],axis=1)
y=data["Attrition"]


# In[45]:


X_train, X_test, y_train, y_test=train_test_split(X,y ,test_size=0.2, random_state=42)


# In[46]:


rf=RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred= rf.predict(X_test)


# In[47]:


from sklearn.metrics import classification_report, confusion_matrix


# In[48]:


cm=confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=["Stayed", "left"], yticklabels=["Stayed", "left"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix - Random Forest")
st.pyplot(fig)  


# In[49]:


st.subheader("Random Forest Report")
st.text(classification_report(y_test,y_pred))


# In[50]:


importances=rf.feature_importances_
indices=np.argsort(importances)[::-1]
fig2, ax2 = plt.subplots(figsize=(10,6))
sns.barplot(x=importances[indices][:10], y=X.columns[indices][:10], palette="viridis", ax=ax2)
ax2.set_title("Top 10 Important Features - Default Model")
ax2.set_xlabel("Feature Importance Score")
ax2.set_ylabel("Feature")
st.pyplot(fig2) 


# In[51]:


# Streamlit User Interface
st.sidebar.title("Employee Attrition Prediction")

# User input for each feature (example: select boxes, sliders)
st.sidebar.subheader("Enter Employee Details:")

def user_input_features():
    Age = st.sidebar.slider('Age', int(X['Age'].min()), int(X['Age'].max()), int(X['Age'].mean()))
    DistanceFromHome = st.sidebar.slider('Distance From Home', int(X['DistanceFromHome'].min()), int(X['DistanceFromHome'].max()), int(X['DistanceFromHome'].mean()))
    MonthlyIncome = st.sidebar.slider('Monthly Income', int(X['MonthlyIncome'].min()), int(X['MonthlyIncome'].max()), int(X['MonthlyIncome'].mean()))
    JobSatisfaction = st.sidebar.slider('Job Satisfaction', 1, 4, 2)
    TotalWorkingYears = st.sidebar.slider('Total Working Years', int(X['TotalWorkingYears'].min()), int(X['TotalWorkingYears'].max()), int(X['TotalWorkingYears'].mean()))
    YearsAtCompany = st.sidebar.slider('Years At Company', int(X['YearsAtCompany'].min()), int(X['YearsAtCompany'].max()), int(X['YearsAtCompany'].mean()))

    # Add more sliders/select boxes as needed
    input_data = {
        'Age': Age,
        'DistanceFromHome': DistanceFromHome,
        'MonthlyIncome': MonthlyIncome,
        'JobSatisfaction': JobSatisfaction,
        'TotalWorkingYears': TotalWorkingYears,
        'YearsAtCompany': YearsAtCompany,
        # Add more fields here matching model features
    }
    return pd.DataFrame([input_data])

input_df = user_input_features()

# Predict Button
if st.button("Predict Attrition"):
    result = rf.predict(input_df)
    if result[0] == 0:
        st.success("Prediction: The employee is likely to LEAVE.")
    else:
        st.success("Prediction: The employee is likely to STAY.")


# In[ ]:



# In[ ]:





# In[ ]:





# In[ ]:




