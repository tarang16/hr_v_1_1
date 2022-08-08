#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pickle
import numpy as np


# In[7]:


def load_model():
    with open('saved_steps_v_2.pkl', 'rb') as file:
        data = pickle.load(file)
    return data


# In[9]:


data = load_model()

classifier = data["model"]
le_college = data["le_college"]
le_salary= data["le_salary"]


# In[10]:


def show_predict_page():
    st.title(" HR Portal")
    st.write("""### We need some information to predict whether the Candidate will be Shortlisted or Not""")
    College=('T-1','T-2','T-3')
    Salary=('0-3 Lakhs','6-10 Lakhs','10-15 Lakhs','15-25 Lakhs','25-50 Lakhs','50-75 Lakhs','75-100 Lakhs','1crore+')
    age = st.slider("Age of the candidate", 16, 50, 16)
    salary = st.selectbox("Salary", Salary)
    Expericence = st.slider("Years of Experience", 0, 50, 3)
    offer_in_hand=st.slider("Offer in Hand",0,10,0)
    years_in_current_department=st.slider("Years in Current Department",0,50,1)
    NumCompaniesWorked=st.slider("Number of Companies Worked",0,10,1)
    perofExpectedhike=st.slider("% of Expected Hike",0,400,5)
    College = st.selectbox("College", College)
    ok = st.button("Predict Shortlisting")
    if ok:
        X = np.array([[age, salary,College,offer_in_hand,Expericence,years_in_current_department,NumCompaniesWorked,perofExpectedhike ]])
        X[:, 2] = le_college.transform(X[:,2])
        X[:,1 ]=le_salary.transform(X[:,1])
        X = X.astype(float)
        offer = classifier.predict(X)
        if offer[0]==0:
            result="Not Shortlisted"
        else:
            result="Shortlised"
        st.subheader(f"The following candidate has been {result}")
        
    


# In[ ]:




