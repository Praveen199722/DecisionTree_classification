#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as matlab


# In[2]:


dataset=pd.read_csv("CKD.csv")
dataset


# In[3]:


dataset=pd.get_dummies(dataset,drop_first=True)
dataset


# In[4]:


dataset.columns


# In[5]:


dataset['classification_yes'].value_counts()


# In[6]:


independent=dataset[['age', 'bp', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hrmo', 'pcv',
       'wc', 'rc', 'sg_b', 'sg_c', 'sg_d', 'sg_e', 'rbc_normal', 'pc_normal',
       'pcc_present', 'ba_present', 'htn_yes', 'dm_yes', 'cad_yes',
       'appet_yes', 'pe_yes', 'ane_yes']]
dependent=dataset[['classification_yes']]


# In[7]:


independent.shape


# In[8]:


dependent


# In[9]:


#split into training set and test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(independent,dependent,test_size=1/3,)


# In[10]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[11]:


from sklearn.linear_model import LogisticRegression


# In[20]:


from sklearn.model_selection import GridSearchCV
param_grid={'solver':['newton-cg','lbfgs','liblinear','saga'],'penalty':['l2']}
grid=GridSearchCV(LogisticRegression(),param_grid,refit=True,verbose=3,n_jobs=-1,scoring='f1_weighted')
#fitting the model for grid search
grid.fit(X_train,Y_train)


# In[21]:


from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,Y_train)


# In[22]:


y_pred=classifier.predict(X_test)
y_pred


# In[26]:


re=grid.cv_results_
grid_predictions=grid.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,y_pred)


# In[27]:


y_pred


# In[28]:


print(cm)


# In[29]:


from sklearn.metrics import roc_auc_score
roc_auc_score(Y_test,grid.predict_proba(X_test)[:,1])


# In[30]:


table=pd.DataFrame.from_dict(re)
table


# In[31]:


age_input=float(input("Age:"))
bp_input=float(input("BP:"))
al_input=float(input("AL:"))
su_input=float(input("SU:"))
bgr_input=float(input("BGR:"))
bu_input=float(input("BU:"))
sc_input=float(input("SC:"))
sod_input=float(input("SOD:"))
pot_input=float(input("POT:"))
hrmo_input=float(input("HRMO:"))
pcv_input=float(input("PCV:"))
wc_input=float(input("WC:"))
rc_input=float(input("RC:"))
sg_b_input=float(input("SG_B:"))
sg_c_input=float(input("SG_C:"))
sg_d_input=float(input("SG_D:"))
sg_e_input=float(input("SG_E:"))
rbc_normal_input=float(input("RBC Normal 0 or 1:"))
pc_normal_input=float(input("PC Normal 0 or 1:"))
pcc_present_input=float(input("PC Present 0 or 1:"))
ba_present_input=float(input("BA_Present 0 or 1:"))
htn_yes_input=float(input("HTN 0 or 1:"))
dm_yes_input=float(input("DM 0 or 1:"))
cad_yes_input=float(input("CAD 0 or 1:"))
appet_yes_input=float(input("Appet 0 or 1:"))
pe_yes_input=float(input("PE 0 or 1:"))
ane_yes_input=float(input("ANE 0 or 1:"))


# In[32]:


Future_Prediction=grid.predict([[age_input,bp_input, al_input, su_input, bgr_input, bu_input, sc_input, sod_input, pot_input, hrmo_input, pcv_input,
       wc_input, rc_input, sg_b_input, sg_c_input, sg_d_input, sg_e_input, rbc_normal_input, pc_normal_input,
       pcc_present_input, ba_present_input, htn_yes_input, dm_yes_input, cad_yes_input,
       appet_yes_input, pe_yes_input, ane_yes_input]])
print("Future_Prediction={}".format(Future_Prediction))


# In[ ]:




