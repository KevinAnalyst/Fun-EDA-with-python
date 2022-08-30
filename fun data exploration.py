#!/usr/bin/env python
# coding: utf-8

# # This data exploration was not done orderly it just for fun.
# # If need to your self to do practice use data provide by copying the path and replace with mine because of different storage area in the pc.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[9]:


data=pd.read_csv("D:\CLAS\WorkSalaries.csv")


# In[10]:


data


# In[11]:


data.describe()


# In[12]:


data.shape


# In[13]:


data.info()


# In[16]:


data.columns


# In[17]:


data.max()


# In[18]:


plt.hist(data.sex)


# In[23]:


plt.bar(data.sex,data.salary)


# In[25]:


plt.bar(data.yrs_since_phd,data.salary)


# In[82]:


plt.bar(data.yrs_service,data.salary)
plt.xlabel('yrs_service')
plt.ylabel('salary')
plt.title('bar graph')
plt.figure(figsize=(40,20))


# In[27]:


data.corr()


# In[35]:


plt.boxplot(data.salary)


# In[39]:


x=data[['yrs_since_phd','yrs_service']]
y=data['salary']


# In[40]:


from sklearn.linear_model import LinearRegression


# In[41]:


data1=LinearRegression()


# In[42]:


data1.fit(x,y)


# In[44]:


data1.score(x,y)


# In[45]:


data1.coef_


# In[46]:


data1.intercept_


# In[48]:


predicted_data=data1.predict(x)


# In[49]:


predicted_data


# In[52]:


x


# In[53]:


y


# In[54]:


y.sum()


# In[55]:


y.min()


# In[56]:


y.std()


# In[57]:


y.mean()


# In[58]:


y.median()


# In[59]:


y.describe()


# In[65]:


plt.hist(y)
plt.xlabel('salary')
plt.ylabel('Fequency')
plt.title('Salary graph')
plt.figure(figsize=(30,15))


# In[71]:


y.duplicated()


# In[72]:


x.duplicated()


# In[80]:


plt.scatter(data.salary,data.yrs_service)
plt.xlabel('salary')
plt.ylabel('yrs service')
plt.title('Scatter plot')


# In[83]:


#Visualizing data using seaborn
import seaborn as sns


# In[85]:


# check the variable relationship
sns.scatterplot(x=data.salary,y=data.yrs_service)


# In[89]:


# use box to check the all descriptive statistics and the outliers
sns.boxplot(x=data.sex,y=data.salary)


# In[91]:


# use box to check the all descriptive statistics and the outliers
sns.boxplot(x=data.discipline,y=data.salary)


# In[94]:


sns.barplot(x=data.discipline,y=data.salary)


# In[ ]:




