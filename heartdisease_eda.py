#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


heart_df= pd.read_csv('C:\\Users\\MERINA ANGEL\\Downloads\\heart.csv')
#importing the file of heart diease from US report 2020 usind pandas.
# the dataset is extracted from kaggle.


# ###  UNDERSTANDING THE DATASET

# In[3]:


heart_df.head() #printing 5 rows from dataset


# In[4]:


heart_df.shape #printing rows and columns


# In[5]:


heart_df.info()
#printing information of the dataset which gives information about total rows non null values, datatype, shape


# In[6]:


heart_df.isnull().sum() #printing null values in the dataset


# In[7]:


heart_df['PhysicalHealth'].value_counts() #printing physical health column values to check how many are in 0 values


# In[8]:


heart_df['MentalHealth'].value_counts() #printing Mental health column values to check how many are in 0 values


# In[9]:


205401/319795 #checking how much does the 0 values are persent from overall values


# In[10]:


#here we can see the 0 values are of higher amount, where we can't replace or delete them without the advice of expert domain.


# In[11]:


heart_df.describe() #5 point summary of numeric data


# In[12]:


numeric=heart_df.select_dtypes(include=['int','float']) #printing only numerical columns


# In[13]:


numeric


# In[14]:


categorical=heart_df.select_dtypes(include='object') #printing only categorical columns
categorical


# In[15]:


heart_df.dtypes #checking the datatypes of all the columns


# ### DATA VISUALIZATION

# Data visualization helps in understanding the data in an effective way

# In[16]:


corr=heart_df.corr()
corr #calculation of correlation helps in knowing which features having more correlation between them


# In[17]:


sns.heatmap(corr, cmap="crest",annot=True) #visualizing the correlation


# In[18]:


#checking outliers of all independent columns
j=1
plt.figure(figsize=(10,20))
col=numeric.columns
for i in col:
    plt.subplot(len(col)+1,2,j)
    sns.boxplot(numeric[i])
    plt.title(i,color='white')
    j=j+1 #boxplot is used to check the outliers


# In[19]:


j=1
plt.figure(figsize=(10,20))
col=numeric.columns
for i in col:
    plt.subplot(len(col)+1,2,j)
    sns.histplot(numeric[i],bins=10,kde=True)
    plt.title(i,color='white')
    j=j+1
    #histplot and kdeplot is used to check the variance in univariate continuous data


# In[20]:


j=1
plt.figure(figsize=(10,20))
col=numeric.columns
for i in col:
    plt.subplot(len(col)+1,2,j)
    sns.violinplot(numeric[i])
    plt.title(i,color='white')
    j=j+1
    #violin plot is used for knowing the distribution of data, median, etc


# In[21]:


heart_df['HeartDisease'].value_counts().plot(kind='bar')
#visualzing the heartdisease column values using bar graph
#Most of the people are not suffering with heart disease i.e., 280000


# In[22]:


heart_df['GenHealth'].value_counts().plot(kind='bar') 
#visualzing the GENHEALTH column values using bar graph. Here we can see that 
#Most of them having very good health i.e., above 100000


# In[23]:


import matplotlib
colors = ['red']
heart_df['Sex'].value_counts().plot(kind='bar', cmap=matplotlib.colors.ListedColormap(colors))
plt.show()
#visualizing the sex of the people in US suffering with heart disease with the help of bar graph
#we can see most of the females are suffering with this disease


# In[24]:


sns.catplot(data=heart_df, y="Smoking", x="HeartDisease", hue='Sex')

#visualizing the heartdisease with smoking with respect to their gender


# In[25]:


heart_df['Race'].value_counts().plot(kind='bar')
#race of the people getting diagonised with the heart disease
#we can see most of the white people are getting diagonised with heart disease


# ### LABEL ENCODING

# In[26]:


# Import label encoder
from sklearn import preprocessing
  
# label_encoder object knows how to understand word labels.
le = preprocessing.LabelEncoder()
  
heart_df=heart_df.apply(le.fit_transform) #fitting and transforming the data


# ### FEATURE SCALING

# In[27]:


#feature scaling helps in scale the data into certain range. Here I have used minmaxscaler to scale the data into 0-1 range.
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# transform data
scaled = scaler.fit_transform(heart_df)
print(scaled)


# # printing the entire dataset which is been cleaned using EDA process.
# 
# #### Now the dataset is ready for model building

# In[28]:


heart_df


# In[ ]:




