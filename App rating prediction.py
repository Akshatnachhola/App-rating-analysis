#!/usr/bin/env python
# coding: utf-8

# In[148]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_squared_error as mse


# In[149]:


data = pd.read_csv('googleplaystore.csv')


# In[150]:


data.head()


# In[151]:


data.info()


# In[152]:


data.shape


# #Drop records with null in any of the columns.

# In[153]:


data.isnull().any()


# In[154]:


data.isnull().sum()


# In[155]:


data = data.dropna()


# In[156]:


data.isnull().any()


# In[157]:


data.shape


# #Scaling and cleaning size of installation

# In[158]:


data["Size"] = [ float(i.split('M')[0]) if 'M' in i else float(0) for i in data["Size"]]


# In[159]:


data.head()


# In[160]:


data["Size"] = 1000*data["Size"]


# In[161]:


data


# In[162]:


data.info()


# In[163]:


data["Reviews"] = data ["Reviews"].astype(float)


# In[164]:


data.info()


# #Sanity checks

# In[165]:


data["Installs"] = [ float(i.replace('+','').replace(',', '')) if '+' in i or ',' in i else float(0) for i in data["Installs"] ]


# In[166]:


data.head()


# In[167]:


data.info()


# In[168]:


data["Installs"] = data["Installs"].astype(int)


# In[169]:


data.info()


# In[170]:


data['Price'] = [ float(i.split('$')[1]) if '$' in i else float(0) for i in data['Price'] ]


# In[171]:


data.head()


# In[172]:


data.info()


# In[173]:


data["Price"] = data["Price"].astype(int)


# In[174]:


data.info()


# In[175]:


data.shape


# In[176]:


data.drop(data[(data['Reviews'] < 1) & (data['Reviews'] > 5 )].index, inplace = True)


# In[177]:


data.drop(data[data['Installs'] < data['Reviews'] ].index, inplace = True)


# In[178]:


data.shape


# In[179]:


data.drop(data[(data['Type'] =='Free') & (data['Price'] > 0 )].index, inplace = True)


# #Univariate Analysis

# In[180]:


sns.set(rc={'figure.figsize':(10,6)})


# In[181]:


sns.boxplot(data['Price'])
plt.xlabel('Price')


# In[182]:


sns.boxplot(data['Reviews'])
plt.xlabel('Review')


# In[183]:


sns.boxplot(data['Rating'])


# In[184]:


sns.boxplot(data['Size'])


# #Outliers treatment

# In[185]:


more = data.apply(lambda x : True
            if x['Price'] > 200 else False, axis = 1) 


# In[186]:


more_count = len(more[more == True].index)


# In[187]:


data.drop(data[data['Price'] > 200].index, inplace = True)


# In[188]:


data.shape


# In[189]:


data.drop(data[data['Reviews'] > 2000000].index, inplace = True)


# In[190]:


data.shape


# In[191]:


data.quantile([.1, .25, .5, .70, .90, .95, .99], axis = 0) 

# dropping more than 10000000 Installs value
# In[192]:


data.drop(data[data['Installs'] > 10000000].index, inplace = True)


# In[193]:


data.shape


# #Bivariate Analysis

# In[194]:


sns.scatterplot(x='Rating',y='Price',data=data)


# Yes, Paid apps are higher ratings comapre to free apps.

# In[195]:


sns.scatterplot(x='Rating',y='Size',data=data)


# yes, it is clear that heavior apps are rated better.

# In[196]:


sns.scatterplot(x='Rating',y='Reviews',data=data)


# In[197]:


sns.boxplot(x="Rating", y="Content Rating", data=data)


# Apps which are for everyone has more bad ratings compare to other sections as it has so much outliers value, while 18+ apps have better ratings

# In[198]:


sns.boxplot(x="Rating", y="Category", data=data)


# Events category has best ratings compare to others
# Data Processing
# In[199]:


inp1 = data


# In[200]:


inp1.head()


# In[201]:


inp1.skew()


# In[202]:


reviewskew = np.log1p(inp1['Reviews'])
inp1['Reviews'] = reviewskew


# In[203]:


reviewskew.skew()


# In[204]:


installsskew = np.log1p(inp1['Installs'])
inp1['Installs']


# In[205]:


installsskew.skew()


# In[206]:


inp1.head()


# In[207]:


inp1.drop(["Last Updated","Current Ver","Android Ver","App","Type"],axis=1,inplace=True)



# In[208]:


inp1.head()


# In[209]:


inp1.shape


# In[210]:


inp2 = inp1


# In[211]:


inp2.head()

# Let's apply Dummy EnCoding on Column "Category"
# In[212]:


#get unique values in Column "Category"
inp2.Category.unique()


# In[213]:


inp2.Category = pd.Categorical(inp2.Category)

x = inp2[['Category']]
del inp2['Category']

dummies = pd.get_dummies(x, prefix = 'Category')
inp2 = pd.concat([inp2,dummies], axis=1)
inp2.head()


# In[214]:


inp2.shape


# In[215]:


#get unique values in Column "Genres"
inp2["Genres"].unique()

Since there are too many genre-specific categories. As a result, we'll aim to consolidate several categories with a small number of samples into one new category called Other .
# In[216]:


lists = []
for i in inp2.Genres.value_counts().index:
    if inp2.Genres.value_counts()[i]<20:
        lists.append(i)
inp2.Genres = ['Other' if i in lists else i for i in inp2.Genres] 


# In[217]:


inp2["Genres"].unique()


# In[218]:


inp2.Genres = pd.Categorical(inp2['Genres'])
x = inp2[["Genres"]]
del inp2['Genres']
dummies = pd.get_dummies(x, prefix = 'Genres')
inp2 = pd.concat([inp2,dummies], axis=1)


# In[219]:


inp2.head()


# In[220]:


#get unique values in Column "Content Rating"
inp2["Content Rating"].unique()


# In[221]:


inp2['Content Rating'] = pd.Categorical(inp2['Content Rating'])

x = inp2[['Content Rating']]
del inp2['Content Rating']

dummies = pd.get_dummies(x, prefix = 'Content Rating')
inp2 = pd.concat([inp2,dummies], axis=1)
inp2.head()


# In[222]:


inp2.shape


# #Model Building

# In[223]:


d1 = inp2
X = d1.drop('Rating',axis=1)
y = d1['Rating']

Xtrain, Xtest, ytrain, ytest = tts(X,y, test_size=0.3, random_state=5)


# In[224]:


reg_all = LR()
reg_all.fit(Xtrain,ytrain)


# In[225]:


R2_train = round(reg_all.score(Xtrain,ytrain),3)
print("The R2 value of the Training Set is : {}".format(R2_train))


# In[226]:


R2_test = round(reg_all.score(Xtest,ytest),3)
print("The R2 value of the Testing Set is : {}".format(R2_test))

