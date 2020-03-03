#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[18]:


train = pd.read_csv("C:\\Users\\harsh\\Desktop\\HSNProjects\\ML\\House Price\\input\\train.csv")
test = pd.read_csv("C:\\Users\\harsh\\Desktop\\HSNProjects\\ML\\House Price\\input\\test.csv")


# In[19]:


print ("Train data shape:", train.shape)
print ("Test data shape:", test.shape)


# In[20]:


train.head()
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)


# In[21]:


train.SalePrice.describe()


# In[22]:


print ("Skew is:", train.SalePrice.skew())


# In[23]:


plt.hist(train.SalePrice, color='blue')
plt.show()


# In[24]:


target = np.log(train.SalePrice)
print ("Skew is:", target.skew())
plt.hist(target, color='blue')
plt.show()


# In[26]:


numfeats = train.select_dtypes(include=[np.number])
print(numfeats.dtypes)
corr = numfeats.corr()
print(corr['SalePrice'].sort_values(ascending=False)[:5], '\n')
print(corr['SalePrice'].sort_values(ascending=False)[-5:])
train.OverallQual.unique()


# In[27]:


qp = train.pivot_table(index='OverallQual',
                                  values='SalePrice', aggfunc=np.median)
print(qp)
qp.plot(kind='bar', color='blue')
plt.xlabel('Overall Quality')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()


# In[28]:


plt.scatter(x=train['GrLivArea'], y=target)
plt.ylabel('Sale Price')
plt.xlabel('Above grade (ground) living area square feet')
plt.show()


# In[29]:


plt.scatter(x=train['GarageArea'], y=target)
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
plt.show()


# In[30]:


train = train[train['GarageArea'] < 1200]
plt.scatter(x=train['GarageArea'], y=np.log(train.SalePrice))
plt.xlim(-200,1600) 
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
plt.show()


# In[31]:


nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)


# In[32]:


print("Unique values are:", train.MiscFeature.unique())
categoricals=train.select_dtypes(exclude=[np.number])
print(categoricals.describe())
print("Original: \n") 


# In[33]:


print(train.Street.value_counts(), "\n")
train['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
test['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
print('Encoded: \n') 
print(train.enc_street.value_counts())


# In[34]:


condition_pivot = train.pivot_table(index='SaleCondition',
                                    values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()


# In[37]:


def encode(x):
    if x=='Partial':
        return 1  
    else:
        return 0
    
    


# In[39]:


train['enc_condition'] = train.SaleCondition.apply(encode)
test['enc_condition'] = test.SaleCondition.apply(encode)
cp = train.pivot_table(index='enc_condition', values='SalePrice', aggfunc=np.median)
cp.plot(kind='bar', color='blue')
plt.xlabel('Encoded Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()
data = train.select_dtypes(include=[np.number]).interpolate().dropna() 
y = np.log(train.SalePrice)
X = data.drop(['SalePrice', 'Id'], axis=1)


# In[40]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, random_state=42, test_size=.33)


# In[41]:


from sklearn import linear_model
lr = linear_model.LinearRegression()


# In[42]:


model = lr.fit(X_train, y_train)


# In[43]:


print ("R^2 is: \n", model.score(X_test, y_test))
predictions = model.predict(X_test)


# In[44]:


from sklearn.metrics import mean_squared_error
print ('RMSE is: \n', mean_squared_error(y_test, predictions))
actual_values = y_test
plt.scatter(predictions, actual_values, alpha=.75,
            color='b') 
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Linear Regression Model')
plt.show()


# In[47]:


submission = pd.DataFrame()
submission['Id'] = test.Id
feats = test.select_dtypes(
        include=[np.number]).drop(['Id'], axis=1).interpolate()
predictions = model.predict(feats)
final_predictions = np.exp(predictions)
submission['SalePrice'] = final_predictions
submission.to_csv('C:/Users/harsh/Desktop/HSNProjects/ML/House Price//output//op.csv', index=False)


# In[ ]:




