#!/usr/bin/env python
# coding: utf-8

# # Rental Bikes Demand Prediction

# # Loading Dataset and Importing Modules

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from datetime import datetime
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split

from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error


# In[3]:


import warnings
warnings.filterwarnings('ignore')


# 
# # Import the datset

# In[4]:


rental = pd.read_csv("Rental_Bike_Prediction.csv",encoding='latin')


# In[5]:


rental


# In[6]:


#for displaying the top ten rows in dataset
rental.head(10)


# In[7]:


#for displaying the bottom ten rows in dataset
rental.tail(10)


# In[8]:


#statistical detail of dataset
rental.describe().T


# In[9]:


#shape of the dataset
rental.shape


# In[10]:


#shows the columns of dataset
rental.columns


# In[11]:


#information of the dataset
rental.info()


# In[12]:


#shows the no of unique values in each column of dataset
rental.nunique()


# In[13]:


#gives the null values
rental.isna().sum()


# # Description of dataset 

# # Breakdown of Our Features:
# 
# Date : The date of the day, during 365 days from 01/12/2017 to 30/11/2018, formating in DD/MM/YYYY, type : str, we need to convert into datetime format.
# 
# Rented Bike Count : Number of rented bikes per hour which our dependent variable and we need to predict that, type : int
# 
# Hour: The hour of the day, starting from 0-23 it's in a digital time format, type : int, we need to convert it into category data type.
# 
# Temperature(°C): Temperature in Celsius, type : Float
# 
# Humidity(%): Humidity in the air in %, type : int
# 
# Wind speed (m/s) : Speed of the wind in m/s, type : Float
# 
# Visibility (10m): Visibility in m, type : int
# 
# Dew point temperature(°C): Temperature at the beggining of the day, type : Float
# 
# Solar Radiation (MJ/m2): Sun contribution, type : Float
# 
# Rainfall(mm): Amount of raining in mm, type : Float
# 
# Snowfall (cm): Amount of snowing in cm, type : Float
# 
# Seasons: *Season of the year, type : str, there are only 4 season's in data *.
# 
# Holiday: If the day is holiday period or not, type: str
# 
# Functioning Day: If the day is a Functioning Day or not, type : str

# In[14]:


#shows the no of duplicated values
rental.duplicated().sum()


# In[15]:


#changing the column name
rental=rental.rename(columns={'Rented Bike Count':'Rented_Bike_Count',
                                'Temperature(°C)':'Temperature',
                                'Humidity(%)':'Humidity',
                                'Wind speed (m/s)':'Wind_speed',
                                'Visibility (10m)':'Visibility',
                                'Dew point temperature(°C)':'Dew_point_temperature',
                                'Solar Radiation (MJ/m2)':'Solar_Radiation',
                                'Rainfall(mm)':'Rainfall',
                                'Snowfall (cm)':'Snowfall',
                                'Functioning Day':'Functioning_Day'})


# In[16]:


#displays the dataset after updating the columns
rental


# In[17]:


# Changing the "Date" column into three (year,month,day) column
rental['Date'] = rental['Date'].apply(lambda x: 
                                    dt.datetime.strptime(x,"%d/%m/%Y"))


# In[18]:


rental['year'] = rental['Date'].dt.year
rental['month'] = rental['Date'].dt.month
rental['day'] = rental['Date'].dt.day_name()


# In[19]:


#creating a new column of "weekdays_weekend" and drop the column "Date","day","year"
rental['weekdays_weekend']=rental['day'].apply(lambda x : 1 if x=='Saturday' or x=='Sunday' else 0 )
rental=rental.drop(columns=['Date','day','year'],axis=1)


# In[20]:


#displays the top 5 row in dataset
rental.head()


# In[21]:


#updated information of the dataset
rental.info()


# In[22]:


#count of weekdays and weekend
rental['weekdays_weekend'].value_counts()


# In[23]:


#gives the updated unique values of a column
rental.nunique()


# In[24]:


#Change the int64 column into category column
cols=['Hour','month','weekdays_weekend']
for col in cols:
    rental[col]=rental[col].astype('category')


# In[25]:


#gives the information of the data type
rental.info()


# In[26]:


#gives the updated column names
rental.columns


# In[27]:


#unique values in (weekdays_weekend) column
rental['weekdays_weekend'].unique()


# # Month wise Analysis

# In[28]:


#anlysis of data by vizualisation
fig,ax=plt.subplots(figsize=(20,8))
sns.barplot(data=rental,x='month',y='Rented_Bike_Count')
plt.title('Count of Rented bikes acording to Month ')


# # weekend wise Analysis

# In[29]:


#anlysis of data by vizualisation according to weekend by boxplot
fig,ax=plt.subplots(figsize=(10,8))
sns.barplot(data=rental,x='weekdays_weekend',y='Rented_Bike_Count')
plt.title('Count of Rented bikes acording to weekdays and weekend ')


# In[30]:


#anlysis of data by vizualisation according to weekend by pointplot
fig,ax=plt.subplots(figsize=(20,8))
sns.pointplot(data=rental,x='Hour',y='Rented_Bike_Count',hue='weekdays_weekend')
plt.title('Count of Rented bikes acording to weekdays_weekend ')


# # Hour wise Analysis

# In[31]:


#anlysis of data by vizualisation according to hour by boxplot
fig,ax=plt.subplots(figsize=(20,8))
sns.barplot(data=rental,x='Hour',y='Rented_Bike_Count')
plt.title('Count of Rented bikes acording to Hour ')


# # Functioning day wise Analysis

# In[32]:


#anlysis of data by vizualisation according to Functioning day by boxplot
fig,ax=plt.subplots(figsize=(10,8))
sns.barplot(data=rental,x='Functioning_Day',y='Rented_Bike_Count')
plt.title('Count of Rented bikes acording to Functioning Day ')


# In[33]:


#anlysis of data by vizualisation according to Functioning day by pointplot
fig,ax=plt.subplots(figsize=(20,8))
sns.pointplot(data=rental,x='Hour',y='Rented_Bike_Count',hue='Functioning_Day')
plt.title('Count of Rented bikes acording to Functioning Day ')


# # Season wise Analysis

# In[34]:


#anlysis of data by vizualisation according to seasons by boxplot
fig,ax=plt.subplots(figsize=(15,8))
sns.barplot(data=rental,x='Seasons',y='Rented_Bike_Count')
plt.title('Count of Rented bikes acording to Seasons ')


# In[35]:


#anlysis of data by vizualisation according to seasons by pointplot
fig,ax=plt.subplots(figsize=(20,8))
sns.pointplot(data=rental,x='Hour',y='Rented_Bike_Count',hue='Seasons')
plt.title('Count of Rented bikes acording to seasons ')


# 
# # Holiday wise Analysis

# In[36]:


#anlysis of data by vizualisation according to holiday by boxplot
fig,ax=plt.subplots(figsize=(15,8))
sns.barplot(data=rental,x='Holiday',y='Rented_Bike_Count')
plt.title('Count of Rented bikes acording to Holiday ')


# In[37]:


#anlysis of data by vizualisation according to holiday by pointplot
fig,ax=plt.subplots(figsize=(20,8))
sns.pointplot(data=rental,x='Hour',y='Rented_Bike_Count',hue='Holiday')
plt.title('Count of Rented bikes acording to Holiday ')


# # Analysis of Numerical variables distplots

# In[38]:


#assign the numerical coulmn to variables
numerical_columns=list(rental.select_dtypes(['int64','float64']).columns)
numerical_features=pd.Index(numerical_columns)
numerical_features


# In[39]:


#printing displots to analyze the distribution of all numerical features
for col in numerical_features:
    plt.figure(figsize=(10,6))
    sns.distplot(x=rental[col])
    plt.xlabel(col)
plt.show()


# # Numerical vs.Rented_Bike_Count

# In[40]:


#print the plot to analyze the relationship between "Rented_Bike_Count" and "Temperature" 
rental.groupby('Temperature').mean()['Rented_Bike_Count'].plot()


# In[41]:


#print the plot to analyze the relationship between "Rented_Bike_Count" and "Dew_point_temperature" 
rental.groupby('Dew_point_temperature').mean()['Rented_Bike_Count'].plot()


# In[42]:


#print the plot to analyze the relationship between "Rented_Bike_Count" and "Solar_Radiation" 
rental.groupby('Solar_Radiation').mean()['Rented_Bike_Count'].plot()


# In[43]:


#print the plot to analyze the relationship between "Rented_Bike_Count" and "Snowfall" 
rental.groupby('Snowfall').mean()['Rented_Bike_Count'].plot()


# In[44]:


#print the plot to analyze the relationship between "Rented_Bike_Count" and "Rainfall" 
rental.groupby('Rainfall').mean()['Rented_Bike_Count'].plot()


# In[45]:


#print the plot to analyze the relationship between "Rented_Bike_Count" and "Wind_speed" 
rental.groupby('Wind_speed').mean()['Rented_Bike_Count'].plot()


# # Checking of outliers 

# In[46]:


#Boxplot of Rented Bike Count to check outliers
plt.figure(figsize=(10,6))
plt.ylabel('Rented_Bike_Count')
sns.boxplot(x=rental['Rented_Bike_Count'])
plt.show()


# In[47]:


#After applying sqrt on Rented Bike Count check wheater we still have outliers 
plt.figure(figsize=(10,6))
plt.ylabel('Rented_Bike_Count')
sns.boxplot(x=np.sqrt(rental['Rented_Bike_Count']))
plt.show()


# # Correlation of data after clearing the outliers

# In[48]:


rental.corr()


# # Checking in Ordinary Least Squares regression (OLS) Model

# Ordinary least squares (OLS) regression is a statistical method of analysis that estimates the relationship between one or more independent variables and a dependent variable

# In[49]:


#import the module
#assign the 'x','y' value
import statsmodels.api as sm
X = rental[[ 'Temperature','Humidity',
       'Wind_speed', 'Visibility','Dew_point_temperature',
       'Solar_Radiation', 'Rainfall', 'Snowfall']]
Y = rental['Rented_Bike_Count']


# In[50]:


#displays the top 5 rows in the data
rental.head()


# In[51]:


#add a constant to the column
X = sm.add_constant(X)
X


# # fitting an Ordinary least squares (OLS) model 

# In[52]:


model= sm.OLS(Y, X).fit()
model.summary()


# # Finding the correlation of splitted data stored in x

# In[53]:


X.corr()


# # Heatmap

# visualizing the strength of correlation among variables.

# In[54]:


#plotting the Correlation matrix
plt.figure(figsize=(20,8))
correlation=rental.corr()
mask = np.triu(np.ones_like(correlation, dtype=bool))
sns.heatmap((correlation),mask=mask, annot=True,cmap='coolwarm')


# In[55]:


#drop the Dew point temperature column
rental=rental.drop(['Dew_point_temperature'],axis=1)


# In[56]:


#after dropping the Dew point temperature column checking the information of the data
rental.info()


# # Creating the dummies for variables which are in object and categorical values

# In[57]:


#Assign all catagoriacal features to a variable
categorical_features=list(rental.select_dtypes(['object','category']).columns)
categorical_features=pd.Index(categorical_features)
categorical_features


# # One hot encoding

# In[58]:


#create a copy
rental_copy = rental

def one_hot_encoding(data, column):
    data = pd.concat([data, pd.get_dummies(data[column], prefix=column, drop_first=True)], axis=1)
    data = data.drop([column], axis=1)
    return data

for col in categorical_features:
    rental_copy = one_hot_encoding(rental_copy, col)    


# In[59]:


rental_copy.head()   


# # Splitting the dataset for Train & Test for regression

# In[60]:


#Assign the value in X and Y
X = rental_copy.drop(columns=['Rented_Bike_Count'], axis=1)
y = np.sqrt(rental_copy['Rented_Bike_Count'])


# In[61]:


#displays the top 10 rows of the x dataset
X.head(10)


# In[62]:


#displays the top 5 values in the y dataset
y.head()


# In[63]:


#Create test and train data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)


# In[64]:


#displays the shape of train and test data
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[65]:


#displays the columns
rental_copy.columns


# In[66]:


#gives the statistical description of the data
rental_copy.describe()


# # LINEAR REGRESSION

# Regression models a target prediction value based on independent variables. It is mostly used for finding out the relationship between variables and forecasting

# In[67]:


#importing the package
from sklearn.linear_model import LinearRegression


# In[68]:


#fitting the model
lr_model= LinearRegression().fit(X_train, y_train)


# In[69]:


#calculating the model score
lr_model.score(X_train, y_train)


# In[70]:


#coefficient of the model
lr_model.coef_


# In[71]:


#get the X_train and X-test value
y_pred_train=lr_model.predict(X_train)
y_pred_test=lr_model.predict(X_test)


# In[72]:


y_pred_train


# In[73]:


y_pred_test


# # Calculating for train data

# In[74]:


#import the packages
from sklearn.metrics import mean_squared_error
#calculate MSE
MSE_lr= mean_squared_error((y_train), (y_pred_train))
print("MSE :",MSE_lr)

#calculate RMSE
RMSE_lr=np.sqrt(MSE_lr)
print("RMSE :",RMSE_lr)

#calculate MAE
MAE_lr= mean_absolute_error(y_train, y_pred_train)
print("MAE :",MAE_lr)

#import the packages
from sklearn.metrics import r2_score
#calculate r2 and adjusted r2
#Adjusted R2 is a corrected goodness-of-fit (model accuracy) measure for linear models.
r2_lr= r2_score(y_train, y_pred_train)
print("R2 :",r2_lr)
Adjusted_R2_lr = (1-(1-r2_score(y_train, y_pred_train))*((X_test.shape[0]-1)/(X_test.shape[0]-X_test.shape[1]-1)) )
print("Adjusted R2 :",Adjusted_R2_lr )


# In[75]:


# storing the test set metrics value in a dataframe for later comparison
dict1={'Model':'Linear regression ',
       'MAE':round((MAE_lr),3),
       'MSE':round((MSE_lr),3),
       'RMSE':round((RMSE_lr),3),
       'R2_score':round((r2_lr),3),
       'Adjusted R2':round((Adjusted_R2_lr ),2)
       }
training=pd.DataFrame(dict1,index=[1])


# # Calculating for test data

# In[76]:


#import the packages
from sklearn.metrics import mean_squared_error
#calculate MSE
MSE_lr= mean_squared_error(y_test, y_pred_test)
print("MSE :",MSE_lr)

#calculate RMSE
RMSE_lr=np.sqrt(MSE_lr)
print("RMSE :",RMSE_lr)


#calculate MAE
MAE_lr= mean_absolute_error(y_test, y_pred_test)
print("MAE :",MAE_lr)


#import the packages
from sklearn.metrics import r2_score
#calculate r2 and adjusted r2
r2_lr= r2_score((y_test), (y_pred_test))
print("R2 :",r2_lr)
Adjusted_R2_lr = (1-(1-r2_score((y_test), (y_pred_test)))*((X_test.shape[0]-1)/(X_test.shape[0]-X_test.shape[1]-1)))
print("Adjusted R2 :",Adjusted_R2_lr )


# In[77]:


# storing the test set metrics value in a dataframe for later comparison
dict2={'Model':'Linear regression ',
       'MAE':round((MAE_lr),3),
       'MSE':round((MSE_lr),3),
       'RMSE':round((RMSE_lr),3),
       'R2_score':round((r2_lr),3),
       'Adjusted R2':round((Adjusted_R2_lr ),2)
       }
testing=pd.DataFrame(dict2,index=[1])


# # RANDOM FOREST REGRESSOR

#  Random Forest is an ensemble technique capable of performing regression with the use of multiple decision trees and a technique called Bootstrap and Aggregation, commonly known as bagging.

# In[78]:


#import the packages
from sklearn.ensemble import RandomForestRegressor
# Create an instance of the RandomForestRegressor
rf_model = RandomForestRegressor()
rf_model.fit(X_train,y_train)


# In[79]:


# Making predictions on train and test data

y_pred_train_r = rf_model.predict(X_train)
y_pred_test_r = rf_model.predict(X_test)


# # Calculating for train data

# In[80]:


#import the packages
from sklearn.metrics import mean_squared_error
print("Model Score:",rf_model.score(X_train,y_train))

#calculate MSE
MSE_rf= mean_squared_error(y_train, y_pred_train_r)
print("MSE :",MSE_rf)

#calculate RMSE
RMSE_rf=np.sqrt(MSE_rf)
print("RMSE :",RMSE_rf)


#calculate MAE
MAE_rf= mean_absolute_error(y_train, y_pred_train_r)
print("MAE :",MAE_rf)


#import the packages
from sklearn.metrics import r2_score
#calculate r2 and adjusted r2
r2_rf= r2_score(y_train, y_pred_train_r)
print("R2 :",r2_rf)
Adjusted_R2_rf=(1-(1-r2_score(y_train, y_pred_train_r))*((X_test.shape[0]-1)/(X_test.shape[0]-X_test.shape[1]-1)) )
print("Adjusted R2 :",Adjusted_R2_rf )


# In[81]:


# storing the test set metrics value in a dataframe for later comparison
dict1={'Model':'Random forest regression ',
       'MAE':round((MAE_rf),3),
       'MSE':round((MSE_rf),3),
       'RMSE':round((RMSE_rf),3),
       'R2_score':round((r2_rf),3),
       'Adjusted R2':round((Adjusted_R2_rf ),2)}
training=training.append(dict1,ignore_index=True)


# # Calculating for test data

# In[82]:


#import the packages
from sklearn.metrics import mean_squared_error
#calculate MSE
MSE_rf= mean_squared_error(y_test, y_pred_test_r)
print("MSE :",MSE_rf)

#calculate RMSE
RMSE_rf=np.sqrt(MSE_rf)
print("RMSE :",RMSE_rf)


#calculate MAE
MAE_rf= mean_absolute_error(y_test, y_pred_test_r)
print("MAE :",MAE_rf)


#import the packages
from sklearn.metrics import r2_score
#calculate r2 and adjusted r2
r2_rf= r2_score((y_test), (y_pred_test_r))
print("R2 :",r2_rf)
Adjusted_R2_rf=(1-(1-r2_score((y_test), (y_pred_test_r)))*((X_test.shape[0]-1)/(X_test.shape[0]-X_test.shape[1]-1)) )
print("Adjusted R2 :",1-(1-r2_score((y_test), (y_pred_test_r)))*((X_test.shape[0]-1)/(X_test.shape[0]-X_test.shape[1]-1)) )


# In[83]:


# storing the test set metrics value in a dataframe for later comparison
dict2={'Model':'Random forest regression ',
       'MAE':round((MAE_rf),3),
       'MSE':round((MSE_rf),3),
       'RMSE':round((RMSE_rf),3),
       'R2_score':round((r2_rf),3),
       'Adjusted R2':round((Adjusted_R2_rf ),2)}
testing=testing.append(dict2,ignore_index=True)


# In[84]:


rf_model.feature_importances_


# In[85]:


importances = rf_model.feature_importances_
importance_dict = {'Feature' : list(X_train.columns),
                   'Feature Importance' : importances}
importance_df = pd.DataFrame(importance_dict)


# In[86]:


importance_df['Feature Importance'] = round(importance_df['Feature Importance'],2)


# In[87]:


importance_df.sort_values(by=['Feature Importance'],ascending=False)


# In[88]:


#FIT THE MODEL
rf_model.fit(X_train,y_train)


# In[89]:


features = X_train.columns
importances = rf_model.feature_importances_
indices = np.argsort(importances)


# # BAR GRAPH VISUALIZATION

# In[90]:


#Plot the figure
plt.figure(figsize=(10,20))
plt.title('Feature Importance')
plt.barh(range(len(indices)), importances[indices], color="green", align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')

plt.show()


# # GradientBoostingRegressor

# It builds an additive model . It allows to do the optimization of arbitrary differentiable loss functions. In each stage a regression tree is fit on the negative gradient of the given loss function

# In[91]:


#import the packages
from sklearn.ensemble import GradientBoostingRegressor
# Create an instance of the GradientBoostingRegressor
gb_model = GradientBoostingRegressor()


# In[92]:


#fitting the model
gb_model.fit(X_train,y_train)


# In[93]:


# Making predictions on train and test data

y_pred_train_g = gb_model.predict(X_train)
y_pred_test_g = gb_model.predict(X_test)


# # Calculating for train data

# In[94]:


#import the packages
from sklearn.metrics import mean_squared_error
print("Model Score:",gb_model.score(X_train,y_train))
#calculate MSE
MSE_gb= mean_squared_error(y_train, y_pred_train_g)
print("MSE :",MSE_gb)

#calculate RMSE
RMSE_gb=np.sqrt(MSE_gb)
print("RMSE :",RMSE_gb)


#calculate MAE
MAE_gb= mean_absolute_error(y_train, y_pred_train_g)
print("MAE :",MAE_gb)


#import the packages
from sklearn.metrics import r2_score
#calculate r2 and adjusted r2
r2_gb= r2_score(y_train, y_pred_train_g)
print("R2 :",r2_gb)
Adjusted_R2_gb = (1-(1-r2_score(y_train, y_pred_train_g))*((X_test.shape[0]-1)/(X_test.shape[0]-X_test.shape[1]-1)) )
print("Adjusted R2 :",1-(1-r2_score(y_train, y_pred_train_g))*((X_test.shape[0]-1)/(X_test.shape[0]-X_test.shape[1]-1)) )


# In[95]:


# storing the test set metrics value in a dataframe for later comparison
dict1={'Model':'Gradient boosting regression ',
       'MAE':round((MAE_gb),3),
       'MSE':round((MSE_gb),3),
       'RMSE':round((RMSE_gb),3),
       'R2_score':round((r2_gb),3),
       'Adjusted R2':round((Adjusted_R2_gb ),2),
       }
training=training.append(dict1,ignore_index=True)


# # Calculating for test data

# In[96]:


#import the packages
from sklearn.metrics import mean_squared_error
#calculate MSE
MSE_gb= mean_squared_error(y_test, y_pred_test_g)
print("MSE :",MSE_gb)

#calculate RMSE
RMSE_gb=np.sqrt(MSE_gb)
print("RMSE :",RMSE_gb)


#calculate MAE
MAE_gb= mean_absolute_error(y_test, y_pred_test_g)
print("MAE :",MAE_gb)


#import the packages
from sklearn.metrics import r2_score
#calculate r2 and adjusted r2
r2_gb= r2_score((y_test), (y_pred_test_g))
print("R2 :",r2_gb)
Adjusted_R2_gb = (1-(1-r2_score((y_test), (y_pred_test_g)))*((X_test.shape[0]-1)/(X_test.shape[0]-X_test.shape[1]-1)))
print("Adjusted R2 :",1-(1-r2_score((y_test), (y_pred_test_g)))*((X_test.shape[0]-1)/(X_test.shape[0]-X_test.shape[1]-1)) )


# In[97]:


# storing the test set metrics value in a dataframe for later comparison
dict2={'Model':'Gradient boosting regression ',
       'MAE':round((MAE_gb),3),
       'MSE':round((MSE_gb),3),
       'RMSE':round((RMSE_gb),3),
       'R2_score':round((r2_gb),3),
       'Adjusted R2':round((Adjusted_R2_gb ),2),
       }
testing=testing.append(dict2,ignore_index=True)


# In[98]:


gb_model.feature_importances_


# In[99]:


importances = gb_model.feature_importances_

importance_dict = {'Feature' : list(X_train.columns),
                   'Feature Importance' : importances}

importance_df = pd.DataFrame(importance_dict)


# In[100]:


importance_df['Feature Importance'] = round(importance_df['Feature Importance'],2)


# In[101]:


importance_df.head(10)


# In[102]:


importance_df.sort_values(by=['Feature Importance'],ascending=False)


# In[103]:


gb_model.fit(X_train,y_train)


# In[104]:


features = X_train.columns
importances = gb_model.feature_importances_
indices = np.argsort(importances)


# In[105]:


#Plot the figure
plt.figure(figsize=(10,20))
plt.title('Feature Importance')
plt.barh(range(len(indices)), importances[indices], color='green', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')

plt.show()


# # Hyperparameter tuning:
# Before proceding to try next models, let us try to tune some hyperparameters and see if the performance of our model improves.
# 
# Hyperparameter tuning is the process of choosing a set of optimal hyperparameters for a learning algorithm. A hyperparameter is a model argument whose value is set before the learning process begins. The key to machine learning algorithms is hyperparameter tuning.
# 
# # Using GridSearchCV:
# 
# GridSearchCV helps to loop through predefined hyperparameters and fit the model on the training set. So, in the end, we can select the best parameters from the listed hyperparameters.

# # Gradient Boosting Regressor with GridSearchCV

# # Provide the range of values for chosen hyperparameters

# In[106]:


# Number of trees
n_estimators = [50,80,100]

# Maximum depth of trees
max_depth = [4,6,8]

# Minimum number of samples required to split a node
min_samples_split = [50,100,150]

# Minimum number of samples required at each leaf node
min_samples_leaf = [40,50]

# HYperparameter Grid
param_dict = {'n_estimators' : n_estimators,
              'max_depth' : max_depth,
              'min_samples_split' : min_samples_split,
              'min_samples_leaf' : min_samples_leaf}


# In[107]:


param_dict


# # Importing Gradient Boosting Regressor

# In[108]:


from sklearn.model_selection import GridSearchCV
# Create an instance of the GradientBoostingRegressor
gb_model = GradientBoostingRegressor()

# Grid search
gb_grid = GridSearchCV(estimator=gb_model,param_grid = param_dict,cv = 5, verbose=2)


# In[109]:


gb_grid.fit(X_train,y_train)


# In[110]:


gb_grid.best_estimator_


# In[111]:


gb_optimal_model = gb_grid.best_estimator_


# In[112]:


gb_grid.best_params_


# In[113]:


# Making predictions on train and test data

y_pred_train_g_g = gb_optimal_model.predict(X_train)
y_pred_g_g= gb_optimal_model.predict(X_test)


# # Calculating the Train data

# In[114]:


#import the package
from sklearn.metrics import mean_squared_error


# In[115]:


#model Score
print("Model Score:",gb_optimal_model.score(X_train,y_train))


# In[116]:


#calculate MSE
MSE_gbh= mean_squared_error(y_train, y_pred_train_g_g)
print("MSE :",MSE_gbh)
#calculate RMSE
RMSE_gbh=np.sqrt(MSE_gbh)
print("RMSE :",RMSE_gbh)
#calculate MAE
MAE_gbh= mean_absolute_error(y_train, y_pred_train_g_g)
print("MAE :",MAE_gbh)
#import package
from sklearn.metrics import r2_score
#calculate r2 and adjusted r2
r2_gbh= r2_score(y_train, y_pred_train_g_g)
print("R2 :",r2_gbh)
Adjusted_R2_gbh = (1-(1-r2_score(y_train, y_pred_train_g_g))*((X_test.shape[0]-1)/(X_test.shape[0]-X_test.shape[1]-1)) )
print("Adjusted R2 :",1-(1-r2_score(y_train, y_pred_train_g_g))*((X_test.shape[0]-1)/(X_test.shape[0]-X_test.shape[1]-1)) )


# In[117]:


# storing the test set metrics value in a dataframe for later comparison
dict1={'Model':'Gradient Boosting gridsearchcv ',
       'MAE':round((MAE_gbh),3),
       'MSE':round((MSE_gbh),3),
       'RMSE':round((RMSE_gbh),3),
       'R2_score':round((r2_gbh),3),
       'Adjusted R2':round((Adjusted_R2_gbh ),2)
      }
training=training.append(dict1,ignore_index=True)


# # Calculating the test data

# In[118]:


#import packages
from sklearn.metrics import mean_squared_error
#calculate MSE
MSE_gbh= mean_squared_error(y_test, y_pred_g_g)
print("MSE :",MSE_gbh)
#calculate RMSE
RMSE_gbh=np.sqrt(MSE_gbh)
print("RMSE :",RMSE_gbh)
#calculate MAE
MAE_gbh= mean_absolute_error(y_test, y_pred_g_g)
print("MAE :",MAE_gbh)
#import package
from sklearn.metrics import r2_score
#calculate r2 and adjusted r2
r2_gbh= r2_score((y_test), (y_pred_g_g))
print("R2 :",r2_gbh)
Adjusted_R2_gbh = (1-(1-r2_score(y_test, y_pred_g_g))*((X_test.shape[0]-1)/(X_test.shape[0]-X_test.shape[1]-1)) )
print("Adjusted R2 :",1-(1-r2_score((y_test), (y_pred_g_g)))*((X_test.shape[0]-1)/(X_test.shape[0]-X_test.shape[1]-1)) )


# In[119]:


# storing the test set metrics value in a dataframe for later comparison
dict2={'Model':'Gradient Boosting gridsearchcv ',
       'MAE':round((MAE_gbh),3),
       'MSE':round((MSE_gbh),3),
       'RMSE':round((RMSE_gbh),3),
       'R2_score':round((r2_gbh),3),
       'Adjusted R2':round((Adjusted_R2_gbh ),2)
      }
testing=testing.append(dict2,ignore_index=True)


# In[120]:


gb_optimal_model.feature_importances_


# In[121]:


importances = gb_optimal_model.feature_importances_

importance_dict = {'Feature' : list(X_train.columns),
                   'Feature Importance' : importances}

importance_df = pd.DataFrame(importance_dict)


# In[122]:


importance_df['Feature Importance'] = round(importance_df['Feature Importance'],2)


# In[123]:


importance_df.head(10)


# In[124]:


importance_df.sort_values(by=['Feature Importance'],ascending=False)


# In[125]:


gb_model.fit(X_train,y_train)


# In[126]:


features = X_train.columns
importances = gb_model.feature_importances_
indices = np.argsort(importances)


# In[127]:


#Plot the figure
plt.figure(figsize=(10,20))
plt.title('Feature Importance')
plt.barh(range(len(indices)), importances[indices], color='red', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')

plt.show()


# # Model Comparison

#  During the time of our analysis, we initially did EDA on all the features of our datset. We first analysed our dependent variable, 'Rented Bike Count' and also transformed it. Next we analysed categorical variable and dropped the variable who had majority of one class, we also analysed numerical variable, found out the correlation, distribution and their relationship with the dependent variable. We also removed some numerical features who had mostly 0 values and hot encoded the categorical variables. Next we implemented 4 machine learning algorithms Linear Regression,decission tree, Random Forest . We did hyperparameter tuning to improve our model performance. The results of our evaluation are:

# In[128]:


# displaying the results of evaluation metric values for all models
result=pd.concat([training,testing],keys=['Training set','Test set'])
result


# # conclusion
# • No overfitting is seen.
# 
# • Random forest Regressor and Gradient Boosting gridsearchcv gives the highest R2 score of 99% and 95% recpectively for Train Set and 92% for Test set.
# 
# • Feature Importance value for Random Forest and Gradient Boost are different.
# 
# • We can deploy this model.
