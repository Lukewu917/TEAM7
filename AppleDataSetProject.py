
# %%
import stat
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from warnings import filterwarnings
filterwarnings("ignore")
#!pip3 install ppscore
import ppscore as pps
#Import Library RobustScaler
from sklearn.preprocessing import RobustScaler
#Cluster Model
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import shapiro
from  scipy.stats import spearmanr


# %%
#load_data
import os
os.getcwd()
#os.chdir(r'C:\Users\wpy12\OneDrive\Documents\clone\TEAM7')
appdata = pd.read_csv('AppleStore.csv' ,sep =',' , encoding = 'utf8' )
appdata.head()


#%% [markdown]
# # Mobile App Store 
# Our research is focused on the...
#
# Variables: 

# * "id" : App ID
# * "track_name": App Name
# * "size_bytes": Size (in Bytes)
# * "currency": Currency Type
# * "price": Price amount
# * "ratingcounttot": User Rating counts (for all version)
# * "ratingcountver": User Rating counts (for current version)
# * "user_rating" : Average User Rating value (for all version)
# * "userratingver": Average User Rating value (for current version)
# * "ver" : Latest version code
# * "cont_rating": Content Rating
# * "prime_genre": Primary Genre
# * "sup_devices.num": Number of supporting devices
# * "ipadSc_urls.num": Number of screenshots showed for display
# * "lang.num": Number of supported languages
# * "vpp_lic": Vpp Device Based Licensing Enabled

#%% [markdown]
# # Exploratory Data Analysis
# Look at data and basic descriptive stats
appdata.head()
appdata.shape
#(7197,16)
appdata.columns
# 16 columns in our dataset: five categorical variables, eleven numerical variables(three floats and nine intergers )
appdata.info()
# our appdata set has 7197 rows and the non-null count is 7197 which indicates that there is no missing value in our dataset.
appdata["user_rating"].describe()


#drop unrelated columns
appdata=appdata.drop(columns=['Unnamed: 0'], axis=1)

#double check missing value
appdata.isnull().sum()
# No missing values found 

# #rename some variables in order to make it simple and more sense
# appdata.rename(columns={"size_bytes":"Size","price":"Price","rating_count_tot":"Rating_Count", "user_rating":"Rating", "cont_rating": "Content_Rating", "prime_genre":"App_type", "sup_devices.num":"Devices_Count", "lang.num":"language_Count" } ,inplace=True)
# appdata.head()
# #appdata['Content_Rating'] = [x.strip('+') for x in appdata.Content_Rating]
# appdata['Rating'] = [round(x,1) for x in appdata['Rating']]

#%% [markdown]
# # Managing our Outliers
import matplotlib.pyplot as plt
f,axs = plt.subplots(8,2,figsize=(15,15))
plt.subplot(3,2,1)
appdata['user_rating'].plot(kind='hist', title='user_rating')
plt.subplot(3,2,2)
appdata['user_rating'].plot(kind= 'box')
plt.subplot(3,2,3)
appdata['size_bytes'].plot(kind='hist', title='size_bytes')
plt.subplot(3,2,4)
appdata['size_bytes'].plot(kind= 'box')
plt.subplot(3,2,5)
appdata['price'].plot(kind='hist', title='Price')
plt.subplot(3,2,6)
appdata['price'].plot(kind= 'box')

# f,axs = plt.subplots(3,2,figsize=(15,15))
# plt.subplot(3,2,1)
# appdata['rating_count_tot'].plot(kind='hist', title='rating_count_tot')
# plt.subplot(3,2,2)
# appdata['rating_count_tot'].plot(kind= 'box')
# # plt.subplot(4,2,3)
# # appdata['Content_Rating'].plot(kind='hist', title='Content_Rating')
# # plt.subplot(4,2,4)
# # appdata['Content_Rating'].plot(kind= 'box')
# plt.subplot(3,2,3)
# appdata['sup_devices.num'].plot(kind='hist', title='sup_devices.num')
# plt.subplot(3,2,4)
# appdata['sup_devices.num'].plot(kind= 'box')
# plt.subplot(3,2,5)
# appdata['lang.num'].plot(kind='hist', title='lang.num')
# plt.subplot(3,2,6)
# appdata['lang.num'].plot(kind= 'box')

# Rating data is not normally distributed. Instead, the Rating data seems to be left-skewed. We will try to log-transform the data into log(Rating).


#%%
# appdata['Rating'].apply(np.log).hist()
# Wanted to use log transformation first but some errors ourrced, need to fix it 
#appdata['Rating'].apply(np.exp).hist()
# expenentially tranformation is not fit here.
appdata['user_rating'].apply(np.log).plot(kind='box')
#after normalizting it, it turned out there is some outliers in the transformed Rating data. We need to indentiy these data points before deciding to remove or keep the ourliers. 

# Adding the transformed Rating data to the dataset, and name it log_rating.
appdata['log_rating'] = appdata['user_rating'].apply(np.log)

# Define a function to locate outliers: to find data with difference from sample mean bigger than twice the standard deviation
# def locate_outliers(data,n):
#     return data[abs(data[n] -np.mean(data[n])) > 2 * np.std(data[n])]
# locate_outliers(appdata,'log_rating').head(5)
# there is no data with difference from sample mean bigger than twice the standard deviation. The outliers in the boxplot are NOT satisfied as outliers with our ourlier definition.
# locate_outliers(appdata,'Price').head(5)
# locate_outliers(appdata,'Price').describe()
# # the min of outlier is 13.99.




# print(np.where(appdata['price']>50))
new_appdata = appdata[appdata['price'] <13.99]
print(new_appdata.head(15))
# Since the min of outlier is 13.99 and we want to exclude all the outliers that over 13.99 


new_appdata.shape
new_appdata.describe()
new_appdata["price"].describe()
new_appdata.info()

newprice = list(new_appdata.price)
plt.boxplot(newprice)
plt.show()

plt.hist(new_appdata['price'])


# sns.boxplot(appdata['log_rating'])
# # Position of the Outlier
# print(np.where(appdata['log_rating']<2))
# sns.boxplot(appdata['Price'])



#%% [markdown]
# # Checking for Normality
# * Null hypothesis: Below variables are normally distributed
# * Alt hypothesis: Below variables are NOT normally distributed

print(new_appdata.info())
stat, p = shapiro(new_appdata['price'])
print('Price, Statistics=%.3f, p=%.3f' %(stat,p))

stat, p = shapiro(new_appdata['size_bytes'])
print('App size, Statistics=%.3f, p=%.3f' %(stat,p))

stat, p = shapiro(new_appdata['rating_count_tot'])
print('Count of ratings, Statistics=%.3f, p=%.3f' %(stat,p))

stat, p = shapiro(new_appdata['rating_count_ver'])
print('Count of ratings for current version, Statistics=%.3f, p=%.3f' %(stat,p))

stat, p = shapiro(new_appdata['user_rating'])
print('User ratings, Statistics=%.3f, p=%.3f' %(stat,p))


stat, p = shapiro(new_appdata['sup_devices.num'])
print('Number of Supporting Devices, Statistics=%.3f, p=%.3f' %(stat,p))

stat, p = shapiro(new_appdata['lang.num'])
print('Number of Supported languages , Statistics=%.3f, p=%.3f' %(stat,p))

stat, p = shapiro(new_appdata['ipadSc_urls.num'])
print('Number of Screenshots showed for Display , Statistics=%.3f, p=%.3f' %(stat,p))


# * As shown above, the p-values are less than 0.05, so we reject the null hypothesis which states that the variables are normally distributed. 
# * The variables are NOT normally distributed.

#%%
#create new var based on price
price_categories = new_appdata['price'].map(lambda price: 'free' if price <= 0 else 'paid')
new_appdata['price_cat'] = price_categories

#create new var based on price
rating_categories = new_appdata['user_rating'].map(lambda ratings: 'high' if ratings >=4 else 'low')
new_appdata['ratings_cat'] = rating_categories

#create new var based on ratings
new_appdata.groupby('price_cat').describe()
new_appdata.groupby('ratings_cat').describe()


# Correlation Matrix
#%% 
new_appdata.corr()
corrMatrix = new_appdata.corr()
print (corrMatrix)


corrMatrix = new_appdata.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()


#weak correlation between size and price
price_sizecorr = np.corrcoef(new_appdata.size_bytes, new_appdata.price)
price_sizecorr

new_appdata.plot('price', 'size_bytes', kind='scatter', marker='o') # OR
# plt.plot(df.age, df.rincome, 'o') # if you put marker='o' here, you will get line plot?
plt.ylabel('App size')
plt.xlabel('Price')
plt.show()


#Graph 1
#%%
maxvals = new_appdata.max()
print(maxvals)

pricecategory = new_appdata['price_cat'].value_counts()
pricecategory.plot(kind='bar')
plt.title("Number of Apps, by Price Category")
plt.xlabel("Price Category")
plt.ylabel("Number of Apps")
current_values = plt.gca().get_yticks()
plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values])
plt.show()


# Graph 2

#%%
ratingscol = 'ratings_cat'
new_appdata.groupby(['price_cat', ratingscol]).size().unstack(level=1).plot(kind='bar')
current_values = plt.gca().get_yticks()
plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values])
plt.title('Number of Apps by Price Category and Ratings Category')
plt.ylabel('Number of Apps')
plt.show()

# Graph 3

#%%

plt.style.use('dark_background')
fig = plt.figure(figsize=(10, 5))
highratings = new_appdata[new_appdata['ratings_cat'] == 'high']
lowratings = new_appdata[new_appdata['ratings_cat'] == 'low']


ax = sns.kdeplot(highratings['price'], color ='White', label='Apps with High Ratings', shade=False)
ax = sns.kdeplot(lowratings['price'], color='Red', label='Apps with Low Ratings', shade=False)

plt.yticks([])
plt.title('Price Distribution by User Ratings')
plt.ylabel('')
plt.xlabel('Price')
plt.xlim(0, 10)
plt.legend(loc="upper left")
current_values = plt.gca().get_xticks()
plt.gca().set_xticklabels(['{:,.0f}'.format(x) for x in current_values])
plt.show()

# Graph 4

#%%
# remianing Graphs
# Total apps per group(var)
# average ratings per group(var)
# App Price by genre category (what genre cost the most?)
# avg ratings, free vs paid

#%%
# from statsmodels.formula.api import ols
# modelratingpre = ols(formula='user_rating ~ rating_count_tot + prime_genre + cont_rating', data=dfas)
# modelratingpreFit = modelratingpre.fit()
# print( type(modelratingpreFit) )
# print( modelratingpreFit.summary() )


import statsmodels.formula.api as smf
x=new_appdata[['size_bytes', 'price', 'rating_count_tot']]
y=new_appdata[['user_rating']]
model = smf.ols(formula= 'user_rating ~ size_bytes + price + rating_count_tot ', data=new_appdata)
results_formula = model.fit()
print(model.fit().summary())
# the multi regression equation is : user_rating = 3.375 + 1.964e-10 size_bytes + 0.064 price + 1.812e-06 rating_count_tot. However this model doesn't fit the data very well because the value of r-squared is 0.02 which is very low. R-squared is the proportion of variance explained, so we want R-squared as close to one as possible.

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
from sklearn.metrics import mean_absolute_error
print("MAE",mean_absolute_error(y_test,y_pred))








#%% [markdown]

# Adhithya Kiran' code - will re arrange when complete 



new_appdata.columns

#%% [markdown]

# # Mean user rating of an app is :

print(new_appdata['user_rating'].mean())


print(new_appdata['user_rating'].describe())

i = 0

for element in new_appdata['user_rating']:

    if (element >= 4.5):

        new_appdata['Good_Bad'][i] = 1

        i = i+1

    else :

        new_appdata['Good_Bad'][i] = 0

        i = i+1




# source :  https://www.business2community.com/mobile-apps/app-ratings-and-reviews-2021-benchmarks-02396808


# %%
