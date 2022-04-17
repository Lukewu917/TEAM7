
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
os.chdir(r'C:\Users\wpy12\OneDrive\Documents\clone\TEAM7')
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
# 16 columns in our dataset: five categorical variables, eleven numerical variables(three floats and eight intergers )
appdata.info()
# our appdata set has 7197 rows and the non-null count is 7197 which indicates that there is no missing value in our dataset.
appdata["user_rating"].describe()

#drop unrelated columns
appdata=appdata.drop(columns=['Unnamed: 0'], axis=1)

#appdata.shape
print(appdata.isnull())
appdata.isnull().sum()
print(appdata.dropna())
appdata.shape


#%% [markdown]
# # Managing our Outliers
# from scipy.stats import zscore

# price = list(appdata.price)
# plt.boxplot(price)
# plt.show()

print(np.where(appdata['price']>50))
new_appdata = appdata[appdata['price'] <5]
print(new_appdata.head(15))

new_appdata.shape
new_appdata.describe()
new_appdata["price"].describe()
new_appdata.info()

newprice = list(new_appdata.price)
plt.boxplot(newprice)
plt.show()

plt.hist(new_appdata['price'])


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
appdata['price'].plot(kind='hist', title='price')
plt.subplot(3,2,6)
appdata['price'].plot(kind= 'box')

f,axs = plt.subplots(4,2,figsize=(15,15))
plt.subplot(4,2,1)
appdata['rating_count_tot'].plot(kind='hist', title='rating_count_tot')
plt.subplot(4,2,2)
appdata['rating_count_tot'].plot(kind= 'box')
#plt.subplot(4,2,3)
# appdata['cont_rating'].plot(kind='hist', title='cont_rating')
# plt.subplot(4,2,4)
# appdata['cont_rating'].plot(kind= 'box')
# get rid of '+' for con_rating 
plt.subplot(4,2,5)
appdata['sup_devices.num'].plot(kind='hist', title='sup_devices.num')
plt.subplot(4,2,6)
appdata['sup_devices.num'].plot(kind= 'box')
plt.subplot(4,2,7)
appdata['lang.num'].plot(kind='hist', title='lang.num')
plt.subplot(4,2,8)
appdata['lang.num'].plot(kind= 'box')


sns.boxplot(appdata['user_rating'])
# Position of the Outlier
print(np.where(appdata['user_rating']<2))
sns.boxplot(appdata['price'])

#rename some variables in order to make it simple and more sense
new_appdata.rename(columns={"size_bytes":"Size","price":"Price","rating_count_tot":"Rating_Count", "user_rating":"Rating", "cont_rating": "Content_Rating", "prime_genre":"App_type", "sup_devices.num":"Devices_Count", "lang.num":"language_Count" } ,inplace=True)
new_appdata.head()
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
from statsmodels.formula.api import ols
modelratingpre = ols(formula='user_rating ~ rating_count_tot + prime_genre + cont_rating', data=dfas)
modelratingpreFit = modelratingpre.fit()
print( type(modelratingpreFit) )
print( modelratingpreFit.summary() )












'''
#********************************SOMEONE'S OLD CODE*******************************

#%% [markdown]

# Top Price in important Category (Business , Navigation , Education , Productivity )
#  in another side price for all of apps less than 50 USD
# Education Apps has a higher price 
# Shopping Apps has a lower price
#%%
#
plt.figure(figsize=(10,5))
plt.scatter(y=paid_apps.prime_genre ,x=paid_apps.price,c='DarkBlue')
plt.title('Price & Category')
plt.xlabel('Price')
plt.ylabel('Category')
plt.show()


#%% [markdown]

#from this graph The number of apps that have a price greater than 50 is few compared to before 50 USD
# %%


Top_Apps=paid_apps[paid_apps.price>50][['track_name','price','prime_genre','user_rating']]
Top_Apps
#7 Top apps with price, prime_genre and user rating


# %% [markdown]

# Top 7 apps on the basis of price


def visualizer(x, y, plot_type, title, xlabel, ylabel, rotation=False, rotation_value=60, figsize=(15,8)):
    plt.figure(figsize=figsize)
    
    if plot_type == "bar":  
        sns.barplot(x=x, y=y)
    elif plot_type == "count":  
        sns.countplot(x)
   
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.yticks(fontsize=13)
    if rotation == True:
        plt.xticks(fontsize=13,rotation=rotation_value)
    plt.show()

# %%

Top_Apps = Top_Apps.sort_values('price', ascending=False)

visualizer(Top_Apps.price,Top_Apps.track_name, "bar", "TOP 7 APPS ON THE BASIS OF PRICE","Price (in USD)","APP NAME")
#names of track in y axis to be readable    


# %%

paid_apps.head(2)

# %%

#sum of all paid apps 
sum_paid = paid_apps.price.value_counts().sum()
sum_paid

# %%

#sum of all free apps
sum_free = free_apps.price.value_counts().sum()
sum_free


#%% [markdown]

# How does the price distribution get affected by category ?

data.prime_genre.value_counts()

#%% [markdown]

# Top app category is Games Games  is 3862 and Entertainment  is 535

data.head()

#%%

new_data_cate = data.groupby([data.prime_genre])[['id']].count().reset_index().sort_values('id' ,ascending = False)
new_data_cate.columns = ['prime_genre','# of Apps']
new_data_cate.head()
#Categories and number of apps in each category


#%%

#Top_Categories accorrding number of apps
new_data_cate.head(10)


sns.barplot(y = 'prime_genre',x = '# of Apps', data=new_data_cate.head(10))


#Lower Categories according number of apps Categories unpopular
new_data_cate.tail(10)

#%%

sns.barplot(x= '# of Apps' , y = 'prime_genre' , data = new_data_cate.tail(10))
'''
