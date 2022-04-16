
# %%

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from warnings import filterwarnings
filterwarnings("ignore")
!pip3 install ppscore
import ppscore as pps
#Import Library RobustScaler
from sklearn.preprocessing import RobustScaler
#Cluster Model
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score

# %%
#load_data
data = pd.read_csv('AppleStore.csv' ,sep =',' , encoding = 'utf8' )
data.head()

# %%
#drop column (Unnamed) as semiler ID column
#data.drop(['Unnamed: 0'], axis=1 ,inplace=True)
#show data after drop

# %%
data.head(2)


data.info()


# %%

#show shape of data 7197 Row and 16 columns
data.shape
# %%

data.isnull().sum().sum()



# %%

data.currency.value_counts()
# %%

data.nunique()

#%% [markdown]

# Exploratory Data Analaysis

#  # Visualize price distribution of paid apps ?

data.price.value_counts()


# %%

free_apps = data[(data.price==0.00)]

paid_apps  = data[(data.price>0)]

free_apps.head(2)

#%%

paid_apps.head(2)

paid_apps.price.value_counts()

#%% [markdown]

# The number of apps decreases with increasing his price

free_apps.price.value_counts()

# %%

sns.distplot(free_apps['price'])

# %%

sns.distplot(paid_apps['price'])

# %%

sns.histplot(paid_apps['price'])

# %%

plt.style.use('fivethirtyeight')
plt.figure(figsize=(6,4))

plt.subplot(2,1,2)
plt.title('Visual price distribution')
sns.stripplot(data=paid_apps,y='price',jitter= True,orient = 'h' ,size=6)
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


#%%

plt.figure(figsize=(10,5))
plt.scatter(y=paid_apps.prime_genre ,x=paid_apps.price,c='DarkBlue')
plt.title('Price & Category')
plt.xlabel('Price')
plt.ylabel('Category')
plt.show()


#%% [markdown]

# Top Price in important Category (Business , Navigation , Education , Productivity )
#  in another side price for all of apps less than 50 USD
# Education Apps has a higher price 
# Shopping Apps has a lower price

