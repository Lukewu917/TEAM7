
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
os.chdir('/Users/b/Desktop/DataScience/TEAM7')
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

appdata.shape
appdata.describe()
appdata["price"].describe()

appdata.info()
appdata=appdata.drop(columns=['Unnamed: 0'], axis=1)

appdata.shape
appdata.isnull()
appdata.dropna()
appdata.shape


#%% [markdown]
# # Managing our Outliers
from scipy.stats import zscore


# * Price Outliers
price = list(appdata.price)
plt.style.use('dark_background')
sns.boxplot(price)
plt.show()
sns.distplot(appdata['price'], bins=20)
plt.savefig('pricebeforeoutliers.png')

# Mean and Standard Deviation
mean = appdata.price.mean()
std= appdata.price.std() 

# identify outliers
cutoff = 2.5*std
lowerbound= mean-cutoff
upperbound=mean+cutoff
appdata.shape
new_appdata = appdata[appdata['price'] >=lowerbound]
new_appdata = appdata[appdata['price'] <=upperbound]

new_appdata.shape
new_appdata.describe()
new_appdata["price"].describe()
new_appdata.info()

newprice = list(new_appdata.price)
plt.boxplot(newprice)
plt.show()
sns.distplot(new_appdata['price'], bins=20)
plt.savefig('priceafteroutliers.png')


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

#create new var based on rating
rating_categories = new_appdata['user_rating'].map(lambda ratings: 'high' if ratings >=4 else 'low')
new_appdata['ratings_cat'] = rating_categories

#look into new variables
new_appdata.groupby('price_cat').describe()
new_appdata.groupby('ratings_cat').describe()

#%% [markdown]
# #Correlation Matrix
# Since we have data that is not normally distributed, we use Spearman
import seaborn as sns # For pairplots and heatmaps
import matplotlib.pyplot as plt

corrMatrix = new_appdata.corr(method="spearman")
print (corrMatrix)

plt.figure(figsize=(10,6))
heatmap = sns.heatmap(new_appdata.corr(), vmin=-1,vmax=1, annot=True)
plt.title("Spearman Correlation")
plt.savefig('corrplot.png') 

#weak correlation between size and price
price_sizecorr = np.corrcoef(new_appdata.size_bytes, new_appdata.price)
price_sizecorr

#%% [markdown]
# #Graph 1

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


#%% [markdown]
# #Graph 2

ratingscol = 'ratings_cat'
new_appdata.groupby(['price_cat', ratingscol]).size().unstack(level=1).plot(kind='bar')
current_values = plt.gca().get_yticks()
plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values])
plt.title('Number of Apps, by Price Category and Ratings Category')
plt.ylabel('Number of Apps')
plt.xlabel('Price Category')
plt.savefig('g2.png') 



#%% [markdown]
# #Graph 3


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
plt.savefig('g3.png') 


#%% [markdown]
# #Graph 4
new_appdata.info()

pricecategory = new_appdata['prime_genre'].value_counts()
pricecategory.plot(kind='bar')
plt.title("Number of Apps, by Genre")
plt.xlabel("")
plt.ylabel("Number of Apps")
current_values = plt.gca().get_yticks()
plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values])
plt.tight_layout()
plt.savefig('g4.png') 

#%% [markdown]
# #Graph 5

from matplotlib.ticker import StrMethodFormatter

new_appdata.info()


new_appdata_graph = new_appdata.groupby('prime_genre').mean()
plt.bar(new_appdata_graph.index, new_appdata_graph['user_rating'])
plt.xlabel('Genre')
plt.ylabel('Average Ratings')
plt.title('Average Ratings, by Genre')
current_values = plt.gca().get_yticks()
plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values])
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) 
plt.xticks(rotation = 90)
plt.tight_layout()
plt.savefig('g5.png') 





#%% [markdown]
# #Graph 6
# App Price by genre category (what genre cost the most?)

new_appdata_graph = new_appdata.groupby('prime_genre').mean()
plt.bar(new_appdata_graph.index, new_appdata_graph['price'])
plt.xlabel('Genre')
plt.ylabel('Average Price')
plt.title('Average Price, by Genre')
current_values = plt.gca().get_yticks()
plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values])
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) 
plt.xticks(rotation = 90)
plt.tight_layout()
plt.savefig('g6.png') 


#%% [markdown]
# #Graph 7
# avg ratings, free vs paid
new_appdata_graphf = new_appdata.groupby('price_cat').mean()
plt.bar(new_appdata_graphf.index, new_appdata_graphf['user_rating'], color='grey', edgecolor='darkblue')
plt.xlabel('Price Category')
plt.ylabel('Average Ratings')
plt.title('Average Ratings, Free vs. Paid')
current_values = plt.gca().get_yticks()
plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values])
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) 
plt.savefig('g7.png') 

#%%
new_appdata.info()

#var for models
new_appdata['price_cat'].replace(['free','paid'],[0,1],inplace=True)
new_appdata['ratings_cat'].replace(['low','high'],[0,1],inplace=True)



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

# source :  https://www.business2community.com/mobile-apps/app-ratings-and-reviews-2021-benchmarks-02396808


# # Mean user rating of an app is :

print(new_appdata['user_rating'].mean())


print(new_appdata['user_rating'].describe())

i = 0

new_appdata['Good_Bad'] = 1

for element in new_appdata['user_rating']:

    if (element >= 4.5):

        new_appdata['Good_Bad'][i] = 1

        i = i+1

    else :

        new_appdata['Good_Bad'][i] = 0

        i = i+1



print(new_appdata['Good_Bad'])

#%% [markdown]

print(new_appdata['track_name'].describe()) 

# Unique elements are 7109 out of 7111 count, so dropping.


#%% [markdown]

#to catagorical

new_appdata['Good_Bad'] = pd.Categorical(new_appdata.Good_Bad)

new_appdata['prime_genre'] = pd.Categorical(new_appdata.prime_genre)

new_appdata['price_cat'] = pd.Categorical(new_appdata.price_cat)


new_appdata['ratings_cat'] = pd.Categorical(new_appdata.ratings_cat)


#%% [markdown]




#%% [markdown]


# label encoding for categorcial varibales 

# from sklearn.preprocessing import OrdinalEncoder
# OrdinalEncoder = OrdinalEncoder()

# # Assigning numerical values and storing in another column
# new_appdata['price_cat_en'] = OrdinalEncoder.fit_transform(new_appdata['price_cat'])
# # print(new_appdata['price_cat_en'])


# new_appdata['ratings_cat_en'] = labelencoder.fit_transform(new_appdata['ratings_cat'])
# print(new_appdata['ratings_cat_en'])


# new_appdata['prime_genre_en'] = labelencoder.fit_transform(new_appdata['prime_genre'])
# print(new_appdata['prime_genre_en'])

# new_appdata['cont_rating_en'] = labelencoder.fit_transform(new_appdata['cont_rating'])
# print(new_appdata['cont_rating_en'])

# new_appdata['Good_Bad'] = 0

# i = 0

# for e in new_appdata['cont_rating']:

#     if (e == 4+):

#         new_appdata['Good_Bad'] = 0

#         i = i+1

#     if (e == 12+):

#         new_appdata['Good_Bad'] = 0

#         i = i+1

#      if (e == 9+):

#         new_appdata['Good_Bad'] = 0

#         i = i+1

#      if (e == 17+):

#         new_appdata['Good_Bad'] = 0

#         i = i+1


#     else :

#         return              



from sklearn.preprocessing import OrdinalEncoder

ord_enc = OrdinalEncoder()
new_appdata["genre_en"] = ord_enc.fit_transform(new_appdata[["prime_genre"]])
new_appdata[["prime_genre", "genre_en"]].head(11)

new_appdata["cont_rating_en"] = ord_enc.fit_transform(new_appdata[["cont_rating"]])
new_appdata[["cont_rating", "cont_rating_en"]].head(11)


#%% 

new_appdata["price_cat_en"] = ord_enc.fit_transform(new_appdata[["price_cat"]])
new_appdata[["price_cat", "price_cat_en"]].head(11)

new_appdata["ratings_cat_en"] = ord_enc.fit_transform(new_appdata[["ratings_cat"]])
new_appdata[["ratings_cat", "ratings_cat_en"]].head(11)

#%% [markdown]

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

new_appdata['log_rating'].round(decimals = 2)

X = new_appdata.drop('Good_Bad', axis=1)
X = X.drop('id', axis=1)
X = X.drop('track_name', axis=1)
X = X.drop('currency', axis=1)
X = X.drop('price_cat', axis=1)
X = X.drop('ratings_cat', axis=1)
X = X.drop('prime_genre', axis=1)
X = X.drop('cont_rating', axis=1)
X = X.drop('rating_count_tot', axis=1)
X = X.drop('ver', axis=1)
X.replace([np.inf, -np.inf], np.nan, inplace=True)
Y.replace([np.inf, -np.inf], np.nan, inplace=True)

Y = new_appdata['Good_Bad']

X['log_rating'].round(decimals = 2)

X['size_bytes'] = X['size_bytes'].div(1000000).round(2)
X.fillna(-99999, inplace=True)

model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)
print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))



# %%
