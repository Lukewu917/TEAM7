
# %%
import stat
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from warnings import filterwarnings

from sqlalchemy import false
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


#%%
# Data Wrangling
# Look at data and basic descriptive stats
appdata.shape
appdata.describe()
appdata["price"].describe()
appdata.isnull().sum()
appdata.info()
appdata=appdata.drop(columns=['Unnamed: 0'], axis=1)

appdata.shape
appdata.isnull()

appdata.dropna()
appdata.shape

appdata['cont_rating'] = [x.strip('+') for x in appdata['cont_rating']]
appdata.cont_rating = pd.to_numeric( appdata.cont_rating)
print(appdata.cont_rating.dtypes)
appdata.rename(columns = {'sup_devices.num':'sup_devices_num', 'lang.num':'lang_num', 'ipadSc_urls.num': 'ipadSc_urls_num'}, inplace = True)



#%% [markdown]
# # Exploratory Data Analysis
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

stat, p = shapiro(new_appdata['sup_devices_num'])
print('Number of Supporting Devices, Statistics=%.3f, p=%.3f' %(stat,p))

stat, p = shapiro(new_appdata['lang_num'])
print('Number of Supported languages , Statistics=%.3f, p=%.3f' %(stat,p))

stat, p = shapiro(new_appdata['ipadSc_urls_num'])
print('Number of Screenshots showed for Display , Statistics=%.3f, p=%.3f' %(stat,p))


# * As shown above, the p-values are less than 0.05, so we reject the null hypothesis which states that the variables are normally distributed. 
# * The variables are NOT normally distributed.

#%%
#create new var based on price
price_categories = new_appdata['price'].map(lambda price: 'free' if price <= 0 else 'paid')
new_appdata['price_cat'] = price_categories

#create new var based on rating
rating_categories = new_appdata['user_rating'].map(lambda ratings: 'high' if ratings >=4.5 else 'low')
new_appdata['ratings_cat'] = rating_categories

#look into new variables
new_appdata.groupby('price_cat').describe()
new_appdata.groupby('ratings_cat').describe()

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
# what is the y-value?
plt.xlabel('Price')
plt.xlim(0, 16)
#Luke changed 10 to 16 because the highest price in ourdata set is 15.99
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

# price cat by genre category
sns.barplot(x='prime_genre',y='user_rating',hue='price_cat',data=new_appdata)
plt.xticks(rotation = 90)



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
# Gragh 8 (rating count on current version vs user rating)
sns.relplot(x="rating_count_ver", y="user_rating", data=new_appdata)
new_appdata.loc[new_appdata["rating_count_ver"]>150000]
new_appdata.loc[new_appdata["rating_count_ver"]>50000]
# we can see if the rating count on current verion is over 50000, the app's user rating won't below 4 except one outlier in this gragh. One interesting thing we found in graph 9 is those outlier with user rating above 4 and rating_count_ver over 50000. We want to see what kind of features does those app have to make itself outstanding. The app's name is Infinity Blade. size_byts is pretty big, 634107810; price is much lower than apps average price, only 0.99, cont_rating is 12+ and it is a game app that supports 43 devices and 13 langues. 

#%%
# Graph 9  (size bytes vs user rating)
plt.scatter(x="size_bytes", y="user_rating", data=new_appdata)
plt.xlabel('size')
plt.ylabel('user rating')
plt.show()
# It seems that there is no apparent correlation between app size and user rating.

#%%
# #Graph 10 (user rating on current verison vs user rating)
sns.relplot(x="user_rating_ver", y="user_rating", data=new_appdata)
plt.scatter(x="user_rating_ver", y="user_rating", data=new_appdata)
plt.xlabel('user rating ver')
plt.ylabel('user rating')
plt.show()
# It seems that there is no apparent correlation between user_rating_ver and user rating.


#%%
# chi-square test 
# pip install scipy
from scipy.stats import chi2_contingency
import scipy.stats as stats

pd.crosstab(new_appdata['cont_rating'],new_appdata['user_rating'])
data = [new_appdata['cont_rating'], new_appdata['user_rating']]
stat, p, dof, expected = chi2_contingency(data)
stat, p, dof, expected 
# p value is less than 0.05. Therefore, we reject H0, that is, the variables have a significant relation.


data = [new_appdata['size_bytes'], new_appdata['user_rating']]
stat, p, dof, expected = chi2_contingency(data)
stat, p, dof, expected 
# p value is less than 0.05. Therefore, we reject H0, that is, the variables have a significant relation.


# data = [new_appdata['user_rating_ver'], new_appdata['user_rating']]
# stat, p, dof, expected = chi2_contingency(data)
# stat, p, dof, expected 

# data = [new_appdata['rating_count_ver'],new_appdata['user_rating']]
# stats.chi2_contingency(data)

# data = [new_appdata['price'], new_appdata['user_rating']]
# stat, p, dof, expected = chi2_contingency(data)
# stat, p, dof, expected 


#%% [markdown]
# #Graph 8
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

#%%
new_appdata.info()

#var for models
new_appdata['price_cat'].replace(['free','paid'],[0,1],inplace=True)
new_appdata['ratings_cat'].replace(['low','high'],[0,1],inplace=True)


#%%
# Preditcting rating using size_bytes, price and rating_count_tot via multilinear regression 
import statsmodels.formula.api as smf
x=new_appdata[['size_bytes', 'price', 'rating_count_tot']]
y=new_appdata[['user_rating']]
model1 = smf.ols(formula= 'user_rating ~ size_bytes + price + rating_count_tot ', data=new_appdata)
results_formula = model1.fit()
print(model1.fit().summary())
# the multi regression equation is : user_rating = 3.392 + 1.573e-10 size_bytes + 0.054 price + 1.8e-06 rating_count_tot. However this model doesn't fit the data very well because the value of r-squared is 0.017 which is very low. R-squared is the proportion of variance explained, so we want R-squared as close to one as possible. 

#%%
# Adding one more predictor to the model 
x=new_appdata[['size_bytes', 'price', 'rating_count_tot','cont_rating']]
y=new_appdata[['user_rating']]
model2 = smf.ols(formula= 'user_rating ~ size_bytes + price + rating_count_tot + C(cont_rating) ', data=new_appdata)
results_formula = model2.fit()
print(model2.fit().summary())
# for cont_rating is 17+, the multi regression equation is : user_rating = 3.462 - 0.79 + 1.474e-10 size_bytes + 0.0462price + 1.745e-06 rating_count_tot.
# for cont_rating is 12+, the multi regression equation is : user_rating = 3.462 - 0.038 + 1.474e-10 size_bytes + 0.0462price + 1.745e-06 rating_count_tot.
# for cont_rating is 9+, the multi regression equation is : user_rating = 3.462 - 0.14 + 1.474e-10 size_bytes + 0.0462price + 1.745e-06 rating_count_tot.
#  R-squared of this model is 0.042 which is better than the first model. However R-squared is still low. We will try add one more variable to see how it will change.

#%%
# changing other variables as predictors
from sklearn.model_selection import train_test_split
x=new_appdata[['size_bytes', 'user_rating_ver','cont_rating', 'sup_devices_num', 'lang_num','ipadSc_urls_num']]
y=new_appdata[['user_rating']]


model3 = smf.ols(formula= 'user_rating ~ user_rating_ver + size_bytes + ipadSc_urls_num + sup_devices_num + lang_num ', data=new_appdata)
results_formula = model3.fit()
print(model3.fit().summary())
# Predicitng model has been improved because R-squred of this model is much higher than previous model's. We got R squared of 0.607 for pur model. We conclude that using  size_bytes, price, rating_count_tot, cont_rating, sup_devices_num, and lang_num as predictors is the best model I can approach.




from sklearn.model_selection import train_test_split
from sklearn import linear_model
X_train1, X_test1, y_train1, y_test1 = train_test_split(x, y, test_size = 0.250, random_state=333)
full_split1 = linear_model.LinearRegression() 
full_split1.fit(X_train1, y_train1)
y_pred1 = full_split1.predict(X_test1)
full_split1.score(X_test1, y_test1)

print('score (train):', full_split1.score(X_train1, y_train1)) # 0.6023509986393917
print('score (test):', full_split1.score(X_test1, y_test1)) # 0.6204260945573832
print('intercept:', full_split1.intercept_) # 1.63872717
print('coef_:', full_split1.coef_)  # [-5.67008024e-11  6.32893695e-01  1.57839702e-03 -9.68894846e-03 7.04789432e-03  4.11500645e-02]

test_sizes = [0.2, 0.1, 0.05]



X_train2, X_test2, y_train2, y_test2 = train_test_split(x, y, test_size = 0.2, random_state=333)
full_split2 = linear_model.LinearRegression()
full_split2.fit(X_train2, y_train2)
y_pred2 = full_split2.predict(X_test2)
full_split2.score(X_test2, y_test2)

print('score (train):', full_split2.score(X_train2, y_train2)) # 0.6011932976343275
print('score (test):', full_split2.score(X_test2, y_test2)) # 0.6293198958593318
print('intercept:', full_split2.intercept_) # 1.60945962
print('coef_:', full_split2.coef_)  # [-5.35620736e-11  6.32288165e-01  3.24451279e-03 -9.18205627e-03 7.24639110e-03  4.08507605e-02]


X_train3, X_test3, y_train3, y_test3 = train_test_split(x, y, test_size = 0.1, random_state=333)
full_split3 = linear_model.LinearRegression() 
full_split3.fit(X_train3, y_train3)
y_pred3 = full_split3.predict(X_test3)
full_split2.score(X_test3, y_test3)

print('score (train):', full_split3.score(X_train3, y_train3)) # 0.6060129342797743
print('score (test):', full_split3.score(X_test3, y_test3)) # 0.6145086921403948
print('intercept:', full_split3.intercept_) # 1.56020116
print('coef_:', full_split3.coef_)  # [-5.38258262e-11  6.35560705e-01  3.73249018e-03 -8.39264471e-03 6.72794028e-03  4.24404807e-02]


X_train4, X_test4, y_train4, y_test4 = train_test_split(x, y, test_size = 0.05, random_state=333)
full_split4 = linear_model.LinearRegression()
full_split4.fit(X_train4, y_train4)
y_pred4 = full_split4.predict(X_test4)
full_split2.score(X_test4, y_test4)

print('score (train):', full_split4.score(X_train4, y_train4)) # 0.6076565014159364
print('score (test):', full_split4.score(X_test4, y_test4)) # 0.5927036744032501
print('intercept:', full_split4.intercept_) # 1.56653661
print('coef_:', full_split4.coef_)  # [-5.69613690e-11  6.36089613e-01  3.18376976e-03 -8.54181868e-03 6.70983520e-03  4.29414016e-02]


#%%
# x = new_appdata['user_rating']
# new_appdata['user_rating'] = (x-x.min())/ (x.max() - x.min())

# model4 = smf.ols(formula= 'user_rating ~ user_rating_ver + size_bytes + ipadSc_urls_num + sup_devices_num + lang_num ', data=new_appdata)
# results_formula = model4.fit()
# print(model4.fit().summary())



# #%%
# x1 = new_appdata['user_rating_ver']
# new_appdata['user_rating_ver'] = (x1-x1.min())/ (x1.max() - x1.min())
# x2 = new_appdata['size_bytes']
# new_appdata['size_bytes'] = (x-x.min())/ (x.max() - x.min())
# x3 = new_appdata['ipadSc_urls_num']
# new_appdata['ipadSc_urls_num'] = (x-x.min())/ (x.max() - x.min())
# x4 = new_appdata['sup_devices_num']
# new_appdata['sup_devices_num'] = (x-x.min())/ (x.max() - x.min())
# x5 = new_appdata['lang_num']
# new_appdata['lang_num'] = (x-x.min())/ (x.max() - x.min())


# model4 = smf.ols(formula= 'user_rating ~ user_rating_ver + size_bytes + ipadSc_urls_num + sup_devices_num + lang_num ', data=new_appdata)
# results_formula = model4.fit()
# print(model4.fit().summary())







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
