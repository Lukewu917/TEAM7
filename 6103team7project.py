#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

#%%
import pandas as pd
dfapplestore = pd.read_csv(r'C:\Users\wpy12\OneDrive\Documents\clone\22SP_6103_11T_iDM\project\projectdataset\AppleStore.csv')
# %%
print(dfapplestore.describe)
# %%
dfas = dfapplestore.drop(columns=['id','currency','ver','ipadSc_urls.num','lang.num','vpp_lic',])
dfas
# %%
# Checking the missing value 
dfas.isna().sum()
# %%
# covert categorical variables into numerical variables 
# print(dfas.prime_genre.describe())
# print(dfas.prime_genre.value_counts())
# dfas['OffenderR_convert'] = dfas.OFFENDER_RACE.map(lambda x: '1' if x.strip()=='White' else '2' if x.strip()=='Black or African American' else '3' if x.strip()=='Multiple' else '4' if x.strip()== 'Asian' else '5' if x.strip()== 'American Indian or Alaska Native' else '6' if x.strip()== 'Native Hawaiian or Other Pacific Islander' else np.nan)
# print( CleanHC.OffenderR_convert.value_counts(dropna=False) )

# CleanHC.OffenderR_convert = pd.to_numeric( CleanHC.OffenderR_convert)
# print(CleanHC.OffenderR_convert.dtypes)
#%%
#rename for columns
dfas.rename(columns = {'sup_devices.num':'devices'}, inplace = True)

#%%
from statsmodels.formula.api import ols
modelratingpre = ols(formula='user_rating ~ rating_count_tot + prime_genre + cont_rating', data=dfas)
modelratingpreFit = modelratingpre.fit()
print( type(modelratingpreFit) )
print( modelratingpreFit.summary() )
# %%

# %%

# %%
CleanHC['BIAS_DESC'].tolist()