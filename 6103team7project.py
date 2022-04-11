#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%%
HCdata = pd.read_csv(r'C:\Users\wpy12\OneDrive\Documents\clone\22SP_6103_11T_iDM\project\dataset.csv')
# %%
print(HCdata.describe)
# %%
HCdatadropped = HCdata.drop(columns=['INCIDENT_ID','ORI','STATE_NAME','PUB_AGENCY_NAME','PUB_AGENCY_UNIT','AGENCY_TYPE_NAME',
'DIVISION_NAME','REGION_NAME','POPULATION_GROUP_CODE','POPULATION_GROUP_DESC','ADULT_VICTIM_COUNT',
'JUVENILE_VICTIM_COUNT','TOTAL_OFFENDER_COUNT','ADULT_OFFENDER_COUNT','JUVENILE_OFFENDER_COUNT',
'OFFENDER_ETHNICITY','VICTIM_COUNT','TOTAL_INDIVIDUAL_VICTIMS','VICTIM_TYPES','MULTIPLE_OFFENSE',
'MULTIPLE_BIAS'])
HCdatadropped
# %%
# Checking the missing value 
HCdatadropped.isna().sum()
# remove the rows that have missing value
CleanHC = HCdatadropped.dropna()
# re-check the missing value
CleanHC.isna().sum()
# Now we have 0 null rows as we have dropped them.
# %%
# covert categorical variables into numerical variables 
print(CleanHC.OFFENDER_RACE.describe())
print(CleanHC.OFFENDER_RACE.value_counts())
CleanHC['OffenderR_convert'] = CleanHC.OFFENDER_RACE.map(lambda x: '1' if x.strip()=='White' else '2' if x.strip()=='Black or African American' else '3' if x.strip()=='Multiple' else '4' if x.strip()== 'Asian' else '5' if x.strip()== 'American Indian or Alaska Native' else '6' if x.strip()== 'Native Hawaiian or Other Pacific Islander' else np.nan)
print( CleanHC.OffenderR_convert.value_counts(dropna=False) )

CleanHC.OffenderR_convert = pd.to_numeric( CleanHC.OffenderR_convert)
print(CleanHC.OffenderR_convert.dtypes)


print(CleanHC.BIAS_DESC.describe())
print(CleanHC.BIAS_DESC.value_counts())
pd.crosstab(index=CleanHC['DATA_YEAR'],columns=CleanHC['BIAS_DESC'])
CleanHC['Possible_victimT'] = CleanHC.BIAS_DESC.map(lambda x: '1' if x.strip()=='Anti-Black or African American' else '2' if x.strip()=='Anti-Jewish' else '3' if x.strip()=='Anti-White' else '4' if x.strip()=='Anti-Gay (Male)' else '5' if x.strip()== 'Anti-Hispanic or Latino' else '6' if x.strip()== 'Anti-Asian' else '7' if x.strip()== 'Anti-Lesbian (Female)' else '8' if x.strip()=='Anti-Lesbian, Gay, Bisexual, or Transgender (Mixed Group)' else '9' if  x.strip()== 'Anti-Arab' else '10' if x.strip()=='Anti-Islamic (Muslim)' else '11' if x.strip()=='Anti-Protestant' else '12' if x.strip()=='Anti-Catholic' else '13' if  x.strip()=='Anti-Other Race/Ethnicity/Ancestry' else '14' if  x.strip()=='Anti-Multiple Races, Group' else '15' if x.strip()=='Anti-Other Religion' else '16')
print( CleanHC.Possible_victimT.value_counts(dropna=False) )

CleanHC.Possible_victimT = pd.to_numeric( CleanHC.Possible_victimT)
print(CleanHC.Possible_victimT.dtypes)
# %%
# extract year and month from incident date
CleanHC['Year'] = pd.DatetimeIndex(CleanHC['INCIDENT_DATE']).year
CleanHC['Month'] = pd.DatetimeIndex(CleanHC['INCIDENT_DATE']).month
# %%
sns.countplot(CleanHC.OffenderR_convert)  
plt.show()  
# %%
CleanHC['BIAS_DESC'].tolist()