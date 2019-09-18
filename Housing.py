# -*- coding: utf-8 -*-
"""
Created on Wed May  8 00:21:37 2019

@author: Gowtham G
"""

import numpy as np
import pandas as p
import matplotlib as plt
import seaborn as sps
import matplotlib.pyplot as pp

dataSet=p.read_csv("housing.csv")

dataSet.head()
dataSet.describe()
dataSet.info()

dataSet['ocean_proximity'].value_counts()

dataSet.hist(bins=50,figsize=(20,20))
plt.show()

from sklearn.model_selection import train_test_split
#train_set=p.DataFrame(train_set)
#test_set=p.DataFrame(test_set)
train_set,test_set =train_test_split(dataSet,test_size=0.2,random_state=42)


fig,axs=pp.subplots(1,2)
axs[0].hist(dataSet['median_income'])


dataSet['income_cat']=np.ceil(dataSet['median_income']/1.5)
dataSet['income_cat'].where(dataSet['income_cat']<5,5.0,inplace=True)


fig,axs=pp.subplots(1,2)
axs[0].hist(dataSet['income_cat'])

from sklearn.model_selection import StratifiedShuffleSplit

split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

for train_index,test_index in split.split(dataSet,dataSet['income_cat']):
    strat_train_set=dataSet.iloc[train_index]
    strat_test_set=dataSet.iloc[test_index]
    

dataSet['income_cat'].value_counts()/len(dataSet)


#visualization

dataSet.plot(kind="scatter",x='longitude',y='latitude')
dataSet.plot(kind="scatter",x='longitude',y='latitude',alpha=0.1)

dataSet.plot(kind="scatter",x='longitude',y='latitude',alpha=0.1,s=dataSet['population']/100,
             label='population',figsize=(10,7),c='median_house_value',cmap=plt.cm.get_cmap("jet"),colorbar=True)
plt.legend()

corr_mat=dataSet.corr()
corr_mat['median_house_value'].sort_values(ascending=False)


dataSet.plot(kind='scatter',x='median_income',y='median_house_value',alpha=0.1)

dataSet['rooms_per_household']=dataSet['total_rooms']/dataSet['households']
dataSet['bedrooms_per_household']=dataSet['total_bedrooms']/dataSet['total_rooms']
dataSet['population_per_household']=dataSet['population']/dataSet['households']


corr_mat=dataSet.corr()
corr_mat['median_house_value'].sort_values(ascending=False)

housing=strat_train_set.drop('median_house_value',axis=1)
housing_labels=strat_train_set["median_house_value"].copy()

#filling null in total bedrooms

median=housing['total_bedrooms'].median()
housing['total_bedrooms'].fillna(median, inplace=True)

#filling missing values using imputer

from sklearn.preprocessing import Imputer

imputer=Imputer(strategy='median')
housing_num=housing.drop('ocean_proximity',axis=1)
imputer.fit(housing_num)

imputer.statistics_
housing_num.median().values
X=imputer.transform(housing_num)
housing_tr=p.DataFrame(X,columns=housing_num.columns)

#handling categorical and text categories

from sklearn.preprocessing import LabelEncoder

labelencoder=LabelEncoder()
housing_cat=housing['ocean_proximity']
housing_cat_encoded=labelencoder.fit_transform(housing_cat)
print(labelencoder.classes_)

from sklearn.preprocessing import OneHotEncoder

onehot=OneHotEncoder()
housing_cat_1hot=onehot.fit_transform(housing_cat_encoded.reshape(-1,1))
housing_cat_1hot.toarray()

from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
housing_cat_1hot = encoder.fit_transform(housing_cat)


#custom transformers 

from sklearn.base import BaseEstimator,TransformerMixin
rooms_ix,bedrooms_ix,population_ix,household_ix=3,4,5,6
class CombinedAttributesAdder(BaseEstimator,TransformerMixin):
    def __init__(self,add_bedrooms_per_room=True):
        self.add_bedrooms_per_room=add_bedrooms_per_room
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        rooms_per_household=X[:,rooms_ix]/X[:,household_ix]
        population_per_household=X[:,population_ix]/X[:,household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room=X[:,bedrooms_ix]/X[:,household_ix]
            return np.c_[X,rooms_per_household,population_per_household,bedrooms_per_room]
        else:
            return np.c_[X,rooms_per_household,population_per_household]
        
attr_adder=CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs=attr_adder.transform(housing.values)    

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline=Pipeline([
        ('imputer',Imputer(strategy="median")),
        ('attribs_adder',CombinedAttributesAdder()),
        ('std_scaler',StandardScaler()),])

housing_num_tr=num_pipeline.fit_transform(housing_num)

#from sklearn.preprocessing import StandardScaler
#stdscalar=StandardScaler()

from sklearn.base import TransformerMixin #gives fit_transform method for free
class MyLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)
    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self
    def transform(self, x, y=0):
        return self.encoder.transform(x)


from sklearn.base import BaseEstimator , TransformerMixin 
class DataFrameSelector(BaseEstimator,TransformerMixin):
    def __init__(self,attribute_names):
        self.attribute_names=attribute_names
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        return X[self.attribute_names].values
    
from sklearn.preprocessing import LabelBinarizer    
num_attribs=list(housing_num)
cat_attribs=["ocean_proximity"]
num_pipeline=Pipeline([('selector',DataFrameSelector(num_attribs)),
                       ('imputer',Imputer(strategy="median")),
                       ('attribs_adder',CombinedAttributesAdder()),
                       ('std_scaler',StandardScaler()),])
cat_pipeline=Pipeline([('selector',DataFrameSelector(cat_attribs)),
                       ('label_binarizer',MyLabelBinarizer()),])
    
'''housing_cat = p.DataFrame(housing["ocean_proximity"])
cat_pipeline_new = cat_pipeline.fit_transform(housing_cat)  '''  
    
from sklearn.pipeline import FeatureUnion
full_pipeline=FeatureUnion(transformer_list=[("num_pipeline",num_pipeline),
                                             ("cat_pipeline",cat_pipeline)])  

housing_prepared=full_pipeline.fit_transform(housing)
housing_prepared

from sklearn.linear_model import LinearRegression

lin_reg=LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)

some_data=housing.iloc[:5]
some_labels=housing_labels.iloc[:5]

some_data_prepared=full_pipeline.transform(some_data)
print("predicted:",lin_reg.predict(some_data_prepared))
print("labels:",list(some_labels))


from sklearn.metrics import mean_squared_error

housing_prediction=lin_reg.predict(housing_prepared)
lin_mse=mean_squared_error(housing_prediction,housing_labels)
lin_rmse=np.sqrt(lin_mse)
print(lin_rmse)


from sklearn.tree import DecisionTreeRegressor

dec_reg=DecisionTreeRegressor()
dec_reg.fit(housing_prepared,housing_labels)

housing_predictiondec=dec_reg.predict(housing_prepared)
dec_mse=mean_squared_error(housing_predictiondec,housing_labels)
dec_rmse=np.sqrt(dec_mse)
print(dec_rmse)

from sklearn.model_selection import cross_val_score

scores=cross_val_score(dec_reg,housing_prepared,housing_labels,scoring='neg_mean_squared_error',cv=10)
rmse_scores=np.sqrt(-scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(rmse_scores)


lin_scores=cross_val_score(lin_reg,housing_prepared,housing_labels,scoring='neg_mean_squared_error',cv=10)
lin_rmse_score=np.sqrt(-lin_scores)

display_scores(lin_rmse_score)



from sklearn.ensemble import RandomForestRegressor


ran_reg=RandomForestRegressor()
ran_reg.fit(housing_prepared,housing_labels)

housing_predictionRan=ran_reg.predict(housing_prepared)
ran_mse=mean_squared_error(housing_predictionRan,housing_labels)

ran_rmse=np.sqrt(ran_mse)
print(ran_rmse)


ran_scores=cross_val_score(ran_reg,housing_prepared,housing_labels,scoring='neg_mean_squared_error',cv=10)
ran_rmse_score=np.sqrt(-ran_scores)

display_scores(ran_rmse_score)


