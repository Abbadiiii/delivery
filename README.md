# delivery
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor 
train = pd.read_csv('/content/train.csv.zip')
test=pd.read_csv('/content/test.csv.zip')
train['Weatherconditions'] = train['Weatherconditions'].map(lambda x: str(x)[11:])
test['Weatherconditions'] = test['Weatherconditions'].map(lambda x: str(x)[11:])

train['Time_taken(min)'] = train['Time_taken(min)'].map(lambda x: str(x)[6:])
for i in train.columns:
    train[i].loc[train[i] == 'NaN '] = np.nan
    train[i].loc[train[i] == 'NaN'] = np.nan

for j in test.columns:
    test[j].loc[test[j] == 'NaN '] = np.nan
    test[j].loc[test[j] == 'NaN'] = np.nan
    
train.isnull().sum()
test.isnull().sum()

train.dropna(subset=['Time_Orderd'], axis=0, inplace=True)
test.dropna(subset=['Time_Orderd'], axis=0, inplace=True)


train = train.fillna(method='ffill')
test = test.fillna(method='ffill')
train.isnull().any()
test.isnull().any()
features = ['Delivery_person_Age', 'Delivery_person_Ratings', 'multiple_deliveries', 'Time_taken(min)']
features1 =  ['Delivery_person_Age', 'Delivery_person_Ratings', 'multiple_deliveries']
for i in features:
    train[i] = train[i].astype(str).astype(float)
    for j in features1:
        test[j] = test[j].astype(str).astype(float)

train['Ordered_Time'] = train['Order_Date'] + str(' ') + train['Time_Orderd']
train['Picked_Time'] = train['Order_Date'] + str(' ') + train['Time_Order_picked']

test['Ordered_Time'] = test['Order_Date'] + str(' ') + test['Time_Orderd']
test['Picked_Time'] = test['Order_Date'] + str(' ') + test['Time_Order_picked']

# convert into datetime format
train['Ordered_Time'] = pd.to_datetime(train['Ordered_Time'])
train['Picked_Time'] = pd.to_datetime((train['Picked_Time']))

test['Ordered_Time'] = pd.to_datetime(test['Ordered_Time'])
test['Picked_Time'] = pd.to_datetime((test['Picked_Time']))
train['Time_Ordered_picked'] = ((train['Picked_Time'] - train['Ordered_Time'])/pd.Timedelta(1,'min')).fillna(0).astype(int)
test['Time_Ordered_picked'] = ((test['Picked_Time'] - test['Ordered_Time'])/pd.Timedelta(1, 'min')).fillna(0).astype(int)

train['Time_Ordered_picked'].value_counts()
test['Time_Ordered_picked'].value_counts()

