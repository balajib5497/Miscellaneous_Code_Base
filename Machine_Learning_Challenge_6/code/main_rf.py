import numpy as np
np.random.seed(34)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
#from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report, confusion_matrix


import util_rf
basedir = 'D:/competitions/Hackerearth/Machine_Learning_Challenge_6/Dataset/'
write_dir = 'D:/competitions/Hackerearth/Machine_Learning_Challenge_6/Result/'

def load_data():
    train = pd.io.parsers.read_csv(basedir + 'train.csv', iterator = True, chunksize = 300)
    train = pd.concat(train, ignore_index=True)
    
    test = pd.io.parsers.read_csv(basedir + 'test.csv', iterator = True, chunksize = 300)
    test = pd.concat(test, ignore_index=True)
    
    owner = pd.io.parsers.read_csv(basedir + 'Building_Ownership_Use.csv', iterator = True, chunksize = 300)
    owner = pd.concat(owner, ignore_index=True)
    
    struct = pd.io.parsers.read_csv(basedir + 'Building_Structure.csv', iterator = True, chunksize = 300)
    struct = pd.concat(struct, ignore_index=True)
    
    return train, test, owner, struct
    

train, test, owner, struct = load_data()

x, y = util_rf.get_train_data(train, owner, struct) 
testx, building_id_test = util_rf.get_test_data(x, test, owner, struct)

#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

##############################################
#x_train.reset_index(drop=True, inplace=True)
#y_train.reset_index(drop=True, inplace=True)
#x_test.reset_index(drop=True, inplace=True)
#y_test.reset_index(drop=True, inplace=True)

#y_train_1 = y_train.copy()
#y_train_1[(y_train_1 != 1) & (y_train_1 != 5)] = 0

#x_train_2 = x_train.iloc[y_train[(y_train != 1) & (y_train != 5)].index, :]
#y_train_2 = y_train[(y_train != 1) & (y_train != 5)]

##############################################
model1 = RandomForestClassifier(n_estimators=100, 
                               max_depth=25,
                               max_features = 0.8)
print('fitting model:', model1)
model1.fit(x, y)
#model1.fit(x_train, y_train)
##
#pred1 = model1.predict(x_test)
#print(classification_report(pred1, y_test, digits=5))
#
#
#model2 = XGBClassifier(n_estimators=100, 
#                       objective='multi:softmax',
#                       max_depth=25)
#
#model2.fit(x_train, y_train)
#pred2 = model2.predict(x_test)
#print(classification_report(pred2, y_test, digits=5))


#model3 = GradientBoostingClassifier(n_estimators=100, 
#                       max_depth=25)


#model3.fit(x_train, y_train)
#pred3 = model3.predict(x_test)
#print(classification_report(pred3, y_test, digits=5))


#model.fit(x,y)
predy = model1.predict(testx)
test_pred = pd.DataFrame()
#test_pred['damage_grade'] = predy.apply(lambda e:'Grade ' + str(e))
test_pred['damage_grade'] = ['Grade ' + str(e) for e in predy]
test_result = pd.concat([building_id_test, test_pred], axis=1)
test_result.to_csv(write_dir + 'test_results_rf.csv',index=False)


