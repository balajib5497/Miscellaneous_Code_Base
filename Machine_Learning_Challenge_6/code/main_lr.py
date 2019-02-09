import numpy as np
np.random.seed(34)
import pandas as pd

#from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

import util_lr


from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

basedir = 'D:/competitions/Hackerearth/Machine_Learning_Challenge_6/Dataset/'
write_dir = 'D:/competitions/Hackerearth/Machine_Learning_Challenge_6/Result/'

scaler = None

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

def get_scaled_data(data, is_training = True):
    global scaler 
    if is_training:
        scaler = StandardScaler()
        return scaler.fit_transform(data)
    return scaler.transform(data)

    
train, test, owner, struct = load_data()
util_lr.drop_columns(owner, struct)

x, y = util_lr.get_train_data(train, owner, struct) 
testx, building_id_test = util_lr.get_test_data(x, test, owner, struct)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

x_train = get_scaled_data(x_train)
x_test = get_scaled_data(x_test, is_training=False)


#print('running logistic regression....')
#model = LogisticRegression()
#model.fit(x_train, y_train)
#
#y_pred = model.predict(x_test)
#print(classification_report(y_test, y_pred, digits=5))


print('running logistic regression....')
model = LogisticRegression()
model.fit(x, y)

predy = model.predict(testx)

test_pred = pd.DataFrame()
#test_pred['damage_grade'] = predy.apply(lambda e:'Grade ' + str(e))
test_pred['damage_grade'] = ['Grade ' + str(e) for e in predy]
test_result = pd.concat([building_id_test, test_pred], axis=1)
test_result.to_csv(write_dir + 'test_results_lr.csv',index=False)


#print('running model...')
#searchCV = LogisticRegressionCV(
#        Cs=[0.001, 0.01, 1, 10],
#        penalty='l2',
#        cv=5,
#        random_state=256,
#        fit_intercept=True,
#        max_iter=10000
#    )
#
#searchCV.fit(x_train, y_train)
#y_pred = searchCV.predict(x_test)
#print(classification_report(y_test, y_pred, digits=5))





