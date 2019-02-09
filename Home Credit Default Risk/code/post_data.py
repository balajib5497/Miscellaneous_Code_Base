import numpy as np
np.random.seed(34)
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from xgboost import XGBClassifier

import time

scaler, pca = None,  None
path = '../Dataset/'

def load_data():
    tp = pd.io.parsers.read_csv(path + 'application_train.csv', iterator=True, chunksize=1000)
    appln_train = pd.concat(tp, ignore_index=True)

    tp = pd.io.parsers.read_csv(path + 'application_test.csv', iterator=True, chunksize=1000)
    appln_test = pd.concat(tp, ignore_index=True)
    
    bureau_p = pd.read_csv('./bureau_p.csv')
    prev_appln_p = pd.read_csv('./prev_appln_p.csv')
    
    return appln_train, appln_test, bureau_p, prev_appln_p
    

def clean_data(data, is_trainig=True):
    
    data.drop(columns = [ 
                        'FONDKAPREMONT_MODE', # .68
                        'HOUSETYPE_MODE', # .50
                        'WALLSMATERIAL_MODE', # .50
#                        'EMERGENCYSTATE_MODE', # .47
                    ], axis=1, inplace=True)
    
    data = pd.get_dummies(data,
                          columns = ['NAME_TYPE_SUITE', 'OCCUPATION_TYPE', 'EMERGENCYSTATE_MODE'],
                          drop_first=False)
    
    data = pd.get_dummies(data, 
                          columns = data.select_dtypes(include=['object']).columns, 
                          drop_first=True)
    
    data.drop(columns=[
                    'OWN_CAR_AGE',
                    
                    'EXT_SOURCE_1',
                    'EXT_SOURCE_2',
                    'EXT_SOURCE_3',
                    'APARTMENTS_AVG',
                    'BASEMENTAREA_AVG',
                    'YEARS_BEGINEXPLUATATION_AVG',
                    'YEARS_BUILD_AVG',
                    'COMMONAREA_AVG',
                    'ELEVATORS_AVG',
                    'ENTRANCES_AVG',
                    'FLOORSMAX_AVG',
                    'FLOORSMIN_AVG',
                    'LANDAREA_AVG',
                    'LIVINGAPARTMENTS_AVG',
                    'LIVINGAREA_AVG',
                    'NONLIVINGAPARTMENTS_AVG',
                    'NONLIVINGAREA_AVG',
                    'APARTMENTS_MODE',
                    'BASEMENTAREA_MODE',
                    'YEARS_BEGINEXPLUATATION_MODE',
                    'YEARS_BUILD_MODE',
                    'COMMONAREA_MODE',
                    'ELEVATORS_MODE',
                    'ENTRANCES_MODE',
                    'FLOORSMAX_MODE',
                    'FLOORSMIN_MODE',
                    'LANDAREA_MODE',
                    'LIVINGAPARTMENTS_MODE',
                    'LIVINGAREA_MODE',
                    'NONLIVINGAPARTMENTS_MODE',
                    'NONLIVINGAREA_MODE',
                    'APARTMENTS_MEDI',
                    'BASEMENTAREA_MEDI',
                    'YEARS_BEGINEXPLUATATION_MEDI',
                    'YEARS_BUILD_MEDI',
                    'COMMONAREA_MEDI',
                    'ELEVATORS_MEDI',
                    'ENTRANCES_MEDI',
                    'FLOORSMAX_MEDI',
                    'FLOORSMIN_MEDI',
                    'LANDAREA_MEDI',
                    'LIVINGAPARTMENTS_MEDI',
                    'LIVINGAREA_MEDI',
                    'NONLIVINGAPARTMENTS_MEDI',
                    'NONLIVINGAREA_MEDI',
                    'TOTALAREA_MODE',
                    
                    'OBS_60_CNT_SOCIAL_CIRCLE',
                    
                    'STATUS_C_bure_bal_bure',
                    'STATUS_X_bure_bal_bure',
                    'STATUS_DPD_bure_bal_bure',
                    
                    'AMT_GOODS_PRICE', # 0.98
                    
                    'AMT_CREDIT_prev', # 0.97
                    
                    'AMT_GOODS_PRICE_prev', # 0.92
                    
                    'CNT_INSTALMENT_FUTURE_pos', # 0.95
                    
                    'NAME_CONTRACT_TYPE_prev_Cash_loans', # 0.92
                    'NAME_CONTRACT_TYPE_prev_Consumer_loans', # 0.97
                    'NAME_CONTRACT_TYPE_prev_Revolving_loans',  # 0.90
                    
                    'NAME_GOODS_CATEGORY_prev_Clothing_and_Accessories', # 0.94
                    'NAME_GOODS_CATEGORY_prev_Furniture', # 0.90
                    'NAME_GOODS_CATEGORY_prev_XNA', # 0.90
                    
                    'CHANNEL_TYPE_prev_Car_dealer', # 0.97
                    
                    'CODE_REJECT_REASON_prev_CLIENT',
                    'ORGANIZATION_TYPE_XNA', 
                    'NAME_INCOME_TYPE_Pensioner'
                    ],axis=1, inplace=True)
    
    
    if is_trainig:
        threshold = int(0.4 * (data.shape[1] - 1))
        
        missing_values = pd.DataFrame(data.isnull().sum(axis=1), columns=['count'])
        missing_values['target'] = data['TARGET']
            
        data.drop(index = missing_values[(missing_values['count'] >= threshold)].index,
                           axis=0, inplace=True)
                
        data.reset_index(drop=True, inplace=True)
    
    for col in data.columns:
        value = np.nanmean(data[col])
        data[col].fillna(value = value, inplace = True)
        
    return data
    
def get_scaled_data(data, num_comps = 10, is_training = True):
    global scaler, pca 
    if is_training:
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(data)
    else:
        scaled_data = scaler.transform(data)
    
    if is_training:
        pca = PCA(n_components = num_comps)
        pcs = pca.fit_transform(scaled_data) 
    else:
        pcs = pca.transform(scaled_data) 
    return pcs

def add_missing_columns(appln_train_p, appln_test_p):
    
    missing_cols = set(appln_train_p.columns) - set(appln_test_p.columns)
    
    for col in missing_cols:
        appln_test_p[col] = 0.0
    appln_test_p = appln_test_p[appln_train_p.columns]
    
    return appln_test_p

def get_train_test_data(appln_train_p, test_size = 0.20):
    appln_train_p = appln_train_p.iloc[np.random.permutation(appln_train_p.shape[0]), :]
    
    target_1_data = appln_train_p[appln_train_p['TARGET'] == 1]
    target_1_data.reset_index(drop=True, inplace=True)
    split_at = int(target_1_data.shape[0] * (1 - test_size))
    
    train = target_1_data.iloc[:split_at, :]
    test = target_1_data.iloc[split_at:, :]
    
    target_0_data = appln_train_p[appln_train_p['TARGET'] == 0]
    target_0_data.reset_index(drop=True, inplace=True)
    split_at = int(target_0_data.shape[0] * (1 - test_size))
    
    train =  pd.concat([train, 
                        target_0_data.iloc[:split_at, :]], axis=0)
    train = train.iloc[np.random.permutation(train.shape[0]), :]
    train.reset_index(drop=True, inplace=True)
    
    test =  pd.concat([test, 
                        target_0_data.iloc[split_at:, :]], axis=0)
    test = test.iloc[np.random.permutation(test.shape[0]), :]
    test.reset_index(drop=True, inplace=True)
    
    x_train = train.drop('TARGET', axis=1)
    y_train = train['TARGET']
    
    x_test = test.drop('TARGET', axis=1)
    y_test = test['TARGET']
    
    return x_train, y_train, x_test, y_test

def get_folds(x, y, k, risk_data_size = 0.5 ):
    risk_x = x.iloc[y[y == 1].index, :].reset_index(drop=True)
    risk_y = y[y == 1].reset_index(drop=True)
    risk = pd.concat([risk_x, risk_y], axis=1)
    risk.reset_index(drop=True, inplace=True)
    
    size = int(risk.shape[0] * risk_data_size)
#    risk = risk.sample(n=size).reset_index(drop=True)
    
    normal_x = x.iloc[y[y == 0].index, :].reset_index(drop=True)
    normal_y = y[y == 0].reset_index(drop=True)
    normal = pd.concat([normal_x, normal_y], axis=1)
    normal.reset_index(drop=True, inplace=True)
    
    rand_index = np.random.randint(0,normal.shape[0],[k, size])
    rand_index_r =  np.random.randint(0,risk.shape[0],[k, size])
    folds = [None] * k
    
    for i in range(k):
        tmp = pd.concat([normal.iloc[rand_index[i],:],
                         risk.iloc[rand_index_r[i],:]], axis = 0)
        tmp = tmp.sample(frac=1).reset_index(drop=True)
        
        folds[i] = tmp
    return folds
    
appln_train, appln_test, bureau_p, prev_appln_p = load_data()

appln_train_p = pd.merge(appln_train, bureau_p, how='left', on='SK_ID_CURR') 
appln_train_p = pd.merge(appln_train_p, prev_appln_p, how='left', on='SK_ID_CURR') 

appln_test_p = pd.merge(appln_test, bureau_p, how='left', on='SK_ID_CURR') 
appln_test_p = pd.merge(appln_test_p, prev_appln_p, how='left', on='SK_ID_CURR')

appln_train_p = clean_data(appln_train_p)

appln_test_p = clean_data(appln_test_p, is_trainig=False)
appln_test_p = add_missing_columns(appln_train_p.drop('TARGET', axis=1), appln_test_p)

x_train, y_train, x_test, y_test = get_train_test_data(appln_train_p)

#x_train = get_scaled_data(x_train, num_comps=100)
#x_test = get_scaled_data(x_test, is_training=False)

print('running model... ')
#model = RandomForestClassifier(n_estimators=100)
model = XGBClassifier(n_estimators=100, max_depth = 60, silent=False, random_state=65)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))    
print(confusion_matrix(y_test, y_pred)) 
#skf = StratifiedKFold(n_splits=3)
#
#i = 1
#for train_index, test_index in skf.split(x_train, y_train):
#    trainx, testx = x_train.iloc[train_index,:], x_train.iloc[test_index,:]
#    trainy, testy = y_train.iloc[train_index], y_train.iloc[test_index]
#    
#    model.fit(trainx, trainy)
#    
#    predy = model.predict(testx)
#    print('fold: ', i)
#    i += 1
#    print(classification_report(testy, predy))    
#    print(confusion_matrix(testy, predy)) 
    
#k = 10
#folds = get_folds(x_train, y_train, k)
#for i in range(k):
#    start_time = time.time()
#    train_folds = folds[:i] + folds[i+1:]
#    
#    for j in range(k-1):
#        trainx = train_folds[j].drop('TARGET', axis=1)
#        trainy = train_folds[j]['TARGET']
#        model.fit(trainx, trainy)
#        
#    print('fold: ',i)
#    testx = folds[i].drop('TARGET', axis=1)
#    testy = folds[i]['TARGET']
#    predy = model.predict(testx)
#    print(confusion_matrix(testy, predy))
#    print(classification_report(testy, predy))
#    print('time taken: ',time.time() - start_time)
#        
##model.fit(x_train, y_train)
#print('results on test data')
#y_pred = model.predict(x_test)
#print(classification_report(y_test, y_pred))    
#print(confusion_matrix(y_test, y_pred)) 
    
    
    
    