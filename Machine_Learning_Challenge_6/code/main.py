import numpy as np
np.random.seed(34)

import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns

from sklearn.decomposition import PCA
#from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler, RobustScaler
#from sklearn.grid_search import GridSearchCV

#from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
#from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


#from sklearn.feature_selection import VarianceThreshold, RFE

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import classification_report, confusion_matrix

#import time

from xgboost import XGBClassifier

basedir = 'D:/Hackerearth/Machine_Learning_Challenge_6/Dataset/'
write_dir = 'D:/Hackerearth/Machine_Learning_Challenge_6/'

building_id_train, building_id_test = None, None
scaler, pca = None, None


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

def drop_columns(owner, struct):
    owner.drop(columns=['vdcmun_id', 'ward_id', 'district_id'], axis=1, inplace=True)
    struct.drop(columns=['vdcmun_id', 'ward_id', 'district_id'], axis=1, inplace=True)
    

def merge_data(owner, struct, data, shuffle=False):
    merged = pd.merge(data, owner, how='left', on='building_id')
    merged = pd.merge(merged, struct, how='left', on='building_id')
    
    if shuffle:
        merged = merged.iloc[np.random.permutation(merged.shape[0]),:]
        
    merged.reset_index(drop=True, inplace=True)
    return merged

def missing_value_treatment(data, y=None, is_training=True):
    count = 0
    for col in data.columns:
        count += data[col].isnull().sum()
        
#        if col == 'has_repair_started' and is_training:
#            indices = data[data[col].isnull()].index
#            data.drop(index=indices, axis=0, inplace=True)
#            y.drop(index=indices, axis=0, inplace=True)
            
        if col in ['count_families', 'count_floors_pre_eq', 'count_floors_post_eq']:
            data[col].fillna(value = data[col].mode().values[0], inplace = True)
            
        elif col in ['age_building', 'plinth_area_sq_ft', 'height_ft_pre_eq', 'height_ft_post_eq']:
            data[col].fillna(value = np.nanmean(data[col]), inplace = True)
            
        elif col in ['district_id', 'vdcmun_id', 'ward_id']:
            if data[col].isnull().sum() > 0:
                print('warning: ',col, ' has missing values!')
                
            if type(data[col].any()) == str:
                data[col].fillna(value = str(0.5), inplace = True)
            else:
                data[col].fillna(value = 0.5, inplace = True)
        
        else:
            if type(data[col].any()) == str:
                data[col].fillna(value = str(0.5), inplace = True)
            else:
                data[col].fillna(value = 0.5, inplace = True)
    print('Total number of missing value replaced: ', count)
    return data

def categorical_to_numeric(data):
    data = pd.get_dummies(data, columns = data.select_dtypes(include=['object']).columns, drop_first=False)   
    return data

def pre_processing(data):
    data = missing_value_treatment(data)
    data = categorical_to_numeric(data)
    return data

#def concat_features(data):
#    
#    data['risk'] = 8 * data.has_geotechnical_risk
#    data['risk'] += 7 * data.has_geotechnical_risk_landslide
#    data['risk'] += 6 * data.has_geotechnical_risk_fault_crack
#    data['risk'] += 5 * data.has_geotechnical_risk_rock_fall
#    data['risk'] += 4 * data.has_geotechnical_risk_land_settlement
#    data['risk'] += 3 * data.has_geotechnical_risk_flood
#    data['risk'] += 2 * data.has_geotechnical_risk_liquefaction
#    data['risk'] += data.has_geotechnical_risk_other
#    
#    data.drop(columns=['has_geotechnical_risk', 
#                    'has_geotechnical_risk_landslide', 
#                    'has_geotechnical_risk_fault_crack',
#                    'has_geotechnical_risk_rock_fall', 
#                    'has_geotechnical_risk_land_settlement', 
#                    'has_geotechnical_risk_flood',
#                    'has_geotechnical_risk_liquefaction', 
#                    'has_geotechnical_risk_other'
#                   ], axis=1, inplace=True)
#    
#    data['usage'] =  5 * data.has_secondary_use
#    data['usage'] +=  4 * data.has_secondary_use_agriculture
#    data['usage'] +=  3 * data.has_secondary_use_hotel
#    data['usage'] +=  2 * (data.has_secondary_use_other + data.has_secondary_use_rental)
#    data['usage'] +=  data.has_secondary_use_gov_office + data.has_secondary_use_health_post + data.has_secondary_use_industry + \
#        data.has_secondary_use_institution + data.has_secondary_use_school + data.has_secondary_use_use_police
#        
#    data.drop(columns=['has_secondary_use', 
#                    'has_secondary_use_agriculture', 
#                    'has_secondary_use_hotel', 
#                    'has_secondary_use_other',
#                    'has_secondary_use_rental', 
#                    'has_secondary_use_gov_office', 
#                    'has_secondary_use_health_post', 
#                    'has_secondary_use_industry',
#                    'has_secondary_use_institution', 
#                    'has_secondary_use_school', 
#                    'has_secondary_use_use_police'], axis=1, inplace=True)
#    
#    data['structure'] = 10 * data.has_superstructure_mud_mortar_stone 
#    data['structure'] += 6 * data.has_superstructure_timber
#    data['structure'] += 5 * data.has_superstructure_adobe_mud
#    data['structure'] += 4 * data.has_superstructure_bamboo
#    data['structure'] += 3 * data.has_superstructure_mud_mortar_brick
#    data['structure'] += 2 * data.has_superstructure_stone_flag
#    data['structure'] += data.has_superstructure_other
#
#    data['structure'] -= 3 * data.has_superstructure_mud_mortar_stone
#    data['structure'] -= 2 * data.has_superstructure_rc_non_engineered
#    data['structure'] -= data.has_superstructure_mud_mortar_stone
#    
#    data.drop(columns=['has_superstructure_mud_mortar_stone', 
#                    'has_superstructure_timber', 
#                    'has_superstructure_adobe_mud', 
#                    'has_superstructure_bamboo',
#                    'has_superstructure_mud_mortar_brick', 
#                    'has_superstructure_stone_flag', 
#                    'has_superstructure_other', 
#                    'has_superstructure_mud_mortar_stone',
#                    'has_superstructure_rc_non_engineered', 
#                    'has_superstructure_rc_engineered'], axis=1, inplace=True)
    
def get_train_data(train, owner, struct):
    global building_id_train
    
    train.drop(columns=['vdcmun_id'], axis=1, inplace=True)
    train_data = merge_data(owner, struct, train, shuffle=True)
    
    building_id_train = pd.DataFrame(train_data['building_id'], columns=['building_id'])
    train_data.drop('building_id', axis = 1, inplace = True)
    
    x, y = train_data.drop('damage_grade', axis = 1), train_data['damage_grade']
    x = missing_value_treatment(x, y)
    x = categorical_to_numeric(x)
    x.columns = list(map(lambda e:e.replace(' ', '_'), x.columns))
#    concat_features(x)
    y = y.apply(lambda e:int(e[-1]))
    return x, y

def get_test_data(test, owner, struct):
    global building_id_test
    
    test.drop(columns=['vdcmun_id'], axis=1, inplace=True)
    test_data = merge_data(owner, struct, test)
    print('test_data.shape: ', test_data.shape)
    
    building_id_test = pd.DataFrame(test_data['building_id'], columns=['building_id'])
    test_data.drop('building_id', axis = 1, inplace = True)
    
    test_data = missing_value_treatment(test_data, is_training=False)
    test_data = categorical_to_numeric(test_data)
    test_data.columns = list(map(lambda e:e.replace(' ', '_'), test_data.columns))
#    concat_features(test_data)
    print('test_data.shape: ', test_data.shape)
    return test_data

def add_missing_columns(x, testx):
    
    missing_cols = set(x.columns) - set(testx.columns)
    
    for col in missing_cols:
        testx[col] = 0.0
    testx = testx[x.columns]
    
    return testx

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


#def get_scaled_data(data,  is_training = True):
#    global scaler, pca 
#    if is_training:
#        scaler = RobustScaler()
#        scaled_data = scaler.fit_transform(data)
#    else:
#        scaled_data = scaler.transform(data)
#    return scaled_data

def dimention_reduction(x):
    x.drop(columns=[
#            'vdcmun_id',
#            'ward_id',
    
#            'has_geotechnical_risk_landslide',
#            'has_secondary_use_agriculture',
#            'height_ft_pre_eq',
#            'height_ft_post_eq',
#            'condition_post_eq_Damaged-Rubble_clear',
#            'foundation_type_RC',
#            'other_floor_type_RCC/RB/RBC',
            
            'has_secondary_use_use_police',
            'plan_configuration_H-shape',
            'has_secondary_use_gov_office',
            'plan_configuration_E-shape',
            'plan_configuration_Building_with_Central_Courtyard',
            'has_secondary_use_health_post',
            'has_secondary_use_school',
            'condition_post_eq_Covered_by_landslide',
            'plan_configuration_U-shape',
            'plan_configuration_Others',
            'has_secondary_use_institution'
            ], axis=1, inplace=True)
    

def plot_data_2d(x, y):
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(x)
    
    pca = PCA(n_components = 2)
    pcs = pca.fit_transform(scaled_data) 
#    plt.xlim([-15,15])
#    plt.ylim([-15,15])
    plt.scatter(pcs[:,0], pcs[:,1],  c=y, cmap = 'magma')
    
train, test, owner, struct = load_data()

drop_columns(owner, struct)

x, y = get_train_data(train, owner, struct) 

dimention_reduction(x)

#plot_data_2d(x,y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

x_train.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
x_test.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

#y_train_1 = y_train.copy()
#y_train_1[(y_train_1 != 1) & (y_train_1 != 5)] = 0
#
#x_train_2 = x_train.iloc[y_train[(y_train != 1) & (y_train != 5)].index, :]
#y_train_2 = y_train[(y_train != 1) & (y_train != 5)]


#x_train = get_scaled_data(x_train, num_comps=40)

#x_train = get_scaled_data(x_train, num_comps=50)
#x_test = get_scaled_data(x_test, is_training=False)



testx = get_test_data(test, owner, struct)
testx = add_missing_columns(x, testx)

#model = AdaBoostClassifier(base_estimator = LogisticRegression(),
#                           n_estimators = 100)
#model = KNeighborsClassifier()


#print('running model 1 ...')
#model1 = XGBClassifier(n_estimators=100,
#                      objective='multi:softmax',
##                     colsample_bytree=0.3,
##                     colsample_bylevel=0.2,
#                      max_delta_step = 30,
#                      max_depth=10,
#                      silent=False,
#                      random_state=65
#                      )
#model1.fit(x_train, y_train_1)

#print('running model 2 ...')
##model2 = LogisticRegression()
#model2 =  XGBClassifier(n_estimators=100,
#                      objective='multi:softmax',
##                     colsample_bytree=0.3,
##                     colsample_bylevel=0.2,
#                      max_delta_step = 40,
#                      max_depth=10,
#                      silent=False,
#                      random_state=32
#                      )
#model2.fit(x_train_2, y_train_2)

#x_test = get_scaled_data(x_test, is_training=False)
#y_pred = model1.predict(x_test)
#y_pred = pd.Series(y_pred)
#
#
#y_value = pd.concat([y_test, y_pred], axis=1)

#dam_123 = y_value[(y_value['damage_grade'] != 1) & 
#                                (y_value['damage_grade'] != 5) & 
#                                (y_value[0] != 5) &
#                                (y_value[0] != 5)]


#dam_123 = y_value[(y_value[0] != 0)]


#dam_123 = y_value[(y_value['damage_grade'] != 1) & 
#                                (y_value['damage_grade'] != 5) & 
#                                (y_value[0] != 0)]
#
#                  
#x_test_2 =  x_test.iloc[dam_123.index, :]
#
#y_test_2 = dam_123['damage_grade']
#
#y_pred_2 = model2.predict(x_test_2)
#y_pred_2 = pd.Series(y_pred_2)
#
#print('result: 1')
#print(classification_report(y_test_2, y_pred_2))
#
#j = 0
#for i in dam_123.index:
#    y_pred[i] = y_pred_2[j]
#    j += 1
#
#print('result: 2')
#print(classification_report(y_test, y_pred))

#model3 = KNeighborsClassifier(n_neighbors = 3,
#                              )


#model3 = RandomForestClassifier(n_estimators=100)
#model3 = GaussianNB()
#model3 = GradientBoostingClassifier(n_estimators=100,
#                                    max_depth = 15)

#model3 = RandomForestClassifier(n_estimators = 100)

#model3 =  XGBClassifier(n_estimators=100,
#                      objective='multi:softmax',
##                     colsample_bytree=0.3,
##                     colsample_bylevel=0.2,
#                      max_delta_step = 10,
#                      max_depth=30,
#                      silent=False,
#                      random_state=32
#                      )

#model3 = LogisticRegression()
#
#skf = StratifiedKFold(n_splits=10)
#
#i=1
#for train_index, test_index in skf.split(x_train, y_train):
#    print('fold:', i)
#    model3.fit(x_train.iloc[train_index,:], y_train[train_index])
#    
#    pred = model3.predict(x_train.iloc[test_index,:])
#    print(classification_report(y_train[test_index], pred))
#    i += 1
#
##model3.fit(x_train, y_train)
##
##
#y_pred_3 = model3.predict(x_test)
#print(classification_report(y_test, y_pred_3))
#
#model4 = KNeighborsClassifier()
#model4.fit(x_train, y_train)
#y_pred_4 = model4.predict(x_test)
#print(classification_report(y_test, y_pred_4))


model = make_pipeline(PolynomialFeatures(2), LogisticRegression())
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))


#
#model =  XGBClassifier(n_estimators=100,
#                      objective='multi:softmax',
##                     colsample_bytree=0.3,
##                     colsample_bylevel=0.2,
#                      max_delta_step = 10,
#                      max_depth=30,
#                      silent=False,
#                      random_state=32
#                      )

#model = RandomForestClassifier(n_estimators=100)
#model = knn = KNeighborsClassifier(n_neighbors = 5,
#                                     algorithm = 'ball_tree')

#model = GaussianNB()
#model.fit(x, y)
#

#predy = model.predict(testx)

 # collecting results
#test_pred = pd.DataFrame()
##test_pred['damage_grade'] = predy.apply(lambda e:'Grade ' + str(e))
#test_pred['damage_grade'] = ['Grade ' + str(e) for e in predy]
#test_result = pd.concat([building_id_test, test_pred], axis=1)
#test_result.to_csv(write_dir + 'test_results.csv',index=False)








dict = {}


for val in test.building_id:
    for c in val:
        dict[c] = dict.get(c, 0) + 1