import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler
age_scaler = MinMaxScaler()

def merge_data(owner, struct, data, shuffle=False):
    merged = pd.merge(data, owner, how='left', on='building_id')
    merged = pd.merge(merged, struct, how='left', on='building_id')
    
    if shuffle:
        merged = merged.iloc[np.random.permutation(merged.shape[0]),:]
        
    merged.reset_index(drop=True, inplace=True)
    return merged

def hexa_to_deci(hexa):
    try:
        deci = int(hexa, 16)
    except ValueError:
        deci = -1
    return deci

def categorical_to_numeric(data):
    data = pd.get_dummies(data, columns = data.select_dtypes(include=['object']).columns, 
                          drop_first=False)   
    return data

def get_scaled_data(data, scaler, is_training = True):
    if is_training:
        return scaler.fit_transform(data)
    return scaler.transform(data)

def missing_value_treatment(data, y=None, is_training=True):
    count = 0
    for col in data.columns:
#        if col == 'has_repair_started':
#            continue
        
        count += data[col].isnull().sum()
            
        if col in ['count_families', 'count_floors_pre_eq', 'count_floors_post_eq']:
            data[col].fillna(value = data[col].mode().values[0], inplace = True)
            
        elif col in ['age_building', 'plinth_area_sq_ft', 'height_ft_pre_eq', 'height_ft_post_eq']:
            data[col].fillna(value = np.nanmean(data[col]), inplace = True)
        
        else:
            if type(data[col].any()) == str:
                data[col].fillna(value = str(0.5), inplace = True)
            else:
                data[col].fillna(value = 0.5, inplace = True)
    return data

def add_missing_columns(x, testx):
    missing_cols = set(x.columns) - set(testx.columns)
    for col in missing_cols:
        testx[col] = 0.0
    testx = testx[x.columns]
    return testx

def get_new_vcdmun_id(values):
    district_id = str(values[0])
    vcdmun_id = str(values[1])
    new_vcdmun_id = int(vcdmun_id[len(district_id):])
    return new_vcdmun_id

def get_new_ward_id(values):
    vcdmun_id = str(values[0])
    ward_id = str(values[1])
    new_ward_id = int(ward_id[len(vcdmun_id):])
    return new_ward_id
    
def get_new_building_id(values):
    ward_id = str(values[0])
    building_id = str(values[1])
    new_building_id = int(building_id[len(ward_id):])
    return new_building_id

def get_train_data(train, owner, struct):
    owner = owner.copy()
    struct = struct.copy()
    owner.drop(columns=['vdcmun_id', 'ward_id', 'district_id'], axis=1, inplace=True)
    struct.drop(columns=['vdcmun_id', 'district_id'], axis=1, inplace=True)
    
    train_data = merge_data(owner, struct, train, shuffle=True)
    x, y = train_data.drop('damage_grade', axis = 1), train_data['damage_grade']
    x = missing_value_treatment(x, y)
    x['building_id'] = x['building_id'].apply(hexa_to_deci)
    x = categorical_to_numeric(x)
    x.columns = list(map(lambda e:e.replace(' ', '_'), x.columns))
    
    x['count_floors_diff'] = x['count_floors_pre_eq'] - x['count_floors_post_eq']
    x['height_ft_diff'] = x['height_ft_pre_eq'] - x['height_ft_post_eq']
    
    x['age_building'] = get_scaled_data(x['age_building'].values.reshape(-1,1), age_scaler)
    x['plinth_area_sq_ft'] = x['plinth_area_sq_ft'].apply(lambda a:np.log1p(a))
    x['height_ft_pre_eq'] = x['height_ft_pre_eq'].apply(lambda a:np.log1p(a))
    x['height_ft_post_eq'] = x['height_ft_post_eq'].apply(lambda a:np.log1p(a))
    
    x['new_vdcmun_id'] = x[['district_id', 'vdcmun_id']].apply(get_new_vcdmun_id, axis=1)
    x['new_ward_id'] = x[['vdcmun_id', 'ward_id']].apply(get_new_ward_id, axis=1)
    x['new_building_id'] = x[['ward_id', 'building_id']].apply(get_new_building_id, axis=1)
    
    y = y.apply(lambda e:int(e[-1]))
    
    return x, y

def get_test_data(x, test, owner, struct):
    owner = owner.copy()
    struct = struct.copy()
    owner.drop(columns=['vdcmun_id', 'ward_id', 'district_id'], axis=1, inplace=True)
    struct.drop(columns=['vdcmun_id', 'district_id'], axis=1, inplace=True)
    
    test_data = merge_data(owner, struct, test)
    test_data = missing_value_treatment(test_data, is_training=False)
    building_id_test = test_data['building_id'].copy()
    test_data['building_id'] = test_data['building_id'].apply(hexa_to_deci)
    test_data = categorical_to_numeric(test_data)
    test_data.columns = list(map(lambda e:e.replace(' ', '_'), test_data.columns))
    test_data = add_missing_columns(x, test_data)
    
    test_data['count_floors_diff'] = test_data['count_floors_pre_eq'] - test_data['count_floors_post_eq']
    test_data['height_ft_diff'] = test_data['height_ft_pre_eq'] - test_data['height_ft_post_eq']
    
    test_data['age_building'] = get_scaled_data(test_data['age_building'].values.reshape(-1,1),
                                                 age_scaler, is_training=False)
    test_data['plinth_area_sq_ft'] = test_data['plinth_area_sq_ft'].apply(lambda a:np.log1p(a))
    test_data['height_ft_pre_eq'] = test_data['height_ft_pre_eq'].apply(lambda a:np.log1p(a))
    test_data['height_ft_post_eq'] = test_data['height_ft_post_eq'].apply(lambda a:np.log1p(a))
    
    test_data['new_vdcmun_id'] = test_data[['district_id', 'vdcmun_id']].apply(get_new_vcdmun_id, axis=1)
    test_data['new_ward_id'] = test_data[['vdcmun_id', 'ward_id']].apply(get_new_ward_id, axis=1)
    test_data['new_building_id'] = test_data[['ward_id', 'building_id']].apply(get_new_building_id, axis=1)
    return test_data, building_id_test



