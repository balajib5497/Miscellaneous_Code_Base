import numpy as np
np.random.seed(34)

import pandas as pd

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


def categorical_to_numeric(data):
    data = pd.get_dummies(data, columns = data.select_dtypes(include=['object']).columns, drop_first=False)   
    return data

def dimention_reduction(x):
    x.drop(columns=[
            'vdcmun_id',
    
            'has_geotechnical_risk_landslide',
            'has_secondary_use_agriculture',
            'height_ft_pre_eq',
            'height_ft_post_eq',
            'condition_post_eq_Damaged-Rubble_clear',
            'foundation_type_RC',
            'other_floor_type_RCC/RB/RBC',
            
#            'has_secondary_use_use_police',
#            'plan_configuration_H-shape',
#            'has_secondary_use_gov_office',
#            'plan_configuration_E-shape',
#            'plan_configuration_Building_with_Central_Courtyard',
#            'has_secondary_use_health_post',
#            'has_secondary_use_school',
#            'condition_post_eq_Covered_by_landslide',
#            'plan_configuration_U-shape',
#            'plan_configuration_Others',
#            'has_secondary_use_institution',
#            
#            'has_geotechnical_risk_flood',
#            'has_geotechnical_risk_liquefaction',
#            'has_geotechnical_risk_other',     
#            'has_secondary_use_rental',     
#            'has_secondary_use_industry',     
#            'has_secondary_use_other',     
#            'area_assesed_Interior',     
#            'legal_ownership_status_Institutional',
#            'legal_ownership_status_Other',
#            'foundation_type_Other',
#            'ground_floor_type_Other',
#            'ground_floor_type_Timber',
#            'position_Attached-3_side',
#            'plan_configuration_Multi-projected',
#            'plan_configuration_T-shape'
            ], axis=1, inplace=True)
    

def add_missing_columns(x, testx):
    missing_cols = set(x.columns) - set(testx.columns)
    
    for col in missing_cols:
        testx[col] = 0.0
    testx = testx[x.columns]
    
    return testx

def get_train_data(train, owner, struct):
    train_data = merge_data(owner, struct, train, shuffle=True)
    train_data.drop('building_id', axis = 1, inplace = True)
    x, y = train_data.drop('damage_grade', axis = 1), train_data['damage_grade']
    x = missing_value_treatment(x, y)
    x = categorical_to_numeric(x)
    x.columns = list(map(lambda e:e.replace(' ', '_'), x.columns))
    dimention_reduction(x)
    y = y.apply(lambda e:int(e[-1]))
    return x, y


def get_test_data(x, test, owner, struct):
    test_data = merge_data(owner, struct, test)
    building_id_test = test_data['building_id'].copy()
    test_data.drop('building_id', axis = 1, inplace = True)
    test_data = missing_value_treatment(test_data, is_training=False)
    test_data = categorical_to_numeric(test_data)
    test_data.columns = list(map(lambda e:e.replace(' ', '_'), test_data.columns))
    test_data = add_missing_columns(x, test_data)
    return test_data, building_id_test