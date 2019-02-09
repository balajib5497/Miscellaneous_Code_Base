import numpy as np
import pandas as pd


def merge_data(owner, struct, data, shuffle=False):
    merged = pd.merge(data, owner, how='left', on='building_id')
    merged = pd.merge(merged, struct, how='left', on='building_id')
    
    if shuffle:
        merged = merged.iloc[np.random.permutation(merged.shape[0]),:]
        
    merged.reset_index(drop=True, inplace=True)
    return merged

def categorical_to_numeric(data):
    data = pd.get_dummies(data, columns = data.select_dtypes(include=['object']).columns, 
                          drop_first=False)   
    return data


def missing_value_treatment_random_forest(data):
    count = 0
    for col in data.columns:
        count += data[col].isnull().sum()
            
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


def missing_value_treatment(model, x, y=None, is_training=True):
    if model == 'random_forest':
        return missing_value_treatment_random_forest(x)
    
        




