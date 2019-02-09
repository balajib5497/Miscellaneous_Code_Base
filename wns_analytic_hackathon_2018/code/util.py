import numpy as np
#np.random.seed(34)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, train_test_split

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.decomposition import PCA

def plot_data_2d(x, y=None):
    pca = PCA(n_components = 2)
    pcs = pca.fit_transform(x) 
    if y is None:
        plt.scatter(pcs[:,0], pcs[:,1], cmap = 'viridis')
    else:
        plt.scatter(pcs[:,0], pcs[:,1], c=y, cmap = 'viridis')
   
train = pd.read_csv('../dataset/train_LZdllcl.csv')
test = pd.read_csv('../dataset/test_2umaH9m.csv')

# 'department', 'region', 'education', 'gender', 'recruitment_channel', 
# 'previous_year_rating', 'KPIs_met >80%', 'awards_won?'

# no_of_trainings, age, length_of_service, avg_training_score

scaler1 = StandardScaler()
scaler2 = MinMaxScaler()
cols_to_scale = ['no_of_trainings', 'age', 'length_of_service', 'avg_training_score']

train['age'] = train['age'].apply(np.log1p)
train['avg_training_score'] = train['avg_training_score'].apply(np.log1p)
train['length_of_service'] = train['length_of_service'].apply(lambda x: x**(2))
train[cols_to_scale] = scaler1.fit_transform(train[cols_to_scale])
train[cols_to_scale] = scaler2.fit_transform(train[cols_to_scale])

test['age'] = test['age'].apply(np.log1p)
test['avg_training_score'] = test['avg_training_score'].apply(np.log1p)
test['length_of_service'] = test['length_of_service'].apply(lambda x: x**(2))
test[cols_to_scale] = scaler1.transform(test[cols_to_scale])
test[cols_to_scale] = scaler2.transform(test[cols_to_scale])

train = pd.get_dummies(train, columns=['education', 'previous_year_rating'], drop_first=False)
train.drop('employee_id', axis=1, inplace=True)
train = pd.get_dummies(train, columns=train.select_dtypes('object').columns, drop_first=True)
train = train.sample(frac=1).reset_index(drop=True)

test = pd.get_dummies(test, columns=['education', 'previous_year_rating'], drop_first=False)
test_employee_id = test['employee_id']
test.drop('employee_id', axis=1, inplace=True)
test = pd.get_dummies(test, columns=test.select_dtypes('object').columns, drop_first=True)

def get_folds(train, k=6, bal=0.7):
    train_class_0 = train[train['is_promoted'] == 0]
    train_class_0.reset_index(drop=True, inplace=True)
    
    train_class_1 = train[train['is_promoted'] == 1]
    train_class_1.reset_index(drop=True, inplace=True)
    
    sample_size = round(train_class_1.shape[0] * bal * 2)
    folds = [None]*k
    for i in range(k):
        t =  train_class_0.loc[i*sample_size:(i+1)*sample_size,:]
        t = pd.concat([t, train_class_1.sample(frac=bal).reset_index(drop=True)], axis=0)
        folds[i] = t
    return folds
    
def get_train_test_split(train, test_size):
    x_train, x_test, y_train, y_test = train_test_split(train[train['is_promoted'] == 0].drop('is_promoted', axis=1), 
                                                        train[train['is_promoted'] == 0]['is_promoted'], 
                                                        test_size=test_size, random_state=42)
    
    x_train2, x_test2, y_train2, y_test2 = train_test_split(train[train['is_promoted'] == 1].drop('is_promoted', axis=1), 
                                                            train[train['is_promoted'] == 1]['is_promoted'], 
                                                            test_size=test_size, random_state=42)
    
    x_train = pd.concat([x_train, x_train2],  axis=0, ignore_index=True)
    x_train.reset_index(drop=True, inplace=True)
    y_train = pd.concat([y_train, y_train2],  axis=0, ignore_index=True)
    y_train.reset_index(drop=True, inplace=True)
    
    index = np.random.permutation(x_train.shape[0])
    x_train = x_train.loc[index, :]
    y_train = y_train[index]
    
    x_test = pd.concat([x_test, x_test2],  axis=0)
    y_test = pd.concat([y_test, y_test2],  axis=0)
    
    return x_train, x_test, y_train, y_test

x = train.drop('is_promoted', axis=1)
y = train['is_promoted']
plot_data_2d(x,y)

x_train, x_test, y_train, y_test = get_train_test_split(train, 0.2)

#pca = PCA(n_components = 20)
#x_train = pd.DataFrame(pca.fit_transform(x_train))
#x_test = pd.DataFrame(pca.transform(x_test))
#x_train.columns = x_train.columns.astype(str)
#x_test.columns = x_test.columns.astype(str)

