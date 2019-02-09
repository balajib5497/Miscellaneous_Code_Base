import numpy as np
#np.random.seed(34)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, train_test_split, StratifiedKFold

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.decomposition import PCA

from imblearn.under_sampling import RandomUnderSampler, NearMiss, CondensedNearestNeighbour
from imblearn.under_sampling import EditedNearestNeighbours, RepeatedEditedNearestNeighbours, TomekLinks

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek

from imblearn.ensemble import BalanceCascade, EasyEnsemble

from sklearn.ensemble import AdaBoostClassifier

from xgboost import XGBClassifier

def get_folds(train, k=6, bal=0.7):
    train_class_0 = train[train['is_promoted'] == 0]
    train_class_0.reset_index(drop=True, inplace=True)
    
    train_class_1 = train[train['is_promoted'] == 1]
    train_class_1.reset_index(drop=True, inplace=True)
    
    sample_size = round(train_class_1.shape[0] * bal)
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

def plot_data_2d(x, y=None):
    pca = PCA(n_components = 2)
    pcs = pca.fit_transform(x) 
    if y is None:
        plt.scatter(pcs[:,0], pcs[:,1], cmap = 'viridis')
    else:
        plt.scatter(pcs[:,0], pcs[:,1], c=y, cmap = 'viridis')
   
def oversampling(train, top_n=100):
    data = train.sort_values(by=['avg_training_score', 'previous_year_rating', 'KPIs_met >80%', 'awards_won?'],
                       ascending=[False, False, False, False]).head(top_n)
    
    avg_training_score_max = max(train['avg_training_score'])
    avg_training_score_min = min(train['avg_training_score'])
    
    previous_year_rating_max = max(train['previous_year_rating'])
    previous_year_rating_min = min(train['previous_year_rating'])
    
    new_samples = []
    for i in data.index:
        if data.loc[i, 'is_promoted'] == 1:
            if data.loc[i, 'avg_training_score'] < avg_training_score_max:
                t = data.loc[i, :]
                t['avg_training_score'] += 1
                new_samples.append(t)
        
                
            if data.loc[i, 'avg_training_score'] > avg_training_score_min:
                t = data.loc[i, :]
                t['avg_training_score'] -= 1
                new_samples.append(t)
                
            if data.loc[i, 'previous_year_rating'] < previous_year_rating_max:
                t = data.loc[i, :]
                t['previous_year_rating'] += 1
                new_samples.append(t)
                
                
            if data.loc[i, 'previous_year_rating'] > previous_year_rating_min:
                t = data.loc[i, :]
                t['previous_year_rating'] -= 1
                new_samples.append(t)
               
#            t = data.loc[i, :]
#            t['KPIs_met >80%'] = abs(t['KPIs_met >80%'] - 1)
#            new_samples.append(t)
#            
#            t = data.loc[i, :]
#            t['awards_won?'] = abs(t['awards_won?'] - 1)
#            new_samples.append(t)
                
    return pd.DataFrame(new_samples, columns=train.columns)
    
train = pd.read_csv('../dataset/train_LZdllcl.csv')
test = pd.read_csv('../dataset/test_2umaH9m.csv')

#train.drop(columns=['region'], axis=1, inplace=True)
#test.drop(columns=['region'], axis=1, inplace=True)

#train['KPI_and_Award'] = train['KPIs_met >80%'] + train['awards_won?']
#test['KPI_and_Award'] = test['KPIs_met >80%'] +  test['awards_won?']
#
#age_min = min(train['age'])
#age_max = max(train['age'])
#length_min = min(train['length_of_service'])
#length_max = max(train['length_of_service'])
#
#train['Age_and_length'] = 0.5 * ((train['age'] - age_min)/(age_max-age_min)) + \
#            0.5 * ((train['length_of_service'] - length_min)/(length_max-length_min))
#    
#corr = train.corr()

x = train.drop('is_promoted', axis=1)
y = train['is_promoted']

x_train, x_test, y_train, y_test = get_train_test_split(train, 0.2)

#new_samples = oversampling(pd.concat([x_train, y_train], axis=1), top_n = 100)
#
#x_train = pd.concat([x_train, new_samples.drop('is_promoted', axis=1)],  axis=0, ignore_index=True)
#x_train.reset_index(drop=True, inplace=True)
#
#y_train = pd.concat([y_train, new_samples['is_promoted']], axis=0, ignore_index=True)
#y_train.reset_index(drop=True, inplace=True)
#
#index = np.random.permutation(x_train.shape[0])
#x_train = x_train.loc[index, :]
#y_train = y_train[index]
    
#tr = train.sort_values(by=['avg_training_score', 'previous_year_rating', 'KPIs_met >80%', 'awards_won?'],
#                       ascending=[False, False, False, False])

#tr.head(1000).is_promoted.value_counts()
#train.drop(columns = ['region'], axis=1, inplace=True)
#test.drop(columns = ['region'], axis=1, inplace=True)
# 'department', 'region', 'education', 'gender', 'recruitment_channel', 
# 'previous_year_rating', 'KPIs_met >80%', 'awards_won?'

# no_of_trainings, age, length_of_service, avg_training_score
############################################################################################
scaler = StandardScaler()
cols_to_scale = ['no_of_trainings', 'age', 'length_of_service', 'avg_training_score']

x_train['age'] = x_train['age'].apply(np.log1p)
x_train['avg_training_score'] = x_train['avg_training_score'].apply(np.log1p)
x_train['length_of_service'] = x_train['length_of_service'].apply(lambda x: x**(2))
x_train[cols_to_scale] = scaler.fit_transform(x_train[cols_to_scale])

x_test['age'] = x_test['age'].apply(np.log1p)
x_test['avg_training_score'] = x_test['avg_training_score'].apply(np.log1p)
x_test['length_of_service'] = x_test['length_of_service'].apply(lambda x: x**(2))
x_test[cols_to_scale] = scaler.transform(x_test[cols_to_scale])

x_train = pd.get_dummies(x_train, columns=['education', 'previous_year_rating'], drop_first=False)
x_train.drop('employee_id', axis=1, inplace=True)
x_train = pd.get_dummies(x_train, columns=x_train.select_dtypes('object').columns, drop_first=True)

x_test = pd.get_dummies(x_test, columns=['education', 'previous_year_rating'], drop_first=False)
x_test.drop('employee_id', axis=1, inplace=True)
x_test = pd.get_dummies(x_test, columns=x_test.select_dtypes('object').columns, drop_first=True)

############################################################################################
scaler = StandardScaler()
cols_to_scale = ['no_of_trainings', 'age', 'length_of_service', 'avg_training_score']

train['age'] = train['age'].apply(np.log1p)
train['avg_training_score'] = train['avg_training_score'].apply(np.log1p)
train['length_of_service'] = train['length_of_service'].apply(lambda x: x**(2))
train[cols_to_scale] = scaler.fit_transform(train[cols_to_scale])

test['age'] = test['age'].apply(np.log1p)
test['avg_training_score'] = test['avg_training_score'].apply(np.log1p)
test['length_of_service'] = test['length_of_service'].apply(lambda x: x**(2))
test[cols_to_scale] = scaler.transform(test[cols_to_scale])

train = pd.get_dummies(train, columns=['education', 'previous_year_rating'], drop_first=False)
train.drop('employee_id', axis=1, inplace=True)
train = pd.get_dummies(train, columns=train.select_dtypes('object').columns, drop_first=True)
train = train.sample(frac=1).reset_index(drop=True)

test = pd.get_dummies(test, columns=['education', 'previous_year_rating'], drop_first=False)
test_employee_id = test['employee_id']
test.drop('employee_id', axis=1, inplace=True)
test = pd.get_dummies(test, columns=test.select_dtypes('object').columns, drop_first=True)

############################################################################################
x = train.drop('is_promoted', axis=1)
y = train['is_promoted']
plot_data_2d(x,y)

#us = RandomUnderSampler(ratio=0.5, random_state=1)
#us = NearMiss(ratio=0.5, size_ngh=3, version=1, random_state=1)
#us = NearMiss(ratio=0.5, size_ngh=3, version=2, random_state=1)
#us = NearMiss(ratio=0.5, size_ngh=3, ver3_samp_ngh=3, version=3, random_state=1)
#us = CondensedNearestNeighbour(random_state=1)
#us = EditedNearestNeighbours(size_ngh=5, random_state=1)
#us = RepeatedEditedNearestNeighbours(size_ngh=5, random_state=1)
#us = TomekLinks()
#x_train, y_train = us.fit_sample(x_train, y_train)
#x_train_res, y_train_res = us.fit_sample(x, y)

#os = RandomOverSampler(ratio=1)
#os = SMOTE(ratio=1, k=5, random_state=1)
#os = SMOTETomek(ratio=0.5, k=5, random_state=1)
#os = SMOTEENN(ratio=0.5, k=5, size_ngh=5, random_state=1)
#x_train, y_train = os.fit_sample(x_train, y_train)
#x_train_res, y_train_res = os.fit_sample(x, y)

#ens = EasyEnsemble()
#x_train_res, y_train_res = ens.fit_sample(x_train, y_train)
#x_train_res, y_train_res = os.fit_sample(x, y)


#x_train = pd.DataFrame(x_train)
#y_train = pd.Series(y_train)
#train_data = pd.concat([x_train, y_train], axis=1)
#model = RandomForestClassifier()

#k = 6
#folds = get_folds(train, k, bal=0.7)
#
#for i in range(k):
#    train_folds = folds[:i] + folds[i+1:]
#    
#    for j in range(k-1):
#        trainx = train_folds[j].drop('is_promoted', axis=1)
#        trainy = train_folds[j]['is_promoted']
#        model.fit(trainx, trainy)
#        
#    print('fold: ',i)
#    testx = folds[i].drop('is_promoted', axis=1)
#    testy = folds[i]['is_promoted']
#    
#    predy = model.predict(testx)
##    print(confusion_matrix(testy, predy))
#    print(classification_report(testy, predy))
    
#model = DecisionTreeClassifier()
#model.fit(x_train, y_train)

    
#model = RandomForestClassifier(n_estimators=100, 
#                               class_weight='balanced',
#                               max_features=0.8,
#                               min_samples_split=3,
#                               random_state=1
#                               )
model = SVC()
#k = 5
#folds = get_folds(pd.concat([x_train, y_train], axis=1),  k, bal=0.5)
#for i in range(k):
#    train_folds = folds[:i] + folds[i+1:]
#    
#    for j in range(k-1):
#        trainx = train_folds[j].drop('is_promoted', axis=1)
#        trainy = train_folds[j]['is_promoted']
##        us = RandomUnderSampler(ratio=0.5, random_state=1)
#        us = NearMiss(ratio=0.5, size_ngh=3, version=1, random_state=1)
#            #us = NearMiss(ratio=0.5, size_ngh=3, version=2, random_state=1)
#            #us = NearMiss(ratio=0.5, size_ngh=3, ver3_samp_ngh=3, version=3, random_state=1)
#            #us = CondensedNearestNeighbour(random_state=1)
#            #us = EditedNearestNeighbours(size_ngh=5, random_state=1)
#            #us = RepeatedEditedNearestNeighbours(size_ngh=5, random_state=1)
#            #us = TomekLinks()
#        trainx, trainy = us.fit_sample(trainx, trainy)
#    
#        model.fit(trainx, trainy)
#        
#    print('fold: ',i)
#    testx = folds[i].drop('is_promoted', axis=1)
#    testy = folds[i]['is_promoted']
#    predy = model.predict(testx)
#    print(classification_report(testy, predy))


#kf = StratifiedKFold(n_splits=6)
#i=1
#for train_index, test_index in kf.split(x_train, y_train):
#    print('fold: ', i)
#    i += 1
#    trainx, testx = x_train.loc[train_index,:], x_train.loc[test_index,:]
#    trainy, testy = y_train[train_index], y_train[test_index]
#    
#    print(trainy.value_counts())
#    model.fit(trainx, trainy)
#    
#    predy = model.predict(testx)
#    print(classification_report(testy, predy))
    
#model.fit(x_train_res, y_train_res)
model.fit(x_train, y_train)
#model.fit(x, y)
pred_y = model.predict(x_test)
print(classification_report(y_test, pred_y))
print(confusion_matrix(y_test, pred_y))

#m = DecisionTreeClassifier()
#m.fit(x, y)

#pred_y = model.predict(xtest)
#results = pd.DataFrame()
#results['employee_id'] = test_employee_id
#results['is_promoted'] = pred_y
##
#results.to_csv('../results/predictions.csv', index=False)
#contigency_table = pd.crosstab(index=train['region'], 
#                               columns=train[''],
#                               normalize='index',
#                               margins=True)
#
#sns.countplot(y='department', hue='is_promoted', data=train)
#
#tr = train.sort_values(by=['KPIs_met >80%', 'awards_won?','avg_training_score','previous_year_rating',
#                           'no_of_trainings', 'age', 'length_of_service'], 
#                       ascending=[False, False, False, False, True, True, True])
#
#train[train['previous_year_rating'].isnull()]['is_promoted'].value_counts() * 100 / train[train['previous_year_rating'].isnull()].shape[0]
#
#corr = train.corr()