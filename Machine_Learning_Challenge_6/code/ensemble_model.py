import numpy as np
np.random.seed(49)

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
#from xgboost import XGBClassifier

class models:
    def __init__(self):
        
        log_reg = LogisticRegression()
        
#        knn = KNeighborsClassifier(n_neighbors = 5)
        
        rf = RandomForestClassifier(n_estimators=100,
                            max_depth = 25)
        
#        gb = GradientBoostingClassifier(n_estimators=100)
        
        dt = DecisionTreeClassifier(min_samples_leaf = 30,
                                    max_depth = 50)
        
#        xgb = XGBClassifier(n_estimators=100,
#                                        max_depth=30)
        
        self.models = {
                'logistric_regression': log_reg,
#                'knn':knn,
                'random_forest':rf,
#                'gradient_boosting':gb,
                'decision_tree':dt,
#                'xgboost':xgb
                }
        
        
    def fit(self, x, y):
        for key in self.models.keys():
            print('training model:', key)
            self.models[key].fit(x,y)
            
            
    def predict(self, x):
        m = len(self.models)
        n = len(x)
        pred = pd.DataFrame(np.zeros([n,m]), columns=self.models.keys())
        
        for key in self.models.keys():
            pred.loc[:, key] = self.models[key].predict(x)
            
        print('predicted result:\n', pred)
        
        def final_result(frame):
            res = frame.value_counts().head(1)
            if res.sum() > 1:
                return res.index[0]
            else:
                return frame['random_forest']
                
        pred.to_csv('./ensemble_predictions.csv')
        y_pred = pred.apply(final_result, axis=1)
        print('final result:\n', y_pred)
        
        return y_pred
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        