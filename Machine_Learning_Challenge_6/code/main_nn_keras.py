import numpy as np
np.random.seed(34)

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import softmax, relu, tanh
from keras.losses import logcosh

import util_nn_keras

basedir = 'D:/competitions/Hackerearth/Machine_Learning_Challenge_6/Dataset/'
write_dir = 'D:/competitions/Hackerearth/Machine_Learning_Challenge_6/Result/'
scaler, enc, pca =  None, None, None

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

x, y = util_nn_keras.get_train_data(train, owner, struct) 
testx, building_id_test = util_nn_keras.get_test_data(x, test, owner, struct)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

one_hot_y_train = util_nn_keras.one_hot_encoding(y_train)
one_hot_y_test = util_nn_keras.one_hot_encoding(y_test, is_train=False)

learning_rate = 1e-3
num_input = x.shape[1]
num_classes = 5
batch_size = 30
epochs = 10

model = Sequential()
model.add(Dense(input_dim=num_input,
                units=90, 
                activation=LeakyReLU(),
                kernel_initializer=keras.initializers.random_normal(mean=0.0, stddev=0.01),
                bias_initializer=keras.initializers.Ones(),
                ))

model.add(Dense(units=90, 
                activation=LeakyReLU(),
                kernel_initializer=keras.initializers.random_normal(mean=0.0, stddev=0.01),
                bias_initializer=keras.initializers.Ones()
                ))

model.add(Dense(units=90, 
                activation=LeakyReLU(),
                kernel_initializer=keras.initializers.random_normal(mean=0.0, stddev=0.01),
                bias_initializer=keras.initializers.Ones()
                ))

model.add(Dense(units=64, 
                activation=LeakyReLU(),
                kernel_initializer=keras.initializers.random_normal(mean=0.0, stddev=0.01),
                bias_initializer=keras.initializers.Ones()
                ))

model.add(Dense(units=32, 
                activation=LeakyReLU(),
                kernel_initializer=keras.initializers.random_normal(mean=0.0, stddev=0.01),
                bias_initializer=keras.initializers.Ones()
                ))

model.add(Dense(units=num_classes, 
                activation=softmax,
                kernel_initializer=keras.initializers.random_normal(mean=0.0, stddev=0.01),
                bias_initializer=keras.initializers.Ones()
                ))

opt = keras.optimizers.Adam(lr=learning_rate)

model.compile(loss='categorical_crossentropy', 
              optimizer=opt, 
              metrics=['accuracy'])


#-----------------------------------------------------------
x_train.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
x_test.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

y_train_1 = y_train.copy()
y_train_1[(y_train_1 != 1) & (y_train_1 != 5)] = 0

x_train_2 = x_train.iloc[y_train[(y_train != 1) & (y_train != 5)].index, :]
y_train_2 = y_train[(y_train != 1) & (y_train != 5)]


model_105 =  RandomForestClassifier(n_estimators=100, 
                                            max_depth=25,
                                            max_features = 0.8)
print('fitting model: model_105', model_105)
model_105.fit(x_train, y_train_1)

#model_234 =  RandomForestClassifier(n_estimators=100, 
#                                            max_depth=25,
#                                            max_features = 0.8)

model_234 = MLPClassifier(hidden_layer_sizes=(90, 60, 30), 
                       activation = 'relu',
                       verbose = True)

print('fitting model: model_234', model_234)
model_234.fit(x_train_2, y_train_2)

y_pred_1 = model_105.predict(x_test)
y_pred_1 = pd.Series(y_pred_1)

#y_value = pd.concat([y_test, y_pred], axis=1)

#dam_123 = 


#dam_123 = y_value[(y_value[0] != 0)]


#dam_123 = y_value[(y_value['damage_grade'] != 1) & 
#                                (y_value['damage_grade'] != 5) & 
#                                (y_value[0] != 0)]
#
#                  
x_test_2 =  x_test.iloc[y_pred_1[y_pred_1 == 0].index, :]
#y_test_2 = y_test[y_pred_1[y_pred_1 == 0].index]

y_pred_2 = model_234.predict(x_test_2)
y_pred_2 = pd.Series(y_pred_2)

#y_test_1 = y_test.copy()
#y_test_1[(y_test_1 != 1) & (y_test_1 != 5)] = 0
#
#print('result: 1')
#print(classification_report(y_test_2, y_pred_2))

j = 0
for i in x_test_2.index:
    y_pred_1[i] = y_pred_2[j]
    j += 1

print('result: 2')
print(classification_report(y_test, y_pred_1, digits=5))




#
#print('fitting MLP model...')
#
#model.fit(x=x_train, y=one_hot_y_train, batch_size=batch_size, epochs=epochs)
#pred = model.predict(x_test)
#
#pred = [p.argmax() for p in pred]
#true = [y.argmax() for y in one_hot_y_test] 
#print(classification_report(true, pred))
#
#model2 = MLPClassifier(hidden_layer_sizes=(90, 60, 30), 
#                       activation = 'relu',
#                       verbose = True)
#model2.fit(x_train, y_train)
#
#pred2 = model2.predict(x_test)
#print(classification_report(y_test, pred2, digits=5))


#model.fit(x,y)
#predy = model.predict(testx)
#test_pred = pd.DataFrame()
#test_pred['damage_grade'] = predy.apply(lambda e:'Grade ' + str(e))
#test_pred['damage_grade'] = ['Grade ' + str(e) for e in predy]
#test_result = pd.concat([building_id_test, test_pred], axis=1)
#test_result.to_csv(write_dir + 'test_results_rf.csv',index=False)


