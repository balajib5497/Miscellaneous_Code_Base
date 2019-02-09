import numpy as np
np.random.seed(34)

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

#import keras
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers.advanced_activations import LeakyReLU
#from keras.activations import softmax, relu, tanh
#from keras.losses import logcosh

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

#one_hot_y_train = util_nn_keras.one_hot_encoding(y_train)
#one_hot_y_test = util_nn_keras.one_hot_encoding(y_test, is_train=False)
#
#learning_rate = 1e-3
#num_input = x.shape[1]
#num_classes = 5
#batch_size = 30
#epochs = 10
#
#model = Sequential()
#model.add(Dense(input_dim=num_input,
#                units=90, 
#                activation=LeakyReLU(),
#                kernel_initializer=keras.initializers.random_normal(mean=0.0, stddev=0.01),
#                bias_initializer=keras.initializers.Ones(),
#                ))
#
#model.add(Dense(units=90, 
#                activation=LeakyReLU(),
#                kernel_initializer=keras.initializers.random_normal(mean=0.0, stddev=0.01),
#                bias_initializer=keras.initializers.Ones()
#                ))
#
#model.add(Dense(units=90, 
#                activation=LeakyReLU(),
#                kernel_initializer=keras.initializers.random_normal(mean=0.0, stddev=0.01),
#                bias_initializer=keras.initializers.Ones()
#                ))
#
#model.add(Dense(units=64, 
#                activation=LeakyReLU(),
#                kernel_initializer=keras.initializers.random_normal(mean=0.0, stddev=0.01),
#                bias_initializer=keras.initializers.Ones()
#                ))
#
#model.add(Dense(units=32, 
#                activation=LeakyReLU(),
#                kernel_initializer=keras.initializers.random_normal(mean=0.0, stddev=0.01),
#                bias_initializer=keras.initializers.Ones()
#                ))
#
#model.add(Dense(units=num_classes, 
#                activation=softmax,
#                kernel_initializer=keras.initializers.random_normal(mean=0.0, stddev=0.01),
#                bias_initializer=keras.initializers.Ones()
#                ))
#
#opt = keras.optimizers.Adam(lr=learning_rate)
#
#model.compile(loss='categorical_crossentropy', 
#              optimizer=opt, 
#              metrics=['accuracy'])

#model = MLPClassifier(hidden_layer_sizes=(70, 30, 10), 
#                       activation = 'relu',
#                       batch_size = 30,
#                       verbose = True)
#
#print('fitting model: model_234', model)
#model.fit(x_train, y_train)
#y_pred = model.predict(x_test)
#
#print(classification_report(y_test, y_pred, digits=5))

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


