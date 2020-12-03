import pickle
from sklearn import tree
import pickle
import numpy as np
import numpy as np
from scenario import scenario
#from scenario import index2target
from calcMSE import calcMSE
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class learner:
  def classify(N,K,alpha,gamma,w,G,pmin,pmax,sensor_seed,offgrid_coeff,training_size,test_size,train):
      filename = 'trained_learner.sav'
      if train == True:
        r = np.zeros((K,training_size),dtype=np.float)
        idx = np.zeros((N,training_size),dtype=np.float)
        for i in range(0,training_size):
            target_seed = i
            r[:,[i]], idx[0:N,[i]], dummy1, dummay2 = scenario(N,K,alpha,gamma,w,G,pmin,pmax,target_seed,sensor_seed,offgrid_coeff)
        X = r; Y = idx 
        scaler_X = StandardScaler()
        scaler_X.fit(X.T)
        X = scaler_X.transform(X.T).T
        #clf = RandomForestClassifier(n_estimators=50, random_state=0)
        clf = tree.DecisionTreeClassifier()
        clf.fit(X.transpose(),Y.transpose())
        pickle.dump([clf,scaler_X], open(filename, 'wb'))

      loaded_clf, scaler_X = pickle.load(open(filename, 'rb'))
      r_test = np.zeros((K,test_size),dtype=np.float)
      idx_test = np.zeros((N,test_size),dtype=np.float)
      for i in range(0,test_size):
          target_seed = i+training_size
          r_test[:,[i]], idx_test[0:N,[i]], dummy1, dummay2 = scenario(N,K,alpha,gamma,w,G,pmin,pmax,target_seed,sensor_seed,offgrid_coeff)
      Tx, Ty = index2target(idx_test,w,G)
      X_test = r_test; Y_test = idx_test
      X_test = scaler_X.transform(X_test.T).T; 
      #idx_hat = learner.classify(X_test)
      idx_hat = loaded_clf.predict(X_test.transpose()).transpose()
      Tx_hat, Ty_hat = index2target(idx_hat,w,G)
      MSE = calcMSE(Tx,Ty,Tx_hat,Ty_hat)
      return MSE, Tx, Ty, Tx_hat,Ty_hat

  def regression(N,K,alpha,gamma,w,G,pmin,pmax,sensor_seed,offgrid_coeff,training_size,test_size,train):
      filename = 'trained_regression.sav'
      gamma = gamma*1e-3;  ### dB/m
      beta = 10**(-gamma/10);  ### rain factor
      if train == True:
        r = np.zeros((K,training_size),dtype=np.float)
        Tx = np.zeros((N,training_size),dtype=np.float)
        Ty = np.zeros((N,training_size),dtype=np.float)
        target_seed = 1
        r, Tx, Ty = scenario(N,K,alpha,beta,training_size,pmax,pmin,w,sensor_seed,target_seed)
#
#        for i in range(0,training_size):
#            target_seed = i
#            r[:,[i]], dummy, Tx[0:N,[i]], Ty[0:N,[i]] = scenario(N,K,alpha,gamma,w,G,pmin,pmax,target_seed,sensor_seed,offgrid_coeff)
        X = r; Y = np.concatenate([Tx,Ty],axis=0)
        scaler_X = StandardScaler()
        scaler_Y = StandardScaler()
        scaler_X.fit(X.T)
        scaler_Y.fit(Y.T)
        X = scaler_X.transform(X.T).T
        Y = scaler_Y.transform(Y.T).T
      
        #clf = RandomForestRegressor(n_estimators=20, random_state=0)
        clf = tree.DecisionTreeRegressor()
        #clf = MLPRegressor(activation='logistic',max_iter=1000)
        clf.fit(X.transpose(),Y.transpose())
        pickle.dump([clf,scaler_X,scaler_Y], open(filename, 'wb'))

      loaded_clf, scaler_X, scaler_Y = pickle.load(open(filename, 'rb'))
      r_test = np.zeros((K,test_size),dtype=np.float)
      Tx = np.zeros((N,test_size),dtype=np.float)
      Ty = np.zeros((N,test_size),dtype=np.float)
      target_seed = training_size*N + K + 1
      r_test, Tx, Ty = scenario(N,K,alpha,beta,test_size,pmax,pmin,w,sensor_seed,target_seed)
#      for i in range(0,test_size):
#          target_seed = i+training_size
#          r_test[:,[i]], dummy, Tx[0:N,[i]], Ty[0:N,[i]] = scenario(N,K,alpha,gamma,w,G,pmin,pmax,target_seed,sensor_seed,offgrid_coeff)
      X_test = r_test; 
      X_test = scaler_X.transform(X_test.T).T; 
      z_hat = loaded_clf.predict(X_test.transpose()).transpose()
      z_hat = scaler_Y.inverse_transform(z_hat.T).T
      Tx_hat = z_hat[0:int(np.divide(N,2))+1,:]
      Ty_hat = z_hat[int(np.divide(N,2))+1:,:]
      MSE = calcMSE(Tx,Ty,Tx_hat,Ty_hat)
      return MSE, Tx, Ty, Tx_hat,Ty_hat

  def NN(N,K,alpha,gamma,w,G,pmin,pmax,sensor_seed,offgrid_coeff,training_size,test_size,train):
      filename = 'trained_regression_NN.sav'
      EPOCH = 1000
      if train == True:
        r = np.zeros((K,training_size),dtype=np.float)
        Tx = np.zeros((N,training_size),dtype=np.float)
        Ty = np.zeros((N,training_size),dtype=np.float)
        for i in range(0,training_size):
            target_seed = i
            r[:,[i]], dummy, Tx[0:N,[i]], Ty[0:N,[i]] = scenario(N,K,alpha,gamma,w,G,pmin,pmax,target_seed,sensor_seed,offgrid_coeff)
        X = r; Y = np.concatenate([Tx,Ty],axis=0)
        scaler_X = StandardScaler()
        scaler_Y = StandardScaler()
        scaler_X.fit(X.T)
        scaler_Y.fit(Y.T)
        X = scaler_X.transform(X.T).T
        Y = scaler_Y.transform(Y.T).T
      
        clf = learner.build_model(X,N)
        print(clf.summary())
        clf.fit(X.transpose(),Y.transpose(),epochs=EPOCH,validation_split=0.3)
        #pickle.dump([clf,scaler_X,scaler_Y], open(filename, 'wb'))

      #loaded_clf, scaler_X, scaler_Y = pickle.load(open(filename, 'rb'))
      loaded_clf = clf
      r_test = np.zeros((K,test_size),dtype=np.float)
      Tx = np.zeros((N,test_size),dtype=np.float)
      Ty = np.zeros((N,test_size),dtype=np.float)
      for i in range(0,test_size):
          target_seed = i+training_size
          r_test[:,[i]], dummy, Tx[0:N,[i]], Ty[0:N,[i]] = scenario(N,K,alpha,gamma,w,G,pmin,pmax,target_seed,sensor_seed,offgrid_coeff)
      X_test = r_test; 
      X_test = scaler_X.transform(X_test.T).T; 
      z_hat = loaded_clf.predict(X_test.transpose()).transpose()
      z_hat = scaler_Y.inverse_transform(z_hat.T).T
      Tx_hat = z_hat[0:int(np.divide(N,2))+1,:]
      Ty_hat = z_hat[int(np.divide(N,2))+1:,:]
      MSE = calcMSE(Tx,Ty,Tx_hat,Ty_hat)
      return MSE, Tx, Ty, Tx_hat,Ty_hat

  def build_model(train_dataset,N):
    model = keras.Sequential([
      layers.Dense(64, activation=tf.nn.relu, input_shape=(train_dataset.shape[0],)),
      layers.Dense(164, activation=tf.nn.relu),
      layers.Dense(164, activation=tf.nn.relu),
      layers.Dense(164, activation=tf.nn.relu),
      layers.Dense(164, activation=tf.nn.relu),
      layers.Dense(164, activation=tf.nn.relu),
      layers.Dense(164, activation=tf.nn.relu),
      layers.Dense(2*N, activation=tf.nn.sigmoid) ]) 
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mean_squared_error',optimizer=optimizer,metrics=['mean_absolute_error', 'mean_squared_error'])
    return model

