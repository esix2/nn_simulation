### Test
import os
import numpy as np
import numpy.matlib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.preprocessing import StandardScaler, MinMaxScaler

import pandas as pd
import matplotlib.pyplot as plt
from scenario import scenario


import pickle

from calcMSE import calcMSE
from statsmodels.distributions.empirical_distribution import ECDF

from rss_transform import rss_transform
from tensorflow.keras.models import model_from_json
from build_model_FF import build_model

M = int(1e4)
N = 2
K = 20
dim = 2
alpha = 2
beta = 1
w = 1000
target_seed = 1723871923
sensor_seed = 1
pmin = 1
pmax = 1
scaler_file = 'minmax_scaler.sav'
log_base = np.exp(1)

if_y_scale = False
if_x_scale = False
if_sensor_scale = False
if_y_scale = True
if_x_scale = True
if_sensor_scale = True
scaling_type = "maxmin"
scaling_type = "multiply"
scaler_file = 'minmax_scaler.sav'
scaler_X, scaler_Y = pickle.load(open(scaler_file, 'rb'))




rss_test, pos_target_test, pos_sensor = scenario(N,K,dim,alpha,beta,M,pmax,pmin,w,sensor_seed,target_seed)

Xt = rss_transform(rss_test,log_base)
if if_x_scale ==True:
    Xt = scaler_X.transform(Xt)



# load json and create model
#    json_file = open('model.json', 'r')
#    loaded_model_json = json_file.read()
#    json_file.close()
#    loaded_model = model_from_json(loaded_model_json)
loaded_model = build_model(Xt,K,N,dim)
# load weights into new model
filepath = "saved-model.hdf5"
loaded_model.load_weights("saved-model.hdf5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

Xt = Xt.reshape(Xt.shape[0],-1,1)
result = loaded_model.predict(Xt)
result = np.transpose(result, axes=(0,2,1))
result = result.reshape(M,N*dim)
if if_y_scale == True:
    if scaling_type == "maxmin":
        result = scaler_Y.inverse_transform(result)
    elif scaling_type == "multiply":
            result = result*w
            pos_sensor = pos_sensor*w
Tx_hat = result[:,:N]
Ty_hat = result[:,N:]

Yt = np.squeeze(pos_target_test,axis=2)
Tx_test = Yt[:,:,0]
Ty_test = Yt[:,:,1]

MSE = calcMSE(Tx_test.T,Ty_test.T,Tx_hat.T,Ty_hat.T)

mse = MSE.reshape(1,-1)[0]
print("Mean error: "+str(np.mean(mse)))
print("Mean error: "+str(np.var(mse)))

cdf = ECDF(tuple(mse))
plt.semilogx(cdf.x, 1-cdf.y)
plt.xlim(1e-1,np.max(cdf.x))
plt.ylim(0,1e0)
plt.grid(True)


plt.figure()
plt.scatter(Tx_test[:,:],Ty_test[:,:])
plt.scatter(Tx_hat[:,:],Ty_hat[:,:])
# plt.scatter(pos_sensor[:,:,:,0],pos_sensor[:,:,:,1])
# print(Y[:1,:])
# plt.xlim([-w,w])
# plt.ylim([-w,w])

plt.figure()
#    plt.plot(Tx_test[:,0]-Ty_test[:,0],'-o')
#    plt.plot(Tx_hat[:,0]-Ty_hat[:,0],'-x')
plt.plot(np.abs(Tx_hat-Ty_hat)-np.abs(Tx_test-Ty_test),'-r')
# plt.plot(Tx_test[:,1]-Ty_test[:,1])
# plt.plot(Tx_hat[:,1]-Ty_hat[:,1])
plt.xlim([0,M-1])
# plt.ylim([-w,w])
for e in [0.1,1,10,100,1000]:
  prob=(mse > e).sum()/M*100
  print("Pr(e>"+str(e)+"m)= "+str(prob)+"%")
