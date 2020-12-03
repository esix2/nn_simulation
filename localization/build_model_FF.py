import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# # Budiling the model
def build_model(X,K,N,dim):
  Nnodes = 2048
  Nhidden_layers = 2
  Dense = tf.keras.layers.Dense
  Reshape = tf.keras.layers.Reshape
  Dropout = tf.keras.layers.Dropout
  Lambda = tf.keras.layers.Lambda
  Conv = keras.layers.Conv1D

  initilization_method = 'random_uniform' #'random_uniform' ,'random_normal','TruncatedNormal' ,'glorot_uniform', 'glorot_nomral', 'he_normal', 'he_uniform'

  x1 = layers.Input((X.shape[1],1))
  model1 = keras.models.Sequential()
  filters = 4
  kernel_size =4
  X = X.reshape(X.shape[0],-1,1)
  model1.add(Conv(16, kernel_size, strides=1,padding='valid',
                               activation='tanh', input_shape=(X.shape[1],1)))
  model1.add(Conv(4, kernel_size, strides=1,padding='valid',
                               activation='tanh'))
  model1.add(Conv(2, kernel_size, strides=1,padding='valid',
                               activation='tanh'))
  model1.add(tf.keras.layers.Flatten())
  model1.add(Dense(N * dim, kernel_initializer = initilization_method,activation='tanh'))
  y1 = model1(x1)

  l0 = layers.Flatten()(x1)



  l2 = layers.Concatenate(axis=1)([l0,y1])
  l2 = layers.Reshape([l2.shape[1],1])(l2)
  model2 = keras.models.Sequential()
  model2.add(Conv(8, kernel_size, strides=1,padding='valid',activation='tanh', input_shape=(l2.shape[1],1)))
  model2.add(Conv(4, kernel_size, strides=1,padding='valid',activation='tanh'))
  model2.add(Conv(2, kernel_size, strides=1,padding='valid',activation='tanh'))
  model2.add(tf.keras.layers.Flatten())
# model2.add(Dense(N * dim, kernel_initializer = initilization_method,activation='tanh'))
  y2 = model2(l2)

#  l2 = layers.Concatenate(axis=1)([l0,y1])
#  model2 = keras.models.Sequential()
#  model2.add(Dense(Nnodes, activation= 'selu', input_shape=(l2.shape[1],)))
#  #model2.add(Dense(Nnodes, activation= 'selu'))
#  for l in range(0,Nhidden_layers):
#      model2.add(Dense(Nnodes, activation = 'relu'))
#  model2.add(Dense(N * dim))
#  model2.add(Lambda(lambda x:keras.backend.clip(x, -1, 1)))
#  #model2.add(Reshape(target_shape=(N,dim)))
#  y2 = model2(l2)


  l3 = layers.Concatenate(axis=1)([l0,y2])
  model3 = keras.models.Sequential()
  model3.add(Dense(Nnodes, activation= 'selu', input_shape=(l3.shape[1],)))
  #model3.add(Dense(Nnodes, activation= 'selu'))
  for l in range(0,Nhidden_layers):
      model3.add(Dense(Nnodes, activation = 'relu'))
  model3.add(Dense(N * dim))
  model3.add(Lambda(lambda x:keras.backend.clip(x, -1, 1)))
  #model3.add(Reshape(target_shape=(N,dim)))
  y3 = model3(l3)

  l4 = layers.Concatenate(axis=1)([l0,y3])
  model4 = keras.models.Sequential()
  model4.add(Dense(Nnodes, activation= 'selu', input_shape=(l4.shape[1],)))
  #model4.add(Dense(Nnodes, activation= 'selu'))
  for l in range(0,Nhidden_layers):
      model4.add(Dense(Nnodes, activation = 'relu'))
  model4.add(Dense(N * dim))
  model4.add(Lambda(lambda x:keras.backend.clip(x, -1, 1)))
  #model4.add(Reshape(target_shape=(N,dim)))
  y4 = model4(l4)

  l5 = layers.Concatenate(axis=1)([l0,y4])
  model5 = keras.models.Sequential()
  model5.add(Dense(Nnodes, activation= 'selu', input_shape=(l5.shape[1],)))
  #model5.add(Dense(Nnodes, activation= 'selu'))
  for l in range(0,Nhidden_layers):
      model5.add(Dense(Nnodes, activation = 'relu'))
  model5.add(Dense(N * dim))
  model5.add(Lambda(lambda x:keras.backend.clip(x, -1, 1)))
  #model5.add(Reshape(target_shape=(N,dim)))
  y5 = model5(l5)

  l6 = layers.Concatenate(axis=1)([l0,y5])
  model6 = keras.models.Sequential()
  model6.add(Dense(Nnodes, activation= 'selu', input_shape=(l6.shape[1],)))
  #model6.add(Dense(Nnodes, activation= 'selu'))
  for l in range(0,Nhidden_layers):
      model6.add(Dense(Nnodes, activation = 'relu'))
  model6.add(Dense(N * dim))
  model6.add(Lambda(lambda x:keras.backend.clip(x, -1, 1)))
  #model6.add(Reshape(target_shape=(N,dim)))
  y6 = model6(l6)

  l7 = layers.Concatenate(axis=1)([l0,y6])
  model7 = keras.models.Sequential()
  model7.add(Dense(Nnodes, activation= 'selu', input_shape=(l7.shape[1],)))
  model7.add(Dense(Nnodes, activation= 'selu'))
  for l in range(0,Nhidden_layers):
      model7.add(Dense(Nnodes, activation = 'relu'))
  model7.add(Dense(N * dim))
  model7.add(Lambda(lambda x:keras.backend.clip(x, -1, 1)))
  #model7.add(Reshape(target_shape=(N,dim)))
  y7 = model7(l7)

  l8 = layers.Concatenate(axis=1)([l0,y7])
  model8 = keras.models.Sequential()
  model8.add(Dense(Nnodes, activation= 'selu', input_shape=(l8.shape[1],)))
  model8.add(Dense(Nnodes, activation= 'selu'))
  for l in range(0,Nhidden_layers):
      model8.add(Dense(Nnodes, activation = 'relu'))
  model8.add(Dense(N * dim))
  model8.add(Lambda(lambda x:keras.backend.clip(x, -1, 1)))
  model8.add(Reshape(target_shape=(N,dim)))
  y8 = model8(l8)

  model = keras.Model(x1,y8)
  return model
