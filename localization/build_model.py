import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# # Budiling the model
def build_model(X,K,N,dim):
  Nnodes = 512
  Nhidden_layers = 5
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
  model1.add(Conv(16, kernel_size, strides=1,padding='valid',activation='tanh', input_shape=(X.shape[1],1)))
#  model1.add(Conv(4, kernel_size, strides=1,padding='valid', activation='tanh'))
#  model1.add(Conv(2, kernel_size, strides=1,padding='valid', activation='tanh'))
  model1.add(tf.keras.layers.Flatten())
  #model1.add(Dense(N * dim, kernel_initializer = initilization_method,activation='tanh'))
  y1 = model1(x1)

  l0 = layers.Flatten()(x1)



  l2 = layers.Concatenate(axis=1)([l0,y1])
  l2 = layers.Reshape([l2.shape[1],1])(l2)
  model2 = keras.models.Sequential()
  model2.add(Conv(16, kernel_size, strides=1,padding='valid',activation='tanh', input_shape=(l2.shape[1],1)))
  #model2.add(Conv(4, kernel_size, strides=1,padding='valid',activation='tanh'))
  #model2.add(Conv(2, kernel_size, strides=1,padding='valid',activation='tanh'))
  model2.add(tf.keras.layers.Flatten())
  #model2.add(Dense(N * dim, kernel_initializer = initilization_method,activation='tanh'))
  y2 = model2(l2)


  l3 = layers.Concatenate(axis=1)([l0,y2])
  l3 = layers.Reshape([l3.shape[1],1])(l3)
  model3 = keras.models.Sequential()
  model3.add(Conv(16, kernel_size, strides=1,padding='valid',activation='tanh', input_shape=(l3.shape[1],1)))
  #model3.add(Conv(4, kernel_size, strides=1,padding='valid',activation='tanh'))
  #model3.add(Conv(3, kernel_size, strides=1,padding='valid',activation='tanh'))
  model3.add(tf.keras.layers.Flatten())
  #model3.add(Dense(N * dim, kernel_initializer = initilization_method,activation='tanh'))
  y3 = model3(l3)

  l4 = layers.Concatenate(axis=1)([l0,y3])
  l4 = layers.Reshape([l4.shape[1],1])(l4)
  model4 = keras.models.Sequential()
  model4.add(Conv(16, kernel_size, strides=1,padding='valid',activation='tanh', input_shape=(l4.shape[1],1)))
  #model4.add(Conv(4, kernel_size, strides=1,padding='valid',activation='tanh'))
  #model4.add(Conv(4, kernel_size, strides=1,padding='valid',activation='tanh'))
  model4.add(tf.keras.layers.Flatten())
  model4.add(Dense(N * dim, kernel_initializer = initilization_method,activation='tanh'))
  #model4.add(Lambda(lambda x:keras.backend.clip(x, -1, 1)))
  #model4.add(Reshape(target_shape=(N,dim)))
  y4 = model4(l4)

  l5 = layers.Concatenate(axis=1)([l0,y4])
  model5 = keras.models.Sequential()
  model5.add(Dense(Nnodes, activation= 'selu', input_shape=(l5.shape[1],)))
  model5.add(Dense(Nnodes, activation= 'selu'))
  for l in range(0,Nhidden_layers):
      model5.add(Dense(Nnodes, activation = 'relu'))
  model5.add(Dense(N * dim))
  model5.add(Lambda(lambda x:keras.backend.clip(x, -1, 1)))
  model5.add(Reshape(target_shape=(N,dim)))
  y5 = model5(l5)

  model = keras.Model(x1,y5)
  return model
