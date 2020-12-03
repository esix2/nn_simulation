def myloss(y_true,y_pred):
    mse1 = keras.losses.mean_squared_error(y_true,y_pred)
    perm_mat = np.array([[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]],dtype=np.float32)
    y_perm = tf.matmul(y_true,perm_mat)
    mse2 = keras.losses.mean_squared_error(y_perm,y_pred)
    mse = tf.minimum(mse1,mse2)
    return mse 

