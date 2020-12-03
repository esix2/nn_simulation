import numpy as np
import scipy as cp

def scenario(N,K,dim,alpha,beta,M,pmax,pmin,w,sensor_seed,target_seed):
    np.random.seed(target_seed)
    #np.random.seed()
    
#    Tx = w*(-1+2*np.random.rand(N,M))
#    Ty = w*(-1+2*np.random.rand(N,M))
#
##    Tx[0,:] = w*(0.5+np.random.rand(1,M))
##    Ty[0,:] = w*(0.5+np.random.rand(1,M))
##    Tx[1,:] = -w*(0.5+np.random.rand(1,M))
##    Ty[1,:] = -w*(0.5+np.random.rand(1,M))
##    if N > 2:
##      Tx[2,:] = -w*(np.random.rand(1,M))
##      Ty[2,:] = w*(np.random.rand(1,M))
##      if N > 3:
##        Tx[3,:] = w*(np.random.rand(1,M))
##        Ty[3,:] = -w*(np.random.rand(1,M))
#
#    p = pmin+(pmax-pmin)*np.random.rand(N,M)
#
#    np.random.seed(sensor_seed)
#    Sx = w*(-1+2*np.random.rand(K,1,1))
#    Sy = w*(-1+2*np.random.rand(K,1,1))
#
#    SSx = np.repeat(Sx, N, axis=1)
#    SSx = np.repeat(SSx, M, axis=2)
#    SSy = np.repeat(Sy, N, axis=1)
#    SSy = np.repeat(SSy, M, axis=2)
#
#    TTx = np.expand_dims(Tx,axis=0)
#    TTx = np.repeat(TTx, K, axis=0)
#    TTy = np.expand_dims(Ty,axis=0)
#    TTy = np.repeat(TTy, K, axis=0)
#    pp = np.expand_dims(p,axis=0)
#    pp = np.repeat(pp, K, axis=0)
#    D = np.power(np.power(SSx-TTx,2)+np.power(SSy-TTy,2), 0.5)
#    #r = np.multiply((np.power(D,-alpha)*np.power(beta,D)),p).sum(axis=1)
#    r = ((np.power(D,-alpha)*np.power(beta,D))*p).sum(axis=1)

#    Dcenter = np.power(np.power(-1000-Tx,2)+np.power(-1000-Ty,2), 0.5)
#    idx = np.argsort(Dcenter, axis=0, order=None)
#    Tx_sorted = np.array(list(map(lambda i : Tx[idx[:,i],i], range(0,M)))).T
#    Ty_sorted = np.array(list(map(lambda i : Ty[idx[:,i],i], range(0,M)))).T
#    Dcenter_sorted = np.power(np.power(Tx_sorted,2)+np.power(Ty_sorted,2), 0.5)
#    Tx = Tx_sorted
#    Ty = Ty_sorted

    np.random.seed(sensor_seed)
    pos_sensor = w*(-1+2*np.random.rand(1,1,K,dim))

    np.random.seed(target_seed)
    pos_target = w*(-1+2*np.random.rand(M,N,1,dim))

#    pos_target = w*(np.random.rand(M,1,1,dim))
#    pos_target = np.concatenate([pos_target,w*(-1*np.random.rand(M,1,1,dim))],axis=1)

    pow_target = pmin+(pmax-pmin)*np.random.rand(M,N,1)
    pow_target = pmin+(pmax-pmin)*np.random.rand(M,N,1)
    distance = np.linalg.norm(pos_target  - pos_sensor , ord = 2, axis = 3)
    rss = np.sum(pow_target*(distance ** -alpha), axis = 1)
    return rss, pos_target, pos_sensor


#def scenario(N,K,alpha,gamma,w,G,pmin,pmax,target_seed,sensor_seed,offgrid_coeff):
#    gamma = gamma*1e-3;  ### dB/m
#    beta = 10**(-gamma/10);  ### rain factor
#    Tx, Ty, p , i, j= createTarget(N,w,G,pmin,pmax,target_seed,offgrid_coeff)
#    Sx, Sy = createSensor(K,w,sensor_seed)
#    D = calcDist(Tx, Ty, Sx, Sy)
#    r = calcRSS(D,p,alpha,beta)
#    idx = i*G+j
#    return r, idx, Tx, Ty#Sx, Sy, D
#def createTarget(N,w,G,pmin,pmax,target_seed,offgrid_coeff):
#    np.random.seed(target_seed)
#    delta = 2*w/(G-1)
#    ###if h
#    i = np.random.randint(0,G-1,(2,1),dtype=int)
#    j = np.random.randint(0,G-1,(2,1),dtype=int)
#    Tx = -w+i*delta
#    Ty = -w+j*delta
#    dx = offgrid_coeff*delta*(-1+2*np.random.rand(N,1))
#    dy = offgrid_coeff*delta*(-1+2*np.random.rand(N,1))
#    Tx = Tx + dx
#    Ty = Ty + dy
##    Tx = -w+2*w*np.random.rand(N,1)
##    Ty = -w+2*w*np.random.rand(N,1)
#    p = pmin+(pmax-pmin)*np.random.rand(N,1)
#    return Tx,Ty, p, i ,j
#
#def createSensor(K,w,sensor_seed):
#    np.random.seed(sensor_seed)
#    Sx = -w+2*w*np.random.rand(K,1)
#    Sy = -w+2*w*np.random.rand(K,1)
#    return Sx,Sy
#
#def calcDist(Tx,Ty,Sx,Sy):
#    N = np.size(Tx)
#    K = np.size(Sx)
#    TTx = np.repeat(Tx.transpose(),K,axis=0)
#    TTy = np.repeat(Ty.transpose(),K,axis=0)
#    SSx = np.repeat(Sx,N,axis=1)
#    SSy = np.repeat(Sy,N,axis=1)
#    D = np.power(np.power(TTx-SSx,2)+np.power(TTy-SSy,2),0.5)
#    return D
#
#def calcRSS(D,p,alpha,beta):
#    r = (np.power(D,-alpha)*np.power(beta,D)).dot(p)
#    return r
#
#def index2target(idx,w,G):
#    delta = 2*w/(G-1)
#    N, J = np.shape(idx)
#    Tx = np.zeros((N,J),dtype=np.float)
#    Ty = np.zeros((N,J),dtype=np.float)
#    for n in range(0,N):
#        for j in range(0,J):
#            row = np.floor(idx[n][j]/G)
#            col = idx[n][j]-row*G
#            Tx[n][j] = -w+row*delta
#            Ty[n][j] = -w+col*delta
#    return Tx, Ty

