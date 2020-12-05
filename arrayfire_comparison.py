import arrayfire as af
import numpy as np
import torch as tr
from numpy import linalg as npla
from tensorflow import linalg as tfla
from numpy import matmul as matmul
from timeit import default_timer as timer
import pandas as pd

A = np.random.rand(200,200)
d1 = 2000; d2 = d1; d3 = d1

A = af.random.randn(d1,d2); 
B = af.random.randn(d2,d3); 
start = timer(); C = af.matmul(A,B); end = timer(); af_t1 = 1000*(end-start);
start = timer(); C = af.inverse(A); end = timer(); af_t2 = 1000*(end-start);
print(' ArrayFire:\t Matrix multplication {:.4f} ms, \t Matrix inverse {:.4f} ms'.format(af_t1,af_t2))

A = np.random.randn(d1,d2); 
B = np.random.randn(d2,d3); 
start = timer(); C = np.matmul(A,B); end = timer(); np_t1 =1000*(end-start);
start = timer(); C = npla.inv(A); end = timer(); np_t2 = 1000*(end-start);
print('     Numpy:\t Matrix multplication {:.4f} ms, \t Matrix inverse {:.4f} ms'.format(np_t1,np_t2))



A = np.random.randn(d1,d2); 
B = np.random.randn(d2,d3); 
start = timer(); C = tfla.matmul(A,B); end = timer(); tf_t1 = 1000*(end-start);
start = timer(); C = tfla.inv(A); end = timer(); tf_t2 = 1000*(end-start);
print('TensorFlow:\t Matrix multplication {:.4f} ms, \t Matrix inverse {:.4f} ms'.format(tf_t1,tf_t2))


A = torch.from_numpy(A)
B = torch.from_numpy(B)
start = timer(); C = tr.matmul(A,B); end = timer(); tr_t1 = 1000*(end-start);
start = timer(); C = tr.inverse(A); end = timer(); tr_t2 = 1000*(end-start);
print('   PyTorch:\t Matrix multplication {:.4f} ms, \t Matrix inverse {:.4f} ms'.format(tr_t1,tr_t2))

df = pd.DataFrame(np.array([[af_t1, af_t2], [np_t1, np_t2], [tf_t1, tf_t2], [tr_t1, tr_t2]]),
                  columns=['Matrix Multiplication (ms)', 'Matrix Inverse (ms)'],
                   index=['ArrayFire', 'Numpy', 'TensorFlow', 'PyTorch'])
df.to_html("arrayfire_comparison.html")
