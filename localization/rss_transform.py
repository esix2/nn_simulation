import numpy as np

def rss_transform(r,b):    
    r_transformed = np.divide(np.log(r),np.log(b))    
    return r_transformed
