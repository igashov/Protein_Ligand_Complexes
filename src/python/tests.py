
# coding: utf-8

# In[1]:

import time
import numpy as np

from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# ### Для Ki

# In[2]:

start_time = time.time()

X_nat_train = np.loadtxt('../../data/pdb/refined/ki/X_nat_train.data')
s_train = np.loadtxt('../../data/pdb/refined/ki/s_train.data')
X_test = np.loadtxt('../../data/pdb/refined/ki/X_test.data')
s_test = np.loadtxt('../../data/pdb/refined/ki/s_test.data')

print("--- %s seconds ---" % (time.time() - start_time))


# In[6]:

with open("../../data/pdb/refined/ki/w.data", "r") as f:
    w_changed = np.array(f.read().split("\n")[6:-1], dtype=float)
w_changed = w_changed.reshape((w_changed.shape[0], 1))
w_changed.shape


# In[7]:

start_time = time.time()

Cr = 100
A_inv = np.loadtxt('../../data/pdb/refined/ki/A_inv.data')
B = Cr * A_inv @ X_nat_train @ s_train
B = B.reshape((B.shape[0], 1))
w = A_inv @ (w_changed + B)
w = w[:6440] # Отбрасываем лишние компоненты вектора w

print("--- %s seconds ---" % (time.time() - start_time))


# In[8]:

prediction = w.T @ X_test
print("Spearman: ", spearmanr(s_test, np.array(prediction)[0]))
print("Pearson: ", pearsonr(s_test, np.array(prediction)[0]))
print("R2: ", r2_score(s_test, np.array(prediction)[0]))
print("MSE: ", mean_squared_error(s_test, np.array(prediction)[0]))


# ### Для Kd

# In[10]:

start_time = time.time()

X_nat_train = np.loadtxt('../../data/pdb/refined/kd/X_nat_train.data')
s_train = np.loadtxt('../../data/pdb/refined/kd/s_train.data')
X_test = np.loadtxt('../../data/pdb/refined/kd/X_test.data')
s_test = np.loadtxt('../../data/pdb/refined/kd/s_test.data')

print("--- %s seconds ---" % (time.time() - start_time))


# In[11]:

with open("../../data/pdb/refined/kd/w.data", "r") as f:
    w_changed = np.array(f.read().split("\n")[6:-1], dtype=float)
w_changed = w_changed.reshape((w_changed.shape[0], 1))
w_changed.shape


# In[12]:

start_time = time.time()

Cr = 100
A_inv = np.loadtxt('../../data/pdb/refined/kd/A_inv.data')
B = Cr * A_inv @ X_nat_train @ s_train
B = B.reshape((B.shape[0], 1))
w = A_inv @ (w_changed + B)
w = w[:6440] # Отбрасываем лишние компоненты вектора w

print("--- %s seconds ---" % (time.time() - start_time))


# In[13]:

prediction = w.T @ X_test
print("Spearman: ", spearmanr(s_test, np.array(prediction)[0]))
print("Pearson: ", pearsonr(s_test, np.array(prediction)[0]))
print("R2: ", r2_score(s_test, np.array(prediction)[0]))
print("MSE: ", mean_squared_error(s_test, np.array(prediction)[0]))


# In[ ]:



