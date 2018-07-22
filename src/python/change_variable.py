
# coding: utf-8

# In[1]:

import time
import numpy as np
from math import log, exp
from scipy.linalg import sqrtm, inv, norm
from scipy.optimize import minimize


# ### Замена переменных :
# 
# $\begin{aligned}
# & \mathbf{w}'= \mathbf{A}{\mathbf{w}} - \mathbf{B}, \ \text{где} \\
# & \mathbf{A}=\left[\frac{1}{2}\mathbf{I} + C_{r}{\mathbf{X}}{\mathbf{X}}^{\text{T}}\right]^{\frac{1}{2}},\\
# & \mathbf{B}=C_r\left[\frac{1}{2}\mathbf{I} + C_{r}{\mathbf{X}}{\mathbf{X}}^{\text{T}}\right]^{-\frac{1}{2}}{\mathbf{X}}\mathbf{s},\\
# &\hat{\mathbf{X}} = (\mathbf{A}^{-1})^{\text{T}}\mathbf{X}.
# \end{aligned}$

# ### Для Ki:

# In[2]:

X_nat_train = np.loadtxt('../../data/pdb/refined/ki/X_nat_train.data')
X_train = np.loadtxt('../../data/pdb/refined/ki/X_train.data')


# In[3]:

X_nat_train.shape
X_train.shape


# In[4]:

start_time = time.time()

# Подсчет обратной матрицы A_inv и новой матрицы признаков X_changed_train
Cr = 100 # Коэффициент регуляризации
XXT = X_nat_train @ X_nat_train.T
I = np.identity(XXT.shape[0])
A = np.real(sqrtm(0.5 * I + Cr * XXT))
A_inv = inv(A)
X_changed_train = (A_inv.T @ X_train).T

print("--- %s seconds ---" % (time.time() - start_time))


# In[6]:

y_train = np.loadtxt('../../data/pdb/refined/ki/y_train.data')


# In[7]:

# Запись обратной матрицы (понадобится потом)
np.savetxt("../../data/pdb/refined/ki/A_inv.data", A_inv)

start_time = time.time()

# Запись обучающей выборки в файл в формате для liblinear
with open("../../data/pdb/refined/ki/X_changed_train.data", "w") as f:
    for i in range(X_changed_train.shape[0]):
        y_i = ("+1 " if y_train[i] == 1 else "-1 ")
        f.write(y_i)
        for j in range(X_changed_train.shape[1]):
            f.write(str(j + 1) + ":" + str(X_changed_train[i, j]) + " ")
        f.write("\n")
        
print("--- %s seconds ---" % (time.time() - start_time))


# ### Для Kd:

# In[8]:

X_nat_train = np.loadtxt('../../data/pdb/refined/kd/X_nat_train.data')
X_train = np.loadtxt('../../data/pdb/refined/kd/X_train.data')
y_train = np.loadtxt('../../data/pdb/refined/ki/y_train.data')


# In[9]:

X_nat_train.shape
X_train.shape


# In[10]:

start_time = time.time()

# Подсчет обратной матрицы A_inv и новой матрицы признаков X_changed_train 
Cr = 100 # Коэффициент регуляризации
XXT = X_nat_train @ X_nat_train.T
I = np.identity(XXT.shape[0])
A = np.real(sqrtm(0.5 * I + Cr * XXT))
A_inv = inv(A)
X_changed_train = (A_inv.T @ X_train).T

print("--- %s seconds ---" % (time.time() - start_time))


# In[11]:

# Запись обратной матрицы (понадобится потом)
np.savetxt("../../data/pdb/refined/kd/A_inv.data", A_inv)

start_time = time.time()

# Запись обучающей выборки в файл в формате для liblinear
with open("../../data/pdb/refined/kd/X_changed_train.data", "w") as f:
    for i in range(X_changed_train.shape[0]):
        y_i = ("+1 " if y_train[i] == 1 else "-1 ")
        f.write(y_i)
        for j in range(X_changed_train.shape[1]):
            f.write(str(j + 1) + ":" + str(X_changed_train[i, j]) + " ")
        f.write("\n")
        
print("--- %s seconds ---" % (time.time() - start_time))


# In[ ]:



