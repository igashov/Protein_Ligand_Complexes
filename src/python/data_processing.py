
# coding: utf-8

# ## Считываем данные из PDBBind (refined dataset)

# In[1]:

import array, struct, sys, os, tqdm
import numpy as np

def read_binaries(path_binfiles):
    result = {}
    for binfile in tqdm.tqdm(os.listdir(path_binfiles)):
        if (binfile.split('.')[1] != 'bin'):
            continue
        pdbcode = binfile.split('.')[0]                     # name of file (pdbcode)
        F = open('{0}/{1}'.format(path_binfiles, binfile), 'rb')
        n_decoys = struct.unpack('i', F.read(4))[0]         # number of decoys (=19 for this dataset)
        dimension = struct.unpack('i', F.read(4))[0]        # data dimensionality (23 protein types x 40 ligand types x 7 bins for this dataset)
        res = []
        for i in range(n_decoys):
            label = struct.unpack('d', F.read(8))[0]        # label (1 for native, -1 for non-native)
            data = array.array('d')                         
            data.fromfile(F, dimension)                     # feature vector (histograms, can be represented as a 23x40x7 matrix) 
            res.append([label, data])
        result[pdbcode] = res
        F.close()
    return result


# In[2]:

result = read_binaries('../../data/pdb/general-no2013_t14_t3_l7.0_g1.0_r1.0')


# In[3]:

with open('../../data/pdb/affinity_data_refined.csv', 'r') as f:
    data = f.read().split('\n')
    data = data[1:-1]


# In[4]:

datasets = [
    {'name': d.split(',')[0], 'value': d.split(',')[1], 'type': d.split(',')[3]}
    for d in data
]


# In[5]:

Kd_values = []
Ki_values = []
for d in datasets:
    if d['type'] == 'Kd':
        Kd_values.append(d)
    else:
        Ki_values.append(d)


# In[6]:

Kd_data = []
for item in Kd_values:
    Kd_data.append([item['value']] + result[item['name']])


# In[7]:

Ki_data = []
for item in Ki_values:
    Ki_data.append([item['value']] + result[item['name']])


# ## Записываем матрицы признаков и ответов для refined 

# In[8]:

import time
import numpy as np
from math import log, exp
from scipy.linalg import sqrtm, inv, norm
from scipy.optimize import minimize


# ## Известны значения Ki
# #### Путь: 
# $\texttt{data/pdb/refined/ki/}$
# 
# #### Файлы: 
# 
# $\texttt{X_nat_train.data}$ - обучающая выборка из нативных комплексов (признаки)
# 
# $\texttt{s_train.data}$ - обучающая выборка из нативных комплексов (аффинности)
# 
# $\texttt{X_train.data}$ - обучающая выборка всех комплексов (признаки)
# 
# $\texttt{y_train.data}$ - обучающа выборка всех комплексов (позы)
# 
# $\texttt{X_test.data}$ - тестовая выборка всех комплексов (признаки)
# 
# $\texttt{y_test.data}$ - тестовая выборка всех комплексов (позы)

# In[9]:

start_time = time.time()
data = Ki_data
train = data[:int(len(data) * 0.6)]
test = data[int(len(data) * 0.6):]

# Матрица признаков (для которых аффинности известны)
X_nat_train = []
for i, t in enumerate(train):
    # Повышаем размерность (за счет вектора сдвигов b)
    additional = np.zeros(len(train))
    additional[i] = -1
    X_nat_train.append(np.append(t[1][1], additional))
X_nat_train = np.matrix(X_nat_train).T

# Столбец значений свободной энергии
s_train = np.matrix([
    float(t[0])
    for t in train
]).T
print('--- %s seconds ---' % (time.time() - start_time))


# In[10]:

print(X_nat_train.shape)
print(s_train.shape)


# In[11]:

np.savetxt('../../data/pdb/refined/ki/X_nat_train.data', X_nat_train)
np.savetxt('../../data/pdb/refined/ki/s_train.data', s_train)


# In[12]:

start_time = time.time()
X_train = []
for i, t in enumerate(train):
    # Повышаем размерность (за счет вектора сдвигов b)
    additional = np.zeros(len(train))
    additional[i] = -1
    for pose in t[1:]:
        X_train.append(np.append(pose[1], additional))
        
X_train = np.matrix(X_train).T

y_train = []
for t in train:
    for pose in t[1:]:
        y_train.append(pose[0])

y_train = np.matrix(y_train).T
print("--- %s seconds ---" % (time.time() - start_time))


# In[13]:

print(X_train.shape)
print(y_train.shape)


# In[14]:

np.savetxt('../../data/pdb/refined/ki/X_train.data', X_train)
np.savetxt('../../data/pdb/refined/ki/y_train.data', y_train)


# In[35]:

start_time = time.time()
X_test = []
for t in test:
    X_test.append(t[1][1])
        
X_test = np.matrix(X_test).T
        
s_test = np.matrix([
    float(t[0])
    for t in test
]).T

print("--- %s seconds ---" % (time.time() - start_time))


# In[36]:

print(X_test.shape)
print(s_test.shape)


# In[37]:

np.savetxt('../../data/pdb/refined/ki/X_test.data', X_test)
np.savetxt('../../data/pdb/refined/ki/s_test.data', s_test)


# ## Известны значения Kd
# #### Путь: 
# $\texttt{data/pdb/refined/kd/}$
# 
# #### Файлы: 
# 
# $\texttt{X_nat_train.data}$ - обучающая выборка из нативных комплексов (признаки)
# 
# $\texttt{s_train.data}$ - обучающая выборка из нативных комплексов (аффинности)
# 
# $\texttt{X_train.data}$ - обучающая выборка всех комплексов (признаки)
# 
# $\texttt{y_train.data}$ - обучающа выборка всех комплексов (позы)
# 
# $\texttt{X_test.data}$ - тестовая выборка всех комплексов (признаки)
# 
# $\texttt{y_test.data}$ - тестовая выборка всех комплексов (позы)

# In[39]:

start_time = time.time()
data = Kd_data
train = data[:int(len(data) * 0.6)]
test = data[int(len(data) * 0.6):]

# Матрица признаков (для которых аффинности известны)
X_nat_train = []
for i, t in enumerate(train):
    # Повышаем размерность (за счет вектора сдвигов b)
    additional = np.zeros(len(train))
    additional[i] = -1
    X_nat_train.append(np.append(t[1][1], additional))
X_nat_train = np.matrix(X_nat_train).T

# Столбец значений свободной энергии
s_train = np.matrix([
    float(t[0])
    for t in train
]).T
print('--- %s seconds ---' % (time.time() - start_time))

print(X_nat_train.shape)
print(s_train.shape)

np.savetxt('../../data/pdb/refined/kd/X_nat_train.data', X_nat_train)
np.savetxt('../../data/pdb/refined/kd/s_train.data', s_train)


# In[40]:

start_time = time.time()
X_train = []
for i, t in enumerate(train):
    # Повышаем размерность (за счет вектора сдвигов b)
    additional = np.zeros(len(train))
    additional[i] = -1
    for pose in t[1:]:
        X_train.append(np.append(pose[1], additional))
        
X_train = np.matrix(X_train).T

y_train = []
for t in train:
    for pose in t[1:]:
        y_train.append(pose[0])

y_train = np.matrix(y_train).T
print("--- %s seconds ---" % (time.time() - start_time))

print(X_train.shape)
print(y_train.shape)

np.savetxt('../../data/pdb/refined/kd/X_train.data', X_train)
np.savetxt('../../data/pdb/refined/kd/y_train.data', y_train)


# In[41]:

start_time = time.time()
X_test = []
for t in test:
    X_test.append(t[1][1])
        
X_test = np.matrix(X_test).T
        
s_test = np.matrix([
    float(t[0])
    for t in test
]).T

print("--- %s seconds ---" % (time.time() - start_time))

print(X_test.shape)
print(s_test.shape)

np.savetxt('../../data/pdb/refined/kd/X_test.data', X_test)
np.savetxt('../../data/pdb/refined/kd/s_test.data', s_test)


# Записали все исходные матрицы признаков, поз и ответов для Ki и Kd в отдельные файлы. Далее надо сделать замену переменных, записать новые признаки и считать скоринговый вектор.

# In[ ]:



