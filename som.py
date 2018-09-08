
# coding: utf-8

# In[10]:


# data
import pandas as pd
import math
from scipy.spatial import distance
import numpy as np

df = pd.read_csv('data.csv', names=['x', 'y', 'z'])
df


# In[27]:


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['x'], df['y'], df['z'], c='skyblue', s=10)
ax.view_init(30, 185)
plt.show()


# In[3]:


# Initialize plane
M = 100
sample = df.sample(n=M)
sample.to_csv('sample.csv', sep=',', encoding='utf-8')

# load sample

codebook_df = pd.read_csv('sample.csv').iloc[:, 1:]
codebook_df


# In[4]:


# generate table
table = []
for i in range(10):
    for j in range(10):
        table.append([j, i, 0])
index_df = pd.DataFrame(table, columns=['index_x', 'index_y', 'index_z'])
index_df


# In[7]:


codebook_df = pd.merge(codebook_df, index_df, right_index=True, left_index=True, how='outer')



    
# learning rate initialize
eta_0 = 1
sigma_0 = 1

tau = 1
nu = 1

# itrative time
time = 1
eta = 1

while eta > 0.00001:
    eta = eta_0 * math.exp(-(time/tau))
    sigma = sigma_0 * math.exp(-(time/nu))
    print(time)
    time = time + 1
    for index, row in df.iterrows():

        dis_matrix = distance.cdist(np.matrix(row), np.matrix(codebook_df.iloc[:, :3]), 'euclidean')
        win = pd.DataFrame(dis_matrix).idxmin(1)
        row_bwin = np.matrix(row - codebook_df.iloc[win, :3])
        for ind, b_row in codebook_df.iterrows():
            h = math.exp(-math.pow(distance.euclidean(codebook_df.iloc[win, -3:], b_row.iloc[3:]), 2) / sigma ** 2)
            tmp =  pd.DataFrame(np.matrix(row_bwin) * (eta * h) + np.matrix(codebook_df.iloc[ind,:3]))
            codebook_df.loc[ind, 'x'] = tmp.iloc[0,0]
            codebook_df.loc[ind, 'y'] = tmp.iloc[0,1]
            codebook_df.loc[ind, 'z'] = tmp.iloc[0,2]
codebook_df.to_csv('codebook_df.csv', sep=',', encoding='utf-8')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(codebook_df['x'], codebook_df['y'], codebook_df['z'], c='skyblue', s=10)
ax.view_init(30, 185)
plt.show()



