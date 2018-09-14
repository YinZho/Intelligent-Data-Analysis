

# data
import pandas as pd
import math
from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('wine-norm.csv').iloc[:,1:]



# add label "rate"
df.to_csv('wind-add-quality.csv', sep=',', encoding='utf-8')



M = 100
codebook_df = df.sample(n=M)

codebook_df = codebook_df.reset_index(drop=True)

dic = []
for i in range(10):
    for j in range(10):
        dic.append([j, i, 0])
index_df = pd.DataFrame(dic, columns=['index_x', 'index_y', 'index_z'])

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
    eta = eta_0 * math.exp(-(time / tau))
    sigma = sigma_0 * math.exp(-(time / nu))
    print(time)
    time = time + 1
    for index, row in df.iterrows():
        row = row.iloc[:11]
        dis_matrix = distance.cdist(np.matrix(row), np.matrix(codebook_df.iloc[:, :11]), 'euclidean')
        win = pd.DataFrame(dis_matrix).idxmin(1)
        row_bwin = np.matrix(row - codebook_df.iloc[win, :11])
        for ind, b_row in codebook_df.iterrows():
            h = math.exp(-math.pow(distance.euclidean(codebook_df.iloc[win, -3:], b_row.iloc[12:]), 2) / sigma ** 2)
            tmp = pd.DataFrame(np.matrix(row_bwin) * (eta * h) + np.matrix(codebook_df.iloc[ind, :11]))
            codebook_df.iloc[ind, 0] = tmp.iloc[0,0]
            codebook_df.iloc[ind, 1] = tmp.iloc[0,1]
            codebook_df.iloc[ind, 2] = tmp.iloc[0,2]
            codebook_df.iloc[ind, 3] = tmp.iloc[0,3]
            codebook_df.iloc[ind, 4] = tmp.iloc[0,4]
            codebook_df.iloc[ind, 5] = tmp.iloc[0,5]
            codebook_df.iloc[ind, 6] = tmp.iloc[0,6]
            codebook_df.iloc[ind, 7] = tmp.iloc[0,7]
            codebook_df.iloc[ind, 8] = tmp.iloc[0,8]
            codebook_df.iloc[ind, 9] = tmp.iloc[0,9]


codebook_df.to_csv('codebook_df.csv', sep=',', encoding='utf-8')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(codebook_df['x'], codebook_df['y'], codebook_df['z'], c='skyblue', s=10)
ax.view_init(30, 185)
plt.show()

