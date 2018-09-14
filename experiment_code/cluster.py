
# coding: utf-8

# In[1]:


# cluster

import pandas as pd
from scipy.spatial import distance
import numpy as np
import math


def cluster(i, M):
    # load Data
    path = "wine-norm.csv"
    df = pd.read_csv(path)
    df = df.iloc[:, 1:12]
    df

    
    # M - the number of codebook

    # Random pick data
    codebook_vectors = df.sample(n=M)


    # Iterate
    eta_0 = 1
    tao = 1

    time = 1
    eta = 1
    while eta > 0.00001:
        eta = eta_0 * math.exp(-(time/tao))
        time = time + 1
        for index, row in df.iterrows():
            dis_matrix = distance.cdist(np.matrix(row), np.matrix(codebook_vectors), 'euclidean')
            # closest codebook vector index
            win = pd.DataFrame(dis_matrix).idxmin(1)
            b_old = codebook_vectors.iloc[win, :]
            codebook_vectors.iloc[win, :] = b_old + eta * (row - b_old)
    txt_path = './test/' + str(M) + '-' + str(i) + '-' + 'cv' +'.txt'
    np.savetxt(txt_path, codebook_vectors)
    codebook_vectors.to_csv('codebook_vectors.csv', sep=',', encoding='utf-8')

    # add label type

    def label_type(row):
         dis_matrix = distance.cdist(np.matrix(row), np.matrix(codebook_vectors), 'euclidean')
         win = pd.DataFrame(dis_matrix).idxmin(1)
         return win
    df['type'] = df.apply(lambda row: label_type(row), axis=1)
    df.to_csv('out.csv', sep=',', encoding='utf-8')

    # error

    error = 0
    df_ = pd.read_csv("./out.csv")
    df_b = pd.read_csv("./codebook_vectors.csv")
    df_ = df_.iloc[:, 1:13]
    for index, row in df_.iterrows():
        type = int(row[11])
        error += float(distance.cdist(np.matrix(row.iloc[:11]), np.matrix(df_b.iloc[type,1:]), 'euclidean'))
    file.write(str(M)+", "+str(error))
    print(str(M)+", "+str(error))



file = open('./test/test-output.txt', 'w')
# for j in range(1, 11):
#     for i in range(0,5):
#         cluster(i, j)
cluster(0, 3)
file.close()

# E(M)-M How the codebook vectors are placed for M = 1 ~ 10 
# df = pd.read_csv('cluster-error.csv', names=['error'])
# import matplotlib.pyplot as plt
# plt.plot([i for i in range(1, 11)], df)
# plt.plot([i for i in range(1, 11)], df,'s') 
# plt.suptitle('W - Error(W)')
# plt.savefig('E(M)-M.png')

