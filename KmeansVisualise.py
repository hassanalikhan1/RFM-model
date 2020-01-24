import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from copy import deepcopy
from scipy.spatial import distance
import math
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)


##Data read
retail = pd.read_excel('Online Retail.xlsx')

print (len(retail))
retail = retail[pd.notnull(retail['CustomerID'])]
print (len(retail))

##DATA PREPROCESSING
#remove all the negative values
retail = retail[retail.Quantity >= 0]

##CREATE A TABLE OF RECENT,MONETARY AND FREQUENCY

retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'])
retail['TotalRevenue'] = retail.UnitPrice * retail.Quantity
recentDate=dt.datetime(2011,12,10)

rfmTable = retail.groupby('CustomerID').agg({'InvoiceDate': lambda x: (recentDate - x.max()).days, 
                                           'InvoiceNo': lambda x: len(x), 
                                           'TotalRevenue': lambda x: x.sum()})

rfmTable['InvoiceDate'] = rfmTable['InvoiceDate'].astype(int)
rfmTable.rename(columns={'InvoiceDate': 'recency', 
                         'InvoiceNo': 'frequency', 
                         'TotalRevenue': 'monetary'}, inplace=True)

print (rfmTable.head())

##CREATE A TBALE OF SCORE BASED ON THE VALUES OF RMF: from 1-5
f_score = []
m_score = []
r_score = []

columns = ['frequency', 'monetary']
scores_str = ['f_score', 'm_score']
scores = [f_score, m_score]


for n in range(len(columns)):
    # Order by column
    rfmTable = rfmTable.sort_values(columns[n], ascending=False)
    
    # Create new index
    refs = np.arange(1,4340)
    rfmTable['refs'] = refs
    
    # Add score
    for i, row in rfmTable.iterrows():
        if row['refs'] <= 866:
            scores[n].append(5)
        elif row['refs'] > 866 and row['refs'] <= 866*2:
            scores[n].append(4)
        elif row['refs'] > 866*2 and row['refs'] <= 866*3:
            scores[n].append(3)
        elif row['refs'] > 866*3 and row['refs'] <= 866*4:
            scores[n].append(2)
        else: 
            scores[n].append(1)

    # Create f_score column
    rfmTable[scores_str[n]] = scores[n]
    
# For recency, we do the opposite: most recents are better, so we order as ascending
rfmTable = rfmTable.sort_values('recency', ascending=True)
    
# Recreate index
refs = np.arange(1,4340)
rfmTable['refs'] = refs
    
# Add score
for i, row in rfmTable.iterrows():
    if row['refs'] <= 866:
        r_score.append(5)
    elif row['refs'] > 866 and row['refs'] <= 866*2:
        r_score.append(4)
    elif row['refs'] > 866*2 and row['refs'] <= 866*3:
        r_score.append(3)
    elif row['refs'] > 866*3 and row['refs'] <= 866*4:
        r_score.append(2)
    else: 
        r_score.append(1)

# Create f_score column
rfmTable['r_score'] = r_score
    
rfmTableScores = rfmTable.drop(['frequency', 'monetary', 'recency', 'refs'], axis=1)


#IMPLEMENT KMEANS CLUSTERING

X=np.array(list(zip(rfmTableScores['f_score'],rfmTableScores['r_score'],rfmTableScores['m_score'])))


k = 10 #number of clusters

	# X coordinates of random centroids
C_x = np.random.randint(0, np.max(X[0]), size=k)
# Y coordinates of random centroids
C_y = np.random.randint(0, np.max(X[1]), size=k)
C_z = np.random.randint(0, np.max(X[2]), size=k)

C = np.array(list(zip(C_x, C_y,C_z)), dtype=np.int8)
print (C)

# To store the value of centroids when it updates
C_old = np.zeros(C.shape)
# Cluster Lables(0, 1, 2)
clusters = np.zeros(len(X))
# Error func. - Distance between new centroids and old centroids
error = dist(C, C_old,None)
print (error)
# Loop will run till the error becomes zero
while error != 0:
    # Assigning each value to its closest cluster
    for i in range(len(X)):
        distances = dist(X[i], C)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    # Storing the old centroid values
    C_old = deepcopy(C)
    # Finding the new centroids by taking the average value
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    error = dist(C, C_old,None)

rfmTable['clusters']=clusters

print (rfmTableScores)





##PLOT THE CLUSTERS WITH THE CENTROIDS

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111, projection='3d')

fig = plt.figure(figsize=(15,10))
dx = fig.add_subplot(111, projection='3d')
colors = ['blue', 'yellow', 'green', 'red','black','cyan','magenta','purple','salmon','plum']


for i in range(0,k):
    dx.scatter(rfmTable[rfmTable.clusters == i].recency, 
               rfmTable[rfmTable.clusters == i].frequency, 
               rfmTable[rfmTable.clusters == i].monetary, 
               c = colors[i], 
               label = 'Cluster ' + str(i+1), 
               s=50)

dx.set_title('Clusters of clients')
dx.set_xlabel('Recency')
dx.set_ylabel('Frequency')
dx.set_zlabel('Monetary')
dx.legend()

plt.show()



