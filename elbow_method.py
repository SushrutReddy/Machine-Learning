import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data_points = np.array([[1, 2], [2, 3], [4, 3], [5, 6], [9, 1], [8, 5], [7, 7]])
k_values = range(1,len(data_points))
inertia = []
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_points)
    inertia.append(kmeans.inertia_)

plt.plot(k_values, inertia, 'bo-')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.grid()
plt.show()
