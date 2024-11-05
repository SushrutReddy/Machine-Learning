import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

data_points = np.array([[1, 2], [2, 3], [4, 3], [5, 6], [9, 1], [8, 5], [7, 7]])
linked = linkage(data_points, method='ward')
agg_cluster = AgglomerativeClustering(n_clusters=3)
labels = agg_cluster.fit_predict(data_points)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
dendrogram(linked, labels=range(1, len(data_points) + 1))

plt.title('Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Euclidean Distance')

plt.subplot(1, 2, 2)
plt.scatter(data_points[:, 0], data_points[:, 1], c=labels, cmap='viridis')

plt.title('Agglomerative Clustering')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid()
plt.tight_layout()
plt.show()