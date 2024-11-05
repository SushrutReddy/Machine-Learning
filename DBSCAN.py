import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
data = np.array([
    [1, 2],[3, 4],[2.5, 4], [1.5,2.5], [3, 5], [2.8,4.5], [2.5,4.5],
    [1.2,2.5], [1,3], [1, 5],[1, 2.5], [5, 6], [4, 3]
])
eps = 0.6
min_samples = 4
db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
labels = db.labels_
# plt.figure(figsize=(8, 5))

for k in set(labels):
    if k == -1:
        plt.scatter(data[labels == k][:, 0], data[labels == k][:, 1], c='r', s=100, label='Outliers', marker='x')
    else:
        plt.scatter(data[labels == k][:, 0], data[labels == k][:, 1], s=100, label=f'Cluster {k}')

plt.title(f'DBSCAN Clustering (eps={eps})')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()

print(f"Cluster labels: {labels}")
