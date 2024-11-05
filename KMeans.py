import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


data_points = np.array([[1,2],[2,3],[4,3],[5,6],[9,1],[8,5],[7,7]])
k=3
kmeans = KMeans(n_clusters=k,random_state=42)
kmeans.fit(data_points)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
print("Cluster Labels:", labels)
print("Cluster Centers:\n", centroids)
new_data = np.array([[5,5],[6,3],[2,2]])
predicted_clusters = kmeans.predict(new_data)

print("Original Data Points and Their Clusters:")
for point, cluster in zip(data_points, labels):
    print(f'Point {point} is in cluster {cluster}')

print("\nNew Data Points and Their Predicted Clusters:")
for point, cluster in zip(new_data, predicted_clusters):
    print(f'Point {point} is in cluster {cluster}')

print("\nCentroids of the Clusters:")
for i, centroid in enumerate(centroids):
    print(f'Centroid {i}: {centroid}')

# Plotting
plt.scatter(data_points[:, 0], data_points[:, 1], c=labels, cmap='viridis', label='Original Data Points')
plt.scatter(new_data[:, 0], new_data[:, 1], c='red', marker='x', label='New Data Points')
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=200, label='Centroids')

plt.title('K-means Clustering with Closely Grouped Data Points')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid()
plt.show()
