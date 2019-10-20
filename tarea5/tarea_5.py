from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

np.random.seed(42)

digits = load_digits()

#REMOVE NORMALIZATION
data = digits.data

n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels = digits.target

sample_size = 300

print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))


print(82 * '_')
print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')

def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_,
                                                average_method='arithmetic'),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))

bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
              name="k-means++", data=data)

bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10),
              name="random", data=data)

# in this case the seeding of the centers is deterministic, hence we run the
# kmeans algorithm only once with n_init=1
pca = PCA(n_components=n_digits).fit(data)
bench_k_means(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1),
              name="PCA-based",
              data=data)
print(82 * '_')




###############################################################################
# Plotting centroids as images

def plot_centroids(centroids):
  k = len(centroids)
  fig = plt.figure(figsize=(k, 2))
  plt.title("Centroids for k = " + str(k))
  plt.xticks(())
  plt.yticks(())

  rows = 1;
  columns = k;

  #Subfigures for each centroid
  for i in range(1, columns*rows +1):
    img = centroids[i-1].reshape(8,8);
    fig.add_subplot(rows, columns, i);
    plt.imshow(img,  cmap='gray');
    plt.xticks(())
    plt.yticks(())



# #############################################################################
# Visualize the results on PCA-reduced data

#PCA model to plot in 2D
model = PCA(n_components=2).fit(data)
reduced_data = model.transform(data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))


###############################################################################
# Iterate over different k's

ks = [3, 10, 20]


for k in ks:

  #Kmeans in 64D
  kmeans64 = KMeans(init='k-means++', n_clusters=k, n_init=10)
  kmeans64.fit(data)

  #Plotting centroids as images
  plot_centroids(kmeans64.cluster_centers_)
  reduced_cluster_centers = model.transform(kmeans64.cluster_centers_)

  #Kmeans in 2D for original image
  kmeans = KMeans(init='k-means++', n_clusters=k, n_init=10)
  kmeans.fit(reduced_data)

  # Obtain labels for each point in mesh. Use last trained model.
  Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

  Z = Z.reshape(xx.shape)
  plt.figure()
  plt.clf()
  plt.imshow(Z, interpolation='nearest',
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            cmap=plt.cm.Paired,
            aspect='auto', origin='lower')

  #Plot points
  plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)

  # Plot the centroids as a white X
  centroids2D = kmeans.cluster_centers_
  plt.scatter(centroids2D[:, 0], centroids2D[:, 1],
              marker='x', s=169,
              color='w', zorder=10)

  # Plot the centroids in 64D reprojected in 2D as a white star
  centroids64D = reduced_cluster_centers
  plt.scatter(centroids64D[:, 0], centroids64D[:, 1],
              marker='*', s=169,
              color='w', zorder=10)

  plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
            'Centroids in 2D and 64D are marked with white cross and star respectively')

  plt.xlim(x_min, x_max)
  plt.ylim(y_min, y_max)
  plt.xticks(())
  plt.yticks(())


plt.show()