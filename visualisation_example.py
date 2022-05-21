# Visualisation example

from minisom import MiniSom
import numpy as np
import pandas as pd

data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt', 
                    names=['area', 'perimeter', 'compactness', 'length_kernel', 'width_kernel',
                   'asymmetry_coefficient', 'length_kernel_groove', 'target'], 
                   sep='\t+', engine='python')
# data extraction
data = data.iloc[:,0:7]
data_2 = data.iloc[:,0:2]


data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
data = data.values

# Initialization and training
som_shape = (1, 3)
som = MiniSom(som_shape[0], som_shape[1], data.shape[1], sigma=.5, learning_rate=.5,
              neighborhood_function='gaussian', random_seed=10)

som.train_batch(data, 500, verbose=True)

# each neuron represents a cluster
winner_coordinates = np.array([som.winner(x) for x in data]).T
# with np.ravel_multi_index we convert the bidimensional
# coordinates to a monodimensional index
cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)

import plotly.graph_objects as go

win_map = som.win_map(data)
size=som.distance_map().shape[0]
qualities=np.empty((size,size))
qualities[:]=np.NaN
for position, values in win_map.items():
    print(values)
    qualities[position[0], position[1]] = np.mean(abs(values-som.get_weights()[position[0], position[1]]))

layout = go.Layout(title='quality plot')
fig = go.Figure(layout=layout)
fig.add_trace(go.Heatmap(z=qualities, colorscale='Viridis'))
fig.show()



# plotting the clusters using the first 2 dimentions of the data
for c in np.unique(cluster_index):
    plt.scatter(data[cluster_index == c, 0],
                data[cluster_index == c, 1], label='cluster='+str(c), alpha=.7)

# plotting centroids
for centroid in som.get_weights():
    plt.scatter(centroid[:, 0], centroid[:, 1], marker='x', 
                s=8, linewidths=15, color='k', label='centroid')
plt.legend()

plt.show()


som.get_weights()

# Two features not 6

# data normalization

data_2 = (data_2 - np.mean(data_2, axis=0)) / np.std(data_2, axis=0)
data_2 = data_2.values

# Initialization and training
som_shape = (1, 3)
som_2 = MiniSom(som_shape[0], som_shape[1], data_2.shape[1], sigma=.5, learning_rate=.5,
              neighborhood_function='gaussian', random_seed=10)

som_2.train_batch(data_2, 500, verbose=True)

# each neuron represents a cluster
winner_coordinates_2 = np.array([som_2.winner(x) for x in data_2]).T
# with np.ravel_multi_index we convert the bidimensional
# coordinates to a monodimensional index
cluster_index_2 = np.ravel_multi_index(winner_coordinates_2, som_shape)



import matplotlib.pyplot as plt
%matplotlib inline

# plotting the clusters using the first 2 dimentions of the data
for c in np.unique(cluster_index_2):
    plt.scatter(data_2[cluster_index_2 == c, 0],
                data_2[cluster_index_2 == c, 1], label='cluster='+str(c), alpha=.7)

# plotting centroids
for centroid in som_2.get_weights():
    plt.scatter(centroid[:, 0], centroid[:, 1], marker='x', 
                s=8, linewidths=15, color='k', label='centroid')
plt.legend()

plt.show()