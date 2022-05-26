
from mdsom_functions import *
import pandas as pd
# from minisom import MiniSom
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
from sklearn import preprocessing
# import math

# Read in our data and prepare it
columns=['area', 'perimeter', 'compactness', 'length_kernel', 'width_kernel',
                   'asymmetry_coefficient', 'length_kernel_groove', 'target']

data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt', 
                    names=columns, 
                   sep='\t+', engine='python')
labels = data['target'].values
label_names = {1:'Kama', 2:'Rosa', 3:'Canadian'}
data = data[data.columns[0:6]]

names = data.columns
d = preprocessing.normalize(data, axis=0)
data_normalised = pd.DataFrame(d, columns=names)

# Create the test train split of our data.
X_train, X_test, y_train, y_test = train_test_split(data_normalised, labels)

# Not sure where this is used.....
# som_shape = [3,3]

# Create out feature collections
X_test_column_names = np.array([[i] for i in X_train.columns ])

# ________________________________ CREATE A SINGLE LAYER MDSOM ____________________________________

# Create our first layer of SOMS
trained_soms_layer_1 = train_som_layer(data = X_train, feature_collections = X_test_column_names)

# Create our training convolutional layer that is used to blend the results from our layer 1 SOM's
convolv_layer_one_test = create_convolution_layer(data = X_train, trained_soms = trained_soms_layer_1,  feature_collections = X_test_column_names)

# Now using the values output from our training cololutional layer (I think I got half way through implementing the addition of the node number as well
# as the distance from the given node) So now my create train SOM has no idea what to do with the god dam outpuut.
final_som = create_train_som(data=convolv_layer_one_test, n_features = convolv_layer_one_test.shape[1], convolutional_layer=True)

# BOOM. We've got our MDSOM. The key elements are the trained soms and the final SOM. 

# Pass our test data through our constructed SOM 
convolution_layer_test = create_convolution_layer(data=X_test, trained_soms=trained_soms_layer_1, feature_collections = X_test_column_names)


# ________________________________ CREATE A SOM TRAINED OFF ALL VALUES ____________________________________

standard_som = create_train_som(data=X_train.values, n_features = 6, convolutional_layer=False)

area_som = trained_soms_layer_1['area'][0]
X_test_area = np.array([[i] for i in X_train['area'].values])

# evaluate_purity(som, X_train, y_train)


# Single layer SOM 
evaluate_purity(standard_som, X_test.values, y_test)

# Single layer SOM 
evaluate_purity(standard_som, X_train.values, y_train)

# 
evaluate_purity(final_som, convolution_layer_test, y_test, convolutional_layer=True)

evaluate_purity(final_som, convolv_layer_one_test, y_train, convolutional_layer=True)

# ________________________________ CREATE A MULTI LAYER MDSOM ____________________________________
X_test_column_names


# Create our first layer of SOMS
trained_soms_layer_1 = train_som_layer(data = X_train, feature_collections = X_test_column_names)

# Create our training convolutional layer that is used to blend the results from our layer 1 SOM's
convolv_layer_one_test = create_convolution_layer(data = X_train, trained_soms = trained_soms_layer_1,  feature_collections = X_test_column_names)

# Create a second layer som based off our initial layer 
layer_2_feature_collection = pd.array([["area", "perimeter"], ["compactness", "length_kernel"], ["width_kernel","asymmetry_coefficient"]])

trained_soms_layer_2 = train_som_layer(data = convolv_layer_one_test, feature_collections = layer_2_feature_collection, convolutional_layer=True)

convolv_layer_two_test = create_convolution_layer(data = convolv_layer_one_test, trained_soms = trained_soms_layer_2,  feature_collections = layer_2_feature_collection, convolutional_layer=True)

final_som = create_train_som(data=convolv_layer_two_test, n_features = convolv_layer_two_test.shape[1], convolutional_layer=True)


evaluate_purity(final_som, convolv_layer_two_test, y_train, convolutional_layer=True)


# Now using the values output from our training cololutional layer (I think I got half way through implementing the addition of the node number as well
# as the distance from the given node) So now my create train SOM has no idea what to do with the god dam outpuut.
final_som = create_train_som(data=convolv_layer_one_test, n_features = convolv_layer_one_test.shape[1], convolutional_layer=True)

# BOOM. We've got our MDSOM. The key elements are the trained soms and the final SOM. 

# Pass our test data through our constructed SOM 
convolution_layer_test = create_convolution_layer(data=X_test, trained_soms=trained_soms_layer_1, feature_collections = layer_2_feature_collection)

evaluate_purity(final_som, convolution_layer_test, y_test, convolutional_layer=True)




import plotly.graph_objects as go

win_map = standard_som.win_map(X_train.values)
size=standard_som.distance_map().shape[0]
qualities=np.empty((size,size))
qualities[:]=np.NaN
for position, values in win_map.items():
    qualities[position[0], position[1]] = np.mean(abs(values-standard_som.get_weights()[position[0], position[1]]))

layout = go.Layout(title='quality plot')
fig = go.Figure(layout=layout)
fig.add_trace(go.Heatmap(z=qualities, colorscale='Viridis'))
fig.show()



# Visualising the results



from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])







