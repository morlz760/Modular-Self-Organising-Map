
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
d = data[data.columns[0:6]]
new_d = data[data.columns[6:7]]

names = d.columns
d = preprocessing.normalize(d, axis=0)
d_normalised = pd.DataFrame(d, columns=names)

new_names = new_d.columns
new_d = preprocessing.normalize(new_d, axis=0)
new_d_normalised = pd.DataFrame(new_d, columns=new_names)

# Create the test train split of our data.
X_train, X_test, y_train, y_test = train_test_split(data_normalised, labels)

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(new_d_normalised, labels)

# Not sure where this is used.....
# som_shape = [3,3]

# Create out feature collections
X_test_column_names = np.array([[i] for i in X_train.columns ])

# Things we need to think about
# - Cross validation
# - Visualisation
# - 

# Our benchmark SOM will always have the same number of nodes as our final SOM .

# ________________________________ CREATE A SOM TRAINED OFF ALL VALUES ____________________________________

# Train the SOM
standard_som = create_train_som(data=X_train.values, n_features = X_train.shape[1], convolutional_layer=False)

# Single layer SOM 
evaluate_purity(standard_som, X_test.values, y_test)


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

evaluate_purity(final_som, convolution_layer_test, y_test, convolutional_layer=True)


# ________________________________ CREATE A SINGLE LAYER MDSOM - Trained on differing featureset ____________________________________

layer_1_feature_collection = pd.array([["area", "perimeter"], ["compactness", "length_kernel"], ["width_kernel","asymmetry_coefficient"]])


# Create our first layer
trained_soms_layer_1 = train_som_layer(data = X_train, feature_collections = layer_1_feature_collection)
convolv_layer_one_test = create_convolution_layer(data = X_train, trained_soms = trained_soms_layer_1,  feature_collections = layer_1_feature_collection)

# Now using the values output from our training cololutional layer (I think I got half way through implementing the addition of the node number as well
# as the distance from the given node) So now my create train SOM has no idea what to do with the god dam outpuut.
final_som = create_train_som(data=convolv_layer_one_test, n_features = convolv_layer_one_test.shape[1], convolutional_layer=True)

# BOOM. We've got our MDSOM. The key elements are the trained soms and the final SOM. 

# Pass our test data through our constructed SOM 
convolution_layer_test = create_convolution_layer(data=X_test, trained_soms=trained_soms_layer_1, feature_collections = layer_1_feature_collection)

evaluate_purity(final_som, convolution_layer_test, y_test, convolutional_layer=True)


# ________________________________ CREATE A MULTI LAYER MDSOM ____________________________________

X_test_column_names

# Create our first layer of SOMS
trained_soms_layer_1 = train_som_layer(data = X_train, feature_collections = X_test_column_names)
# Create our training convolutional layer that is used to blend the results from our layer 1 SOM's
convolv_layer_one_train = create_convolution_layer(data = X_train, trained_soms = trained_soms_layer_1,  feature_collections = X_test_column_names)

# Create a second layer som based off our initial layer 
layer_2_feature_collection = pd.array([["area", "perimeter"], ["compactness", "length_kernel"], ["width_kernel","asymmetry_coefficient"]])
trained_soms_layer_2 = train_som_layer(data = convolv_layer_one_train, feature_collections = layer_2_feature_collection, convolutional_layer=True)
convolv_layer_two_train = create_convolution_layer(data = convolv_layer_one_train, trained_soms = trained_soms_layer_2,  feature_collections = layer_2_feature_collection)

# Create the final layer
final_som = create_train_som(data=convolv_layer_two_train, n_features = convolv_layer_two_train.shape[1], convolutional_layer=True)

# Create the testing layers
convolv_layer_one_test = create_convolution_layer(data = X_test, trained_soms = trained_soms_layer_1,  feature_collections = X_test_column_names)
convolv_layer_two_test = create_convolution_layer(data = convolv_layer_one_test, trained_soms = trained_soms_layer_2,  feature_collections = layer_2_feature_collection, convolutional_layer=True)

evaluate_purity(final_som, convolv_layer_two_test, y_test, convolutional_layer=True)



# ________________________________ CREATE A MDSOM - New Features introduced ____________________________________


# Create our first layer of SOMS
trained_soms_layer_1 = train_som_layer(data = X_train, feature_collections = X_test_column_names)
# Create our training convolutional layer that is used to blend the results from our layer 1 SOM's
convolv_layer_one_train = create_convolution_layer(data = X_train, trained_soms = trained_soms_layer_1,  feature_collections = X_test_column_names)

# Create a second layer som based off our initial layer 
layer_2a_feature_collection = pd.array([["area", "perimeter"], ["compactness", "length_kernel"], ["width_kernel","asymmetry_coefficient"]])
trained_soms_layer_2a = train_som_layer(data = convolv_layer_one_train, feature_collections = layer_2a_feature_collection, convolutional_layer=True)
# convolv_layer_2a_train = create_convolution_layer(data = convolv_layer_one_train, trained_soms = trained_soms_layer_2a,  feature_collections = layer_2a_feature_collection, convolutional_layer=True)

# Here we're introducing some new features
new_data = X_train_b
layer_2b_feature_collection = pd.array([['length_kernel_groove']])
trained_soms_layer_2b = train_som_layer(data = X_train_b, feature_collections = layer_2b_feature_collection, convolutional_layer=False)
# convolv_layer_2b_train = create_convolution_layer(data = X_train_b, trained_soms = trained_soms_layer_2b,  feature_collections = layer_2b_feature_collection, convolutional_layer=False)

# The new data has to have  be the same length, doesn't matter if its completely new data, or the same data that was used to originally train the 2nd layer. 
# With the ne


# Merge the som dictionaries, do I need to do this, I think not, lets try another apporoach
trained_soms_layer_2_complete = trained_soms_layer_2a.copy()
trained_soms_layer_2_complete.update(trained_soms_layer_2b)

# Merge the input data. 
convolv_layer_one_train_ab = convolv_layer_one_train
convolv_layer_one_train_ab['length_kernel_groove'] = X_train_b.values

# Create a featureset that combines all the input things.
layer_2ab_feature_collection = pd.array([["area", "perimeter"], ["compactness", "length_kernel"], ["width_kernel","asymmetry_coefficient"], ["length_kernel_groove"]])

# Create our new convolutional layer
convolv_layer_2_complete_train = create_convolution_layer(data = convolv_layer_one_train_ab, trained_soms = trained_soms_layer_2_complete,  feature_collections = layer_2ab_feature_collection)

# Do i create a merge function?

# Create the final layer
final_som = create_train_som(data=convolv_layer_2_complete_train, n_features = convolv_layer_2_complete_train.shape[1], convolutional_layer=True)

evaluate_purity(final_som, convolv_layer_2_complete_train, y_train, convolutional_layer=True)

# Create the testing layers
convolv_layer_one_test = create_convolution_layer(data = X_test, trained_soms = trained_soms_layer_1,  feature_collections = X_test_column_names)
convolv_layer_two_test = create_convolution_layer(data = convolv_layer_one_test, trained_soms = trained_soms_layer_2,  feature_collections = layer_2_feature_collection, convolutional_layer=True)

evaluate_purity(final_som, convolv_layer_two_test, y_test, convolutional_layer=True)








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







