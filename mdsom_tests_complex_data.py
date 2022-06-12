# This script is going to be an example of how to construct each of the differing MDSOM structures.

from mdsom_functions import *
import pandas as pd
# from minisom import MiniSom
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
from sklearn import preprocessing
import statistics
import plotly.express as px

data = pd.read_csv("../data/winequality-red.csv", sep = ";")
labels = data['quality'].values
# label_names = {1:'Kama', 2:'Rosa', 3:'Canadian'}
d = data[data.columns[0:11]]
new_d = data[data.columns[6:7]]

names = d.columns
d = preprocessing.normalize(d, axis=1)
d_normalised = pd.DataFrame(d, columns=names)


X_train = d_normalised
y_train = labels


# Things we need to think about
# - Cross validation
# - Visualisation
# - Evaluation in terms of nodes

# Our benchmark SOM will always have the same number of nodes as our final SOM .

# ________________________________ CREATE A SOM TRAINED OFF ALL VALUES ____________________________________

col_results_purity = []
for n_cols in range(1,12):
    print(n_cols)
    data_for_evaluation = X_train[X_train.columns[0:n_cols]]
    purity_results = []
    for _ in range(10):
    # Train the SOM
        standard_som = create_train_som(data=data_for_evaluation.values, n_features= data_for_evaluation.shape[1], convolutional_layer=False, grid_size=[24,24])
        purity = evaluate_purity(standard_som, data_for_evaluation.values, y_train)
        purity_results.append(purity)
    col_results_purity.append(statistics.mean(purity_results))


d = {"structure": "SOM", "n_features": list(range(len(X_train.columns))), 'purity': col_results_purity}
dfs = pd.DataFrame(data=d)
dfs["n_features"] = dfs["n_features"].apply(lambda x: x + 1)

standard_som = create_train_som(data=X_train.values, n_features= X_train.shape[1], convolutional_layer=False, grid_size=[24,24])
plot_som_win_map(X_train, y_train, standard_som, title = "Som Win Map - Complex Data", sampled_layer = False,  simple=False)


# ________________________________ CREATE A SINGLE LAYER MDSOM ____________________________________


# Create simple feature collections
feature_collections_1 = np.array([[i] for i in X_train.columns ])
feature_collections_1 = np.array([['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']])
######## Results using only node location as single value index ##############

###########  ################
col_results_purity = []
for n_cols in range(1,12):
    feature_collections = feature_collections_1[0:n_cols]
    purity_results = []
    print(feature_collections)
    for _ in range(10):
        trained_soms_layer_1 = train_som_layer(data = X_train, feature_collections = feature_collections, grid_size=[24,24])
        sampling_layer_one_train = create_sampling_layer_xyw(data = X_train, trained_soms = trained_soms_layer_1,  feature_collections = feature_collections,   normalise = False)
        final_som = create_train_som(data= sampling_layer_one_train, n_features = sampling_layer_one_train.shape[1]*3, convolutional_layer=True, grid_size=[24,24])
        purity = evaluate_purity(final_som, sampling_layer_one_train, y_train, convolutional_layer=True)
        purity_results.append(purity)
    col_results_purity.append(statistics.mean(purity_results))

d = {"structure": "MSOM - Multiple Featuresets", "n_features": list(range(len(X_train.columns))), 'purity': col_results_purity}
dfmsom = pd.DataFrame(data=d)
dfmsom["n_features"] = dfmsom["n_features"].apply(lambda x: x + 1)

plot_som_win_map(sampling_layer_one_train, y_train, final_som, title = "Som Win Map", sampled_layer = True, simple=False)

# Train a MSOM 
trained_soms_layer_1 = train_som_layer(data = X_train, feature_collections = feature_collections_1, grid_size=[24,24])
sampling_layer_one_train = create_convolution_layer_xyw(data = X_train, trained_soms = trained_soms_layer_1,  feature_collections = feature_collections_1, normalise = False)
final_som = create_train_som(data= sampling_layer_one_train, n_features = sampling_layer_one_train.shape[1]*3, convolutional_layer=True, grid_size=[24,24])

plot_som_win_map(sampling_layer_one_train, y_train, final_som, title = "MSOM Win Map - Complex Data", sampled_layer = True, simple=False)

# ________________________________ CREATE A SINGLE LAYER MDSOM (adding features to 1 featureset) ____________________________________

feature_collections_1 = np.array([['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']])
######## Results using only node location as single value index ##############
n_cols = 2
np.array(list(feature_collections_1[0][0:n_cols]))


###########  ################
col_results_purity = []
for n_cols in range(1,12):
    feature_collections = np.array([feature_collections_1[0][0:n_cols]])
    print(feature_collections)
    purity_results = []
    for _ in range(10):
        trained_soms_layer_1 = train_som_layer(data = X_train, feature_collections = feature_collections, grid_size=[20,20])
        convolv_layer_one_train = create_sampling_layer_xyw(data = X_train, trained_soms = trained_soms_layer_1,  feature_collections = feature_collections,   normalise = False)
        final_som = create_train_som(data= convolv_layer_one_train, n_features = convolv_layer_one_train.shape[1]*3, convolutional_layer=True, grid_size=[24,24])
        purity = evaluate_purity(final_som, convolv_layer_one_train, y_train, convolutional_layer=True)
        purity_results.append(purity)
    col_results_purity.append(statistics.mean(purity_results))

d = {"structure": "MSOM - 1 Featureset", "n_features": list(range(len(X_train.columns))), 'purity': col_results_purity}
dfmsom1 = pd.DataFrame(data=d)
dfmsom1["n_features"] = dfmsom1["n_features"].apply(lambda x: x + 1)

plot_som_win_map(convolv_layer_one_train, y_train, final_som, title = "Som Win Map", sampled_layer = True, simple=False)

# Train a MSOM 
trained_soms_layer_1 = train_som_layer(data = X_train, feature_collections = feature_collections_1, grid_size=[24,24])
convolv_layer_one_train = create_convolution_layer_xyw(data = X_train, trained_soms = trained_soms_layer_1,  feature_collections = feature_collections_1, normalise = False)
final_som = create_train_som(data= convolv_layer_one_train, n_features = convolv_layer_one_train.shape[1]*3, convolutional_layer=True, grid_size=[24,24])

plot_som_win_map(convolv_layer_one_train, y_train, final_som, title = "MSOM Win Map - Complex Data", sampled_layer = True, simple=False)


# ----------------------------------- COMBINE all results and Create a chart to evaluate performance ---------------------------
final_results_mdsom_complex = pd.concat([dfs, dfmsom, dfmsom1])
fig = px.line(final_results_mdsom_complex, x="n_features", y="purity", color = "structure", title='Evaluating Purity')
fig.update_layout(
    title='MSOM Evaluating Feature Additions - Complex Data',
    template="simple_white",
    autosize=False,
    width=750,
    height=600,
    yaxis=dict(
        title='Weighted Node Purity %',
        titlefont_size=16,
        tickfont_size=14,
        showgrid=True, 
        mirror=True,
        ticks='outside',
        showline=True,
        gridwidth=0.25,
        gridcolor='#e0e0e0'
    ),
    xaxis=dict(
        title='Features Included',
        titlefont_size=16,
        tickfont_size=14,
        showgrid=True, 
        mirror=True,
        ticks='outside',
        showline=True,
        gridwidth=1,
        gridcolor='#e0e0e0'
    ),
    legend=dict(
        title='Structure',
        x=0.62,
        y=0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    )
)
fig.show()



trained_soms_layer_1 = train_som_layer(data = X_train, feature_collections = feature_collections_1, grid_size=[24,24])
convolv_layer_one_train = create_convolution_layer_xyw(data = X_train, trained_soms = trained_soms_layer_1,  feature_collections = feature_collections_1,   normalise = False)
final_som = create_train_som(data= convolv_layer_one_train, n_features = convolv_layer_one_train.shape[1]*3, convolutional_layer=True, grid_size=[17,17])
    
plot_som_win_map(convolv_layer_one_train, labels, final_som, title = "Som Win Map", sampled_layer = True, simple = False)



# ________________________________ CREATE A SINGLE LAYER MDSOM - Trained on differing featureset ____________________________________

layer_1_feature_collection = pd.array([["area", "perimeter"], ["compactness", "length_kernel"], ["width_kernel","asymmetry_coefficient"]])

# Create our first layer
trained_soms_layer_1 = train_som_layer(data = X_train, feature_collections = layer_1_feature_collection)
convolv_layer_one_train = create_convolution_layer(data = X_train, trained_soms = trained_soms_layer_1,  feature_collections = layer_1_feature_collection, normalise=False)

# Now using the values output from our training cololutional layer (I think I got half way through implementing the addition of the node number as well
# as the distance from the given node) So now my create train SOM has no idea what to do with the god dam outpuut.
final_som = create_train_som(data=convolv_layer_one_train, n_features = convolv_layer_one_train.shape[1], convolutional_layer=True)

evaluate_purity(final_som, convolv_layer_one_train, y_train, convolutional_layer=True)

# BOOM. We've got our MDSOM. The key elements are the trained soms and the final SOM. 

# Pass our test data through our constructed SOM 
convolution_layer_test = create_convolution_layer(data=X_test, trained_soms=trained_soms_layer_1, feature_collections = layer_1_feature_collection)

evaluate_purity(final_som, convolution_layer_test, y_test, convolutional_layer=True)


# ________________________________ CREATE A MULTI LAYER MDSOM ____________________________________

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

