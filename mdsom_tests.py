# This script is going to be an example of how to construct each of the differing MDSOM structures.

from matplotlib.pyplot import grid
from mdsom_functions import *
import pandas as pd
# from minisom import MiniSom
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
from sklearn import preprocessing
import statistics
import plotly.express as px

# Read in our data and prepare it
columns=['area', 'perimeter', 'compactness', 'length_kernel', 'width_kernel',
                   'asymmetry_coefficient', 'length_kernel_groove', 'target']

data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt', 
                    names=columns, 
                   sep='\t+', engine='python')
labels = data['target'].values
label_names = {1:'Kama', 2:'Rosa', 3:'Canadian'}
d = data[data.columns[0:7]]
new_d = data[data.columns[6:7]]

names = d.columns
d = preprocessing.normalize(d, axis=1)
d_normalised = pd.DataFrame(d, columns=names)


new_names = new_d.columns
new_d = preprocessing.normalize(new_d, axis=1)
new_d_normalised = pd.DataFrame(new_d, columns=new_names)

X_train = d_normalised
y_train = labels


# Our benchmark SOM will always have the same number of nodes as our final SOM .
# ________________________________ CREATE A SOM TRAINED OFF ALL VALUES ____________________________________
import plotly.graph_objects as go
from plotly.subplots import make_subplots

col_results_purity = []
for n_cols in range(1,8):
    print(n_cols)
    data_for_evaluation = X_train[X_train.columns[0:n_cols]]
    purity_results = []
    for _ in range(10):
    # Train the SOM
        standard_som = create_train_som(data=data_for_evaluation.values, grid_size=[8,8], n_features = data_for_evaluation.shape[1], convolutional_layer=False)
        purity = evaluate_purity(standard_som, data_for_evaluation.values, y_train)
        purity_results.append(purity)
    col_results_purity.append(statistics.mean(purity_results))
    # plot_som_win_map(data_for_evaluation, y_train, standard_som, title = "MSOM Win Map", sampled_layer = False)


d = {"structure": "SOM", "n_features": list(range(0,7)), 'purity': col_results_purity}
dfs = pd.DataFrame(data=d)
dfs["n_features"] = dfs["n_features"].apply(lambda x: x + 1)

plot_som_win_map(data_for_evaluation, y_train, standard_som, title = "MSOM Win Map", sampled_layer = False)

standard_som = create_train_som(data=X_train.values, grid_size=[8,8], n_features = X_train.shape[1], convolutional_layer=False)
plot_som_win_map(X_train, y_train, standard_som, title = "MSOM Win Map", sampled_layer = False)

# ________________________________ CREATE A SINGLE LAYER MDSOM ____________________________________


# Create simple feature collections
feature_collections_1 = np.array([[i] for i in X_train.columns ])
# feature_collections_1 = pd.array([["area", "perimeter","compactness", "length_kernel","width_kernel","asymmetry_coefficient", "length_kernel_groove"]])
######## Results using only node location as single value index ##############
# grid_size_variations = [[6,6], [7,7] ,[8,8], [9,9],[10,10], [12,12], [16,16], [20,20], [24,24]]

###########  ################
col_results_purity = []
for n_cols in range(1,8):
    feature_collections = feature_collections_1[0:n_cols]
    purity_results = []
    print(feature_collections)
    for _ in range(10):
        trained_soms_layer_1 = train_som_layer(data = X_train, feature_collections = feature_collections, grid_size=[9,9])
        sampling_layer_one_train = create_sampling_layer_xyw(data = X_train, trained_soms = trained_soms_layer_1,  feature_collections = feature_collections,   normalise = False)
        final_som = create_train_som(data= sampling_layer_one_train, grid_size=[9,9], n_features = sampling_layer_one_train.shape[1]*3, convolutional_layer=True)
        purity = evaluate_purity(final_som, sampling_layer_one_train, y_train, convolutional_layer=True)
        purity_results.append(purity)
    col_results_purity.append(statistics.mean(purity_results))

d = {"structure": "MSOM - Multiple Featuresets", "n_features": list(range(len(X_train.columns))), 'purity': col_results_purity}
dfmsom = pd.DataFrame(data=d)
dfmsom["n_features"] = dfmsom["n_features"].apply(lambda x: x + 1)


# ________________________________ CREATE A SINGLE LAYER MDSOM SINGLE Featureset  ____________________________________


# Create simple feature collections
feature_collections_1 = pd.array([["area", "perimeter","compactness", "length_kernel","width_kernel","asymmetry_coefficient", "length_kernel_groove"]])
######## Results using only node location as single value index ##############

###########  ################
col_results_purity = []
for n_cols in range(1,8):
    feature_collections = np.array([feature_collections_1[0][0:n_cols]])
    purity_results = []
    print(feature_collections)
    for _ in range(10):
        trained_soms_layer_1 = train_som_layer(data = X_train, feature_collections = feature_collections, grid_size=[9,9])
        convolv_layer_one_train = create_sampling_layer_xyw(data = X_train, trained_soms = trained_soms_layer_1,  feature_collections = feature_collections,   normalise = False)
        final_som = create_train_som(data= convolv_layer_one_train, grid_size=[9,9], n_features = convolv_layer_one_train.shape[1]*3, convolutional_layer=True)
        purity = evaluate_purity(final_som, convolv_layer_one_train, y_train, convolutional_layer=True)
        purity_results.append(purity)
    col_results_purity.append(statistics.mean(purity_results))

d = {"structure": "MSOM - 1 Featureset", "n_features": list(range(len(X_train.columns))), 'purity': col_results_purity}
dfmsom1 = pd.DataFrame(data=d)
dfmsom1["n_features"] = dfmsom1["n_features"].apply(lambda x: x + 1)


# Creating a win Map
trained_soms_layer_1 = train_som_layer(data = X_train, feature_collections = feature_collections_1, grid_size=[9,9])
convolv_layer_one_train = create_convolution_layer_xyw(data = X_train, trained_soms = trained_soms_layer_1,  feature_collections = feature_collections_1,   normalise = False)
final_som = create_train_som(data= convolv_layer_one_train, grid_size=[9,9], n_features = convolv_layer_one_train.shape[1]*3, convolutional_layer=True)
som = trained_soms_layer_1.get("area")
plot_som_win_map(pd.DataFrame(X_train['area']), y_train, som[0], title = "MSOM Win Map", sampled_layer = False)

plot_som_win_map(convolv_layer_one_train, y_train, final_som, title = "MSOM Win Map", sampled_layer = True)

purity = evaluate_purity(som[0], pd.DataFrame(X_train['area']).values, y_train, convolutional_layer=False)

winmap = som[0].labels_map(, y_train)

# ----------------------------------- COMBINE all results and Create a chart to evaluate performance ---------------------------

final_results_mdsom = pd.concat([dfs, dfmsom, dfmsom1])
fig = px.line(final_results_mdsom, x="n_features", y="purity", color = "structure", title='Evaluating Purity')
fig.update_layout(
    title='MSOM Evaluating Feature Additions - Simple Data',
    template="simple_white",
    autosize=False,
    width=650,
    height=500,
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
        title='Sampling Method',
        x=0.63,
        y=0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    )
)
fig.show()


# ______________________________ CREATE WINMAP PLOTS FOR EVALUATION _________________________________

feature_collections_1 = pd.array([["area", "perimeter","compactness", "length_kernel","width_kernel","asymmetry_coefficient", "length_kernel_groove"]])

n_features = 2
feature_collections = np.array([feature_collections_1[0][0:n_features]])

trained_soms_layer_1 = train_som_layer(data = X_train, feature_collections = feature_collections, grid_size=[9,9])
sampling_layer_one_train = create_sampling_layer_xyw(data = X_train, trained_soms = trained_soms_layer_1,  feature_collections = feature_collections, normalise = False)
final_som = create_train_som(data= sampling_layer_one_train, n_features = sampling_layer_one_train.shape[1]*3, convolutional_layer=True, grid_size=[9,9])

p1 = plot_som_win_map(sampling_layer_one_train, y_train, final_som, title = "MSOM Win Map - Complex Data", sampled_layer = True, simple=True)

feature_collections_1 = np.array([[i] for i in X_train.columns ])
feature_collections = feature_collections_1[0:n_features]

trained_soms_layer_1 = train_som_layer(data = X_train, feature_collections = feature_collections, grid_size=[9,9])
sampling_layer_one_train = create_sampling_layer_xyw(data = X_train, trained_soms = trained_soms_layer_1,  feature_collections = feature_collections, normalise = False)
final_som = create_train_som(data= sampling_layer_one_train, n_features = sampling_layer_one_train.shape[1]*3, convolutional_layer=True, grid_size=[9,9])

p2 = plot_som_win_map(sampling_layer_one_train, y_train, final_som, title = "MSOM Win Map - Complex Data", sampled_layer = True, simple=True)


# Create out feature collections
feature_collections_1 = np.array([[i] for i in X_train.columns ])
feature_collections_1 = pd.array([["area", "perimeter"], ["compactness", "length_kernel"], ["width_kernel","asymmetry_coefficient"]])
feature_collections_1 = pd.array([["area"], ["perimeter"]])
# Results using only node location
purity_results = []
class_results = []


