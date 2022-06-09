# This script is going to be an example of how to construct each of the differing MDSOM structures.

from mdsom_functions import *
import pandas as pd
# from minisom import MiniSom
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
from sklearn import preprocessing
import statistics


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

plot_som_win_map(X_train.values, y_train, standard_som, title = "Som Win Map", sampled_layer = False)

data_values = X_train.values
# Create the dataframe to plot.
winmap = pd.DataFrame()
winmap = standard_som.labels_map(data_values, y_train)
winmapDFT = pd.DataFrame(winmap).T
winmapDFT['class'] = winmapDFT.apply(lambda x: winmapDFT.columns[x.argmax()], axis = 1).astype(str) 
winmapDFT = winmapDFT.reset_index()
winmapDFT["max_val_node"] = winmapDFT[[5,6,7,4,8,3]].max(axis=1)
winmapDFT["total_obs_node"] = winmapDFT[[5,6,7,4,8,3]].sum(axis=1)
winmapDFT["node_purity"] = winmapDFT["max_val_node"] / winmapDFT["total_obs_node"]
# plot the data
fig = px.scatter(winmapDFT, x="level_0", y="level_1", color="class", size = "node_purity")
fig.show()

# ________________________________ CREATE A SINGLE LAYER MDSOM ____________________________________


# Create simple feature collections
feature_collections_1 = np.array([[i] for i in X_train.columns ])
feature_collections_1 = pd.array([["area", "perimeter","compactness", "length_kernel","width_kernel","asymmetry_coefficient", "length_kernel_groove"]])
######## Results using only node location as single value index ##############
# grid_size_variations = [[6,6], [7,7] ,[8,8], [9,9],[10,10], [12,12], [16,16], [20,20], [24,24]]

###########  ################
col_results_purity = []
for n_cols in range(1,12):
    feature_collections = feature_collections_1[0:n_cols]
    print(feature_collections)
    for _ in range(10):
        trained_soms_layer_1 = train_som_layer(data = X_train, feature_collections = feature_collections, grid_size=[9,9])
        convolv_layer_one_train = create_convolution_layer_xyw(data = X_train, trained_soms = trained_soms_layer_1,  feature_collections = feature_collections,   normalise = False)
        final_som = create_train_som(data= convolv_layer_one_train, n_features = convolv_layer_one_train.shape[1]*3, convolutional_layer=True, grid_size=[24,24])
        purity = evaluate_purity(final_som, convolv_layer_one_train, y_train, convolutional_layer=True)
        purity_results.append(purity)
    col_results_purity.append(statistics.mean(purity_results))

d = {"structure": "MDSOM", "n_features": list(range(len(X_train.columns))), 'purity': col_results_purity}
dfmdsom = pd.DataFrame(data=d)
dfmdsom["n_features"] = dfmdsom["n_features"].apply(lambda x: x + 1)

set(winmapDFT[3])
plot_som_win_map(convolv_layer_one_train, y_train, final_som, title = "Som Win Map", sampled_layer = True)
    
data_values = unnest_data(convolv_layer_one_train)
# Create the dataframe to plot.
winmap = pd.DataFrame()
winmap = final_som.labels_map(data_values, labels)
winmapDFT = pd.DataFrame(winmap).T
winmapDFT['class'] = winmapDFT.apply(lambda x: winmapDFT.columns[x.argmax()], axis = 1).astype(str) 
winmapDFT = winmapDFT.reset_index()
winmapDFT["max_val_node"] = winmapDFT[[5,6,7,4,8,3]].max(axis=1)
winmapDFT["total_obs_node"] = winmapDFT[[5,6,7,4,8,3]].sum(axis=1)
winmapDFT["node_purity"] = winmapDFT["max_val_node"] / winmapDFT["total_obs_node"]
# plot the data
fig = px.scatter(winmapDFT, x="level_0", y="level_1", color="class", size = "node_purity")
fig.show()

fig.update_layout(
    title="MDSOM Win Map - Complex Data",
    template="simple_white",
    autosize=False,
    width=750,
    height=600,
    yaxis=dict(
        title='Y coordinate',
        titlefont_size=16,
        tickfont_size=14,
        tickmode='linear',
        showgrid=True, 
        mirror=True,
        ticks='outside',
        showline=True,
        gridwidth=1,
        gridcolor='#e0e0e0'
    ),
    xaxis=dict(
        title='X coordinate',
        titlefont_size=16,
        tickfont_size=14,
        tickmode='linear',
        showgrid=True, 
        mirror=True,
        ticks='outside',
        showline=True,
        gridwidth=1,
        gridcolor='#e0e0e0'
    ),
        legend=dict(
        title='Allocated Class',
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    )
    # paper_bgcolor='rgba(0,0,0,0)',
    # plot_bgcolor='rgba(0,0,0,0)'
)



set(winmapDFT["node_purity"])

winmapDFT[(winmapDFT.node_purity.isnull())]

final_results_mdsom_complex = pd.concat([dfs, dfmdsom])
fig = px.line(final_results_mdsom_complex, x="n_features", y="purity", color = "structure", title='Evaluating Purity')
fig.show()


fig.update_layout(
    title='Evaluating Perforamce Over Differing Grid Sizes - Complex Data',
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
        title='Sampling Method',
        x=0.78,
        y=0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    )
    # paper_bgcolor='rgba(0,0,0,0)',
    # plot_bgcolor='rgba(0,0,0,0)'
)




# Create out feature collections
feature_collections_1 = np.array([[i] for i in X_train.columns ])
feature_collections_1 = pd.array([["area", "perimeter"], ["compactness", "length_kernel"], ["width_kernel","asymmetry_coefficient"]])
feature_collections_1 = pd.array([["area"], ["perimeter"]])
# Results using only node location
purity_results = []
class_results = []

for _ in range(10):
    trained_soms_layer_1 = train_som_layer(data = X_train, feature_collections = feature_collections_1)
    convolv_layer_one_train = create_convolution_layer_only_winning_som(data = X_train, trained_soms = trained_soms_layer_1,  feature_collections = feature_collections_1,   normalise = False)
    final_som = create_train_som(data= convolv_layer_one_train.values, n_features = convolv_layer_one_train.shape[1], convolutional_layer=False)
    purity = evaluate_purity(final_som, convolv_layer_one_train.values, y_train, convolutional_layer=False)
    purity_results.append(purity)
    evaluated_data = label_output(som = final_som, data = X_train, targets = y_train, final_convolution=convolv_layer_one_train,convolutional_layer = False, original = True)
    evaluated_data["correct"] = np.where( (evaluated_data["default_class"] == evaluated_data['evaluated_class']), 1, 0)
    correct_class = sum(evaluated_data["correct"])/len(evaluated_data.index)
    class_results.append(correct_class)

statistics.mean(purity_results)
statistics.mean(class_results)

# Results using only node location normalised
purity_results = []
class_results = []

for _ in range(10):
    trained_soms_layer_1 = train_som_layer(data = X_train, feature_collections = feature_collections_1)
    convolv_layer_one_train = create_convolution_layer_only_winning_som(data = X_train, trained_soms = trained_soms_layer_1,  feature_collections = feature_collections_1,   normalise = True)
    final_som = create_train_som(data= convolv_layer_one_train.values, n_features = convolv_layer_one_train.shape[1], convolutional_layer=False)
    purity = evaluate_purity(final_som, convolv_layer_one_train.values, y_train, convolutional_layer=False)
    purity_results.append(purity)
    evaluated_data = label_output(som = final_som, data = X_train, targets = y_train, final_convolution=convolv_layer_one_train,convolutional_layer = False, original = True)
    evaluated_data["correct"] = np.where( (evaluated_data["default_class"] == evaluated_data['evaluated_class']), 1, 0)
    correct_class = sum(evaluated_data["correct"])/len(evaluated_data.index)
    class_results.append(correct_class)

statistics.mean(purity_results)
statistics.mean(class_results)

# Results using node location and distance
purity_results = []
class_results = []

for _ in range(10):
    trained_soms_layer_1 = train_som_layer(data = X_train, feature_collections = feature_collections_1)
    convolv_layer_one_train = create_convolution_layer(data = X_train, trained_soms = trained_soms_layer_1,  feature_collections = feature_collections_1,   normalise = False)
    final_som = create_train_som(data= convolv_layer_one_train, n_features = convolv_layer_one_train.shape[1], convolutional_layer=True)
    purity = evaluate_purity(final_som, convolv_layer_one_train, y_train, convolutional_layer=True)
    purity_results.append(purity)
    evaluated_data = label_output(som = final_som, data = X_train, targets = y_train, final_convolution=convolv_layer_one_train,convolutional_layer = True, original = True)
    evaluated_data["correct"] = np.where( (evaluated_data["default_class"] == evaluated_data['evaluated_class']), 1, 0)
    correct_class = sum(evaluated_data["correct"])/len(evaluated_data.index)
    class_results.append(correct_class)

statistics.mean(purity_results)
statistics.mean(class_results)

purity_results = []
class_results = []

# Results using node location and distance and location normalised
for _ in range(10):
    trained_soms_layer_1 = train_som_layer(data = X_train, feature_collections = feature_collections_1)
    convolv_layer_one_train = create_convolution_layer(dat  a = X_train, trained_soms = trained_soms_layer_1,  feature_collections = feature_collections_1,   normalise = True)
    final_som = create_train_som(data= convolv_layer_one_train, n_features = convolv_layer_one_train.shape[1], convolutional_layer=True)
    purity = evaluate_purity(final_som, convolv_layer_one_train, y_train, convolutional_layer=True)
    purity_results.append(purity)
    evaluated_data = label_output(som = final_som, data = X_train, targets = y_train, final_convolution=convolv_layer_one_train,convolutional_layer = True, original = True)
    evaluated_data["correct"] = np.where( (evaluated_data["default_class"] == evaluated_data['evaluated_class']), 1, 0)
    correct_class = sum(evaluated_data["correct"])/len(evaluated_data.index)
    class_results.append(correct_class)

statistics.mean(purity_results)
statistics.mean(class_results)




x = pca_plot(data = X_train, target_array= evaluated_data['evaluated_class'].values)
x.show()

# Pass our test data through our constructed SOM 
convolution_layer_test = create_convolution_layer(data=X_test, trained_soms=trained_soms_layer_1, feature_collections = feature_collections_1)

evaluate_purity(final_som, convolution_layer_test, y_test, convolutional_layer=True)

# def pca_plot(som, data, targets, final_convolution = "", convolutional_layer = False):

x = pca_plot(som = final_som, data = X_train,targets = y_train, final_convolution = convolv_layer_one_train, convolutional_layer = True)

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







