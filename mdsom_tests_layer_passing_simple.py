# This script 

from mdsom_function_dev_convo_layer import *
import pandas as pd
# from minisom import MiniSom
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
from sklearn import preprocessing
import statistics
import inspect
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

X_train = d_normalised
y_train = labels


# ________________________________ CREATE A SOM TRAINED OFF ALL VALUES ____________________________________

# Define grid size's
grid_size_variations = [[6,6], [7,7] ,[8,8], [9,9],[10,10], [12,12], [16,16], [20,20], [24,24]]

col_results_purity = []
for grid_size in grid_size_variations:
    print(grid_size)
    # data_for_evaluation = X_train[X_train.columns[0:n_cols]]
    purity_results = []
    for _ in range(10):
        # Train the SOM
        standard_som = create_train_som(data=X_train.values, n_features=X_train.shape[1], grid_size = grid_size, convolutional_layer=False)
        purity = evaluate_purity(standard_som, X_train.values, y_train)
        purity_results.append(purity)
    col_results_purity.append(statistics.mean(purity_results))

d = {"sampling_method": "som", 'grid_size': grid_size_variations, 'purity': col_results_purity}
dfs = pd.DataFrame(data=d)
dfs["grid_size"] = dfs["grid_size"].apply(lambda x: ' x '.join(map(str, x)))
fig = px.line(dfs, x="grid_size", y="purity", text="purity",title='Evaluating Purity')
fig.show()

# ________________________________ CREATE SINGLE LAYER DSOM USING ONLY WINNING VALUE ____________________________________

# Create simple feature collections
feature_collections_1 = pd.array([["area", "perimeter","compactness", "length_kernel","width_kernel","asymmetry_coefficient", "length_kernel_groove"]])
######## Results using only node location as single value index ##############
grid_size_variations = [[6,6], [7,7] ,[8,8], [9,9],[10,10], [12,12], [16,16], [20,20], [24,24]]

col_results_purity = []
for grid_size in grid_size_variations:
    print(grid_size)
    # data_for_evaluation = X_train[X_train.columns[0:n_cols]]
    purity_results = []
    for _ in range(10):
        trained_soms_layer_1 = train_som_layer(data = X_train,feature_collections = feature_collections_1, grid_size=grid_size)
        convolv_layer_one_train = create_convolution_layer_only_winning_som(data = X_train, trained_soms = trained_soms_layer_1,  feature_collections = feature_collections_1,   normalise = False)
        final_som = create_train_som(data= convolv_layer_one_train.values, n_features = convolv_layer_one_train.shape[1], convolutional_layer=False)
        purity = evaluate_purity(final_som, convolv_layer_one_train.values, y_train, convolutional_layer=False)
        purity_results.append(purity)
    
    col_results_purity.append(statistics.mean(purity_results))

d = {"sampling_method": "z", 'grid_size': grid_size_variations, 'purity': col_results_purity}
dfz = pd.DataFrame(data=d)
dfz["grid_size"] = dfz["grid_size"].apply(lambda x: ' x '.join(map(str, x)))
fig = px.line(dfz, x="grid_size", y="purity", text="purity",title='Evaluating Purity')
fig.show()


som = trained_soms_layer_1.get("areaperimetercompactnesslength_kernelwidth_kernelasymmetry_coefficient")

plot = som_map_plot(X_train, y_train, som, convolutional_layer = False)
plot.show() 

import plotly.express as px
import matplotlib.pyplot as plt
if convolutional_layer:
    data_values = unnest_data(convolv_layer_one_train)
else:
    data_values = convolv_layer_one_train.values
winmap = pd.DataFrame()
winmap = final_som.labels_map(data_values, y_train)
winmapDFT = pd.DataFrame(winmap).T
winmapDFT['class'] = winmapDFT.apply(lambda x: winmapDFT.columns[x.argmax()], axis = 1).astype(str) 
winmapDFT = winmapDFT.reset_index()
winmapDFT["max_val_node"] = winmapDFT[[1,2,3]].max(axis=1)
winmapDFT["total_obs_node"] = winmapDFT[[1,2,3]].sum(axis=1)
winmapDFT["node_purity"] = winmapDFT["max_val_node"] / winmapDFT["total_obs_node"]
winmapDFT_pure = winmapDFT[(winmapDFT.node_purity != 1)]
# plt.figure(figsize=(10, 10))
pllot = px.scatter(winmapDFT, x="level_0", y="level_1", color="class")
pllot.update_traces(marker_size=25)
pllot.show()


x = pca_plot(data = data_for_evaluation, target_array= evaluated_data['evaluated_class'].values)
x.savefig('visulisations/standard_som_all_features.png', bbox_inches='tight')
x.show()

########### Results using node location and distance ################

col_results_purity = []
for grid_size in grid_size_variations:
    print(grid_size)
    # data_for_evaluation = X_train[X_train.columns[0:n_cols]]
    purity_results = []
    for _ in range(10):
        trained_soms_layer_1 = train_som_layer(data = X_train, feature_collections = feature_collections_1, grid_size=grid_size)
        convolv_layer_one_train = create_convolution_layer_zw(data = X_train, trained_soms = trained_soms_layer_1,  feature_collections = feature_collections_1,   normalise = False)
        final_som = create_train_som(data= convolv_layer_one_train, n_features = convolv_layer_one_train.shape[1]*2, convolutional_layer=True)
        purity = evaluate_purity(final_som, convolv_layer_one_train, y_train, convolutional_layer=True)
        purity_results.append(purity)
        
    col_results_purity.append(statistics.mean(purity_results))

col_results_purity


d = {"sampling_method": "zw", 'grid_size': grid_size_variations, 'purity': col_results_purity}
dfzw = pd.DataFrame(data=d)
dfzw["grid_size"] = dfzw["grid_size"].apply(lambda x: ' x '.join(map(str, x)))
fig = px.line(dfzw, x="grid_size", y="purity", text="purity",title='Evaluating Purity')
fig.show()

########### Results using node location and distance normalised ################
col_results_purity = []
for grid_size in grid_size_variations:
    print(grid_size)
    # data_for_evaluation = X_train[X_train.columns[0:n_cols]]
    purity_results = []
    for _ in range(10):
        trained_soms_layer_1 = train_som_layer(data = X_train,  feature_collections = feature_collections_1, grid_size=grid_size)
        convolv_layer_one_train = create_convolution_layer_zw(data = X_train, trained_soms = trained_soms_layer_1,  feature_collections = feature_collections_1,   normalise = True)
        final_som = create_train_som(data= convolv_layer_one_train, n_features = convolv_layer_one_train.shape[1]*2, convolutional_layer=True)
        purity = evaluate_purity(final_som, convolv_layer_one_train, y_train, convolutional_layer=True)
        purity_results.append(purity)
        
    col_results_purity.append(statistics.mean(purity_results))

d = {"sampling_method": "zw_n",'grid_size': grid_size_variations, 'purity': col_results_purity}
dfzw_n = pd.DataFrame(data=d)
dfzw_n["grid_size"] = dfzw_n["grid_size"].apply(lambda x: ' x '.join(map(str, x)))
fig = px.line(dfzw_n, x="grid_size", y="purity", text="purity",title='Evaluating Purity')
fig.show()

########### Results using node location as coords and distance ################
col_results_purity = []
for grid_size in grid_size_variations:
    print(grid_size)
    # data_for_evaluation = X_train[X_train.columns[0:n_cols]]
    purity_results = []
    for _ in range(10):
        trained_soms_layer_1 = train_som_layer(data = X_train, feature_collections = feature_collections_1, grid_size=grid_size)
        convolv_layer_one_train = create_convolution_layer_xyw(data = X_train, trained_soms = trained_soms_layer_1,  feature_collections = feature_collections_1,   normalise = False)
        final_som = create_train_som(data= convolv_layer_one_train, n_features = convolv_layer_one_train.shape[1]*3, convolutional_layer=True)
        purity = evaluate_purity(final_som, convolv_layer_one_train, y_train, convolutional_layer=True)
        purity_results.append(purity)
    col_results_purity.append(statistics.mean(purity_results))

d = {"sampling_method": "xyw",'grid_size': grid_size_variations, 'purity': col_results_purity}
dfxyw = pd.DataFrame(data=d)
dfxyw["grid_size"] = dfxyw["grid_size"].apply(lambda x: ' x '.join(map(str, x)))

# Create a visulisation of the final SOM to investigate 
trained_soms_layer_1 = train_som_layer(data = X_train, feature_collections = feature_collections_1, grid_size=[9,9])
convolv_layer_one_train = create_convolution_layer_xyw(data = X_train, trained_soms = trained_soms_layer_1,  feature_collections = feature_collections_1,   normalise = False)
final_som = create_train_som(data= convolv_layer_one_train, n_features = convolv_layer_one_train.shape[1]*3, convolutional_layer=True)

som = trained_soms_layer_1.get("areaperimetercompactnesslength_kernelwidth_kernelasymmetry_coefficientlength_kernel_groove")

plot_som_win_map(X_train, y_train, som[0], title = "Som Win Map", sampled_layer = False)
plot_som_win_map(convolv_layer_one_train, y_train, final_som, title = "MSOM Win Map - Simple Data", sampled_layer = True)


########### Results using node location in vector of 0's ################
col_results_purity = []
for grid_size in grid_size_variations:
    print(grid_size)
    # data_for_evaluation = X_train[X_train.columns[0:n_cols]]
    purity_results = []
    for _ in range(10):
        trained_soms_layer_1 = train_som_layer(data = X_train, feature_collections = feature_collections_1, grid_size=grid_size)
        convolv_layer_one_train = create_convolution_layer_g(data = X_train, trained_soms = trained_soms_layer_1,  feature_collections = feature_collections_1,   normalise = False)
        final_som = create_train_som(data= convolv_layer_one_train, n_features = (grid_size[0]**2), convolutional_layer=True)
        purity = evaluate_purity(final_som, convolv_layer_one_train, y_train, convolutional_layer=True)
        purity_results.append(purity)
    col_results_purity.append(statistics.mean(purity_results))

d = {"sampling_method": "g",'grid_size': grid_size_variations, 'purity': col_results_purity}
dfg = pd.DataFrame(data=d)
dfg["grid_size"] = dfg["grid_size"].apply(lambda x: ' x '.join(map(str, x)))
fig = px.line(dfg, x="grid_size", y="purity", text="purity",title='Evaluating Purity')
fig.show()



########### Results using distance to observation in vector of 0's ################
col_results_purity = []
for grid_size in grid_size_variations:
    print(grid_size)
    # data_for_evaluation = X_train[X_train.columns[0:n_cols]]
    purity_results = []
    for _ in range(10):
        trained_soms_layer_1 = train_som_layer(data = X_train, feature_collections = feature_collections_1, grid_size=grid_size)
        convolv_layer_one_train = create_convolution_layer_g_w(data = X_train, trained_soms = trained_soms_layer_1,  feature_collections = feature_collections_1,   normalise = False)
        final_som = create_train_som(data= convolv_layer_one_train, n_features =  (grid_size[0]**2), convolutional_layer=True)
        purity = evaluate_purity(final_som, convolv_layer_one_train, y_train, convolutional_layer=True)
        purity_results.append(purity)
    col_results_purity.append(statistics.mean(purity_results))

d = {"sampling_method": "gw",'grid_size': grid_size_variations, 'purity': col_results_purity}
dfgw = pd.DataFrame(data=d)
dfgw["grid_size"] = dfgw["grid_size"].apply(lambda x: ' x '.join(map(str, x)))
fig = px.line(dfgw, x="grid_size", y="purity", text="purity",title='Evaluating Purity')
fig.show()



# Combine all of the results into one dataframe and viualise
final_results = pd.concat([dfs, dfz, dfzw, dfxyw, dfg, dfgw])
fig = px.line(final_results, x="grid_size", y="purity",  color='sampling_method',symbol="sampling_method", title='Evaluating Purity')
fig.show()


fig.update_layout(
    title='Evaluating Perforamce Over Differing Grid Sizes - Dataset 1',
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
        title='Grid Dimensions',
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
        x=0,
        y=0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    )
    # paper_bgcolor='rgba(0,0,0,0)',
    # plot_bgcolor='rgba(0,0,0,0)'
)












