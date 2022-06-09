# This script is going to be an example of how to construct each of the differing MDSOM structures.

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


new_names = new_d.columns
new_d = preprocessing.normalize(new_d, axis=1)
new_d_normalised = pd.DataFrame(new_d, columns=new_names)

X_train = d_normalised
y_train = labels

# Things we need to think about
# - Cross validation
# - Visualisation
# - Evaluation in terms of nodes

# Our benchmark SOM will always have the same number of nodes as our final SOM .

# Benchmark PCA plot
x = pca_plot(data = X_train, target_array= y_train)
x.show()

# ________________________________ CREATE A SOM TRAINED OFF ALL VALUES ____________________________________


standard_som = create_train_som(data=X_train.values, n_features=X_train.shape[1], grid_size = [8,8], convolutional_layer=False)


plot_som_win_map(X_train, y_train, standard_som, title = "Som Win Map", sampled_layer = False)
# x = pca_plot(data = data_for_evaluation, target_array= evaluated_data['evaluated_class'].values)
# x.savefig('visulisations/standard_som_all_features.png', bbox_inches='tight')
# x.show()

# What are my options here. 
# I can just use the method from the original paper. Then just run it ten times. Show the two scores and the two plots
# I feel like that isnt enough test cases. Ok so why dont I start with results from a single layer SOM. With all features from DS

# ________________________________ CREATE SINGLE LAYER DSOM using node location as coords and distance____________________________________

# Create simple feature collections
feature_collections_1 = np.array([[i] for i in X_train.columns ])
feature_collections_1 = pd.array([["area", "perimeter","compactness", "length_kernel","width_kernel","asymmetry_coefficient", "length_kernel_groove"]])
######## Results using only node location as single value index ##############
# grid_size_variations = [[6,6], [7,7] ,[8,8], [9,9],[10,10], [12,12], [16,16], [20,20], [24,24]]

###########  ################

purity_results = []
for _ in range(10):
    trained_soms_layer_1 = train_som_layer(data = X_train, feature_collections = feature_collections_1, grid_size=[8,8])
    convolv_layer_one_train = create_convolution_layer_xyw(data = X_train, trained_soms = trained_soms_layer_1,  feature_collections = feature_collections_1,   normalise = False)
    final_som = create_train_som(data= convolv_layer_one_train, n_features = convolv_layer_one_train.shape[1]*3, convolutional_layer=True)
    purity = evaluate_purity(final_som, convolv_layer_one_train, y_train, convolutional_layer=True)
    purity_results.append(purity)
statistics.mean(purity_results)


som = trained_soms_layer_1.get("areaperimetercompactnesslength_kernelwidth_kernelasymmetry_coefficientlength_kernel_groove")

plot_som_win_map(X_train, y_train, som[0], title = "Som Win Map", sampled_layer = False)
# Evaluating Final Layer Performance - DSOM 2 layers

def plot_som_win_map(data, labels, som, title = "Som Win Map", sampled_layer = False):
    import plotly.express as px
    if sampled_layer:
        data_values = unnest_data(data)
    else:
        data_values = data.values
    # Create the dataframe to plot.
    winmap = pd.DataFrame()
    winmap = som.labels_map(data_values, labels)
    winmapDFT = pd.DataFrame(winmap).T
    winmapDFT['class'] = winmapDFT.apply(lambda x: winmapDFT.columns[x.argmax()], axis = 1).astype(str) 
    winmapDFT = winmapDFT.reset_index()
    winmapDFT["max_val_node"] = winmapDFT[[1,2,3]].max(axis=1)
    winmapDFT["total_obs_node"] = winmapDFT[[1,2,3]].sum(axis=1)
    winmapDFT["node_purity"] = winmapDFT["max_val_node"] / winmapDFT["total_obs_node"]
    # plot the data
    fig = px.scatter(winmapDFT, x="level_0", y="level_1", color="class", size="node_purity")
    fig.update_layout(
        title=title,
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
    fig.show()



x = pca_plot(data = data_for_evaluation, target_array= evaluated_data['evaluated_class'].values)
x.savefig('visulisations/standard_som_all_features.png', bbox_inches='tight')
x.show()




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



    # evaluated_data = label_output(som = final_som, data = X_train, targets = y_train, final_convolution=convolv_layer_one_train,convolutional_layer = True, original = True)
    # evaluated_data["correct"] = np.where( (evaluated_data["default_class"] == evaluated_data['evaluated_class']), 1, 0)
    # correct_class = sum(evaluated_data["correct"])/len(evaluated_data.index)
    # class_results.append(correct_class)

)

final_results = pd.concat([dfs, dfz, dfzw, dfzw_n, dfxyw, dfg, dfgw])
fig = px.line(final_results, x="grid_size", y="purity",  color='sampling_method',title='Evaluating Purity')
fig.show()


fig.update_layout(
    title='Evaluating Perforamce Over Differing Grid Sizes - Simple Data',
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







