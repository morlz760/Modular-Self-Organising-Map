
from mdsom_functions import *
import pandas as pd
# from minisom import MiniSom
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
from sklearn import preprocessing
import statistics
import plotly.graph_objects as go
import matplotlib.pyplot as plt


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
d = preprocessing.normalize(d, axis=1)
d_normalised = pd.DataFrame(d, columns=names)


new_names = new_d.columns
new_d = preprocessing.normalize(new_d, axis=1)
new_d_normalised = pd.DataFrame(new_d, columns=new_names)

X_train = d_normalised
y_train = labels

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_train, y_train)

classify(final_som, x_test, x_train, y_train)

# Initialization and training
n_neurons = 9
m_neurons = 9
som = MiniSom(n_neurons, m_neurons, X_train.shape[1], sigma=1.5, learning_rate=.5, 
              neighborhood_function='gaussian', random_seed=0)

som.pca_weights_init(X_train.values)
som.train(X_train.values, 1000, verbose=True)  # random training

# Create simple feature collections
feature_collections_1 = np.array([[i] for i in X_train.columns ])
feature_collections_2 = pd.array([["area", "perimeter", "compactness"], ["length_kernel", "width_kernel","asymmetry_coefficient"]])

trained_soms_layer_1 = train_som_layer(data = X_train_b,  grid_size = [8,8], feature_collections = feature_collections_1)
convolv_layer_one_train = create_convolution_layer(data = X_train_b, trained_soms = trained_soms_layer_1,  feature_collections = feature_collections_1,   normalise = False)
trained_soms_layer_2 = train_som_layer(data = convolv_layer_one_train,  grid_size = [8,8], feature_collections = feature_collections_2,  convolutional_layer=True)
convolv_layer_two_train = create_convolution_layer(data = convolv_layer_one_train, trained_soms = trained_soms_layer_2,  feature_collections = feature_collections_2,   normalise = False)


trained_soms_layer_test = train_som_layer(data = X_test_b,  grid_size = [8,8], feature_collections = feature_collections_1)
convolv_layer_one_test = create_convolution_layer(data = X_test_b, trained_soms = trained_soms_layer_test,  feature_collections = feature_collections_1,   normalise = False)
trained_soms_layer_2_test = train_som_layer(data = convolv_layer_one_test,  grid_size = [8,8], feature_collections = feature_collections_2,  convolutional_layer=True)
convolv_layer_two_test = create_convolution_layer(data = convolv_layer_one_test, trained_soms = trained_soms_layer_2_test,  feature_collections = feature_collections_2,   normalise = False)


unnested_test = unnest_data(convolv_layer_two_test)
unnested_train = unnest_data(convolv_layer_two_train)

classified = classify(final_som, unnested_test, unnested_train, y_train_b)

len(classified)
len(y_test_b)
pd.array(list((zip(classified, y_test_b))))

final_som = create_train_som(data= convolv_layer_two_train, grid_size=[10,10], n_features = convolv_layer_two_train.shape[1], convolutional_layer=True)
purity = evaluate_purity(final_som, convolv_layer_two_train, y_train, convolutional_layer=True)
purity['weighted_node_purity'].sum(axis=0)



evaluated_data = label_output(som = final_som, data = X_train, targets = y_train, final_convolution=convolv_layer_two_train,convolutional_layer = True, original = True)
evaluated_data["correct"] = np.where( (evaluated_data["default_class"] == evaluated_data['evaluated_class']), 1, 0)
correct_class = sum(evaluated_data["correct"])/len(evaluated_data.index)
class_results.append(correct_class)



data_values = unnest_data(convolv_layer_two_train)
winmap = final_som.labels_map(data_values, y_train)

# winmap = som.labels_map(X_train.values, y_train)




winmapDFT = pd.DataFrame(winmap).T
winmapDFT['class'] = winmapDFT.apply(lambda x: winmapDFT.columns[x.argmax()], axis = 1).astype(str) 
winmapDFT = winmapDFT.reset_index()
winmapDFT["max_val_node"] = winmapDFT[[1,2,3]].max(axis=1)
# Create a column that has the total observations for each node
winmapDFT["total_obs_node"] = winmapDFT[[1,2,3]].sum(axis=1)
# Calculate the simple node purity (a more complex purity might include some sort of penalty for having moltiple obs set off)
winmapDFT["node_purity"] = winmapDFT["max_val_node"] / winmapDFT["total_obs_node"]



import plotly.express as px
data_values = unnest_data(convolv_layer_two_train)
winmap = final_som.labels_map(data_values, y_train)
winmapDFT = pd.DataFrame(winmap).T
winmapDFT['class'] = winmapDFT.apply(lambda x: winmapDFT.columns[x.argmax()], axis = 1).astype(str) 
winmapDFT = winmapDFT.reset_index()
winmapDFT["max_val_node"] = winmapDFT[[1,2,3]].max(axis=1)
winmapDFT["total_obs_node"] = winmapDFT[[1,2,3]].sum(axis=1)
winmapDFT["node_purity"] = winmapDFT["max_val_node"] / winmapDFT["total_obs_node"]
winmapDFT_pure = winmapDFT[(winmapDFT.node_purity != 1)]

plt.figure(figsize=(10, 10))
plt = px.scatter(winmapDFT, x="level_0", y="level_1", color="class")
plt.update_traces(marker_size=25)
plt.show()




winmapDFT

df["size"].astype(str) 
# plt.pcolor(final_som.distance_map().T, cmap='bone_r')  # plotting the distance map as background
# plt.colorbar()
import plotly.express as px

plt = px.scatter(x=winmapDFT["level_0"], y=winmapDFT["level_1"])
plt.show()

plt = px.scatter(winmapDFT_pure, x="level_0", y="level_1", color="class")
plt.update_traces(marker_size=25)
plt.show()

# Plotting the response for each pattern in the iris dataset
# different colors and markers for each label
markers = ['o', 's', 'D']
colors = ['C0', 'C1', 'C2']
for row in winmapDFT.shape[0]:
    x = winmapDFT.iloc[row]
    y = winmapDFT.iloc[row]['level_1']
    # palce a marker on the winning position for the sample xx
    plt.plot(x+.5, x+.5, markers[y_train[cnt]-1], markerfacecolor='None',
             markeredgecolor=colors[y_train[cnt]-1], markersize=12, markeredgewidth=2)

plt.show()


winmapDFT = pd.DataFrame(winmap).T.rename_axis('node_coordinants').rename_axis(None, 1)


    # Pull the max value for that node
    winmapDFT["max_val_node"] = winmapDFT.max(axis=1)
    # Create a column that has the total observations for each node
    winmapDFT["total_obs_node"] = winmapDFT.iloc[:, 0:(len(winmapDFT.columns)-1)].sum(axis=1)
    # Calculate the simple node purity (a more complex purity might include some sort of penalty for having moltiple obs set off)
    winmapDFT["node_purity"] = winmapDFT["max_val_node"] / winmapDFT["total_obs_node"]
    # Calculate the weight of each node
    winmapDFT["weight"] = winmapDFT["total_obs_node"] / winmapDFT['total_obs_node'].sum(axis=0)
    # Calculate the weighted node purity
    winmapDFT["weighted_node_purity"] = winmapDFT["node_purity"] * winmapDFT["weight"]
    # Calculate the overall purity for the layer
    node_purity = winmapDFT['node_purity'].mean(axis=0)
    weighted_node_purity = winmapDFT['weighted_node_purity'].sum(axis=0)


# Define the default class
default_class = np.sum(list(winmap.values())).most_common()[0][0]
result_classes = []
# Extract the winning node for each observation.
for d in data_values:
    win_position = final_som.winner(d)
    if win_position in winmap:
        result_classes.append(winmap[win_position].most_common()[0][0])

    else:
        result_classes.append(default_class)

if original:
    output = data.copy(deep=True)
    output["default_class"] = targets
    output["evaluated_class"] = result_classes
    return(output)
else:
    output = final_convolution.copy(deep=True)
    output["default_class"] = targets
    output["evaluated_class"] = result_classes
    return(output)

plt.figure(figsize=(10, 10))

# plt.pcolor(final_som.distance_map().T, cmap='bone_r')  # plotting the distance map as background
# plt.colorbar()

# Plotting the response for each pattern in the iris dataset
# different colors and markers for each label
markers = ['o', 's', 'D']
colors = ['C0', 'C1', 'C2']
for cnt, xx in enumerate(unnest_data(convolv_layer_one_train)):
    w = final_som.winner(xx)  # getting the winner
    # palce a marker on the winning position for the sample xx
    plt.plot(w[0]+.5, w[1]+.5, markers[y_train[cnt]-1], markerfacecolor='None',
             markeredgecolor=colors[y_train[cnt]-1], markersize=12, markeredgewidth=2)

plt.show()



win_map = final_som.win_map(convolv_layer_one_train.values)
size=final_som.distance_map().shape[0]
qualities=np.empty((size,size))
qualities[:]=np.NaN
for position, values in win_map.items():
    qualities[position[0], position[1]] = np.mean(abs(values-final_som.get_weights()[position[0], position[1]]))

layout = go.Layout(title='quality plot')
fig = go.Figure(layout=layout)
fig.add_trace(go.Heatmap(z=qualities, colorscale='Viridis'))
fig.show()



from plotly.subplots import make_subplots
import math
def showPropertyPlot(som, data, columns):
# plots the distances for each different property
    win_map = som.win_map(data)
    size=som.distance_map().shape[0]
    properties=np.empty((size*size,2+data.shape[1]))
    properties[:]=np.NaN
    i=0
    for row in range(0,size):
        for col in range(0,size):
            properties[size*row+col,0]=row
            properties[size*row+col,1]=col
    for position, values in win_map.items():
        properties[size*position[0]+position[1],0]=position[0]
        properties[size*position[0]+position[1],1]=position[1]
        properties[size*position[0]+position[1],2:] = np.mean(values, axis=0)
        i=i+1
    B = ['row', 'col']
    B.extend(columns)
    properties = pd.DataFrame(data=properties, columns=B)
    fig = make_subplots(rows=math.ceil(math.sqrt(data.shape[1])), cols=math.ceil(math.sqrt(data.shape[1])), shared_xaxes=False, horizontal_spacing=0.1, vertical_spacing=0.05, subplot_titles=columns, column_widths=None, row_heights=None)
    i=0
    zmin=min(np.min(properties.iloc[:,2:]))
    zmax=max(np.max(properties.iloc[:,2:]))
    for property in columns:
        fig.add_traces(
            [go.Heatmap(z=properties.sort_values(by=['row', 'col'])[property].values.reshape(size,size), zmax=zmax, zmin=zmin, coloraxis = 'coloraxis2')],
            rows=[i // math.ceil(math.sqrt(data.shape[1])) + 1 ],
            cols=[i % math.ceil(math.sqrt(data.shape[1])) + 1 ]
            )
        i=i+1
    for layout in fig.layout:
        if layout.startswith('xaxis') or layout.startswith('yaxis'):
            fig.layout[layout].visible=False
            fig.layout[layout].visible=False
        if layout.startswith('coloraxis'):
            fig.layout[layout].cmax=zmax
            fig.layout[layout].cmin=zmin
        if layout.startswith('colorscale'):
            fig.layout[layout]={'diverging':'viridis'}
    fig.update_layout(
        height=800
    )
    fig.show()

showPropertyPlot(final_som,convolv_layer_one_train.values, columns[:10])