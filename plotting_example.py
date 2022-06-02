
from mdsom_function_dev_convo_layer import *
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



# Create simple feature collections
feature_collections_1 = np.array([[i] for i in X_train.columns ])


trained_soms_layer_1 = train_som_layer(data = X_train, n_samples=X_train.shape[0], feature_collections = feature_collections_1)
convolv_layer_one_train = create_convolution_layer_only_winning_som(data = X_train, trained_soms = trained_soms_layer_1,  feature_collections = feature_collections_1,   normalise = False)
final_som = create_train_som(data= convolv_layer_one_train.values, n_samples=X_train.shape[0], n_features = convolv_layer_one_train.shape[1], convolutional_layer=False)


evaluated_data = label_output(som = final_som, data = X_train, targets = y_train, final_convolution=convolv_layer_one_train,convolutional_layer = False, original = True)
evaluated_data["correct"] = np.where( (evaluated_data["default_class"] == evaluated_data['evaluated_class']), 1, 0)
correct_class = sum(evaluated_data["correct"])/len(evaluated_data.index)
class_results.append(correct_class)



plt.figure(figsize=(8, 8))

plt.pcolor(final_som.distance_map().T, cmap='bone_r')  # plotting the distance map as background
plt.colorbar()

# Plotting the response for each pattern in the iris dataset
# different colors and markers for each label
markers = ['o', 's', 'D']
colors = ['C0', 'C1', 'C2']
for cnt, xx in enumerate(convolv_layer_one_train.values):
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