import pandas as pd
from minisom import MiniSom
import numpy as np
columns=['area', 'perimeter', 'compactness', 'length_kernel', 'width_kernel',
                   'asymmetry_coefficient', 'length_kernel_groove', 'target']

data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt', 
                    names=columns, 
                   sep='\t+', engine='python')
target = data['target'].values
label_names = {1:'Kama', 2:'Rosa', 3:'Canadian'}
data = data[data.columns[:-1]]
# data normalization
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
data = data.values

data.shape



# Initialization and training
n_neurons = 9
m_neurons = 9
som = MiniSom(n_neurons, m_neurons, data.shape[1], sigma=1.5, learning_rate=.5, 
              neighborhood_function='gaussian', random_seed=0)

som.pca_weights_init(data)
som.train(data, 1000, verbose=True)  # random training

import plotly.graph_objects as go

win_map = som.win_map(data)
size=som.distance_map().shape[0]
qualities=np.empty((size,size))
qualities[:]=np.NaN
for position, values in win_map.items():
    qualities[position[0], position[1]] = np.mean(abs(values-som.get_weights()[position[0], position[1]]))

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
            
    fig.update_layout(height=800)
    fig.show()
    
showPropertyPlot(som, data, columns[:-1])

