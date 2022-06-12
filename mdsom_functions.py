import math

import numpy as np
import pandas as pd
from minisom import MiniSom
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Classifty the output out the function
def classify(som, x_test, x_train, y_train):
    winmap = som.labels_map(x_train, y_train)
    default_class = np.sum(list(winmap.values())).most_common()[0][0]
    result = []
    for d in x_test:
        win_position = som.winner(d)
        if win_position in winmap:
            result.append(winmap[win_position].most_common()[0][0])
        else:
            result.append(default_class)
    return result

# Create and train a SOM this is the old version that works, 
# Below I'm going to add some functionality that will allow it 
# to ingest the DF from a convolv layer.
# def create_train_som(data, n_features):
#     # Create SOM dimensions
#     som_nurons = int((math.sqrt(5*math.sqrt(n_features))))*2
#     x = som_nurons
#     y = som_nurons
#     #Create and train SOM
#     som = MiniSom(x, y, n_features, sigma=0.3, learning_rate=0.5) # initialization of x X y som
#     som.random_weights_init(data)
#     som.train_random(data,100, verbose=False) # training with 100 iterations
#     return som

def unnest_data(data):
        values = data.values
        unnested_data = np.array([np.concatenate(i) for i in values])
        return(unnested_data)

# Create and train a SOM
def create_train_som(data, n_features, grid_size = [8,8], convolutional_layer = False, sigma = 1.5, learning_rate=0.5):
    # Create SOM dimensions
    if convolutional_layer:
        data = unnest_data(data)
        n_features = n_features
    else:
        data = data
    # Create SOM dimensions
    # som_nurons = int((math.sqrt(5*math.sqrt(n_samples))))
    x = grid_size[0]
    y = grid_size[1]
    print("Som Neurons", x*y)
    #Create and train SOM
    som = MiniSom(x, y, n_features, sigma=sigma, learning_rate=learning_rate) # initialization of x X y som
    som.random_weights_init(data)
    som.train_random(data, 10000, verbose=False) # training with 100 iterations
    return som

def data_prep(data):
    data_normal = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    data_normal = data_normal.values
    return(data_normal)

def convert_coordinants(coordinants, som_y_size):
    x = coordinants[0]
    y = coordinants[1]
    unique_num = x * som_y_size + y
    return(unique_num)

def data_vectorisation(data):
    data_values_array = data.values
    if len(data.columns) == 1:
        data_values_array = np.array([[i] for i in data_values_array])
        return(data_values_array)
    else:
        return(data_values_array)

def train_som_layer(data, feature_collections, grid_size = [9,9], convolutional_layer = False):
    # create a dictionary to store the trained SOM
    trained_soms = {}
    # For each feature in the data set we will train a SOM. I intend to update this so that we can pass this function a list of combos we want to 
    # train our som's on.
    for feature_set in feature_collections:
        print("Training SOM on ", feature_set)
        n_features = len(feature_set)
        train_value = data[feature_set]
        # prepare the data
            # Unnest data
        if convolutional_layer:
            train_value_array = unnest_data(train_value)
            n_features = n_features*2
        else:
            train_value_array = train_value.values
        # train_value_array = train_value
        print("n features for creating SOM:", n_features)
        observation_key = "".join(map(str,feature_set))
        som = create_train_som(train_value_array, n_features, grid_size)
        trained_soms.setdefault(observation_key,[]).append(som)
    return(trained_soms)

# The create convolution layer takes the input data the traind soms and the feature collections to create a convolutional layer. 
# This means that you'll use this function in the creation of the MDSOM and then also in the testing process. The Traind_SOMS layer is the 
# static element, the convolutional layer is desitned to change depending on the data that's fed into it. So in the testing process you'll
# have to create a new convolutional layer 
# This keeps the x y coordiants and the weight
def create_sampling_layer_xyw(data, trained_soms, feature_collections, normalise=False):
    # Create empty dataframe to store the winning nodes from our trained SOM's
    dataframe = pd.DataFrame()
    # loop through each of the featuresets that have been used to build the SOM's to extarct the output from these values
    # that will be used to create the convolv layer.
    for feature_set in feature_collections:
        # So for each feature set extarct the corrosponding data from the training data.
        print("Creating convolutional layer for: ", feature_set)
        train_value = data[feature_set]
        if len(train_value.dtypes.unique()) > 1:
            print("Your feature sets have incompatable data formats")
        if (train_value.dtypes == 'O').all():
            train_value_array = unnest_data(train_value)
        else:
            train_value_array = train_value.values
        # Convert that data into an array, if the feature set is only one feature we will need to put it into an array
        # so that we are able to pass it to our SOM and extrac the values.
        # extract the SOM that was trained on the given feature(s)
        observation_key = "".join(map(str,feature_set))
        som = trained_soms.get(observation_key)[0]
        # Extract the y dimension of the given SOM.
        som_y_dim = max(som._neigy) + 1
        print("Y som dims: ",som_y_dim)
        # extract the distance from weights
        distance_map = som._distance_from_weights(train_value_array)
        # Create a winning node array to store the winning nodes for the feature.
        winning_node_value = []
        winning_node_distance = []
        # loop through the array of observations and extract the winning node and its distance for the given observation.
        for observation in train_value_array:
            winning_pos = som.winner(observation)
            node_distance = distance_map[winning_pos]
            # We now convert this coordinant into a numerical value so we can feed it to our next layer, to to do this properly
            # we need the y value of the SOM as that helps us convery coords to a single didget.
            winning_pos = pd.array(winning_pos, dtype=np.int32)
            # print(winning_pos)
            winning_node_value.append(winning_pos)
            # print(winning_node_value)
            winning_node_distance.append(node_distance)
        # Here we evaluate if we want to normalise our data or not. In this example we are normalising column wise. 
        if normalise:
            # print(winning_node_value_array)
            winning_node_value_normalised = preprocessing.normalize(winning_node_value_array, axis = 0).flatten()
            winning_node_distance_array = pd.array(winning_node_distance, dtype=np.float32).reshape(-1,1)
            winning_node_distance_normalised = preprocessing.normalize(winning_node_distance_array, axis = 0).flatten()
            winning_nodes = np.array(list(zip(winning_node_value_normalised, winning_node_distance_normalised))).tolist()
            dataframe[observation_key] = winning_nodes
        else:
            winning_nodes = np.array(list(zip(winning_node_value, winning_node_distance))).tolist()
            winning_node_array = np.array([np.hstack(i) for i in winning_nodes]).tolist()
            dataframe[observation_key] = winning_node_array
    return(dataframe)

# This function is used to evaluate the node purity of the output. One way to measure the effecacy of the algo.
# We need to figure out how to apply this purity function to SOM's that have differing number of nodes
# SOM's with more NODES will inevidebly have a higher purity raiting 

def evaluate_purity(som, data, targets, convolutional_layer=False):
    # Unnest training data
    if convolutional_layer:
        data = unnest_data(data)
    else:
        data = data
    # Extract the winning node for each obseervation
    winmap = som.labels_map(data, targets)    
    # Create a DF based of the winmap and transpose for easier data manipulation
    winmapDF = pd.DataFrame.from_dict(winmap)
    winmapDFT = winmapDF.T
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
    return(weighted_node_purity)

def label_output(som, data, targets, final_convolution = pd.DataFrame(), convolutional_layer = False, original = True):
    # Get the winning values
    if convolutional_layer:
        data_values = unnest_data(final_convolution)
    else:
        data_values = data.values
    winmap = som.labels_map(data_values, targets)
    # Define the default class
    default_class = np.sum(list(winmap.values())).most_common()[0][0]
    result_classes = []
    # Extract the winning node for each observation.
    for d in data_values:
        win_position = som.winner(d)
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


def pca_plot(data, target_array, title = "Principal Component Analysis"):
    df = data.copy(deep=True)
    from sklearn.decomposition import PCA
    # extract the PCA components so we can visualise
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(df)
    pca_review_df = pd.DataFrame(data= principalComponents, columns= ['Component1','Component2'])
    pca_review_df["label"] = target_array
    # Create the plot
    import matplotlib.pyplot as plt
    plt.figure()
    plt.figure(figsize=(10,10))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('Principal Component - 1',fontsize=20)
    plt.ylabel('Principal Component - 2',fontsize=20)
    plt.title(title,fontsize=20)
    targets = list(set(target_array))
    colors = ['r', 'g', 'b']
    for target, color in zip(targets,colors):
        indicesToKeep = pca_review_df['label'] == target
        plt.scatter(pca_review_df.loc[indicesToKeep, 'Component1'], pca_review_df.loc[indicesToKeep, 'Component2'], c = color, s = 50)
    plt.legend(targets,prop={'size': 15})
    return(plt)

def som_map_plot(data, labels, som, convolutional_layer = False):
    import plotly.express as px
    import matplotlib.pyplot as plt
    if convolutional_layer:
        data_values = unnest_data(data)
    else:
        data_values = data.values
    winmap = som.labels_map(data_values, labels)
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
    return(plt)

def plot_som_win_map(data, labels, som, title = "Som Win Map", sampled_layer = False, simple = True):
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
    if simple:
        winmapDFT["max_val_node"] = winmapDFT[[1,2,3]].max(axis=1)
        winmapDFT["total_obs_node"] = winmapDFT[[1,2,3]].sum(axis=1)
    else:
        winmapDFT["max_val_node"] = winmapDFT[[5,6,7,4,8,3]].max(axis=1)
        winmapDFT["total_obs_node"] = winmapDFT[[5,6,7,4,8,3]].sum(axis=1)
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
    )
    fig.show()