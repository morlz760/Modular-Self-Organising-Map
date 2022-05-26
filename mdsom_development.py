# This script is a development script. Here I've been slowly building out the MDSOM process
# I've done this by writing down what I need to create. Hard coding the process. Then once its been validated
# I've turned it into a function. As I've worked down the pipeline I've identified areas that need to go back
# and be reworked and this is all part of that process. The finalised functions have been saved in the "mdsom_functions" file.
# Towards the end of this script I have a rough pipeline that I've been using to evaluate if the process is working.

# __________ IMPORT PACKAGES ___________

import pandas as pd
from minisom import MiniSom
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import preprocessing
import math

# def classify(som, x_test, x_train, y_train):
#     winmap = som.labels_map(x_train, y_train)
#     default_class = np.sum(list(winmap.values())).most_common()[0][0]
#     result = []
#     for d in x_test:
#         win_position = som.winner(d)
#         if win_position in winmap:
#             result.append(winmap[win_position].most_common()[0][0])
#         else:
#             result.append(default_class)
#     return result


# ____________________________ DATA PREP _____________________________________

columns=['area', 'perimeter', 'compactness', 'length_kernel', 'width_kernel',
                   'asymmetry_coefficient', 'length_kernel_groove', 'target']

data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt', 
                    names=columns, 
                   sep='\t+', engine='python')
labels = data['target'].values
label_names = {1:'Kama', 2:'Rosa', 3:'Canadian'}
data = data[data.columns[0:6]]


# Here we will do the two value SOM. 

# ok so how do i implement this? i know i need to extract the given results for X so lets try that 
# So instead of passing the patches to the function we wwill pass it the feature vector.

def create_train_som(x_train, n_features):
    # Create SOM dimensions
    som_nurons = int((math.sqrt(5*math.sqrt(n_features))))*2
    # print("number of neurons ", som_nurons*som_nurons)
    x = som_nurons
    y = som_nurons
    #Create and train SOM
    som = MiniSom(x, y, n_features, sigma=0.3, learning_rate=0.5) # initialization of 6x6 SOM
    som.random_weights_init(x_train)
#     print("Training...")
    som.train_random(x_train,100, verbose=False) # training with 100 iterations
#     print("...ready!")
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


# ___________________________________ Process to train multiple SOM's on one set of data _____________________________________

# create a dictionary to store the trained SOM
trained_soms = {}

# something to think about. How am I going to ensure that the SOM is trained on a given feature that it always recieves that feature. So SOM X is trained on 
# the petal width feature, therefor when passing new information into the process you need to specifically annotate which some addresses which feature. 
# (well we've addressed this with the feature sets issue!)

# maybe we need to also add the option to train a layer based on given combinations of SOM's
# We also need to think about how we're going to add SOM's to a given layer.

def train_som_layer(data, feature_collections):
    # create a dictionary to store the trained SOM
    trained_soms = {}
    
    # For each feature in the data set we will train a SOM. I intend to update this so that we can pass this function a list of combos we want to 
    # train our som's on.
    for feature_set in feature_collections:
        print("Training SOM on ", feature_set)
        print(len(feature_set))
        train_value = data[feature_set]
        # prepare the data
        # train_value_array = data_vectorisation(train_value)
        train_value_array = train_value.values
        # train_value_array = train_value.values
        # train_value_array = np.array([[i] for i in train_value_array])
        # som_shape = [3,3]
                    # observation_key = [index_val, "-", feature]
        observation_key = "".join(map(str,feature_set))
        # som = MiniSom(som_shape[0], som_shape[1], 1, sigma=.5, learning_rate=.5,neighborhood_function='gaussian', random_seed=10)
        # som.random_weights_init(train_value_normal)
        # som.train_random(train_value_normal, 500, verbose=False)
        som = create_train_som(train_value_array, len(feature_set))
        trained_soms.setdefault(observation_key,[]).append(som)
    return(trained_soms)

# ________________________ Now we extract the winning nodes ____________________________

# create dictionary to store the winning nodes for the given 

# where should purity evaluation occur?

# Train SOM's
# Extract winning node to build convolution layer
# Train final SOM on the convolution layer
# Evaluate performance

# loop through each of the patches
def create_convolution_layer(data, trained_soms, feature_collections):
    # Create empty dataframe to store the winning nodes from our trained SOM's
    dataframe = pd.DataFrame()
    # loop through each of the featuresets that have been used to build the SOM's to extarct the output from these values
    # that will be used to create the convolv layer.
    for feature_set in feature_collections:
        # So for each feature set extarct the corrosponding data from the training data.
        print("Creating convolutional layer for: ", feature_set)
        train_value = data[feature_set]
        
        # Convert that data into an array, if the feature set is only one feature we will need to put it into an array
        # so that we are able to pass it to our SOM and extrac the values.
        train_value_array = train_value.values
        
        # extract the SOM that was trained on the given feature(s)
        som = trained_soms.get(feature)[0]
        # extract the distance from weights
        distance_map = som._distance_from_weights(train_value_array)
        # Create a winning node array to store the winning nodes for the feature.
        winning_nodes = []
        # loop through the array of observations and extract the winning node and its distance for the given observation.
        for observation in train_value_array:
            # print("extracting winner for observation ", index_val)
            winning_pos = som.winner(observation)
            node_distance = distance_map[winning_pos]
            # winner_distance = min()
            # We now convert this coordinant into a numerical value so we can feed it to our next layer
            node_value = convert_coordinants(winning_pos, 2)
            # print("winning value", k)
            # observation_key = [index_val, "-", feature]
            # observation_key = "".join(map(str,observation_key))
            #print("saving to index", image_key)    
            # print(mapped_observation,k)   
            output = [node_value, node_distance] 
            winning_nodes.append(output)
        dataframe[feature] = winning_nodes
        # Here we will evaluate the node purity?
    return(dataframe)

# How do I best do this. Option a, just use the value of the winning node. In this instance we do some amazing dimensionality 
# Reduction but is it too much dimensionality reduction? 

# So the idea at the tip of my brain is to build a convolutional layer for each observation. 
# This layer would be a combination of the below SOM nodes, but only the winning ones. So we'd end up 
# With for X observation we have a convolutional layer that has all the SOM's in a grid, however only the 
# Winning node is active in that grid, all other nodes have a weight of 0, however the winning node has a weight 
# of the value of that observation.

# Maybe this is something I want to ask my tutuor. Is this a logical way to construct a second layer? My other question becomes,
# in the paper that I've read are they                                                                                                                                                                                          

# ________________________ Import our Data __________________________________


# import itertools
names = data.columns
d = preprocessing.normalize(data, axis=0)
data_normalised = pd.DataFrame(d, columns=names)


X_train, X_test, y_train, y_test = train_test_split(data_normalised, labels)

som_shape = [3,3]

# Create out feature collections
X_test_column_names = np.array([[i] for i in X_train.columns ])

# Create our mdsom
trained_soms_layer_1 = train_som_layer(data = X_train, feature_collections = X_test_column_names)
# Create our training convolutional layer
convolv_layer_one = create_convolution_layer(data = X_train, trained_soms = trained_soms_layer_1,  feature_collections = X_test_column_names)

# Ok we've had to introduce an intermediate step to transform the data into a format that can be interperated by the train som function
# this is potentially something we could add to our train som function in the future and I would highly reccommend as every time we're going
# to feed it new data from a convolv layer (which will be alot we will need to do this, so well I guess I'm going to do it now.)

value = convolv_layer_one.values

please = ([np.concatenate(i) for i in value])

np.array(please).shape

first_obs = value[1]

np.concatenate(first_obs)

convolv_layer_one.values
flattening_convolv_layer = np.array([[i] for i in value ])
flattening_convolv_layer.shape


# Ok I think the way we will do this we will add a feature to our create SOM layer that takes the column names of the PD dataframe we are passing it
# and it will use that to take the input data and manipulate it so we can pass it to that given SOM. 

final_som = create_train_som(convolv_layer_one.values, convolv_layer_one.shape[1])

# Pass our test data through our constructed SOM 
convolution_layer_test = create_convolution_layer(X_test, trained_soms_layer_1)

# Extract the values for our area some so we can evaluate the purity of our nodes
area_som = trained_soms_layer_1['area'][0]
X_test_area = np.array([[i] for i in X_train['area'].values])


# Here we are evaluating the purity of our SOM AKA evaluating how well its performing.

# the final 
final_som_node_purity = evaluate_purity(final_som, convolv_layer_one.values, y_train)

purity_dict = {}
for feature in X_train:
        train_value = X_train[feature]
        train_value_array = np.array([[i] for i in train_value.values])
        som = trained_soms_layer_1.get(feature)[0]
        purity = evaluate_purity(som, train_value_array, y_train)
        purity_dict.setdefault(feature,[]).append(purity)

print("MDSOM node purity average is: ", final_som_node_purity, " | While the single area average SOM node purity is: ", area_som_node_purity)


# Here we evaluate the effectivness of our algo
print(classification_report(y_test, classify(final_som, convolution_layer_test.values, convolv_layer_one.values, y_train)))

# Testing a single layer SOM
single_layer_som = create_train_som(X_train.values, 6)
print(classification_report(y_test, classify(single_layer_som, X_test.values, X_train.values, y_train)))



k = convert_coordinants(winning_pos, 2)

area_som = trained_soms_layer_1.get('area')[0]
x = X_train['area']
x.unique()
train_value_array = np.array([[i] for i in x.values])

train_value_array.unique()

area_som_distance_from_weights = area_som._distance_from_weights(train_value_array)

train_value_array[1]

for x in train_value_array:
    winning_pos = area_som.winner(x)
    print(convert_coordinants(winning_pos, 2))


convolv_layer_one['area'].values

# Ok how to turn this into a function!


# we need the SOM, the training data, the labels for the training data 

def evaluate_purity(som, X_train, y_train):
    # Extract the winning node for each obseervation
    winmap = som.labels_map(X_train, y_train)    
    # Create a DF based of the winmap and transpose for easier data manipulation
    winmapDF = pd.DataFrame.from_dict(winmap)
    winmapDFT = winmapDF.T
    # Pull the max value for that node
    winmapDFT["max_val_node"] = winmapDFT.max(axis=1)
    # Create a column that has the total observations for each node
    winmapDFT["total_obs_node"] = winmapDFT.iloc[:, 0:3].sum(axis=1)
    # Calculate the simple node purity (a more complex purity might include some sort of penalty for having moltiple obs set off)
    winmapDFT["node_purity"] = winmapDFT["max_val_node"] / winmapDFT["total_obs_node"]
    # Calculate the overall purity for the layer
    node_purity = winmapDFT['node_purity'].mean(axis=0)
    return(node_purity)


evaluate_purity(final_som, convolv_layer_one.values, y_train)

# Ok so does it make a difference how many nodes there are in the SOM? Can I compaire a 3X3 to a 5X5? is thaht fair?
# Next steps are to evaluate the performance of the nodes at each level. Maybe I can also print out the purity of the nofrsd
# at each step of the process when creating the SOM.




np.sum(list(winmap.values())).most_common()[0][0]


min(area_som_distance_from_weights[1])

area_som_distance_from_weights[0,2]


for d in convolution_layer_test.values:
        win_position = final_som.winner(d)
        if win_position in winmap:
            print(winmap[win_position].most_common()[0][0])
        else:
            print("No clear winner")


# Notes

# When i do one feature I get really good results. Like 90 % accurate but then when I add them all together I get worse results.
# Ok. So next steps. 
# First thing I'm going to do is to do some clustering. So I need to figure out how to visualise the output of the SOM.
# I'm going to do the data as a whole through 1 SOM - cluster the output
# Then I will do it on one feature. 
# Then I will do it on the output of my modular SOM.


# Create the ablity to measure the performance of the SOM

final_som



            # observation_key = [index_val, "-", feature]
            # observation_key = "".join(map(str,observation_key))



trained_soms_layer_1 = train_som_layer(X_train, list(list("area", "preimeter"), list("perimeter")))
# create a dictionary to store the trained SOM
trained_soms = {}

# For each feature in the data set we will train a SOM. I intend to update this so that we can pass this function a list of combos we want to 


data_values_array = train_value.values
if len(train_value.columns) == 1:
    data_values_array = np.array([[i] for i in data_values_array])
    print(data_values_array)
else:




#### FInish implementing the feature set ability of the algo

trained_soms_layer_1 = train_som_layer(X_train, [["area", "perimeter"], ["perimeter"]])


def train_som_layer(data, feature_collections):
    # create a dictionary to store the trained SOM
    trained_soms = {}
    # For each feature in the data set we will train a SOM. I intend to update this so that we can pass this function a list of combos we want to 
    # train our som's on.
    for feature_set in feature_collections:
        print("Training SOM on ", feature_set)
        print(len(feature_set))
        train_value = data[[feature_set]]
        # prepare the data
        train_value_array = data_vectorisation(train_value)
        observation_key = "".join(map(str,feature_set))
        som = create_train_som(train_value_array, len(feature_set))
        trained_soms.setdefault(observation_key,[]).append(som)
    return(trained_soms)

# ________________________ Now we extract the winning nodes ____________________________

# create dictionary to store the winning nodes for the given 

# where should purity evaluation occur?

# Train SOM's
# Extract winning node to build convolution layer
# Train final SOM on the convolution layer
# Evaluate performance



# ________________________________ FINAL TESTING OF THE WORKFLOW ___________________________

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
data = data[data.columns[0:6]]

names = data.columns
d = preprocessing.normalize(data, axis=0)
data_normalised = pd.DataFrame(d, columns=names)

# Create the test train split of our data.
X_train, X_test, y_train, y_test = train_test_split(data_normalised, labels)

# Not sure where this is used.....
# som_shape = [3,3]

# Create out feature collections
X_test_column_names = np.array([[i] for i in X_train.columns ])

# ________________________________ CREATE OUR MDSOM ____________________________________

# Create our first layer of SOMS
trained_soms_layer_1 = train_som_layer(data = X_train, feature_collections = X_test_column_names)

# Create our training convolutional layer that is used to blend the results from our layer 1 SOM's
convolv_layer_one_test = create_convolution_layer(data = X_train, trained_soms = trained_soms_layer_1,  feature_collections = X_test_column_names)

# Now using the values output from our training cololutional layer (I think I got half way through implementing the addition of the node number as well
# as the distance from the given node) So now my create train SOM has no idea what to do with the god dam outpuut.
final_som = create_train_som(data=convolv_layer_one_test, n_features = convolv_layer_one.shape[1], convolutional_layer=True)

# BOOM. We've got our MDSOM. The key elements are the trained soms and the final SOM. 

# Pass our test data through our constructed SOM 
convolution_layer_test = create_convolution_layer(data=X_test, trained_soms=trained_soms_layer_1, feature_collections = X_test_column_names)

# Extract the values for our area some so we can evaluate the purity of our nodes
area_som = trained_soms_layer_1['area'][0]
X_test_area = np.array([[i] for i in X_train['area'].values])











# _________________________ Alllowing new data to be add at higher level convolutions ____________________________



X_test_column_names

# Create our first layer of SOMS
trained_soms_layer_1 = train_som_layer(data = X_train, feature_collections = X_test_column_names)
# Create our training convolutional layer that is used to blend the results from our layer 1 SOM's
convolv_layer_one_train = create_convolution_layer(data = X_train, trained_soms = trained_soms_layer_1,  feature_collections = X_test_column_names)

# Create a second layer som based off our initial layer 
layer_2_feature_collection = pd.array([["area", "perimeter"], ["compactness", "length_kernel"], ["width_kernel","asymmetry_coefficient"]])
trained_soms_layer_2 = train_som_layer(data = convolv_layer_one_train, feature_collections = layer_2_feature_collection, convolutional_layer=True)
convolv_layer_two_train = create_convolution_layer(data = convolv_layer_one_train, trained_soms = trained_soms_layer_2,  feature_collections = layer_2_feature_collection, convolutional_layer=True)

# Create the final layer
final_som = create_train_som(data=convolv_layer_two_train, n_features = convolv_layer_two_train.shape[1], convolutional_layer=True)

# Create the testing layers
convolv_layer_one_test = create_convolution_layer(data = X_test, trained_soms = trained_soms_layer_1,  feature_collections = X_test_column_names)
convolv_layer_two_test = create_convolution_layer(data = convolv_layer_one_test, trained_soms = trained_soms_layer_2,  feature_collections = layer_2_feature_collection, convolutional_layer=True)

evaluate_purity(final_som, convolv_layer_two_test, y_test, convolutional_layer=True)


convolv_layer_2_complete_train = create_convolution_layer(data = convolv_layer_one_train_ab, trained_soms = trained_soms_layer_2_complete,  feature_collections = layer_2ab_feature_collection)


def create_convolution_layer(data, trained_soms, feature_collections, normalise=False):
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
        winning_nodes_a = []
        winning_nodes_b = []
        # loop through the array of observations and extract the winning node and its distance for the given observation.
        for observation in train_value_array:
            winning_pos = som.winner(observation)
            node_distance = distance_map[winning_pos]
            # We now convert this coordinant into a numerical value so we can feed it to our next layer, to to do this properly
            # we need the y value of the SOM as that helps us convery coords to a single didget.
            node_value = convert_coordinants(winning_pos, som_y_dim)
            output_a = node_value
            output_b = node_distance
            winning_nodes_a.append(output_a)
            winning_nodes_b.append(output_b)
        if normalise:
            print(winning_nodes_a)
            winning_nodes_a = preprocessing.normalize(winning_nodes_a, axis = 0)
            print(winning_nodes_a)
            winning_nodes_b = preprocessing.normalize(winning_nodes_b, axis = 0)
            winning_nodes_normalised = np.array(list(zip(winning_nodes_a, winning_nodes_b)))
            print(winning_nodes_normalised)
            dataframe[observation_key] = winning_nodes_normalised
        else:
            winning_nodes = zip(winning_nodes_a, winning_nodes_b)
            winning_nodes_out = (list(winning_nodes))
            # winning_nodes = np.concatenate((winning_nodes_a, winning_nodes_b))
            print(data_values_array)
            print(data_values_array.shape)
            dataframe[observation_key] = winning_nodes
    return(dataframe)


convolv_layer_one_train = create_convolution_layer(data = X_train, trained_soms = trained_soms_layer_1,  feature_collections = feature_collections_1,   normalise = False)




numbers = [1, 2, 3]
letters = [1, 1, 1]
zipped = zip(numbers, letters)

np.array(list(zipped)).shape
list(zipped)