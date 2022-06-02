# This script is going to be an example of how to construct each of the differing MDSOM structures.

from mdsom_function_dev_convo_layer import *
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

# Benchmark PCA plot
x = pca_plot(data = X_train, target_array= y_train)
x.show()

# ________________________________ CREATE A SOM TRAINED OFF ALL VALUES ____________________________________

col_results_purity = []
col_results_class = []

for n_cols in range(12):
    print(n_cols)
    data_for_evaluation = X_train[X_train.columns[0:n_cols]]
    purity_results = []
    class_results = []
    for _ in range(10):
        # Train the SOM
        standard_som = create_train_som(data=data_for_evaluation.values, n_samples=data_for_evaluation.shape[0], n_features = data_for_evaluation.shape[1], convolutional_layer=False)
        purity = evaluate_purity(standard_som, data_for_evaluation.values, y_train)
        purity_results.append(purity)
        evaluated_data = label_output(standard_som, data_for_evaluation, y_train, convolutional_layer = False, original = True)
        evaluated_data["correct"] = np.where( (evaluated_data["default_class"] == evaluated_data['evaluated_class']), 1, 0)
        correct_class = sum(evaluated_data["correct"])/len(evaluated_data.index)
        class_results.append(correct_class)
    col_results_class.append(statistics.mean(class_results))
    col_results_purity.append(statistics.mean(purity_results))

col_results_class
col_results_purity

d = {'n_col': list(range(len(col_results_class))) , 'class': col_results_class, 'purity': col_results_purity}
df = pd.DataFrame(data=d)

x = pca_plot(data = data_for_evaluation, target_array= evaluated_data['evaluated_class'].values)
x.savefig('visulisations/standard_som_all_features.png', bbox_inches='tight')
x.show()


# What are my options here. 
# I can just use the method from the original paper. Then just run it ten times. Show the two scores and the two plots
# I feel like that isnt enough test cases. Ok so why dont I start with results from a single layer SOM. With all features from DS

# ________________________________ CREATE SINGLE LAYER MDSOM WITH DIFFERING ____________________________________

# Create simple feature collections
feature_collections_1 = np.array([[i] for i in X_train.columns ])
feature_collections_1 = pd.array([['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar'], ['chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']])



######## Results using only node location as single value index ##############
purity_results = []
class_results = []

for _ in range(10):
    trained_soms_layer_1 = train_som_layer(data = X_train, n_samples=X_train.shape[0], feature_collections = feature_collections_1)
    convolv_layer_one_train = create_convolution_layer_only_winning_som(data = X_train, trained_soms = trained_soms_layer_1,  feature_collections = feature_collections_1,   normalise = False)
    final_som = create_train_som(data= convolv_layer_one_train.values, n_samples=11, n_features = convolv_layer_one_train.shape[1], convolutional_layer=False)
    purity = evaluate_purity(final_som, convolv_layer_one_train.values, y_train, convolutional_layer=False)
    purity_results.append(purity)
    evaluated_data = label_output(som = final_som, data = X_train, targets = y_train, final_convolution=convolv_layer_one_train,convolutional_layer = False, original = True)
    evaluated_data["correct"] = np.where( (evaluated_data["default_class"] == evaluated_data['evaluated_class']), 1, 0)
    correct_class = sum(evaluated_data["correct"])/len(evaluated_data.index)
    class_results.append(correct_class)

statistics.mean(purity_results)
statistics.mean(class_results)

x = pca_plot(data = data_for_evaluation, target_array= evaluated_data['evaluated_class'].values)
x.savefig('visulisations/standard_som_all_features.png', bbox_inches='tight')
x.show()

########### Results using node location and distance ################
purity_results = []
class_results = []

for _ in range(10):
    trained_soms_layer_1 = train_som_layer(data = X_train, n_samples=X_train.shape[0], feature_collections = feature_collections_1)
    convolv_layer_one_train = create_convolution_layer_zw(data = X_train, trained_soms = trained_soms_layer_1,  feature_collections = feature_collections_1,   normalise = False)
    final_som = create_train_som(data= convolv_layer_one_train, n_samples=X_train.shape[0], n_features = convolv_layer_one_train.shape[1]*2, convolutional_layer=True)
    purity = evaluate_purity(final_som, convolv_layer_one_train, y_train, convolutional_layer=True)
    purity_results.append(purity)
    evaluated_data = label_output(som = final_som, data = X_train, targets = y_train, final_convolution=convolv_layer_one_train,convolutional_layer = True, original = True)
    evaluated_data["correct"] = np.where( (evaluated_data["default_class"] == evaluated_data['evaluated_class']), 1, 0)
    correct_class = sum(evaluated_data["correct"])/len(evaluated_data.index)
    class_results.append(correct_class)

statistics.mean(purity_results)
statistics.mean(class_results)

x = pca_plot(data = data_for_evaluation, target_array= evaluated_data['evaluated_class'].values)
x.savefig('visulisations/standard_som_all_features.png', bbox_inches='tight')
x.show()

########### Results using node location and distance normalised ################
purity_results = []
class_results = []

# Results using node location and distance and location normalised

for _ in range(10):
    trained_soms_layer_1 = train_som_layer(data = X_train, n_samples=X_train.shape[0], feature_collections = feature_collections_1)
    convolv_layer_one_train = create_convolution_layer_zw(data = X_train, trained_soms = trained_soms_layer_1,  feature_collections = feature_collections_1,   normalise = True)
    final_som = create_train_som(data= convolv_layer_one_train, n_samples=X_train.shape[0], n_features = convolv_layer_one_train.shape[1]*2, convolutional_layer=True)
    purity = evaluate_purity(final_som, convolv_layer_one_train, y_train, convolutional_layer=True)
    purity_results.append(purity)
    evaluated_data = label_output(som = final_som, data = X_train, targets = y_train, final_convolution=convolv_layer_one_train,convolutional_layer = True, original = True)
    evaluated_data["correct"] = np.where( (evaluated_data["default_class"] == evaluated_data['evaluated_class']), 1, 0)
    correct_class = sum(evaluated_data["correct"])/len(evaluated_data.index)
    class_results.append(correct_class)

statistics.mean(purity_results)
statistics.mean(class_results)

x = pca_plot(data = data_for_evaluation, target_array= evaluated_data['evaluated_class'].values)
x.savefig('visulisations/standard_som_all_features.png', bbox_inches='tight')
x.show()

########### Results using node location as coords and distance ################
purity_results = []
class_results = []

for _ in range(10):
    trained_soms_layer_1 = train_som_layer(data = X_train, n_samples=X_train.shape[0], feature_collections = feature_collections_1)
    convolv_layer_one_train = create_convolution_layer_xyw(data = X_train, trained_soms = trained_soms_layer_1,  feature_collections = feature_collections_1,   normalise = False)
    final_som = create_train_som(data= convolv_layer_one_train, n_samples=X_train.shape[0], n_features = convolv_layer_one_train.shape[1]*3, convolutional_layer=True)
    purity = evaluate_purity(final_som, convolv_layer_one_train, y_train, convolutional_layer=True)
    purity_results.append(purity)
    evaluated_data = label_output(som = final_som, data = X_train, targets = y_train, final_convolution=convolv_layer_one_train,convolutional_layer = True, original = True)
    evaluated_data["correct"] = np.where( (evaluated_data["default_class"] == evaluated_data['evaluated_class']), 1, 0)
    correct_class = sum(evaluated_data["correct"])/len(evaluated_data.index)
    class_results.append(correct_class)

statistics.mean(purity_results)
statistics.mean(class_results)

x = pca_plot(data = data_for_evaluation, target_array= evaluated_data['evaluated_class'].values)
x.savefig('visulisations/standard_som_all_features.png', bbox_inches='tight')
x.show()

########### Results using node location in vector of 0's ################
purity_results = []
class_results = []

for _ in range(10):
    trained_soms_layer_1 = train_som_layer(data = X_train, n_samples=X_train.shape[0], feature_collections = feature_collections_1)
    convolv_layer_one_train = create_convolution_layer_g(data = X_train, trained_soms = trained_soms_layer_1,  feature_collections = feature_collections_1,   normalise = False)
    final_som = create_train_som(data= convolv_layer_one_train, n_samples=X_train.shape[0], n_features = 2156, convolutional_layer=True)
    purity = evaluate_purity(final_som, convolv_layer_one_train, y_train, convolutional_layer=True)
    purity_results.append(purity)
    evaluated_data = label_output(som = final_som, data = X_train, targets = y_train, final_convolution=convolv_layer_one_train,convolutional_layer = True, original = True)
    evaluated_data["correct"] = np.where( (evaluated_data["default_class"] == evaluated_data['evaluated_class']), 1, 0)
    correct_class = sum(evaluated_data["correct"])/len(evaluated_data.index)
    class_results.append(correct_class)

statistics.mean(purity_results)
statistics.mean(class_results)

########### Results using distance to observation in vector of 0's ################
purity_results = []
class_results = []

for _ in range(10):
    trained_soms_layer_1 = train_som_layer(data = X_train, n_samples=X_train.shape[0], feature_collections = feature_collections_1)
    convolv_layer_one_train = create_convolution_layer_g_w(data = X_train, trained_soms = trained_soms_layer_1,  feature_collections = feature_collections_1,   normalise = False)
    final_som = create_train_som(data= convolv_layer_one_train, n_samples=X_train.shape[0], n_features = 392, convolutional_layer=True)
    purity = evaluate_purity(final_som, convolv_layer_one_train, y_train, convolutional_layer=True)
    purity_results.append(purity)
    evaluated_data = label_output(som = final_som, data = X_train, targets = y_train, final_convolution=convolv_layer_one_train,convolutional_layer = True, original = True)
    evaluated_data["correct"] = np.where( (evaluated_data["default_class"] == evaluated_data['evaluated_class']), 1, 0)
    correct_class = sum(evaluated_data["correct"])/len(evaluated_data.index)
    class_results.append(correct_class)

statistics.mean(purity_results)
statistics.mean(class_results)




# Pass our test data through our constructed SOM 
convolution_layer_test = create_convolution_layer(data=X_test, trained_soms=trained_soms_layer_1, feature_collections = feature_collections_1)

evaluate_purity(final_som, convolution_layer_test, y_test, convolutional_layer=True)

# def pca_plot(som, data, targets, final_convolution = "", convolutional_layer = False):

x = pca_plot(som = final_som, data = X_train,targets = y_train, final_convolution = convolv_layer_one_train, convolutional_layer = True)







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







