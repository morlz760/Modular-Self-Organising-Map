import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eig
import pandas as pd
from minisom import MiniSom  
import math
import ipynb
from sklearn.metrics import classification_report
import cv2
from itertools import chain

%matplotlib inline

# Read in our data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Here we have to flatten our data so we are able to feed it to the SOM
x_test_r = flatten_and_reshape(x_test)
x_train_r = flatten_and_reshape(x_train)

# Create kernal
kernel = np.array((
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1]), dtype="int")


# Apply the kernal to the images to extract the wanted area
output = {}
image_index = 0

for image in image_list:

    
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]
    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    
    # Here we actually slide the kernal across. The first for loop allows us to slide across
    # the height or each row as you will of the image, by the padding widthh
    for y in np.arange(pad, iH + pad):
        # This second for loop allows us to slide across the "columns" or te elements in that 
        # sepcific array 
        for x in np.arange(pad, iW + pad):
            # Now that we have the specific coordinants we're looking at on the image 
            # the height and width position (x,y) we can add the padding to get our patch
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
            # Now, normally we would apply the kernal to our patch to augment the overall image
            # however we do not wish to do that. We are aiming to apply a SOM to this specific 
            # region of the image. 
            index = [y - pad,"-",x - pad]
            index = "".join(map(str,index))
            output.setdefault(index,[]).append(roi)
            
            
    image_index = image_index + 1


# here we apply the SOM on the given area

som_y_size = 25

labels = y_train[0:100]



som_results = {}

for key, value in output.items():
    
    
    print("processing patch ", key)
    # Here we have the kernal results for that specific region of each image. We convert to array and reshape so we can feed to SOM
    my_array = np.array(value)
    flat_array = flatten_and_reshape(my_array)
    
    # Using the data we now train our SOM on the specific region of the image
    som = create_train_som(flat_array, som_y_size)
    
    
    index_val = 0
    print("index value ", index_val)
    
    for x in flat_array:
        
        print("extracting winner for image ", index_val)
        winning_pos = som.winner(x)
#         print(winning_pos)
            # We now convert this coordinant into a numerical value so we can feed it to our next layer
        k = convert_coordinants(winning_pos, som_y_size)
            # store the convolved value in the output (x,y)-
            # coordinate of the output image
        print("winning value", k)
        image_key = [index_val, "-", key]
        image_key = "".join(map(str,image_key))
        print("saving to index", image_key)        
        som_results.setdefault(image_key,[]).append(k)
#         output[y - pad, x - pad] = k
    
        index_val = index_val + 1
        
   
# Now were apply a filtered dictionary

def filter_dict(d, filter_string):
for key, val in d.items():
    values = key.split("-")
    image_index = values[0]
    if filter_string != image_index:
        continue
    yield key, val


# Here we constrct the convolution layer
for key, val in filter_dict(som_results, "1"):
#     print(key)
    values = key.split("-")
    image_index = values[0]
#     print(image_index)
    y = int(values[1])
    x = int(values[2])
    output[y - pad, x - pad] = val
    

