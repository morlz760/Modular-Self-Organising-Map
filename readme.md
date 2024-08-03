# functions

Workflow for DEEP SOM

What is a patch
What is a stride
What is a kernal

Step 1 Extract patches

The first step is to extract the patches from the image. 

To do this we use a kenal that we slide over the image at a given stride value. 

The image is given a buffer around the boarder so that all the regions of the image are captured. This buffer is created based off the kernal value. 

Each patch is then added to a dictionary with the coordinants of the patch are used as the key.

This process is repeated for each image in the training set. The patch values are appended to the list of each specific coordiant.

Step 2 Identify winning SOM

Cycle through each of the patch coordiants

Flatten the array dont know  if I should include or not

Train a SOM based on this array

Now extract the winning som node for each patch and save this in its patches coordnants index.


Step 3 construct the convolution layer 

For each image the winning node for each patch of that image is selected and placed in its index. Creating a convolutional layer. 

Step 4

The convoltional layer for each image is then used to train the next SOM. 


Define the kernal

Extract the patches for each of the images when we do this what were doing is extracting each patch for the given region of the image so we will have patch X,Y for image 1:n stored in one dictionary array.

Next we create and train our SOM on each of these patch locations. So a SOM is trained for each X,Y coordinant. 



