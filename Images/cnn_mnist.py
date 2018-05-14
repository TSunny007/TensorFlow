from __future__ import absolute_import
from __future__ import division
from __future__ import print_function 

#Imports
import numpy as np 
import tensorflow as tf


# Convolutional neural networks (CNNs) are the current state-of-the-art model architecture for image classification tasks. 
# CNNs apply a series of filters to the raw pixel data of an image to extract and learn higher-level features, 
# which the model can then use for classification. CNNs contains three components:

# Convolutional layers, which apply a specified number of convolution filters to the image. For each subregion, 
# the layer performs a set of mathematical operations to produce a single value in the output feature map. 
# Convolutional layers then typically apply a ReLU activation function to the output to introduce nonlinearities into the model.


# Pooling layers, which downsample the image data extracted by the convolutional layers to reduce the dimensionality of the feature map
# in order to decrease processing time. A commonly used pooling algorithm is max pooling, which extracts subregions of the feature map 
# (e.g., 2x2-pixel tiles), keeps their maximum value, and discards all other values.

# Dense (fully connected) layers, which perform classification on the features extracted by the convolutional layers 
# and downsampled by the pooling layers. In a dense layer, every node in the layer is connected to every node in the preceding layer.


# Typically, a CNN is composed of a stack of convolutional modules that perform feature extraction. Each module consists of a convolutional layer
# followed by a pooling layer. The last convolutional module is followed by one or more dense layers that perform classification.
# The final dense layer in a CNN contains a single node for each target class in the model (all the possible classes the model may predict),
# with a softmax activation function to generate a value between 0–1 for each node (the sum of all these softmax values is equal to 1).
# We can interpret the softmax values for a given image as relative measurements of how likely it is that the image falls into each target class.
tf.logging.set_verbosity(tf.logging.INFO)

# Our Application logic
def cnn_model_fn(features, labels, mode):
    """ Model funtion for CNN. """
    # Input layer #1
    input_layer = tf.reshape(features["x"], [-1,28,28,1]) # Batch size, image_height, image_width, channels (monochrome=1)
    
    # Convolutional Layer #1
    conv = tf.layers.conv2d (
            inputs=input_layer,
            filters=32,
            kernel_size=[5,5],
            padding="same",
            activation=tf.nn.relu)
    
    
if __name__ == "__main__":
    tf.app.run()