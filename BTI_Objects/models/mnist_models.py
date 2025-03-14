from keras.regularizers import l2
from keras.layers import (Activation, BatchNormalization, Conv2D, Dense,
                          Dropout, Flatten, Input, MaxPooling2D)
from keras.models import Model, Sequential



## MNIST Model
# Implemented a modified LeNet-5.
# LeNet-5 - GradientBased Learning Applied to Document Recognition (Yann LeCun Leon Bottou Yoshua Bengio and Patrick Haffner)
# (http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)
########

def LeNet5v2(input_shape = (28, 28, 1), classes = 10, verbose=False):
    """
    Implementation of a modified LeNet-5.
    Only those layers with learnable parameters are counted in the layer numbering.

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    model = Sequential([
        Conv2D(filters = 32, kernel_size = 5, strides = 1, activation = 'relu', input_shape = (28,28,1), kernel_regularizer=l2(0.0005), name = 'convolution_1'),
        Conv2D(filters = 32, kernel_size = 5, strides = 1, name = 'convolution_2', use_bias=False),
        BatchNormalization(name = 'batchnorm_1'),
        Activation("relu"),
        MaxPooling2D(pool_size = 2, strides = 2, name = 'max_pool_1'),
        Dropout(0.25, name = 'dropout_1'),
        Conv2D(filters = 64, kernel_size = 3, strides = 1, activation = 'relu', kernel_regularizer=l2(0.0005), name = 'convolution_3'),
        Conv2D(filters = 64, kernel_size = 3, strides = 1, name = 'convolution_4', use_bias=False),
        BatchNormalization(name = 'batchnorm_2'),
        Activation("relu"),
        MaxPooling2D(pool_size = 2, strides = 2, name = 'max_pool_2'),
        Dropout(0.25, name = 'dropout_2'),
        Flatten(name = 'flatten'),
        Dense(units = 256, name = 'fully_connected_1', use_bias=False),
        BatchNormalization(name = 'batchnorm_3'),
        Activation("relu"),
        Dense(units = 128, name = 'fully_connected_2', use_bias=False),
        BatchNormalization(name = 'batchnorm_4'),
        Activation("relu"),
        Dense(units = 84, name = 'fully_connected_3', use_bias=False),
        BatchNormalization(name = 'batchnorm_5'),
        Activation("relu"),
        Dropout(0.25, name = 'dropout_3'),
        Dense(units = 10, activation = 'softmax', name = 'output')
    ])

    model._name = 'LeNet5v2'
    if verbose:
        model.summary(show_trainable=True,expand_nested=True)

    return model


if __name__ == '__main__':
    mnist_model = LeNet5v2((28,28,1),10,verbose=True)