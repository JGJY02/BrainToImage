import os.path
import random

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.utils import array_to_img
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

def sample_image_by_label(images, labels, target_label):
    """
    Sample a random image from the dataset with the specified label.

    Parameters:
    images (numpy.ndarray): Array of images.
    labels (numpy.ndarray): Array of labels corresponding to the images.
    target_label (int): The label of the image to sample.

    Returns:
    numpy.ndarray: A randomly sampled image with the specified label.
    """
    # Find indices of images with the target label
    indices = np.where(labels == target_label)[0]

    # Randomly select an index
    random_index = np.random.choice(indices)

    # Return the image at the selected index
    return images[random_index]



def get_balanced_mnist_subset(X_data,y_label,n_samples_per_class=1000, random_state=42):
    """ Fetch a balanced subset of MNIST data.

    Parameters:
    - n_samples_per_class (int): Number of samples to select for each digit class (0-9).
    - random_state (int): Random seed for reproducibility.

    Returns:
    - X (numpy.ndarray): Array of flattened images (n_samples, 784)
    - y (numpy.ndarray): Array of labels (n_samples,)
    """
    # Fetch the full MNIST dataset
    #X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

    # Convert to numpy arrays and ensure correct data types
    #X = X.astype(np.float32)
    #y = y.astype(np.int32)

    # Shuffle the dataset
    X, y = shuffle(X_data, y_label, random_state=random_state)

    # Create a balanced subset
    balanced_X = []
    balanced_y = []

    for digit in range(10):
        # Find indices of the current digit
        digit_indices = np.where(y == digit)[0]

        # Select n_samples_per_class random indices for this digit
        selected_indices = np.random.choice(digit_indices, n_samples_per_class, replace=False)

        # Add selected samples to the balanced subset
        balanced_X.append(X[selected_indices])
        balanced_y.append(y[selected_indices])

    # Concatenate the balanced subsets
    balanced_X = np.concatenate(balanced_X)
    balanced_y = np.concatenate(balanced_y)

    # Shuffle the balanced dataset
    balanced_X, balanced_y = shuffle(balanced_X, balanced_y, random_state=random_state)

    return balanced_X, balanced_y

def sample_images(my_array, num_samples, replace=True):
    """ Randomly sample images from an array.

    This function uses numpy's random.choice to select random samples from the input array.

    Parameters:
    my_array (array-like): The array of images to sample from.
    num_samples (int): The number of samples to select.
    replace (bool, optional): Whether to sample with replacement. Defaults to True.

    Returns:
    numpy.ndarray: An array of randomly sampled images.

    Example:
    >>> images = np.array([1, 2, 3, 4, 5])
    >>> sampled = sample_images(images, 3, replace=False)
    >>> print(sampled)
    [3 1 5]
    """
    # must be a 1d array for this
    #random_samples = np.random.choice(my_array, size=num_samples, replace=replace)
    sample_indices = np.random.choice(my_array.shape[0], size=num_samples, replace=replace)
    #print(sample_indices)
    random_samples = my_array[sample_indices]

    return random_samples

def show_samples(samples):
    if len(samples) > 4:
        raise ValueError

    r, c = 2, 2
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            img = array_to_img(samples[cnt].reshape((28,28,1)))
            axs[i,j].imshow(img,cmap='gray')
            axs[i,j].axis('off')
            cnt += 1

    return fig

def load_local_mnist(data_dir, norm=False, sparse=False):
    x_train = np.load(os.path.join(data_dir, 'x_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    x_test = np.load(os.path.join(data_dir, 'x_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

    if norm == -1: #to normalize between -1 / 1
        x_train = (x_train.astype(np.float32) - 127.5) / 127.5
        x_test = (x_test.astype(np.float32) - 127.5) / 127.5
    elif norm == 1: # to normalize between 0 / 1
        x_train = x_train.astype(np.float32) / 255
        x_test = x_test.astype(np.float32) / 255

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    if sparse: # sparse_categorical_crossentropy
        y_train = y_train.reshape(-1,1)
        y_test = y_test.reshape(-1,1)
    else: # assume categorical_crossentropy
        y_train = np.array(to_categorical(y_train ,10)).astype(np.uint8)
        y_test = np.array(to_categorical(y_test, 10)).astype(np.uint8)

    return (x_train, y_train), (x_test, y_test)

if __name__ == '__main__':
    print(os.getcwd())
    data_dir = "./Datasets/MNIST_dataset"
    # Download the MNIST dataset
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # np.save(os.path.join(data_dir, 'x_train.npy'), x_train)
    # np.save(os.path.join(data_dir, 'y_train.npy'), y_train)
    # np.save(os.path.join(data_dir, 'x_test.npy'), x_test)
    # np.save(os.path.join(data_dir, 'y_test.npy'), y_test)

    # Load the data from .npy files
    #x_train = np.load(os.path.join(data_dir, 'x_train.npy'))
    #y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    x_test = np.load(os.path.join(data_dir, 'x_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

    #show_samples(sample_images(x_train,4,replace=False))

    ## Example to normalize between 0 / 1
    # x_train = x_train.astype('float32') / 255
    # x_test = x_test.astype('float32') / 255

    ## Example to normilize between -1 / 1
    #x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    x_test = (x_test.astype(np.float32) - 127.5) / 127.5

    #x_train = np.expand_dims(x_train, axis=3)
    #x_test = np.expand_dims(x_test, axis=3)
    x_test = x_test[:, :, :, None]

    # Reshape the data to add channel dimension
    # x_train (60000, 28, 28) -> reshape (60000, 28, 28, 1)
    #x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    # this is equivalent
    # x_train = np.expand_dims(x_train, axis=3)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    # Convert class vectors to binary class matrices (one-hot encoding)
    #y_train = to_categorical(y_train)
    #y_test = to_categorical(y_test)
    y_test = y_test.reshape(-1,1)  ## sparse catergorical


    # Set batch size
    batch_size = 32

    # Create data generators
    train_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()

    ## Create iterators
    ## could also be flow_from_directory or flow_from dataframe
    train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
    test_generator = test_datagen.flow(x_test, y_test, batch_size=batch_size)