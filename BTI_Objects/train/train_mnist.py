import os
import pickle
import sys

from keras.callbacks import ReduceLROnPlateau
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

sys.path.append(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))
from models.mnist_models import LeNet5v2
from utils.local_MNIST import get_balanced_mnist_subset, load_local_mnist

## based on paper
# # (http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)

class_labels = [0,1,2,3,4,5,6,7,8,9]
batch_size = 64
epochs = 30
## load the MNIST trainig data
image_data_dir = "Datasets\MNIST_dataset"
print(f" Loading MNIST images from {image_data_dir}")
(x_train, y_train) , (x_test, y_test) = load_local_mnist(image_data_dir,norm=-1,sparse=False)

## paper calls for some image augmentation, which has becaome a standard technique in image classificaiton
data_aug = ImageDataGenerator(
        rotation_range = 10,        # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1,           # Randomly zoom image
        width_shift_range = 0.1,    # randomly shift images horizontally (fraction of total width)
        height_shift_range = 0.1,   # randomly shift images vertically (fraction of total height))
        )

data_aug.fit(x_train)

opt = Adam(0.0001, 0.9, decay=1e-6)
losses = ['categorical_crossentropy']
metric = ['accuracy']
variable_learning_rate = ReduceLROnPlateau(monitor='val_loss', factor = 0.2, patience = 2)

mnist_model = LeNet5v2((28,28,1),10,verbose=False)
mnist_model.compile(optimizer=opt, loss=losses, metrics=metric)

history = mnist_model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size, callbacks = [variable_learning_rate], validation_split=0.25)
with open(f"{image_data_dir}/mnist_history.pkl","wb") as f:
    pickle.dump(history,f)
mnist_model.save_weights(f"{image_data_dir}/mnist_model_weights.h5")