import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import initializers, regularizers
from keras.layers import Input, Layer

######
## Adapted and updated from MoGLayer example given in https://github.com/ptirupat/ThoughtViz/blob/master/layers/mog_layer.py
######

class MoGLayer(Layer):

    def __init__(self,
                 kernel_regularizer=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        super(MoGLayer, self).__init__(**kwargs)

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.std = self.add_weight(shape=(input_dim,),
                                      name='std',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer)

        self.mean = self.add_weight(shape=(input_dim,),
                                    initializer=self.bias_initializer,
                                    name='mean')

        self.built = True

    def call(self, inputs):
        output = inputs * self.std
        output = K.bias_add(output, self.mean)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = input_shape[-1]
        return tuple(output_shape)

    def get_config(self):
        base_config = super(MoGLayer, self).get_config()
        return {
            **base_config,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "kernel_regularizer": self.kernel_regularizer,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

if __name__ == '__main__':
    space = np.random.normal(0,1 ,(1,128))
    inp = Input(shape=(1,128))
    mog = MoGLayer()(space)
    print(mog.shape)
