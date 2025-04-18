# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import glob
import numpy as np
import tensorflow as tf
import tfutil

##


#----------------------------------------------------------------------------
# Parse individual image from a tfrecords file.

def parse_tfrecord_tf(record):
    features = tf.compat.v1.parse_single_example(record, features={
        'shape': tf.compat.v1.FixedLenFeature([3], tf.compat.v1.int64),
        'data': tf.compat.v1.FixedLenFeature([], tf.compat.v1.string)})
    data = tf.compat.v1.decode_raw(features['data'], tf.compat.v1.uint8)
    return tf.compat.v1.reshape(data, features['shape'])

def parse_tfrecord_np(record):
    ex = tf.compat.v1.train.Example()
    ex.ParseFromString(record)
    shape = ex.features.feature['shape'].int64_list.value
    data = ex.features.feature['data'].bytes_list.value[0]
    return np.fromstring(data, np.uint8).reshape(shape)

#----------------------------------------------------------------------------
# Dataset class that loads data from tfrecords files.

class TFRecordDataset:
    def __init__(self,
        tfrecord_dir,               # Directory containing a collection of tfrecords files.
        resolution      = None,     # Dataset resolution, None = autodetect.
        label_file      = None,     # Relative path of the labels file, None = autodetect.
        max_label_size  = 0,        # 0 = no labels, 'full' = full labels, <int> = N first label components.
        repeat          = True,     # Repeat dataset indefinitely.
        shuffle_mb      = 4096,     # Shuffle data within specified window (megabytes), 0 = disable shuffling.
        prefetch_mb     = 2048,     # Amount of data to prefetch (megabytes), 0 = disable prefetching.
        buffer_mb       = 256,      # Read buffer size (megabytes).
        num_threads     = 2,         # Number of concurrent threads.
        signals_file    = None,
        encoded_label_file = None
        ):       

        self.tfrecord_dir       = tfrecord_dir
        self.resolution         = None
        self.resolution_log2    = None
        self.shape              = []        # [channel, height, width]
        self.dtype              = 'uint8'
        self.dynamic_range      = [0, 255]
        self.label_file         = label_file
        self.label_size         = None      # [component]
        self.label_dtype        = None
        self._np_labels         = None
        self._tf_minibatch_in   = None
        self._tf_labels_var     = None
        self._tf_labels_dataset = None
        self._tf_datasets       = dict()
        self._tf_iterator       = None
        self._tf_init_ops       = dict()
        self._tf_minibatch_np   = None
        self._cur_minibatch     = -1
        self._cur_lod           = -1

        self.signals_file       = signals_file
        self.encoded_label_file = encoded_label_file
        self._np_encoded_labels        = None
        self._tf_encoded_labels_var     = None
        self._tf_encoded_labels_dataset = None

        
        # List tfrecords files and inspect their shapes.
        assert os.path.isdir(self.tfrecord_dir)
        tfr_files = sorted(glob.glob(os.path.join(self.tfrecord_dir, '*.tfrecords')))
        assert len(tfr_files) >= 1
        tfr_shapes = []
        for tfr_file in tfr_files:
            tfr_opt = tf.compat.v1.python_io.TFRecordOptions(tf.compat.v1.python_io.TFRecordCompressionType.NONE)
            for record in tf.compat.v1.python_io.tf_record_iterator(tfr_file, tfr_opt):
                tfr_shapes.append(parse_tfrecord_np(record).shape)
                break

        # Autodetect label filename.
        if self.label_file is None:
            guess = sorted(glob.glob(os.path.join(self.tfrecord_dir, '*.labels')))
            if len(guess):
                self.label_file = guess[0]
        elif not os.path.isfile(self.label_file):
            guess = os.path.join(self.tfrecord_dir, self.label_file)
            if os.path.isfile(guess):
                self.label_file = guess

        # Determine shape and resolution.
        max_shape = max(tfr_shapes, key=lambda shape: np.prod(shape))
        self.resolution = resolution if resolution is not None else max_shape[1]
        self.resolution_log2 = int(np.log2(self.resolution))
        self.shape = [max_shape[0], self.resolution, self.resolution]
        tfr_lods = [self.resolution_log2 - int(np.log2(shape[1])) for shape in tfr_shapes]
        assert all(shape[0] == max_shape[0] for shape in tfr_shapes)
        assert all(shape[1] == shape[2] for shape in tfr_shapes)
        assert all(shape[1] == self.resolution // (2**lod) for shape, lod in zip(tfr_shapes, tfr_lods))
        assert all(lod in tfr_lods for lod in range(self.resolution_log2 - 1))

        # Load labels.
        print("The label size is , ", max_label_size)
        assert max_label_size == 'full' or max_label_size >= 0
        self._np_labels = np.zeros([1<<20, 0], dtype=np.float32)
        if self.label_file is not None and max_label_size != 0:
            self._np_labels = np.load(self.label_file)
            assert self._np_labels.ndim == 2
        if max_label_size != 'full' and self._np_labels.shape[1] > max_label_size:
            self._np_labels = self._np_labels[:, :max_label_size]
        self.label_size = self._np_labels.shape[1]
        self.label_dtype = self._np_labels.dtype.name

        #Beginning of Jared additions


        # Load signals (JARED)
        if self.signals_file is None:
            guess = sorted(glob.glob(os.path.join(self.tfrecord_dir, '*.signal')))
            if len(guess):
                self.signals_file = guess[0]
                
        self.eeg_signals = np.load(self.signals_file)
        self.eeg_signals_size = self.eeg_signals.shape[1]
        self.eeg_signals_dtype = self.eeg_signals.dtype.name

        #Encoded Labels
        if self.encoded_label_file is None:
            guess = sorted(glob.glob(os.path.join(self.tfrecord_dir, '*.encodedLabels')))
            if len(guess):
                self.encoded_label_file = guess[0]
        elif not os.path.isfile(self.encoded_label_file):
            guess = os.path.join(self.tfrecord_dir, self.encoded_label_file)
            if os.path.isfile(guess):
                self.encoded_label_file = guess

        # Load labels.
        assert max_label_size == 'full' or max_label_size >= 0
        self._np_encoded_labels = np.zeros([1<<20, 0], dtype=np.float32)
        if self.encoded_label_file is not None and max_label_size != 0:
            self._np_encoded_labels = np.load(self.encoded_label_file)
            assert self._np_encoded_labels.ndim == 2
        if max_label_size != 'full' and self._np_encoded_labels.shape[1] > max_label_size:
            self._np_encoded_labels = self._np_encoded_labels[:, :max_label_size]
        self.label_size = self._np_encoded_labels.shape[1]
        self.label_dtype = self._np_encoded_labels.dtype.name


        # Build TF expressions.
        with tf.compat.v1.name_scope('Dataset'), tf.compat.v1.device('/cpu:0'):
            tf.compat.v1.disable_eager_execution() # error if this is enabled
            self._tf_minibatch_in = tf.compat.v1.placeholder(tf.compat.v1.int64, name='minibatch_in', shape=[])
            tf_labels_init = tf.compat.v1.zeros(self._np_labels.shape, self._np_labels.dtype)
            self._tf_labels_var = tf.compat.v1.Variable(tf_labels_init, name='labels_var')
            tfutil.set_vars({self._tf_labels_var: self._np_labels})
            self._tf_labels_dataset = tf.compat.v1.data.Dataset.from_tensor_slices(self._tf_labels_var)

            ## add 
            tf_signals_init = tf.compat.v1.zeros(self.eeg_signals.shape, self.eeg_signals.dtype)
            self._tf_signals_var = tf.compat.v1.Variable(tf_signals_init, name='signals_var')
            tfutil.set_vars({self._tf_signals_var: self.eeg_signals})
            self._tf_signals_dataset = tf.compat.v1.data.Dataset.from_tensor_slices(self._tf_signals_var)

            tf_encoded_labels_init = tf.compat.v1.zeros(self._np_encoded_labels.shape, self._np_encoded_labels.dtype)
            self._tf_encoded_labels_var = tf.compat.v1.Variable(tf_encoded_labels_init, name='labels_var')
            tfutil.set_vars({self._tf_encoded_labels_var: self._np_encoded_labels})
            self._tf_encoded_labels_dataset = tf.compat.v1.data.Dataset.from_tensor_slices(self._tf_encoded_labels_var)



            for tfr_file, tfr_shape, tfr_lod in zip(tfr_files, tfr_shapes, tfr_lods):
                if tfr_lod < 0:
                    continue
                dset = tf.compat.v1.data.TFRecordDataset(tfr_file, compression_type='', buffer_size=buffer_mb<<20)
                dset = dset.map(parse_tfrecord_tf, num_parallel_calls=num_threads)
                dset = tf.compat.v1.data.Dataset.zip((dset, self._tf_labels_dataset, self._tf_signals_dataset, self._tf_encoded_labels_dataset))
                bytes_per_item = np.prod(tfr_shape) * np.dtype(self.dtype).itemsize
                if shuffle_mb > 0:
                    dset = dset.shuffle(((shuffle_mb << 20) - 1) // bytes_per_item + 1)
                if repeat:
                    dset = dset.repeat()
                if prefetch_mb > 0:
                    dset = dset.prefetch(((prefetch_mb << 20) - 1) // bytes_per_item + 1)
                dset = dset.batch(self._tf_minibatch_in)
                self._tf_datasets[tfr_lod] = dset
            self._tf_iterator = tf.compat.v1.data.Iterator.from_structure(self._tf_datasets[0].output_types, self._tf_datasets[0].output_shapes)
            self._tf_init_ops = {lod: self._tf_iterator.make_initializer(dset) for lod, dset in self._tf_datasets.items()}

    # Use the given minibatch size and level-of-detail for the data returned by get_minibatch_tf().
    def configure(self, minibatch_size, lod=0):
        lod = int(np.floor(lod))
        assert minibatch_size >= 1 and lod in self._tf_datasets
        if self._cur_minibatch != minibatch_size or self._cur_lod != lod:
            self._tf_init_ops[lod].run({self._tf_minibatch_in: minibatch_size})
            self._cur_minibatch = minibatch_size
            self._cur_lod = lod

    # Get next minibatch as TensorFlow expressions.
    def get_minibatch_tf(self): # => images, labels
        return self._tf_iterator.get_next()

    # Get next minibatch as NumPy arrays.
    def get_minibatch_np(self, minibatch_size, lod=0): # => images, labels
        self.configure(minibatch_size, lod)
        if self._tf_minibatch_np is None:
            self._tf_minibatch_np = self.get_minibatch_tf()
        return tfutil.run(self._tf_minibatch_np)

    # Get random labels as TensorFlow expression.
    def get_random_labels_tf(self, minibatch_size): # => labels
        if self.label_size > 0:
            return tf.compat.v1.gather(self._tf_labels_var, tf.compat.v1.random_uniform([minibatch_size], 0, self._np_labels.shape[0], dtype=tf.compat.v1.int32))
        else:
            return tf.compat.v1.zeros([minibatch_size, 0], self.label_dtype)

    # Get random labels as NumPy array.
    def get_random_labels_np(self, minibatch_size): # => labels
        if self.label_size > 0:
            return self._np_labels[np.random.randint(self._np_labels.shape[0], size=[minibatch_size])]
        else:
            return np.zeros([minibatch_size, 0], self.label_dtype)
    
    def get_random_signals(self, minibatch_size):
        return self.eeg_signals[np.random.randint(self.eeg_signals.shape[0], size=[minibatch_size])]

#----------------------------------------------------------------------------
# Base class for datasets that are generated on the fly.

class SyntheticDataset:
    def __init__(self, resolution=1024, num_channels=3, dtype='uint8', dynamic_range=[0,255], label_size=0, label_dtype='float32'):
        self.resolution         = resolution
        self.resolution_log2    = int(np.log2(resolution))
        self.shape              = [num_channels, resolution, resolution]
        self.dtype              = dtype
        self.dynamic_range      = dynamic_range
        self.label_size         = label_size
        self.label_dtype        = label_dtype
        self._tf_minibatch_var  = None
        self._tf_lod_var        = None
        self._tf_minibatch_np   = None
        self._tf_labels_np      = None

        assert self.resolution == 2 ** self.resolution_log2
        with tf.compat.v1.name_scope('Dataset'):
            self._tf_minibatch_var = tf.compat.v1.Variable(np.int32(0), name='minibatch_var')
            self._tf_lod_var = tf.compat.v1.Variable(np.int32(0), name='lod_var')

    def configure(self, minibatch_size, lod=0):
        lod = int(np.floor(lod))
        assert minibatch_size >= 1 and lod >= 0 and lod <= self.resolution_log2
        tfutil.set_vars({self._tf_minibatch_var: minibatch_size, self._tf_lod_var: lod})

    def get_minibatch_tf(self): # => images, labels
        with tf.compat.v1.name_scope('SyntheticDataset'):
            shrink = tf.compat.v1.cast(2.0 ** tf.compat.v1.cast(self._tf_lod_var, tf.compat.v1.float32), tf.compat.v1.int32)
            shape = [self.shape[0], self.shape[1] // shrink, self.shape[2] // shrink]
            images = self._generate_images(self._tf_minibatch_var, self._tf_lod_var, shape)
            labels = self._generate_labels(self._tf_minibatch_var)
            return images, labels

    def get_minibatch_np(self, minibatch_size, lod=0): # => images, labels
        self.configure(minibatch_size, lod)
        if self._tf_minibatch_np is None:
            self._tf_minibatch_np = self.get_minibatch_tf()
        return tfutil.run(self._tf_minibatch_np)

    def get_random_labels_tf(self, minibatch_size): # => labels
        with tf.compat.v1.name_scope('SyntheticDataset'):
            return self._generate_labels(minibatch_size)

    def get_random_labels_np(self, minibatch_size): # => labels
        self.configure(minibatch_size)
        if self._tf_labels_np is None:
            self._tf_labels_np = self.get_random_labels_tf()
        return tfutil.run(self._tf_labels_np)

    def _generate_images(self, minibatch, lod, shape): # to be overridden by subclasses
        return tf.compat.v1.zeros([minibatch] + shape, self.dtype)

    def _generate_labels(self, minibatch): # to be overridden by subclasses
        return tf.compat.v1.zeros([minibatch, self.label_size], self.label_dtype)

#----------------------------------------------------------------------------
# Helper func for constructing a dataset object using the given options.

def load_dataset(class_name='dataset.TFRecordDataset', data_dir=None, verbose=False, **kwargs):
    adjusted_kwargs = dict(kwargs)
    if 'tfrecord_dir' in adjusted_kwargs and data_dir is not None:
        adjusted_kwargs['tfrecord_dir'] = os.path.join(data_dir, adjusted_kwargs['tfrecord_dir'])
        print(adjusted_kwargs['tfrecord_dir'])
    if verbose:
        print('Streaming data using %s...' % class_name)
    dataset = tfutil.import_obj(class_name)(**adjusted_kwargs)
    if verbose:
        print('Dataset shape =', np.int32(dataset.shape).tolist())
        print('Dynamic range =', dataset.dynamic_range)
        print('Label size    =', dataset.label_size)
    return dataset

#----------------------------------------------------------------------------
