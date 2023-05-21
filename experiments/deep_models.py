import keras.initializers.initializers_v2
import numpy as np
import tensorflow as tf
from keras import regularizers
from keras.layers import Input, Dense, LocallyConnected1D, Reshape, Conv2D, MaxPooling2D, Dropout, Flatten, Lambda
from keras.models import Model

# https://github.com/XiaqiongFan/DeepCID
def CNN(input_dim):
    kernel_init = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1)
    bias_init = keras.initializers.initializers_v2.Constant(0.1)
    opt = tf.keras.optimizers.Adam(lr=1e-4)

    input_layer = Input(shape=(1, input_dim, 1), name="Input")

    x = Conv2D(filters=32, kernel_size=5, strides=(1, 2), padding="same", activation='relu',
               kernel_initializer=kernel_init, bias_initializer=bias_init)(input_layer)
    x = MaxPooling2D(pool_size=1, strides=2)(x)
    x = Dropout(rate=0.5)(x)

    x = Conv2D(filters=64, kernel_size=5, strides=(1, 2), padding="same", activation='relu',
               kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    x = MaxPooling2D(pool_size=1, strides=2)(x)
    x = Dropout(rate=0.5)(x)

    x = Flatten()(x)
    
    x = Dense(x.shape[1], activation="relu", kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    x = Dense(1024, activation="relu", kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    output = Dense(1, activation='sigmoid', kernel_initializer=kernel_init,
                   bias_initializer=bias_init, name="Output")(x)

    cnn_model = Model(inputs=input_layer, outputs=output)
    cnn_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])

    return cnn_model

def FCNN(input_dim):
    kernel_init = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1)
    bias_init = keras.initializers.initializers_v2.Constant(0.1)
    opt = tf.keras.optimizers.Adam(lr=1e-4)

    input_layer = Input(shape=(input_dim,), name="Input")
    
    x = Dense(input_dim, activation="relu", kernel_initializer=kernel_init, bias_initializer=bias_init)(input_layer)
    x = Dense(1024, activation="relu", kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    output = Dense(1, activation='sigmoid', kernel_initializer=kernel_init,
                   bias_initializer=bias_init, name="Output")(x)

    fcnn_model = Model(inputs=input_layer, outputs=output)
    fcnn_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])

    return fcnn_model

