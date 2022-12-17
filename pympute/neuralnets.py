
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils

# Hotfix function
def make_keras_picklable():

    def __reduce__(self):
        model_metadata = saving_utils.model_metadata(self)
        training_config = model_metadata.get("training_config", None)
        model = serialize(self)
        weights = self.get_weights()
        return (unpack, (model, training_config, weights))

    cls = Model
    cls.__reduce__ = __reduce__


def dense_model(inp,out,nlayer,batchnorm=False,dropout=False,loss='mean_squared_error',optimizer='adam',kernel_initializer='normal'):
    dims = np.linspace(inp,out,nlayer).astype(int)[1:-1]
    input_img = keras.Input(shape=inp)
    x = input_img
    for dim in dims:
        x = layers.Dense(dim, kernel_initializer=kernel_initializer, activation='relu')(x)
        if batchnorm:
            x = layers.BatchNormalization()(x)
        if dropout!=0:
            x = layers.Dropout(dropout)(x)
    # create model
    if dropout!=0:
        x = layers.Dropout(dropout)(x)
    x = layers.Dense(out, kernel_initializer='normal')(x)
    
    model = keras.Model(input_img, x)
    model.compile(loss=loss, optimizer=optimizer)
    return model