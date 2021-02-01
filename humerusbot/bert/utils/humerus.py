import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
import json
import pickle


def make_model(input_size):
    model = keras.Sequential()
    model.add(layers.Dense(input_size, input_shape=(input_size, ), activation='relu'))
    model.add(layers.Dense(input_size))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

class predictionNetwork:
    """basic binary classifier prediction network for CAH
    """
    def __init__(self, model): 
        self.model = model

    @classmethod
    def from_scratch(cls, input_shape):
        mdl=make_model(input_shape)
        mdl.summary()
        return cls(mdl)

    @classmethod
    def from_file(cls, input_shape, weight_path): 
        # get input shape
        # pass relative path strings
        mdl = make_model(input_shape)
        mdl.load_weights(weight_path)
        return cls(mdl)

    def save_models(self, path):
        # shitty format check
        path = path + '.h5' if path[-3:] != '.h5' else path
        self.model.save(path)

    def train_model(self, x_train, y_train, n_epochs):
        print("-------STARTING TRAINING------")

        print("-----SHAPES-------")
        print("X:", x_train.shape)
        print("Y:", y_train.shape)

        self.model.fit(x=x_train, y=y_train, 
                       epochs=n_epochs, verbose=1, 
                       validation_split = 0.1, shuffle=True) 
        
    def evaluate(self, x):
        # x can be single item or list!
        return self.model.predict(np.expand_dims(x, axis=0), batch_size=1)[0]





