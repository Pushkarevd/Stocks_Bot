import numpy as np
from tensorflow import keras
from tensorflow.keras import layers



# Simple LSTM model
class LSTM_Model:

    def __init__(self, input_size: int):
        self.model = keras.Sequential(
            [
                layers.LSTM(50, return_sequences=True, input_shape=(input_size, 1)),
                layers.LSTM(50, return_sequences=False),
                layers.Dense(25),
                layers.Dense(1),
            ]
        )
        self.model.compile(optimizer='adam', loss='mean_squared_error')

        self._rmse = None

    def fit(self, x_train: np.array, y_train: np.array, epochs=5):
        self.model.fit(x_train, y_train, batch_size=4, epochs=epochs)

    def predict(self, x: np.array):
        return self.model.predict(x)

    def save(self, stock: str):
        self.model.save('./models/{}/lstm_model.h5'.format(stock))

if __name__ == "__main__":
    pass
