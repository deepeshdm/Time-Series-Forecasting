

import pandas as pd
from tensorflow.python.keras.layers import LSTM, Dense, TimeDistributed
from tensorflow.python.keras.models import Sequential
import matplotlib.pyplot as plt
from tensorflow.python.layers.convolutional import Conv1D
from tensorflow.python.layers.core import Flatten
from tensorflow.python.layers.pooling import MaxPooling1D
from tqdm import tqdm
from numpy import array

DATASET_PATH = \
    r"https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
DATE_COLUMN = "Month"
X_COLUMN = "Passengers"

# parse_dates : parses dates from string to DatetimeIndex
# index_col : set specified column as dataset index
df = pd.read_csv(DATASET_PATH, parse_dates=[DATE_COLUMN], index_col=DATE_COLUMN)

# NOTE : You dont need the dates column here
x_train = df[X_COLUMN].values
x_test = df[X_COLUMN].values


# --------------------------------------------------------------------------

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in tqdm(range(len(sequence))):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


n_steps = 4
n_features = 1
# dividing train sequence into input/output samples
x_train, y_train = split_sequence(x_train, n_steps=n_steps)
# dividing test sequence into input/output samples
x_test, y_test = split_sequence(x_test, n_steps=n_steps)

n_seq = 2
n_steps = 2
# reshape from [samples, timesteps] into [samples,subsequences,timesteps, features]
# This is because input is taken by CNN first then passed to LSTM
x_train = x_train.reshape((x_train.shape[0], n_seq, n_steps, n_features))
x_test = x_test.reshape((x_test.shape[0], n_seq, n_steps, n_features))


# --------------------------------------------------------------------------

# define model
model = Sequential()
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'),
                          input_shape=(None, n_steps, n_features)))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(x_train, y_train,epochs=10,validation_data=(x_test,y_test))

# --------------------------------------------------------------------------

# Inference
x_input = array([508,461,390])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)

# Testing on x_test
predictions = model.predict(x_test)
# bringing to similar shape
predictions = predictions.reshape(y_test.shape)

plt.figure(figsize=(15,8))
plt.plot(y_test,label="Actual")
plt.plot(predictions,label="Predicted")
plt.legend()
plt.title("Monthly Air Passengers Ticket Forecast (Vanilla LSTM)")
plt.xlabel("Month")
plt.ylabel("Ticket Sales")
plt.show()

