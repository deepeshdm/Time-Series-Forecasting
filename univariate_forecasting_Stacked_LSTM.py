import pandas as pd
from tensorflow.python.keras.layers import LSTM, Dense
from tensorflow.python.keras.models import Sequential
from tqdm import tqdm
from numpy import array

DATASET_PATH = \
    r"https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
DATE_COLUMN = "Month"
X_COLUMN = "Passengers"

# parse_dates : parses dates from string to DatetimeIndex
# index_col : set specified column as dataset index
df = pd.read_csv(DATASET_PATH, parse_dates=[DATE_COLUMN], index_col=DATE_COLUMN)
df.head()

# NOTE : You dont need the dates column here
x_train = df[X_COLUMN].values
x_test = df[X_COLUMN].values

# WILL USE THIS FOR DISPLAYING FINAL FORECAST
x_train_dis = x_train


# ----------------------------------------------------------------------------


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


n_steps = 3
n_features = 1

# dividing train sequence into input/output samples
x_train, y_train = split_sequence(x_train, n_steps=n_steps)

# dividing test sequence into input/output samples
x_test, y_test = split_sequence(x_test, n_steps=n_steps)

for i in range(len(x_train)):
    print(x_train[i], y_train[i])

# reshape from [samples, timesteps] into [samples, timesteps, features]
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], n_features))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], n_features))

# ----------------------------------------------------------------------------

# define model
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.summary()

# fit model
model.fit(x_train, y_train, epochs=350, validation_data=(x_test, y_test))

# ----------------------------------------------------------------------------

# Inference
x_input = array([508, 461, 390])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)

# Testing on x_test
predictions = model.predict(x_test)
# bringing to similar shape
predictions = predictions.reshape(y_test.shape)

import matplotlib.pyplot as plt

plt.figure(figsize=(15, 8))
plt.plot(y_test, label="Actual")
plt.plot(predictions, label="Predicted")
plt.legend()
plt.title("Monthly Air Passengers Ticket Forecast (Vanilla LSTM)")
plt.xlabel("Month")
plt.ylabel("Ticket Sales")
plt.show()

# ----------------------------------------------------------------------------

# PREDICTING NEXT "N" PERIODS

# Predicting next N values.
x_input = array([508, 461, 390])  # last values from x_train
temp_input = list(x_input)
lst_output = []
i = 0
# predict next 48 months
Predict_N_STEPS = 48
while (i < Predict_N_STEPS):

    if (len(temp_input) > 3):
        x_input = array(temp_input[1:])
        print("{} day input {}".format(i, x_input))
        # print(x_input)
        x_input = x_input.reshape((1, n_steps, n_features))
        # print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i, yhat))
        temp_input.append(yhat[0][0])
        temp_input = temp_input[1:]
        # print(temp_input)
        lst_output.append(yhat[0][0])
        i = i + 1
    else:
        x_input = x_input.reshape((1, n_steps, n_features))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.append(yhat[0][0])
        lst_output.append(yhat[0][0])
        i = i + 1

print(lst_output)

# plotting the forecast with original dataset
predictions = lst_output
import matplotlib.pyplot as plt
from numpy import arange

# creating x for our predictions
x = arange(len(x_train_dis), len(x_train_dis) + Predict_N_STEPS)

plt.figure(figsize=(15, 8))
plt.plot(x_train_dis, label="X_Train")
plt.plot(x, predictions, label="Forecast")
plt.legend()
plt.title("Monthly Air Passengers Ticket Forecast (Stacked LSTM)")
plt.xlabel("Month")
plt.ylabel("Ticket Sales")
plt.show()
