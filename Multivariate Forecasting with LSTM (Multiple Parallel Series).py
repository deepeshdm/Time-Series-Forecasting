from numpy import array
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ------------------------------------------------------------------

# CREATING 2 PARALLEL TIME-SERIES

df = pd.read_csv \
    (r"https://raw.githubusercontent.com/selva86/datasets/master/Raotbl6.csv")

# change column names
df.columns = ["date", "X_1", "X_2", "X_3", "X_4", "X_5", "X_6", "X_7", "X_8"]

# dropping all columns and keeping only two
df = df[["X_1", "X_2"]]

# creating one extra column
df["Y"] = df.apply(lambda row: round((row.X_1 + row.X_2) / 1.9, 1), axis=1)

print(df.head(20))
"""
OUTPUT :
      X_1     X_2       Y
 1606.4  1608.3  1691.9
 1637.0  1622.2  1715.4
 1629.5  1636.2  1718.8
 1643.4  1650.3  1733.5
 1671.6  1664.6  1755.9
 1666.8  1679.0  1760.9
 1668.4  1693.5  1769.4
 1654.1  1708.2  1769.6
 1671.3  1722.9  1786.4
 1692.1  1737.8  1805.2
 1716.3  1752.8  1825.8
 1754.9  1768.0  1854.2
 1777.9  1783.3  1874.3
 1796.4  1798.7  1892.2
 1813.1  1814.3  1909.2
 1810.1  1829.9  1915.8
 1834.6  1845.8  1937.1
 1860.0  1861.7  1958.8
 1892.5  1877.8  1984.4
 1906.1  1894.0  2000.1
"""

df.plot()
plt.show()


# ------------------------------------------------------------------

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


df = df.values
n_steps = 3

# convert into input/output samples
X, y = split_sequences(df, n_steps)
# summarize the data
for i in range(len(X)):
    print(X[i], y[i])

print('SHAPE : ', X.shape, y.shape)

"""
OUTPUT : 
[[1606.4 1608.3 1691.9]
 [1637.  1622.2 1715.4]
 [1629.5 1636.2 1718.8]] [1643.4 1650.3 1733.5]
[[1637.  1622.2 1715.4]
 [1629.5 1636.2 1718.8]
 [1643.4 1650.3 1733.5]] [1671.6 1664.6 1755.9]
[[1629.5 1636.2 1718.8]
 [1643.4 1650.3 1733.5]
 [1671.6 1664.6 1755.9]] [1666.8 1679.  1760.9]
[[1643.4 1650.3 1733.5]
 [1671.6 1664.6 1755.9]
 [1666.8 1679.  1760.9]] [1668.4 1693.5 1769.4]
[[1671.6 1664.6 1755.9]
 [1666.8 1679.  1760.9]
 [1668.4 1693.5 1769.4]] [1654.1 1708.2 1769.6]
[[1666.8 1679.  1760.9]
 [1668.4 1693.5 1769.4]
 [1654.1 1708.2 1769.6]] [1671.3 1722.9 1786.4]
...
SHAPE :  (120, 3, 3) (120, 3)
"""

# -------------------------------------------------------------------

# since we have 3 parallel time series
n_features = 3

# define model
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(n_features))
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(X, y, epochs=200)

# demonstrate prediction
x_input = array([[1666.8, 1679., 1760.9],
                 [1668.4, 1693.5, 1769.4],
                 [1654.1, 1708.2, 1769.6]])

x_input = x_input.reshape((1, n_steps, n_features))
# predicted value must be around [1671.3,1722.9,1786.4]

yhat = model.predict(x_input)
print(yhat)

# [[1697.874  1709.8911 1793.3396]]
