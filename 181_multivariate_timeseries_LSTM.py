# https://youtu.be/tepxdcepTbY
"""
Code tested on Tensorflow: 2.2.0
    Keras: 2.4.3
"""
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, Dropout
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

pd.options.mode.chained_assignment = None  # default='warn'

# Read the csv file
df = pd.read_csv('data/cape1.csv')
print(df.head())  # 10 columns

# Separate ids for future plotting
train_ids = df['Id']
# print(train_ids.tail(15))  # Check last few ids.

# Variables for training
cols = list(df)[1:]

# Id columns is not used in training.
# print(cols) ['sog', 'stw', 'head', 'wspeedms', 'wspeedbf', 'wdir', 'drafta', 'draftf', 'trim', 'me_power']
print(cols)

# New dataframe with only training data - 11 columns
df_for_training = df[cols].astype(float)

# LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
# the dataset
scaler = StandardScaler()
scaler = scaler.fit(df_for_training)
df_for_training_scaled = scaler.transform(df_for_training)

# As required for LSTM networks, we require to reshape an input data into n_samples x timesteps x n_features.
# In this example, the n_features is 10. We will make timesteps = 10 (past record data used for training).

# Empty lists to be populated using formatted training data
trainX = []  # training series
trainY = []  # prediction

n_future = 1  # Number of prediction records we want to look into the future based on the past records.
n_past = 20  # Number of records we want to use to predict the future.

# Reformat input data into a shape: (n_samples x timesteps x n_features)
# In my example, my df_for_training_scaled has a shape (12823, 5)
# 12823 refers to the number of data points and 5 refers to the columns (multi-variables).
for i in range(n_past, len(df_for_training_scaled) - n_future + 1):
    trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
    trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 9])

trainX, trainY = np.array(trainX), np.array(trainY)

print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))

# for cape1:
# trainX has a shape (21255, 20, 10).
# 21255 because we are looking back 20 records (21275 - 20 = 21255).
# Remember that we cannot look back 20 records until we get to the 21st record.
# Also, trainY has a shape (21255, 1). Our model only predicts a single value, but
# it needs multiple variables (10 in my example) to make this prediction.
# This is why we can only predict a single record after our training, the record after where our data ends.
# To predict more records in the future, we need all the 10 variables which we do not have.
# We need to predict all variables if we want to do that.

# define the Autoencoder model

model = Sequential()
# return_sequences=True because we want this lstm to return a sequence fot the next lstm to process
model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(trainY.shape[1]))

model.compile(optimizer='adam', loss='mse')
model.summary()

# fit the model
history = model.fit(trainX, trainY, epochs=5, batch_size=16, validation_split=0.1, verbose=1)

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()

# Predicting...
# us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
# Remember that we can only predict one day in future as our model needs 5 variables
# as inputs for prediction. We only have all 5 variables until the last day in our dataset.
n_past = 20
n_records_for_prediction = 20  # let us predict past 20 records

# predict_period_dates = pd.date_range(list(train_dates)[-n_past], periods=n_days_for_prediction, freq=us_bd).tolist()
predict_record_ids = list(range(list(train_ids)[-n_past], list(train_ids)[-n_past] + n_records_for_prediction, 1))

# Make prediction
prediction = model.predict(trainX[-n_records_for_prediction:])  # shape = (n, 1) where n is the n_days_for_prediction

# Perform inverse transformation to rescale back to original range
# Since we used 10 variables for transform, the inverse expects same dimensions
# Therefore, let us copy our values 10 times and discard them after inverse transform
prediction_copies = np.repeat(prediction, df_for_training.shape[1], axis=-1)
y_pred_future = scaler.inverse_transform(prediction_copies)[:, 0]

# Convert timestamp to date
forecast_ids = []
for id_i in predict_record_ids:
    forecast_ids.append(id_i)

df_forecast = pd.DataFrame({'Id': np.array(forecast_ids), 'me_power': y_pred_future})
df_forecast['Id'] = pd.to_numeric(df_forecast['me_power'])

original = df[['Id', 'me_power']]
original['Id'] = pd.to_numeric(original['me_power'])
original = original.loc[original['Id'] >= 20000]

sns.lineplot(data=original, x='Id', y='me_power')
sns.lineplot(data=df_forecast, x='Id', y='me_power')
matplotlib.pyplot.show()
print('the end')
