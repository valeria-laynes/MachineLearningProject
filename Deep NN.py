#%% Needed for data
import pandas as pd
from math import ceil, sqrt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
#%%
df = pd.read_excel(r'C:\Users\BroodFather\Desktop\School\CAP5610\Project\Smoothed Data\Catagorized.xlsx', sheet_name = 'Stamped')

# Drop non-data
df = df.drop('Timestamp', axis = 1)

#%% Fill NAN's with 50% stdev, completes dataset
df.isna().sum()

#%% Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(df)

#%%
# split into train and test sets
array_df = np.array(scaled)
train = array_df[:ceil(0.33*len(df)), :]
test = array_df[ceil(0.33*len(df)):, :]

train_features = train[:, -12:]
train_labels = np.ravel(train[:,:-12])

test_features = test[:,-12:]
test_labels = np.ravel(test[:, :-12])

#%%
model = Sequential()
model.add(Dense(12, input_dim=12, kernel_initializer='normal', activation='tanh'))
model.add(Dense(7, activation = 'tanh'))
model.add(Dense(1, activation = 'tanh'))
model.summary()
model.compile(loss='mse', optimizer='adam')
nn = model.fit(train_features, train_labels, epochs=200, batch_size=72,  verbose=2)

plt.plot(nn.history['loss'], label='train')
plt.plot(nn.history['val_loss'], label='test')
plt.legend()
plt.show()

#%%
test_X = test_features
test_y = test_labels

# make a prediction
yhat = model.predict(test_X)
# invert scaling for forecast
inv_yhat = np.concatenate((yhat, test_X[:, -12:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, -12:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

#%%
fig_width = 15
fig_height = 5
plt.rcParams["figure.figsize"] = (fig_width,fig_height)

plt.rc('xtick', labelsize=10) 
plt.rc('ytick', labelsize=10) 
plt.rc('axes', linewidth=2)

plt.plot(inv_y, label = 'Actual', linewidth = 0.75)
plt.plot(inv_yhat, label = 'Predicted', color = 'red', linewidth = 0.5, alpha = 0.7)
plt.xlabel('Data points',weight = 'bold')
plt.ylabel('Traffic Count',weight = 'bold')
plt.legend( loc='upper right')
plt.show()

print('R^2 Score:')
print(r2_score(inv_y, inv_yhat))