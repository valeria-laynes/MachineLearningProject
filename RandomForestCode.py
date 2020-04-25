#%% Needed for data
import pandas as pd
from math import ceil, sqrt
# Needed for Random Forest
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error

# Data Visualization
import matplotlib.pyplot as plt
#%%
df = pd.read_excel(r'C:\Users\BroodFather\Desktop\School\CAP5610\Project\Smoothed Data\Catagorized.xlsx', sheet_name = 'Stamped')

# Drop non-data
df = df.drop('Timestamp', axis = 1)

#%% Fill NAN's with 50% stdev, completes dataset
df.isna().sum()
#%%
# split into train and test sets
array_df = np.array(df)
train = array_df[:ceil(0.33*len(df)), :]
test = array_df[ceil(0.33*len(df)):, :]

train_features = train[:, -12:]
train_labels = np.ravel(train[:,:-12])

test_features = test[:,-12:]
test_labels = np.ravel(test[:, :-12])

#%% Random Forest model
rf = RandomForestRegressor(n_estimators = 600, min_samples_split = 10, min_samples_leaf = 1, max_features = 'sqrt', max_depth = 40, bootstrap = False, random_state = 42)
rf.fit(train_features, train_labels)

#%% Predictions
fig_width = 15
fig_height = 5
plt.rcParams["figure.figsize"] = (fig_width,fig_height)

plt.rc('xtick', labelsize=10) 
plt.rc('ytick', labelsize=10) 
plt.rc('axes', linewidth=2)

plt.plot(test_labels, label = 'Actual', linewidth = 0.75)
plt.plot(predictions, label = 'Predicted', color = 'red', linewidth = 0.5, alpha = 0.7)
plt.xlabel('Data points',weight = 'bold')
plt.ylabel('Traffic Count',weight = 'bold')
plt.legend( loc='upper right')
plt.show()

rmse = sqrt(mean_squared_error(test_labels, predictions))
print('Test RMSE: %.3f' % rmse)

print('R^2 Score:')
print(r2_score(test_labels, predictions))

#%% Tuning using RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(train_features, train_labels)