## Imports
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

## Feature selection

df_train = pd.read_csv('training_data_fall2024.csv')
df_test = pd.read_csv('test_data_fall2024.csv')

# Create weather index
df_train['weather_index'] = (df_train['temp'] + (100 - df_train['humidity']) + (100 - df_train['precip']) + df_train['dew']) / 4
df_test['weather_index'] = (df_test['temp'] + (100 - df_test['humidity']) + (100 - df_test['precip']) + df_test['dew']) / 4

# Binarize hour of day
df_train['hour_of_day_binary'] = df_train['hour_of_day'].apply(lambda x: 1 if 7 <= x <= 20 else 0)
df_test['hour_of_day_binary'] = df_test['hour_of_day'].apply(lambda x: 1 if 7 <= x <= 20 else 0)

# Dropping columns: snow
df_train = df_train.drop(columns=['snow'])
df_test = df_test.drop(columns=['snow'])

## Train and save output of test set
x_train = df_train.drop(columns=['increase_stock'])
y_train = df_train['increase_stock']
x_test = df_test
columns = ['increase_stock_pred']
output = pd.DataFrame(columns=columns)
random_state = 43

# Feature Scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# ** Initialize the Random forest with specified parameters
random_forest = RandomForestClassifier(random_state=random_state,max_depth=None,min_samples_leaf=2, min_samples_split=5,n_estimators=200)

# Train the Random Forest model
random_forest.fit(x_train, y_train)

# Make predictions
output['increase_stock_pred'] = random_forest.predict(x_test)

# Replaced 'low_bike_demand' with 0 and 'high_bike_demand' with 1
output['increase_stock_pred'] = output['increase_stock_pred'].replace({'low_bike_demand': 0, 'high_bike_demand': 1})

# Transposed the DataFrame so that each value is saved as a column
output_transposed = output.T

# Saved the transposed DataFrame to 'predictions.csv' without column headers
output_transposed.to_csv('predictions.csv', header=False, index=False)
