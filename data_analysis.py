import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#$ Read csv
df = pd.read_csv('training_data_fall2024.csv')

## Plot graphs

# Set the font size for titles and labels
plt.rcParams.update({'font.size': 16})

# Create subplots with a 3x3 grid (9 plots in total)
fig, axes = plt.subplots(3, 3, figsize=(18, 18))

# Plot demand by hour of the day (Numerical x, Categorical hue)
sns.countplot(x='hour_of_day', hue='increase_stock', data=df, ax=axes[0, 0])
axes[0, 0].set_xticks(range(0, 24, 1))
axes[0, 0].set_xticklabels([str(i) if i % 6 == 0 else '' for i in range(24)])
# axes[0, 0].set_title('Bike Demand by Hour of Day')

# Plot demand by day of the week (Numerical x, Categorical hue)
sns.countplot(x='day_of_week', hue='increase_stock', data=df, ax=axes[0, 1])
# axes[0, 1].set_title('Bike Demand by Day of Week')

# Plot demand by month of the year (Numerical x, Categorical hue)
sns.countplot(x='month', hue='increase_stock', data=df, ax=axes[0, 2])
# axes[0, 2].set_title('Bike Demand by Month')

# Plot demand by holiday (Boolean, Categorical hue)
sns.countplot(x='holiday', hue='increase_stock', data=df, ax=axes[1, 0])
# axes[1, 0].set_title('Bike Demand on Holidays')

# Plot temperature vs. bike demand (Continuous y, Categorical x)
sns.boxplot(x='increase_stock', y='temp', data=df, ax=axes[1, 1])
# axes[1, 1].set_title('Temperature vs Bike Demand')

# Plot dew point vs. bike demand (Continuous y, Categorical x)
sns.boxplot(x='increase_stock', y='dew', data=df, ax=axes[1, 2])
# axes[1, 2].set_title('Dew Point vs Bike Demand')

# Plot humidity vs. bike demand (Continuous y, Categorical x)
sns.boxplot(x='increase_stock', y='humidity', data=df, ax=axes[2, 0])
# axes[2, 0].set_title('Humidity vs Bike Demand')

# Plot precipitation vs. bike demand (Continuous y, Categorical x)
sns.boxplot(x='increase_stock', y='precip', data=df, ax=axes[2, 1])
# axes[2, 1].set_title('Precipitation vs Bike Demand')

# Plot snow depth vs. bike demand (Continuous y, Categorical x)
sns.boxplot(x='increase_stock', y='snowdepth', data=df, ax=axes[2, 2])
# axes[2, 2].set_title('Snow Depth vs Bike Demand')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()


## Feature selection
# Create weather index
df['weather_index'] = (df['temp'] + (100 - df['humidity']) + (100 - df['precip']) + df['dew']) / 4
# Binarize hour of day
df['hour_of_day_binary'] = df['hour_of_day'].apply(lambda x: 1 if 7 <= x <= 20 else 0)
# Dropping columns: snow
df = df.drop(columns=['snow'])
