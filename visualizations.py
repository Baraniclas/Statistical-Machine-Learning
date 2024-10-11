import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
import sklearn.discriminant_analysis as skl_da
import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
import sklearn.model_selection as skl_ms
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import seaborn as sns

data = pd.read_csv('/content/training_data.csv', dtype={'ID': str}).dropna().reset_index(drop=True)

# Map categorical values to numerical values
data['increase_stock'] = data['increase_stock'].map({'low_bike_demand': 0, 'high_bike_demand': 1})


data.shape
display(data)
data.info()
data.describe()

# Count occurrences of increase_stock=1 for each hour_of_day
hourly_counts_all = data[data['increase_stock'] == 1]['hour_of_day'].value_counts().sort_index()

# Create a Series containing counts of increase_stock=1 for all hours (including those with count 0)
all_hours_counts = hourly_counts_all.reindex(range(24), fill_value=0)

# Create the bar plot
plt.figure(figsize=(12, 6))
sns.barplot(x=all_hours_counts.index, y=all_hours_counts.values)
plt.title('Count of increase_stock=1 per hour of the day (All hours)')
plt.xlabel('Hour of the Day')
plt.ylabel('Count where increase_stock=1')
plt.show()

# Filter the DataFrame for increase_stock=1
increase_stock_1 = data[data['increase_stock'] == 1]

# Drop rows with missing values in the 'day_of_week' column
increase_stock_1 = increase_stock_1.dropna(subset=['day_of_week'])

# Count occurrences of increase_stock=1 for each day_of_week
day_of_week_counts = increase_stock_1['day_of_week'].value_counts().sort_index()

# Create the bar plot
plt.figure(figsize=(12, 6))
sns.barplot(x=day_of_week_counts.index, y=day_of_week_counts.values)
plt.title('Count of high bike demand per day_of_week')
plt.xlabel('day_of_week')
plt.ylabel('Count where increase_stock=1')
plt.show()

# Filter the DataFrame for increase_stock=1
increase_stock_1 = data[data['increase_stock'] == 1]

# Count occurrences of increase_stock=1 for each temp
temp_counts = increase_stock_1['temp'].value_counts().sort_index()

# Find the range of temp values in the dataset
temp_range = range(int(data['temp'].min()), int(data['temp'].max()) + 1)

# Fill in missing temp counts with zeros
for temp in temp_range:
    if temp not in temp_counts.index:
        temp_counts[temp] = 0

# Sort the index after adding the missing values
temp_counts = temp_counts.sort_index()

# Create the bar plot
plt.figure(figsize=(60, 6))
sns.barplot(x=temp_counts.index, y=temp_counts.values)
plt.title('Count of high bike demand per temperature')
plt.xlabel('Temperature')
plt.ylabel('Count where increase_stock=1')
plt.show()

# Filter the DataFrame for increase_stock=1
increase_stock_1 = data[data['increase_stock'] == 1]

# Count occurrences of increase_stock=1 for each dew
dew_counts = increase_stock_1['dew'].value_counts().sort_index()

# Find the range of dew values in the dataset
dew_range = range(int(data['dew'].min()), int(data['dew'].max()) + 1)

# Fill in missing dew counts with zeros
for dew in dew_range:
    if dew not in dew_counts.index:
        dew_counts[dew] = 0

# Sort the index after adding the missing values
dew_counts = dew_counts.sort_index()

# Create the bar plot
plt.figure(figsize=(60, 6))
sns.barplot(x=dew_counts.index, y=dew_counts.values)
plt.title('Count of high bike demand per dew')
plt.xlabel('Dew')
plt.ylabel('Count where increase_stock=1')
plt.show()

# Filter the DataFrame for increase_stock=1
increase_stock_1 = data[data['increase_stock'] == 1]

# Count occurrences of increase_stock=1 for each humidity
humidity_counts = increase_stock_1['humidity'].value_counts().sort_index()

# Find the range of humidity values in the dataset
humidity_range = range(int(data['humidity'].min()), int(data['humidity'].max()) + 1)

# Fill in missing humidity counts with zeros
for humidity in humidity_range:
    if humidity not in humidity_counts.index:
        humidity_counts[humidity] = 0

# Sort the index after adding the missing values
humidity_counts = humidity_counts.sort_index()

# Create the bar plot
plt.figure(figsize=(200, 6))
sns.barplot(x=humidity_counts.index, y=humidity_counts.values)
plt.title('Count of high bike demand per humidity')
plt.xlabel('Humidity')
plt.ylabel('Count where increase_stock=1')
plt.show()

# Filter the DataFrame for increase_stock=1
increase_stock_1 = data[data['increase_stock'] == 1]

# Count occurrences of increase_stock=1 for each windspeed
windspeed_counts = increase_stock_1['windspeed'].value_counts().sort_index()

# Find the range of windspeed values in the dataset
windspeed_range = range(int(data['windspeed'].min()), int(data['windspeed'].max()) + 1)

# Fill in missing windspeed counts with zeros
for windspeed in windspeed_range:
    if windspeed not in windspeed_counts.index:
        windspeed_counts[windspeed] = 0

# Sort the index after adding the missing values
windspeed_counts = windspeed_counts.sort_index()

# Create the bar plot
plt.figure(figsize=(60, 6))
sns.barplot(x=windspeed_counts.index, y=windspeed_counts.values)
plt.title('Count of high bike demand per windspeed')
plt.xlabel('Windspeed')
plt.ylabel('Count where increase_stock=1')
plt.show()

# Define the number of bins and create bins for visibility
num_bins = 10  # You can adjust the number of bins as needed
visibility_bins = pd.cut(data['visibility'], bins=num_bins)

# Filter the DataFrame for increase_stock=1
increase_stock_1 = data[data['increase_stock'] == 1]

# Count occurrences of increase_stock=1 for each visibility bin
visibility_counts = increase_stock_1.groupby(visibility_bins)['increase_stock'].count()

# Create the bar plot
plt.figure(figsize=(12, 6))
visibility_counts.plot(kind='bar')
plt.title('Count of increase_stock=1 for each visibility bin')
plt.xlabel('Visibility Bins')
plt.ylabel('Count where increase_stock=1')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()

# Define the number of bins and create bins for snowdepth
num_bins = 100  # You can adjust the number of bins as needed
snowdepth_bins = pd.cut(data['snowdepth'], bins=num_bins)

# Filter the DataFrame for increase_stock=1
increase_stock_1 = data[data['increase_stock'] == 1]

# Count occurrences of increase_stock=1 for each snowdepth bin
snowdepth_counts = increase_stock_1.groupby(snowdepth_bins)['increase_stock'].count()

# Create the bar plot
plt.figure(figsize=(60, 6))
snowdepth_counts.plot(kind='bar')
plt.title('Count of increase_stock=1 for each snowdepth bin')
plt.xlabel('Snowdepth Bins')
plt.ylabel('Count where increase_stock=1')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()

# Define the number of bins and create bins for cloudcover
num_bins = 30  # You can adjust the number of bins as needed
cloudcover_bins = pd.cut(data['cloudcover'], bins=num_bins)

# Filter the DataFrame for increase_stock=1
increase_stock_1 = data[data['increase_stock'] == 1]

# Count occurrences of increase_stock=1 for each cloudcover bin
cloudcover_counts = increase_stock_1.groupby(cloudcover_bins)['increase_stock'].count()

# Create the bar plot
plt.figure(figsize=(12, 6))
cloudcover_counts.plot(kind='bar')
plt.title('Count of increase_stock=1 for each cloudcover bin')
plt.xlabel('Cloudcover Bins')
plt.ylabel('Count where increase_stock=1')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()

!pip install ydata_profiling
!pip install lida==0.0.10 kaleido python-multipart uvicorn lmx==0.0.15a0 tensorflow-probability==0.22.0

from ydata_profiling import ProfileReport


profile = ProfileReport(data,title="Bike sharing data report")
profile.to_notebook_iframe()
profile.to_file("eda.html")