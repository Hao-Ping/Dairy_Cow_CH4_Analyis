import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


# Load the data
df = pd.read_csv('filtered_sensor_data_0810.csv')

# Convert the datetime column to datetime format for plotting
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)  # Set datetime as the index for easier filtering

# Compute the rolling median with a window of, say, 5 data points
window_size = 250
df['ppm_smoothed'] = df['ppm'].rolling(window=window_size, center=True).median()

# Apply Gaussian smoothing to the ppm data
sigma_value = 3  # Adjust this based on your desired level of smoothing
df['ppm_gaussian_smoothed']  = gaussian_filter1d(df['ppm'], sigma=sigma_value)
inverted_smoothed_ppm = -df['ppm_gaussian_smoothed'] 

valleys, _ = find_peaks(inverted_smoothed_ppm, height=[-30,-5], distance=150, width=1, prominence=5) # quite good

# Calculate the gradient of the smoothed data
df['gradient'] = np.gradient(df['ppm_smoothed'])

# Find max and min gradient within morning every day, max is to find the start milking point which is around 04:30:00~05:10:00, min is to find the end of the milking period which is around 06:30:00~07:00:00
morning_max_gradient = df.between_time('04:30:00', '05:10:00').groupby(lambda x: x.date())['gradient'].idxmax()
morning_min_gradient = df.between_time('06:30:00', '07:00:00').groupby(lambda x: x.date())['gradient'].idxmin()

# Find max and min gradient within evening every day
evening_max_gradient = df.between_time('16:15:00', '16:50:00').groupby(lambda x: x.date())['gradient'].idxmax()
evening_min_gradient = df.between_time('17:30:00', '18:45:00').groupby(lambda x: x.date())['gradient'].idxmin()

# Combine the two
morning_milking_period = pd.concat([morning_max_gradient, morning_min_gradient])
# print(morning_milking_period)
evening_milking_period = pd.concat([evening_max_gradient, evening_min_gradient])

all_valleys = []
# Iterate through the periods defined by morning_max_gradient and morning_min_gradient
for start_time, end_time in zip(morning_max_gradient, morning_min_gradient):
    # print(start_time, end_time)
    # Extract the data segment for the current milking period
    segment = df.loc[start_time:end_time, 'ppm_gaussian_smoothed']
    
    # Invert the segment
    inverted_segment = -segment
    
    # Find valleys within the current segment
    valleys, _ = find_peaks(inverted_segment, height=[-30, -5], distance=150, width=1, prominence=5)
    
    # Convert relative indices of the segment to absolute indices in the dataframe
    absolute_valleys = segment.index[valleys]
    
    # Append the detected valleys to the all_valleys list
    all_valleys.extend(absolute_valleys)

for start_time, end_time in zip(evening_max_gradient, evening_min_gradient):
    # print(start_time, end_time)
    # Extract the data segment for the current milking period
    segment = df.loc[start_time:end_time, 'ppm_gaussian_smoothed']
    
    # Invert the segment
    inverted_segment = -segment
    
    # Find valleys within the current segment
    valleys, _ = find_peaks(inverted_segment, height=[-30, -5], distance=150, width=1, prominence=5)
    
    # Convert relative indices of the segment to absolute indices in the dataframe
    absolute_valleys = segment.index[valleys]
    
    # Append the detected valleys to the all_valleys list
    all_valleys.extend(absolute_valleys)

# Plot original vs. smoothed data
# plt.figure(figsize=(12, 6))
# plt.subplot(2, 1, 1)  # Two rows, one column, first plot
plt.plot(df.index, df['ppm'], color='gray', label='Original Data')
# plt.plot(df.index, df['ppm_smoothed'], label='Smoothed Data (Rolling Median)', color='red')
plt.plot(df.index, df['ppm_gaussian_smoothed'] , label='Gaussian Smoothed Data', color='blue')
plt.scatter(all_valleys, df.loc[all_valleys, 'ppm_gaussian_smoothed'], color='r', s=60, label='Valleys within Milking Period')

# Draw vertical lines for max_gradients and min_gradients
for time in morning_milking_period:
    plt.axvline(time, color='green', linestyle='--', linewidth=0.7)

for time in evening_milking_period:
    plt.axvline(time, color='green', linestyle='--', linewidth=0.7)

plt.xlabel('Datetime')
plt.ylabel('ppm (CH4)')
plt.legend()
plt.title('CH4 (ppm) Original vs. Smoothed Data')
plt.xticks(rotation=45)

# Plot gradient
# plt.subplot(2, 1, 2)  # Two rows, one column, second plot
# plt.plot(df.index, df['gradient'], label='Gradient of Smoothed Data', color='blue')

# # Mark max gradient for each day
# for date, time in max_gradients.items():
#     plt.scatter(time, df.at[time, 'gradient'], color='red', s=50, zorder=5)  # zorder ensures the points are plotted on top

# for date, time in min_gradients.items():
#     plt.scatter(time, df.at[time, 'gradient'], color='red', s=50, zorder=5)  # zorder ensures the points are plotted on top


plt.xlabel('Datetime')
plt.ylabel('Gradient')
plt.legend()
plt.title('CH4 (ppm) Data')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
