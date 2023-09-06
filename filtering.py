import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Load and preprocess your data
data = pd.read_csv('filtered_sensor_data_0810.csv')
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

# Smooth the CH4 concentration data using different techniques
window_size = 250  # Adjust this based on your data

# Rolling Median
smoothed_data_median = data['ppm'].rolling(window=window_size, center=True).median()


# Calculate the derivative of the median-filtered data
# derivative_median = smoothed_data_median.diff()

# Calculate the derivative of the smoothed data using np.gradient
# derivative_median = np.gradient(smoothed_data_median, axis=0)

# Find the maximum and minimum gradient values
# max_gradient = np.max(derivative_median)
# min_gradient = np.min(derivative_median)

# print("Maximum Gradient:", max_gradient)
# print("Minimum Gradient:", min_gradient)
# Rolling Mean
# smoothed_data_mean = data['ppm'].rolling(window=window_size, center=True).mean()

# Gaussian Filter
sigma = 50  # Adjust this based on your data
smoothed_data_gaussian = gaussian_filter1d(data['ppm'], sigma)


plt.figure(figsize=(12, 6))

# Plot the original data and smoothed data
plt.plot(data.index, data['ppm'], label='Original Data', color='gray')
plt.plot(smoothed_data_median.index, smoothed_data_median, label='Rolling Median', color='blue')
# plt.plot(smoothed_data_median.index, derivative_median, label='Derivative of Median', color='blue')

# plt.plot(smoothed_data_mean.index, smoothed_data_mean, label='Rolling Mean', color='pink')
plt.plot(data.index, smoothed_data_gaussian, label=f'Gaussian Filter (sigma = {sigma})', color='green')

plt.xlabel('time [hh:mm:ss]')
plt.ylabel('CH4 [ppm]')
plt.legend()
plt.show()
