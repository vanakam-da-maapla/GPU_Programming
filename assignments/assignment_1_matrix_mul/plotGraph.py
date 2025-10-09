import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Data for the bar graph

# # 1.1
# x_labels = ["(1024, 1024)", "(2048, 512)", "(4096, 256)", "(8192, 128)", "(16384, 64)", "(32768, 32)", "(65536, 16)", "(131072, 8)", "(262144, 4)", "(524288, 2)", "(1048576, 1)"] 
# y_values = [1740.192, 2316.319, 1468.703, 1703.871, 1627.776, 1915.776, 3161.983, 5789.184, 9571.488, 17721.279, 35480.351]

# 2
x_labels = [0, 32, 16, 8, 4, 2, 30, 25, 20, 13, 10] 
y_values = [1740.192, 2726.975, 924.767, 922.432, 2834.080, 8343.392, 1650.367, 1515.488, 1189.919, 1291.520, 1301.599]

# Create figure and axes objects
fig, ax = plt.subplots(figsize=(10, 6))

# Create the bar graph using the axes object
ax.bar(x_labels, y_values, color='skyblue')

# Set major ticks to be at every 1 unit
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

# Get the current x-tick labels and positions
ticks = ax.get_xticks()
labels = ax.get_xticklabels()

# Set the new tick labels with rotation and alignment
ax.set_xticklabels(ticks, rotation=45, ha='right')

# Add labels and title using the axes object
ax.set_xlabel("Tile width")
ax.set_ylabel("Kernel Execution time (micro-seconds)")
ax.set_title("Tiled Version of Matrix Transpose")
ax.legend()

# Adjust layout to prevent labels from overlapping
plt.tight_layout() 

# Display the plot
plt.show()