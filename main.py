
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from model import GaussianProcess


# Load data
def load_data():
    """Loads the data from the data file."""

    # Get raw data
    file = 'data/NASDAQ_2019_2023.csv'
    rawdata = np.loadtxt(file, delimiter=',', skiprows=1, dtype=str)

    # Get dates
    dates = np.array([np.datetime64(date) for date in rawdata[:, 0]])

    # Turn datetime into floats
    dates = (dates - dates[0]) / np.timedelta64(1, 'D')

    # Get data
    data = rawdata[:, 1].astype(float)

    # Return
    return dates, data


# Get data
dates, data = load_data()
data -= data.mean()
sigma = (data[1:] - data[:-1]).std()

# Run the model
model = GaussianProcess(dates, data, sigma=sigma)
model.fit()

# Plot the results
model.plot_results()




# Done
print('Done.')
