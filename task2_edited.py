# STAT0032 Group Project
# Task 2:
# # two-sample hypothesis tests on forest fires in Aug. & Sep.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
from scipy.stats import t
from scipy.stats import ks_2samp

# Read the data
data = pd.read_csv('forestfires.csv')  # (517, 13)
# print("The size of the dataset: "+data_raw.shape)
print(data.head(7))

# Find forest fires of large scales
# # entries with 0 values for area represent small fires whose burned area is lower than 0.01ha
area_threshold = 0.01
fires = data.loc[data.area > area_threshold]
# print(len(fires))
# data_summary = fires.describe()
# print(data_summary)


# Plots:
# # Distribution of the number of fires
plt.hist(fires.area, 50)
plt.title("Distribution of Burned Area of Fires")
plt.show()
# plt.savefig("distr.jpg")

# # Distribution of the number of fires after taking log
plt.hist(np.log(fires.area), 50)
plt.title("Distribution of Log of Burned Area of Fires")
plt.show()
# plt.savefig("distr_log.jpg")



# In August and September (given data in log-normal):
fires_aug = fires.area[fires.month == 'aug']
fires_sep = fires.area[fires.month == 'sep']
log_fires_aug = np.log(fires_aug)
log_fires_sep = np.log(fires_sep)
# August fires:
n_f_aug = len(log_fires_aug)
mean_aug = np.mean(log_fires_aug)
sd_aug = np.std(log_fires_aug)
# September fires:
n_f_sep = len(log_fires_sep)
mean_sep = np.mean(log_fires_sep)
sd_sep = np.std(log_fires_sep)

print("""=========================\n
August:\n
Number of Fires: {0}\nMean of Areas of Fires: {1}\nSD of Areas of Fires: {2}\n
-----------\n
September:\n
Number of Fires: {3}\nMean of Areas of Fires: {4}\nSD of Areas of Fires: {5}\n
========================="""
      .format(n_f_aug, mean_aug, sd_aug,
              n_f_sep, mean_sep, sd_sep))

# # Fires in August
plt.hist(log_fires_aug, 20, color='dodgerblue', alpha=0.9, density=True)
# # Fires in September
plt.hist(log_fires_sep, 20, color='orange', alpha=0.9, density=True)
plt.title("Log of Areas of Fires in August and September")
plt.show()


# Plot 3-D Map of Fires
fig = plt.figure(figsize=(8, 3))
# fig = plt.figure(8, 3)
ax1 = fig.add_subplot(121, projection='3d')
x = data.X.values
y = data.Y.values
top = data.area.values
bottom = np.zeros_like(top)
depth = 1
width = 1
ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
ax1.set_title('Number of Fires in Different Locations')
ax1.set_xlabel('x-coordinate')
ax1.set_ylabel('y-coordinate')
plt.show()


# # Replace the names of months with numbers
# months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
# fires.loc[:, 'month'] = fires.loc[:, 'month'].replace(months, list(range(1, 13)))
# # print(fires.loc[:, 'month'])


# # Number of Forest Fires in Each Month
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(fires.month, fires.area)
ax.set(title="Plot of No. of Forest Fires in Each Month",
       xlabel="Month",
       ylabel="No. of Monthly Fires")


# To calculate the t statistic
def get_tstat(n1, n2, mean1, mean2, sd1, sd2):
    # print("++++ Use get_tstat ++++")
    # print(n1, n2, mean1, mean2, sd1, sd2)
    assert (n1 != 0) and (n2 != 0) and (sd1 ** 2 / n1 + sd2 ** 2 / n2 != 0)
    tstat = (mean1 - mean2) / np.sqrt(sd1 ** 2 / n1 + sd2 ** 2 / n2)
    return tstat


t_stat = get_tstat(n_f_aug, n_f_sep, mean_aug, mean_sep, sd_aug, sd_sep)  # t statistics
dof = min(n_f_aug, n_f_sep) - 1  # degree of freedom

# T test
p_val_t = 2 * (1 - t.cdf(abs(t_stat), dof))  # ? - How to calculate p-value
print("t test:")
print("t statistic: {0}; p-value: {1}\n".format(t_stat, p_val_t))

# Wald test for means to follow
p_val_w = 2 * norm.sf(abs(t_stat))
print("Wald test:")
print("Wald statistic (t statistic): {0}; p-value: {1}\n".format(t_stat, p_val_w))

# Kolmogorov-Smirnov test
print("Kolmogorov-Smirnov Test:")
# with the original data:
ks_stat, p_val_ks = ks_2samp(fires_aug, fires_sep)
print(f"On the original data: \nKS statistic: {ks_stat}; p-value: {p_val_ks}")

# with the log data:
ks_stat_l, p_val_ks_l = ks_2samp(log_fires_aug, log_fires_sep)
print(f"On the log data: \nKS statistic: {ks_stat_l}; p-value: {p_val_ks_l}")
