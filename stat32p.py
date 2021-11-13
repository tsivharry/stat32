# This is a prototype code for STAT032 project
#



####s the green button in the gutter to run the script.
#if __name__ == '__main__':

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data_raw = pd.read_csv('forestfires_data.csv')
data_raw.shape
area_threshold = 0.01; # hectares
data = data_raw.loc[data_raw.area > area_threshold]
data_summary = data.describe()
plt.hist(data.area, 50)
plt.show()
plt.hist(np.log(data.area), 50)
plt.show()
plt.hist(np.log(data.area[data.month == 'aug']), 20)
plt.hist(np.log(data.area[data.month == 'sep']), 20)

plt.hist(data.month)
plt.show()
correl = data.corr()
sns.heatmap(correl, vmin=-1, vmax=1, cmap='BrBG')

#This last line doesnt work at the moment but might be quite useful to summarise the data
#a = data.groupby('month').describe().unstack(1).reset_index().pivot(index='month', values=0, columns='level_1')
