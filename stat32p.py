# This is a prototype code for STAT032 project
#



####s the green button in the gutter to run the script.
#if __name__ == '__main__':

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# just
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

monthl=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
#data['monthn']
monthend=[]
for i in data.month:

    monthend.append(monthl.index(i))
data['monthn']=monthend
data=data.sort_values('monthn')

plt.hist(data.month)
fig, ax=plt.subplots(figsize=(10,6))
ax.set(title = "Plot title here",
       xlabel = "X axis label here",
       ylabel = "Y axis label here")

fig, ax=plt.subplots(figsize=(10,6))
ax.bar(data.month,data.area)
 #Set plot title and axes labels
ax.set(title = "Plot of No of Forest fires",
       xlabel = "Month",
       ylabel = " No of monthly fires")
# plot 3d map of fires
# setup the figure and axes
fig = plt.figure(figsize=(8, 3))
ax1 = fig.add_subplot(121, projection='3d')

x=data.X.values
y=data.Y.values

top = data.area.values
bottom = np.zeros_like(top)
width = depth = 1

ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
ax1.set_title('Shaded')
ax1.set_xlabel(' x-coordinate')
ax1.set_ylabel(' y-coordinate')
plt.show()
print(data.head())
#correl = data.corr()
#sns.heatmap(correl, vmin=-1, vmax=1, cmap='BrBG')
plt.show()
#This last line doesnt work at the moment but might be quite useful to summarise the data
#a = data.groupby('month').describe().unstack(1).reset_index().pivot(index='month', values=0, columns='level_1')
