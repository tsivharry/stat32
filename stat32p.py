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
#data=data1.loc[data1.area < 200.]
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
#_listvalue_ for i in list if condition
monthend=[monthl.index(i) for i in data.month]
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
#data=np.log(data.area)
#t-stastic for distributions :Sep-Aug
mean_aug=np.mean(np.log(data.area[data.month == 'aug']))
mean_sep=np.mean(np.log(data.area[data.month == 'sep']))
std_aug=np.std(np.log(data.area[data.month == 'aug']))
std_sep=np.std(np.log(data.area[data.month == 'sep']))
nofir_aug=len(data.area[data.month == 'aug'])
nofir_sep=len(data.area[data.month == 'sep'])
print(mean_aug,mean_sep,std_aug,std_sep)
tstat=(mean_sep-mean_aug)/((std_aug**2/nofir_aug)+(std_sep**2/nofir_sep))**0.5
print(tstat,nofir_aug,nofir_sep)
from scipy.stats import t
t_stat =tstat
dof = min(nofir_aug,nofir_sep)-1
# p-value for 2-sided test
pvalue=2*(1 - t.cdf(abs(t_stat), dof))
print(tstat,nofir_aug,nofir_sep, pvalue)

#Wald test for means to follow
# same test statistic as T-distribution
import scipy
p_valuesW = scipy.stats.norm.sf(abs(t_stat))*2# twosided
print(p_valuesW)

# perform Kolmogorov _Smirnov test
from scipy import stats
from scipy.stats import ks_2samp
data1=np.log(data.area[data.month == 'aug'])
data2=np.log(data.area[data.month == 'sep'])
data1=(data.area[data.month == 'aug'])
data2=(data.area[data.month == 'sep'])
#perform Kolmogorov-Smirnov test
ks_statistic, p_value = ks_2samp(data1, data2)
print(ks_statistic,p_value)
