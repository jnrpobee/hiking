
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.stats import ttest_ind
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multitest import fdrcorrection
from IPython.display import display
from scipy.stats import skew, kurtosis


import seaborn as sns
#from skimpy import skim 

# Step 2: Load data into a DataFrame
df = pd.read_excel('hiking_version2.xlsx')
display(df.head())

#disply the data types of the columns
display(df.dtypes)

#display the devices in the cellphone column
display(df['cellphone'].value_counts())

#display the devices in the other_device column
display(df['other_device'].value_counts())

#display the devices in the non_listed_devices column
display(df['device10'].value_counts())

# remove the word 'Other' from the other_device column
df['other_device'] = df['other_device'].replace('Other', '')
display(df['other_device'].value_counts())

#display the devices in the other_device column in a table format
display(df['other_device'].value_counts(25))

#create a pivot table to display the devices in the other_device column
pivot_table = df.pivot_table(index='other_device', values='cellphone', aggfunc='count')
display(pivot_table)

#delimites the other_device column by the comma
df['other_device'] = df['other_device'].str.split(',')
display(df['other_device'])

#display the devices in the other_device column in a table format
display(df['other_device'].value_counts(25))

print('----------------Common devices----------------')


# Count the number of devices in device1 to device10 columns
# Count the number of devices in each column
device_counts = df[['device1', 'device2', 'device3', 'device4', 'device5', 'device6', 'device7', 'device8', 'device9', 'device10']].stack().value_counts()

# Display the device counts
display(device_counts)

# Convert the Series to a DataFrame
device_counts_df = device_counts.to_frame('Count')
display(device_counts_df)

# Create a pivot table to display the device counts
#pivot_table = device_counts.melt(var_name='Device', value_name='Count').pivot_table(index='Device', values='Count', aggfunc='sum')

#display(pivot_table)

# Count the number of devices in each column
device_counts = df[['device1', 'device2', 'device3', 'device4', 'device5', 'device6', 'device7', 'device8', 'device9', 'device10']].apply(pd.Series.value_counts)
display(device_counts)
# Create a pivot table to display the device counts
device_counts = df[['device1', 'device2', 'device3', 'device4', 'device5', 'device6', 'device7', 'device8', 'device9', 'device10']].stack().value_counts()
#device_counts.columns = ['Device', 'Count']
#display(device_counts.pivot_table(index='Count', values='Device', aggfunc='sum'))

display(device_counts)



# Create a bar graph
device_counts.plot(kind='bar', color=plt.cm.tab20(range(len(device_counts))))
plt.xlabel('Devices')
plt.ylabel('Device Count')
plt.title('Common Devices')
labels = device_counts.index
colors = [plt.cm.tab20(i) for i in range(len(device_counts))]
legend_patches = [plt.Rectangle((0,0),1,1,fc=plt.cm.tab20(i), edgecolor = 'none') for i in range(len(device_counts))]
plt.legend(legend_patches, device_counts.index, loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
# Add data labels to the bars
for p in plt.gca().patches:
    plt.text(p.get_x() + p.get_width()/2, p.get_height() * 1.005, '{}'.format(p.get_height()), ha='center', fontsize=11, color='black')
plt.xticks([])
plt.show()

print('----------all device combined----------------')
#create a dataframe to combine the devices in the device1 to device10 columns
df['all_devices'] = df[['device1', 'device2', 'device3', 'device4', 'device5', 'device6', 'device7', 'device8', 'device9', 'device10']].apply(lambda x: ','.join(x.dropna().astype(str)), axis=1)
#display(df['all_devices'])
#create a pivot table to display the devices carried and the number of participants
pivot_table = df['all_devices'].value_counts().to_frame('Count')
display(pivot_table)

display('----------------Top 10 devices combination carried----------------')
#create a graph to display the first 10 of all devices carried by participants
df['all_devices'].value_counts().head(10).plot(kind='bar', color=plt.cm.tab20(range(10)))
plt.xlabel('Devices')
plt.ylabel('Device Count')
plt.title('10 most common Devices Combination')
labels = df['all_devices'].value_counts().head(10).index
colors = [plt.cm.tab20(i) for i in range(10)]
legend_patches = [plt.Rectangle((0,0),1,1,fc=plt.cm.tab20(i), edgecolor = 'none') for i in range(10)]
plt.legend(legend_patches, df['all_devices'].value_counts().head(10).index, loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=1)
# Add data labels to the bars
for p in plt.gca().patches:
    plt.text(p.get_x() + p.get_width()/2, p.get_height() * 1.005, '{}'.format(p.get_height()), ha='center', fontsize=11, color='black')
plt.xticks([])
plt.show()

print('----------------number of devices carried----------------')
# Create a new column 'all_devices' to merge devices in device1 to device10 columns
#print('----------all device combined----------------')
#df['all_devices'] = df[['device1', 'device2', 'device3', 'device4', 'device5', 'device6', 'device7', 'device8', 'device9', 'device10']].apply(lambda x: ','.join(x.dropna().astype(str)), axis=1)
#display(df['all_devices'])

# Count the number of devices carried by each participant
device_counts = df['all_devices'].value_counts()
print('-----number of devices carried----')
display(device_counts)

#create a pivot table to display the devices carried and the number of participants
pivot_table = device_counts.to_frame('Count')
display(pivot_table)

#count the number of devices in the all_devices column in a table format 
display(df['all_devices'].value_counts(25))

#count the number of devices in the all_devices column 
device_counts_df = df['all_devices'].str.split(',').apply(lambda x: len(x)).value_counts().reset_index()
device_counts_df.columns = ['Number of Devices', 'Count']
display(device_counts_df)

# Create a bar graph to display the number of devices carried by each participant
device_counts_df.plot(kind='bar', x='Number of Devices', y='Count', color=plt.cm.tab20(range(len(device_counts_df))))
plt.xlabel('Number of Devices')
plt.ylabel('Participant Count')
plt.title('Number of Devices Carried')
plt.xticks(rotation=0)
# Add data labels to the bars
for p in plt.gca().patches:
    plt.text(p.get_x() + p.get_width()/2, p.get_height() * 1.005, '{}'.format(p.get_height()), ha='center', fontsize=11, color='black')
plt.show()

display('----------3 devices carried----------------')
#create a dataframe of 3 devices carried by participants in the all_devices column
df['all_devices'] = df['all_devices'].str.split(',')
df['all_devices'] = df['all_devices'].apply(lambda x: x if len(x) == 3 else np.nan)
#display(df['all_devices'])
#create a pivot table to display 3 devices carried and the number of participants
pivot_table = df['all_devices'].value_counts().to_frame('Count')
display(pivot_table)

#create a graph of top 10 3 devices carried by participants
df['all_devices'].value_counts().head(10).plot(kind='bar', color=plt.cm.tab20(range(10)))
plt.xlabel('Devices')
plt.ylabel('Device Count')
plt.title('Top 10 3 Devices Carriers')
labels = df['all_devices'].value_counts().head(10).index
colors = [plt.cm.tab20(i) for i in range(10)]
legend_patches = [plt.Rectangle((0,0),1,1,fc=plt.cm.tab20(i), edgecolor = 'none') for i in range(10)]
plt.legend(legend_patches, df['all_devices'].value_counts().head(10).index, loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=1)
# Add data labels to the bars
for p in plt.gca().patches:
    plt.text(p.get_x() + p.get_width()/2, p.get_height() * 1.005, '{}'.format(p.get_height()), ha='center', fontsize=11, color='black')
plt.xticks([])
plt.show()

display('----------2 devices carried----------------')

# Create a pivot table to display 2 devices carried and the number of participants
pivot_table = df[['device1', 'device2', 'device3', 'device4', 'device5', 'device6', 'device7', 'device8', 'device9', 'device10']].apply(lambda x: x.dropna().tolist(), axis=1).apply(lambda x: x if len(x) == 2 else np.nan).value_counts().to_frame('Count')
display(pivot_table)

#create a graph of top 10 2 devices carried by participants
df[['device1', 'device2', 'device3', 'device4', 'device5', 'device6', 'device7', 'device8', 'device9', 'device10']].apply(lambda x: x.dropna().tolist(), axis=1).apply(lambda x: x if len(x) == 2 else np.nan).value_counts().head(13).plot(kind='bar', color=plt.cm.tab20(range(13)))
plt.xlabel('Devices')
plt.ylabel('Device Count')
plt.title('2 Devices Carried')
labels = df[['device1', 'device2', 'device3', 'device4', 'device5', 'device6', 'device7', 'device8', 'device9', 'device10']].apply(lambda x: x.dropna().tolist(), axis=1).apply(lambda x: x if len(x) == 2 else np.nan).value_counts().head(10).index
colors = [plt.cm.tab20(i) for i in range(10)]
legend_patches = [plt.Rectangle((0,0),1,1,fc=plt.cm.tab20(i), edgecolor = 'none') for i in range(10)]
plt.legend(legend_patches, df[['device1', 'device2', 'device3', 'device4', 'device5', 'device6', 'device7', 'device8', 'device9', 'device10']].apply(lambda x: x.dropna().tolist(), axis=1).apply(lambda x: x if len(x) == 2 else np.nan).value_counts().head(10).index, loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
# Add data labels to the bars
for p in plt.gca().patches:
    plt.text(p.get_x() + p.get_width()/2, p.get_height() * 1.005, '{}'.format(p.get_height()), ha='center', fontsize=11, color='black')
plt.xticks([])
plt.show()


display('----------4 devices carried----------------')
#createt a pivot table to display 4 devices carried and the number of participants
pivot_table = df[['device1', 'device2', 'device3', 'device4', 'device5', 'device6', 'device7', 'device8', 'device9', 'device10']].apply(lambda x: x.dropna().tolist(), axis=1).apply(lambda x: x if len(x) == 4 else np.nan).value_counts().to_frame('Count')
display(pivot_table)

#create a graph of top 10 4 devices carried by participants
df[['device1', 'device2', 'device3', 'device4', 'device5', 'device6', 'device7', 'device8', 'device9', 'device10']].apply(lambda x: x.dropna().tolist(), axis=1).apply(lambda x: x if len(x) == 4 else np.nan).value_counts().head(10).plot(kind='bar', color=plt.cm.tab20(range(10)))
plt.xlabel('Devices')
plt.ylabel('Device Count')
plt.title('4 Devices Carriers')
labels = df[['device1', 'device2', 'device3', 'device4', 'device5', 'device6', 'device7', 'device8', 'device9', 'device10']].apply(lambda x: x.dropna().tolist(), axis=1).apply(lambda x: x if len(x) == 4 else np.nan).value_counts().head(10).index
colors = [plt.cm.tab20(i) for i in range(10)]
legend_patches = [plt.Rectangle((0,0),1,1,fc=plt.cm.tab20(i), edgecolor = 'none') for i in range(10)]
plt.legend(legend_patches, df[['device1', 'device2', 'device3', 'device4', 'device5', 'device6', 'device7', 'device8', 'device9', 'device10']].apply(lambda x: x.dropna().tolist(), axis=1).apply(lambda x: x if len(x) == 4 else np.nan).value_counts().head(10).index, loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=1)
# Add data labels to the bars
for p in plt.gca().patches:
    plt.text(p.get_x() + p.get_width()/2, p.get_height() * 1.005, '{}'.format(p.get_height()), ha='center', fontsize=11, color='black')
plt.xticks([])
plt.show()

















print('----------------App used----------------')

# create a dataframe to count the number of apps used in the app_used column
app_counts = df['app_used'].value_counts()
#display(app_counts)

#create a pivot table to display the apps used and the number of participants
pivot_table = app_counts.to_frame('Count')
display(pivot_table)

#display the top 10 apps used by participants in a table format 
display('----------------Top 10 apps used----------------')
display(app_counts.head(10))
#create a graph to display the top 10 apps used by participants
app_counts.head(10).plot(kind='bar', color=plt.cm.tab20(range(10)))
plt.xlabel('Apps')
plt.ylabel('App Count')
plt.title('Top 10 Apps Used')
labels = app_counts.head(10).index
colors = [plt.cm.tab20(i) for i in range(10)]
legend_patches = [plt.Rectangle((0,0),1,1,fc=plt.cm.tab20(i), edgecolor = 'none') for i in range(10)]
plt.legend(legend_patches, app_counts.head(10).index, loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=1)
# Add data labels to the bars
for p in plt.gca().patches:
    plt.text(p.get_x() + p.get_width()/2, p.get_height() * 1.005, '{}'.format(p.get_height()), ha='center', fontsize=11, color='black')
plt.xticks([])
plt.show()

display('----------------Common Apps usage----------------')
# Create a bar graph to display the apps used by participants
app_counts.plot(kind='bar', color=plt.cm.tab20(range(len(app_counts))))
plt.xlabel('Apps')
plt.ylabel('App Count')
plt.title('Apps Used')
labels = app_counts.index
colors = [plt.cm.tab20(i) for i in range(len(app_counts))]
legend_patches = [plt.Rectangle((0,0),1,1,fc=plt.cm.tab20(i), edgecolor = 'none') for i in range(len(app_counts))]
plt.legend(legend_patches, app_counts.index, loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
# Add data labels to the bars
for p in plt.gca().patches:
    plt.text(p.get_x() + p.get_width()/2, p.get_height() * 1.005, '{}'.format(p.get_height()), ha='center', fontsize=11, color='black')
plt.xticks([])
plt.show()

#create a datafrfame to count each app used in the app_used column
# app_counts_df = df['app_used'].str.split(',').apply(lambda x: len(x) if not pd.isna(x) else 0).value_counts().reset_index()
# app_counts_df.columns = ['Number of Apps', 'Count']
# display(app_counts_df)


# create a dataframe that will split the app_used column by the comma
df['app_used'] = df['app_used'].str.split(',')
#display(df['app_used'])


print('----------------Common Apps usage----------------')
#create a dataframe to count the number of apps used in the app1 to app8 column

print('----------------Common apps----------------')
app_counts = df[['app1', 'app2', 'app3', 'app4', 'app5', 'app6', 'app7', 'app8']].stack().value_counts()
#display(app_counts)

#create a pivot table to display the apps used and the number of participants
pivot_table = app_counts.to_frame('Count')
display(pivot_table)

# Create a bar graph to display the common apps used by participants
app_counts.plot(kind='bar', color=plt.cm.tab20(range(len(app_counts))))
plt.xlabel('Apps')
plt.ylabel('App Count')
plt.title('Common Apps')
labels = app_counts.index
colors = [plt.cm.tab20(i) for i in range(len(app_counts))]
legend_patches = [plt.Rectangle((0,0),1,1,fc=plt.cm.tab20(i), edgecolor = 'none') for i in range(len(app_counts))]
plt.legend(legend_patches, app_counts.index, loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
# Add data labels to the bars
for p in plt.gca().patches:
    plt.text(p.get_x() + p.get_width()/2, p.get_height() * 1.005, '{}'.format(p.get_height()), ha='center', fontsize=11, color='black')
plt.xticks([])
plt.show()





