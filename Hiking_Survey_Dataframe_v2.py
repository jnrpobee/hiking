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
import matplotlib.patches as mpatches
import seaborn as sns
from skimpy import skim 

display("---------------- Apps ---------/n")

import pandas as pd

# Load only Sheet 2
df = pd.read_excel("hiking_data_v2.xlsx", sheet_name="AppUpdate")

display(df.head())  # Display first few rows

# Count the number of apps in each column
# Ensure all columns are of string type before applying value_counts
app_counts_per_column = df.astype(str).apply(pd.Series.value_counts).fillna(0).astype(int)
app_counts_per_column = app_counts_per_column.loc[~app_counts_per_column.index.str.lower().str.contains('nan')]
print('\n-----------------app counts per column------------------')
display(app_counts_per_column)

# Count the number of apps in each column and sum them up
# Sum the number of usage of an app in each row
total_app_counts_per_row = app_counts_per_column.sum(axis=1)
print('\n-----------------total app counts per row------------------')
display(total_app_counts_per_row)


# Get the top 4 most common Apps
top_apps = total_app_counts_per_row.nlargest(13)

# Create a bar graph
plt.figure(figsize=(8, 4.5))
colors = plt.cm.tab20(range(len(top_apps)))
ax = top_apps.plot(kind='bar', color=colors, width=0.95)  # Increase width to reduce distance between bars
plt.xlabel('Apps')
plt.ylabel('App Count')
# plt.title('Frequently Used Apps')
labels = top_apps.index
legend_patches = [plt.Rectangle((0,0),1,1,fc=color, edgecolor='none') for color in colors]
plt.legend(legend_patches, top_apps.index, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)
# Add data labels to the bars
for p in ax.patches:
    plt.text(p.get_x() + p.get_width()/2, p.get_height() * 1.005, '{}'.format(p.get_height()), ha='center', fontsize=11, color='black')
plt.xticks([])
plt.show()


print(' ---- trial with seaborn on the apps -----------')
#create a bar chart of the top 4 apps
plt.figure(figsize=(10, 6))
sns.barplot(
    x=top_apps.index,
    y=top_apps.values,
    hue=top_apps.index,  # Assign x variable to hue
    palette='viridis',
    legend=False
)
plt.xlabel('Apps')
plt.ylabel('App Count')
plt.title('Apps')
for p in plt.gca().patches:
    plt.text(p.get_x() + p.get_width()/2, p.get_height() * 1.005, '{:.0f}'.format(p.get_height()), ha='center', fontsize=11, color='black')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

print('--------end of trial with seaborn on the apps -----------')

print( '-----------------app counts in percentage------------------')
# Convert the app_counts_per_column to percentage (app_counts_per_column / number of respondants * 100)
app_counts_percentage = total_app_counts_per_row / len(df) * 100
app_counts_percentage = app_counts_percentage.round(1).astype(str) + '%'
display(app_counts_percentage)
display('len(df)', len(df))
# display('df', df)

print("---------------- end of apps -----------\n\n")
print("---------------- start of device combination -----------\n\n")

df = pd.read_excel('hiking_data_v2.xlsx', sheet_name='DevicesUpdate')

# display(df.head())  # Display first few rows

# device_counts_per_column = df.astype(str).apply(pd.Series.value_counts).fillna(0).astype(int)
device_counts_per_column = df.drop(columns=['Count'], errors='ignore').astype(str).apply(pd.Series.value_counts).fillna(0).astype(int)
device_counts_per_column = device_counts_per_column.loc[~device_counts_per_column.index.str.lower().str.contains('nan')]

total_device_counts_per_row = device_counts_per_column.sum(axis=1)

top_devices = total_device_counts_per_row.nlargest(13)

# Create a bar graph
plt.figure(figsize=(8, 5))
colors = plt.cm.tab20(range(len(top_devices)))
top_devices.plot(kind='bar', color=colors, width=0.95)
plt.xlabel('Devices')
plt.ylabel('Device Count')
# plt.title('Frequently Used Devices')
labels = top_devices.index
legend_patches = [plt.Rectangle((0,0),1,1,fc=color, edgecolor='none') for color in colors]
plt.legend(legend_patches, top_devices.index, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)  # Adjusted bbox_to_anchor to remove white space
for p in plt.gca().patches:
    plt.text(p.get_x() + p.get_width()/2, p.get_height() * 1.005, '{}'.format(p.get_height()), ha='center', fontsize=11, color='black')
plt.xticks([])
plt.show()

print('\n-----------------device counts per column------------------')
display(device_counts_per_column)

print('\n-----------------device 2 combination------------------')
df = pd.read_excel('hiking_data_v2.xlsx', sheet_name='2devices')

# combine device1 and device2 columns into a new column called 2devices
df['2devices'] = df['device1'] + ', ' + df['device2']

# remove the leading and trailing spaces from the new column
df['2devices'] = df['2devices'].str.strip()

# display(df.head())  # Display first few rows

# Count the number of devices in 2devices column
device_2_counts = df['2devices'].value_counts()
# display(device_2_counts)

#percentage of devices in the 2devices column
device_2_percentage = df['2devices'].value_counts(normalize=True) * 100
device_2_percentage = device_2_percentage.round(1).astype(str) + '%'
display('2 device combinaiton', device_2_percentage.round(1))

# Create a bar graph of the counts in the 2devices column

device_2_colors = plt.cm.tab20(range(len(device_2_counts)))  # Use a colormap for consistent coloring
device_2_counts.plot(kind='bar', color=device_2_colors)
plt.xlabel('Device Combinations')
plt.ylabel('Device Count')
# plt.title('Device Count in 2devices Column')
legend_patches = [plt.Rectangle((0,0),1,1,fc=color, edgecolor='none') for color in device_2_colors]
plt.legend(legend_patches, device_2_counts.index, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)  # Adjusted bbox_to_anchor to remove white space
# Add data labels to the bars
for p in plt.gca().patches:
    plt.text(p.get_x() + p.get_width()/2, p.get_height() * 1.005, '{}'.format(p.get_height()), ha='center', fontsize=11, color='black')
plt.xticks([])
plt.show()

print('\n -----------------devices in 2 device combination------------------')
df = pd.read_excel('hiking_data_v2.xlsx', sheet_name='2devCount')

# display(df.head())  # Display first few rows

# Count the number of devices in the '2_Devices' column
deviceTwo_counts = df['Two_Devices'].value_counts()
# display('device 2 count', deviceTwo_counts)

# percentage of devices in the 2devices column
deviceTwo_percentage = df['Two_Devices'].value_counts(normalize=True) * 100
deviceTwo_percentage = deviceTwo_percentage.round(1).astype(str) + '%'
display('devices in 2 device combination',deviceTwo_percentage.round(1))

device2CountColors = plt.cm.tab20(range(len(device_2_counts)))  # Use a colormap for consistent coloring
deviceTwo_counts.plot(kind='bar', color=device2CountColors)
plt.xlabel('Devices')
plt.ylabel('Count')
# plt.title('Device Count in 2devices Column')
legend_patches = [plt.Rectangle((0,0),1,1,fc=color, edgecolor='none') for color in device2CountColors]
plt.legend(legend_patches, deviceTwo_counts.index, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)  # Adjusted bbox_to_anchor to remove white space
for p in plt.gca().patches:
    plt.text(p.get_x() + p.get_width()/2, p.get_height() * 1.005, '{}'.format(p.get_height()), ha='center', fontsize=11, color='black')
plt.xticks([])
plt.show()

print ('\n-----------------device 3 combination------------------')
df = pd.read_excel('hiking_data_v2.xlsx', sheet_name='3devices')

# combine device1, device2 and device3 columns into a new column called 3devices
df['3devices'] = df['device1'] + ', ' + df['device2'] + ', ' + df['device3']

# remove the leading and trailing spaces from the new column
df['3devices'] = df['3devices'].str.strip()


# display(df.head())  # Display first few rows


# Count the number of devices in 3devices column
deviceThree_counts = df['3devices'].value_counts()
# display(deviceThree_counts)

# percentage of devices in the 3devices column
deviceThree_percentage = df['3devices'].value_counts(normalize=True) * 100
deviceThree_percentage = deviceThree_percentage.round(1).astype(str) + '%'
display('3 device combination ',deviceThree_percentage.round(1))


# Create a bar graph of the counts in the 3devices column
device_3_count = deviceThree_counts.nlargest(8)  # Get the top 13 device combinations

device_3_colors = plt.cm.tab20(range(len(device_3_count)))  # Use a colormap for consistent coloring

device_3_count.plot(kind='bar', color=device_3_colors)
plt.xlabel('Device Combinations')
plt.ylabel('Device Count')
# plt.title('Device Count in 3devices Column')
legend_patches = [plt.Rectangle((0,0),1,1,fc=color, edgecolor='none') for color in device_3_colors]
plt.legend(legend_patches, device_3_count.index, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=1)  # Adjusted bbox_to_anchor to remove white space

# Add data labels to the bars
for p in plt.gca().patches:
    plt.text(p.get_x() + p.get_width()/2, p.get_height() * 1.005, '{}'.format(p.get_height()), ha='center', fontsize=11, color='black')
plt.xticks([])
plt.show()


print('\n-----------------devices in 3 device combination------------------')
df = pd.read_excel('hiking_data_v2.xlsx', sheet_name='3devCount')


# display(df.head())  # Display first few rows


# Count the number of devices in the '3_Devices' column
deviceThree_counts = df['Three_Devices'].value_counts()
# display('device 3 count', deviceThree_counts)

# percentage of devices in the 3devices column
deviceThree_percentage = df['Three_Devices'].value_counts(normalize=True) * 100
deviceThree_percentage = deviceThree_percentage.round(1).astype(str) + '%'
display('device 3 count percentage', deviceThree_percentage.round(1))


device3CountColors = plt.cm.tab20(range(len(deviceThree_counts)))  # Use a colormap for consistent coloring

deviceThree_counts.plot(kind='bar', color=device3CountColors)
plt.xlabel('Devices')
plt.ylabel('Count')
# plt.title('Device Count in 3devices Column')
legend_patches = [plt.Rectangle((0,0),1,1,fc=color, edgecolor='none') for color in device3CountColors]
plt.legend(legend_patches, deviceThree_counts.index, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)  # Adjusted bbox_to_anchor to remove white space
for p in plt.gca().patches:
    plt.text(p.get_x() + p.get_width()/2, p.get_height() * 1.005, '{}'.format(p.get_height()), ha='center', fontsize=11, color='black')
plt.xticks([])
plt.show()


print("---------------- End of 2 and 3 device combination ------fix-----\n\n\n")

# Step 2: Load data into a DataFrame
df = pd.read_excel('hiking_data_v2.xlsx')
#print(df.head()) 

print ("these are the gender counts---")
# Count the number of respondants in the dataset by 
gender_counts = df['Gender'].value_counts()
print('\n-----------------participant counts------------------')
display(gender_counts)

# Calculate the percentage of each row in the Gender column
gender_percentage = df['Gender'].value_counts(normalize=True) * 100
display(gender_percentage.round(1))


#------------------------------------------------#

#------------------------------------------------#

# Split the devices in the name_of_devices column by comma and create new # Split the devices in the name_of_devices column by comma and create new columns
devices_df = df['combined'].str.split(',', expand=True).add_prefix('Device')

# Remove leading and trailing whitespace from the new columns
devices_df = devices_df.apply(lambda x: x.str.strip())

# Concatenate the original dataframe with the new devices dataframe
df = pd.concat([df, devices_df], axis=1)

# Display the updated dataframe
# display(df)

# Count the number of devices in each column
device_counts_per_column = devices_df.apply(pd.Series.value_counts).fillna(0).astype(int)
print('\n-----------------device counts per column------------------')
# display(device_counts_per_column)

# Count the number of devices in each column and sum them up
device_counts = df[devices_df.columns].stack().value_counts()
print('\n-----------------device counts------------------')
display(device_counts)

print('-----------------device counts in percentage------------------')

#convert the device_counts to percentage (device_counts / number of respondants * 100)
device_counts_percentage = device_counts / len(df) * 100
device_counts_percentage = device_counts_percentage.round(1).astype(str) + '%'
device_counts_percentage = device_counts_percentage.to_frame().rename(columns={0: 'Percentage'})
device_counts_percentage['Count'] = device_counts
display(device_counts_percentage)
display('len(df)', len(df))



# device_counts = df[['Device 0', 'Device 1', 'Device 2', 'Device 3', 'Device 4', 'Device 5', 'Device 6', 'Device 7', 'Device 8', 'Device 9']].apply(pd.Series.value_counts)
# display(device_counts)
# # Count the number of devices in each column and sum them up
# device_counts = df[['Device 0', 'Device 1', 'Device 2', 'Device 3', 'Device 4', 'Device 5', 'Device 6', 'Device 7', 'Device 8', 'Device 9']].stack().value_counts()
# display(device_counts)
# Get the top 4 most common devices
top_devices = device_counts.nlargest(4)


"under testing for consistency of the data. especially with the color coding of the devices --- original code"

# Create a table with the device names and their counts
table = pd.DataFrame({'Device': top_devices.index, 'Count': top_devices.values})

# Display the table
# display('\n-----------------device counts------------------', table)

# Get the top 4 most common devices
top_devices = device_counts.nlargest(13)

# Create a bar graph
colors = plt.cm.tab20(range(len(top_devices)))
top_devices.plot(kind='bar', color=colors)
plt.xlabel('Devices')
plt.ylabel('Device Count')
# plt.title('Common Devices')
labels = top_devices.index
legend_patches = [plt.Rectangle((0,0),1,1,fc=color, edgecolor='none') for color in colors]
plt.legend(legend_patches, top_devices.index, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
# Add data labels to the bars
for p in plt.gca().patches:
    plt.text(p.get_x() + p.get_width()/2, p.get_height() * 1.005, '{}'.format(p.get_height()), ha='center', fontsize=11, color='black')
plt.xticks([])
plt.show()



#---- 2 devices testing on the data set ----#

# # Count the number of devices in the 'Device 2' column
# device_1_counts = df['Device0'].value_counts()

# # Display the counts
# display(device_1_counts)

# # Create a bar graph of the counts in the 'Device 2' column
# device_1_colors = [colors[top_devices.index.get_loc(device)] if device in top_devices.index else 'grey' for device in device_1_counts.index]
# device_1_counts.plot(kind='bar', color=device_1_colors)
# plt.xlabel('Devices')
# plt.ylabel('Device Count')
# plt.title('Device Count in Device 2 Column')
# legend_patches = [plt.Rectangle((0,0),1,1,fc=color, edgecolor='none') for color in device_1_colors]
# plt.legend(legend_patches, device_1_counts.index, loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
# # Add data labels to the bars
# for p in plt.gca().patches:
#     plt.text(p.get_x() + p.get_width()/2, p.get_height() * 1.005, '{}'.format(p.get_height()), ha='center', fontsize=11, color='black')
# plt.xticks([])
# plt.show()

#---------------testing--------------#
"under testing for consistency of the data. especially with the color coding of the devices"

# Create a table with the device names and their counts
table = pd.DataFrame({'Device': top_devices.index, 'Count': top_devices.values})

# Display the table
# display(table)

# # Get the top 4 most common devices
# top_devices = device_counts.nlargest(13)

# # Create a bar graph
# colors = plt.cm.tab20(range(len(top_devices)))
# top_devices.plot(kind='bar', color=colors)
# plt.xlabel('Devices')
# plt.ylabel('Device Count')
# plt.title('Common Devices')
# labels = top_devices.index
# legend_patches = [plt.Rectangle((0,0),1,1,fc=color, edgecolor='none') for color in colors]
# plt.legend(legend_patches, top_devices.index, loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
# # Add data labels to the bars
# for p in plt.gca().patches:
#     plt.text(p.get_x() + p.get_width()/2, p.get_height() * 1.005, '{}'.format(p.get_height()), ha='center', fontsize=11, color='black')
# plt.xticks([])
# plt.show()

#---- 2 devices testing on the data set ----#

# Count the number of devices in the 'Device 2' column
device_2_counts = df['Device1'].value_counts()

# Display the counts
print(device_2_counts)

# Create a bar graph of the counts in the 'Device 2' column
device_2_colors = [colors[top_devices.index.get_loc(device)] if device in top_devices.index else 'grey' for device in device_2_counts.index]
device_2_counts.plot(kind='bar', color=device_2_colors)
plt.xlabel('Devices')
plt.ylabel('Device Count')
# plt.title('Device Count in Device 2 Column')
legend_patches = [plt.Rectangle((0,0),1,1,fc=color, edgecolor='none') for color in device_2_colors]
plt.legend(legend_patches, device_2_counts.index, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
# Add data labels to the bars
for p in plt.gca().patches:
    plt.text(p.get_x() + p.get_width()/2, p.get_height() * 1.005, '{}'.format(p.get_height()), ha='center', fontsize=11, color='black')
plt.xticks([])
plt.show()

#-----------2 devices testing on the data set-----------------#
print('\n2 devices\n')
df_2_devices = df['2devices'].value_counts().to_frame().T
print(df_2_devices)
print('\n')

#create a graph of the 2devices dataframe as a bar graph with the devices as the x-axis and the count as the y-axis

df_2_devices.plot(kind='bar')
plt.xlabel('Devices')
plt.ylabel('Count')
plt.title('2 Devices')
legend_patches = [plt.Rectangle((0,0),1,1,fc='blue', edgecolor='none')]
# legend_patches = [mpatches.Patch(color='blue', label='2 Devices')]
plt.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
#label the bars with the count
ax = plt.gca()
for p in plt.gca().patches:
    plt.text(p.get_x() + p.get_width()/2, p.get_height() * 1.005, '{}'.format(p.get_height()), ha='center', fontsize=11, color='black')
plt.xticks([])
plt.show()
print('done\n')

#--------- 3 devices testing on the data set ---------#

# Count the number of devices in the 'Device 3' column
device_3_counts = df['Device3'].value_counts()

# Display the counts
display(device_3_counts)

# Create a bar graph of the counts in the 'Device 3' column
device_3_colors = [colors[top_devices.index.get_loc(device)] if device in top_devices.index else 'grey' for device in device_3_counts.index]
device_3_counts.plot(kind='bar', color=device_3_colors)
plt.xlabel('Devices')
plt.ylabel('Device Count')
# plt.title('Device Count in Device 3 Column')
legend_patches = [plt.Rectangle((0,0),1,1,fc=color, edgecolor='none') for color in device_3_colors]
plt.legend(legend_patches, device_3_counts.index, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
# Add data labels to the bars
for p in plt.gca().patches:
    plt.text(p.get_x() + p.get_width()/2, p.get_height() * 1.005, '{}'.format(p.get_height()), ha='center', fontsize=11, color='black')
plt.xticks([])
plt.show()




































































# Define a mapping from categories to numerical values
mapping = {
    'Once a month': 12,  # 12 times a year
    '2-3 times per year': 2.5,  # Average of 2 and 3
    'Once a week': 52, # once a week
    'Once a year': 1, # once a year
    'Never': 0, # never
}

print ("doing the replacement...")
# Replace the categories with their numerical values
df['hiking_duration_numeric'] = df['hiking_duration'].replace(mapping).infer_objects(copy=False)
#display (df['hiking_duration_numeric'].head(5))
print ("Done")

# Calculate the mean
avg_hiking_duration = df['hiking_duration_numeric'].mean()
print ("mean of all respondants: ")
display(avg_hiking_duration)


print('comparison between 1 and 5 devices')

# make a new dataframe that just contains the rows in df that have 1 in the number_of_devices column
df_1_devices = df[df['number _of_devices'] == 1]
#display (df_1_devices.head(10))
# and get a new dataframe with just the rows with 5 in number_of_devices
df_5_devices = df[df['number _of_devices'] == 5]
#display (df_5_devices.head(10))

# Calculate the mean of the hiking duration for both groups
avg_hiking_duration_1_devices = df_1_devices['hiking_duration_numeric'].mean()
avg_hiking_duration_5_devices = df_5_devices['hiking_duration_numeric'].mean()
print ("mean of 1 devices: ")
display(avg_hiking_duration_1_devices)
print ("mean of 5 devices: ")
display(avg_hiking_duration_5_devices)




print ("---------------grahp of the hiking duration for 1 and 5 devices-----------------")
# Graph histogram of 1 devices and 5 devices
plt.hist(df_1_devices['hiking_duration_numeric'], bins=10, alpha=0.5, label='1 device')
plt.hist(df_5_devices['hiking_duration_numeric'], bins=10, alpha=0.5, label='5 devices')
plt.xlabel('Hiking Frequency')
plt.ylabel('Count')
plt.legend()
plt.show()
print('done')

# Step 3: Create a pivot table
pivot_table = pd.pivot_table(df, values='hiking_duration_numeric', index='number _of_devices', aggfunc='mean')
display(pivot_table)
                   

# Create a pivot table with devices as columns and frequencies as rows
pivot_table = df[df['number _of_devices'].isin([1, 5])].pivot_table(index='hiking_duration_numeric', columns='number _of_devices', aggfunc='size', fill_value=0)

# Plot the pivot table as a heatmap
plt.imshow(pivot_table, cmap='coolwarm', interpolation='nearest', extent=[-0.5, len(pivot_table.columns)-0.5, -0.5, len(pivot_table.index)-0.5])
plt.colorbar(label='Count')
plt.xlabel('Number of Devices')
plt.ylabel('Hiking Duration')
plt.xticks(range(len(pivot_table.columns)), pivot_table.columns)
plt.yticks(range(len(pivot_table.index)), pivot_table.index)

# Display the cell values
for i in range(len(pivot_table.index)):
    for j in range(len(pivot_table.columns)):
        plt.text(j, i, pivot_table.iloc[i, j], ha='center', va='center', color='black')

plt.title('Hiking Duration by Number of Devices')
plt.show()



# Perform chi-square test
print('\nchi-square test on number of devices and hiking duration')
chi2, p, _, expected = stats.chi2_contingency(pivot_table)

# Print p-value
display("p-value:", p)

# Calculate residuals
residuals = (pivot_table - expected) / expected

# Display the residuals in a table
print("Residuals:")
display(residuals)


# Count the number of devices in the number_of_devices column
print('\n-----------------device counts--------------------')
device_counts = df['number _of_devices'].value_counts()
# Ensure 0 is included in the index, even if not present in the data
if 0 not in device_counts.index:
    device_counts.loc[0] = 0
device_counts = device_counts.sort_index()
display(device_counts, '\n')
# Sort the device counts in ascending order
device_counts = device_counts.sort_index()
device_counts.plot(kind='bar')
plt.xlabel('Number of Devices')
plt.ylabel('Count')
# plt.title('Distribution of Devices')
plt.xticks(rotation=0, ha='center')
#plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
# Add data labels to the bars
for p in plt.gca().patches:
    plt.text(p.get_x() + p.get_width()/2, p.get_height() * 1.005, '{}'.format(p.get_height()),ha='center', fontsize=8, color='black')
plt.show()


print('------device counts using palette=tab10---------')
colors = plt.cm.tab10(range(len(device_counts)))
device_counts.plot(kind='bar', color=colors, width=0.95)
plt.xlabel('Number of Devices')
plt.ylabel('Count')
# plt.title('Distribution of Devices')
labels = device_counts.index
# legend_patches = [plt.Rectangle((0,0),1,1,fc=color, edgecolor='none') for color in colors]
# plt.legend(legend_patches, device_counts.index, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
# Add data labels to the bars
for p in plt.gca().patches:
    plt.text(p.get_x() + p.get_width()/2, p.get_height() * 1.005, '{:.0f}'.format(p.get_height()), ha='center', fontsize=9, color='black')
plt.xticks(rotation=0, ha='center')
plt.tight_layout()
plt.show()
print('--------end of device counts using palette=tab10---------')


print('--------trial with seaborn on the distribution of devices---------')
# Create a bar plot using seaborn

# Ensure 0 is included in the number of devices, even if not present in the data
if 0 not in df['number _of_devices'].unique():
    df_with_0 = pd.concat([df, pd.DataFrame({'number _of_devices': [0]})], ignore_index=True)
else:
    df_with_0 = df

plt.figure(figsize=(8, 5))
sns.countplot(data=df_with_0, x='number _of_devices', hue='number _of_devices', palette='tab10', legend=False)
plt.xlabel('Number of Devices')
plt.ylabel('Count')
plt.title('Distribution of Devices')
# Add data labels to the bars (no decimals)
for p in plt.gca().patches:
    plt.text(
        p.get_x() + p.get_width() / 2,
        p.get_height() * 1.005,
        '{:.0f}'.format(p.get_height()),
        ha='center',
        fontsize=9,
        color='black'
    )
plt.xticks(rotation=0, ha='center')
plt.tight_layout()
plt.show()

print('--------end of trial with seaborn on the distribution of devices---------')


# Create a pivot table of the number of devices and the percentage
percentage_devices = df['number _of_devices'].value_counts(normalize=True) * 100
percentage_devices = percentage_devices.round(1).astype(str) + '%'
print("\n------------Percentage of devices:------------")
display(percentage_devices,'\n')


# change the name of yes in the smartphone column to smartphone and no to an empty string
df['smartphone'] = df['smartphone'].replace({'Yes': 'smartphone', 'No': ''})
# print(df['smartphone'].head(10))


# Create a dataframe for the combined column
combined_df = df[['combined']]
#print(combined_df.head())


# combine the two dataframes of smartphone and name_of_devices columns
#print('\ncombined columns created')
#df['smartphone'] = df['smartphone'].str.lower()
#df['name_of_devices'] = df['name_of_devices'].str.lower()
#df['combined'] = df['smartphone'] + ', ' + df['name_of_devices']
#print(df['combined'].head(10))
print('combined column creation completed\n')

# Create a pivot table of the combined column with the number_of_devices column as rows and the count as columns 
print('combined column with number of devices')
pivot_table_combined = df.pivot_table(index='number _of_devices', columns='combined', aggfunc='size', fill_value=0)
# display(pivot_table_combined)



# Create a pivot table of the number of each device in combined column
pivot_table_combined_devices = df.pivot_table(index='combined', aggfunc='size')
# display(pivot_table_combined_devices)



# Count the number of devices in the combined column
device_counts_combined = df['combined'].value_counts()
# display(device_counts_combined)
































# Step 4: Create a pivot table of Female and Male row in the Gender column with devices column and number _of_devices columns as rows and columns
print('--------------------Female and Male with the number of devices---------------------')
pivot_table_gender = df[df['Gender'] != 'Fill in the blank'].pivot_table(index='Gender', columns='number _of_devices', aggfunc='size', fill_value=0)
display(pivot_table_gender)


print('Gender with the number of devices in percentage')
pivot_table_gender = pd.pivot_table(df[df['Gender'] != 'Fill in the blank'], index='Gender', columns='number _of_devices', aggfunc='size', fill_value=0)
total_devices = pivot_table_gender.sum().sum()
pivot_table_gender_percentage = pivot_table_gender.divide(total_devices, axis=0).mul(100).round(1).astype(str) + '%'
display(pivot_table_gender_percentage)


print('--------Gender with the number of devices (each column should add up to 100%)------------------')
# Create a pivot table of the number of devices and the percentage (each column adds up to 100%)
pivot_table_gender_column_percent = pivot_table_gender.div(pivot_table_gender.sum(axis=0), axis=1) * 100
pivot_table_gender_column_percent_numeric = pivot_table_gender_column_percent.round(1)  # Numeric version for plotting
pivot_table_gender_column_percent = pivot_table_gender_column_percent_numeric.astype(str) + '%'
display(pivot_table_gender_column_percent_numeric)



print('-----------------Gender with the number of devices (each row should add up to  100%)------------------')
# Create a pivot table of the number of devices and the percentage (each row adds up to 100%)
pivot_table_gender_row_percent = pivot_table_gender.div(pivot_table_gender.sum(axis=1), axis=0) * 100
pivot_table_gender_row_percent_numeric = pivot_table_gender_row_percent.round(1)  # Numeric version for plotting
pivot_table_gender_row_percent = pivot_table_gender_row_percent_numeric.astype(str) + '%'
display(pivot_table_gender_row_percent)

# Create a bar graph of the pivot table of the number of devices and the percentage
colors = plt.cm.tab20(range(len(pivot_table_gender_row_percent_numeric.columns)))
ax = pivot_table_gender_row_percent_numeric.T.plot(kind='bar', width=0.8, color=colors)
plt.xlabel('Number of Devices')
plt.ylabel('Percentage')
plt.title('Percentage of Devices by Gender')
plt.xticks(rotation=0, ha='center')
# Add data labels to the bars
for p in ax.patches:
    ax.annotate('{:.1f}%'.format(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()), 
                ha='center', va='bottom', fontsize=7, color='black')
plt.legend(title='Gender', loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
plt.tight_layout()
plt.show()


print('-----end of Gender with the number of devices------\n\n')

print('done\n')



# Create a new dataframe that contains rows with Male and Female in the Gender column and the like_to_hike_alone column
print('\n------------Male and Female with like_to_hike_alone----------------')
df_gender_like_to_hike_alone = df[df['Gender'].isin(['Male', 'Female'])][['Gender', 'like_to_hike_alone']]
display(df_gender_like_to_hike_alone.head())

# Create a pivot table of the like_to_hike_alone column with
df_gender_all_columns = df[df['Gender'].isin(['Male', 'Female'])]
#display(df_gender_all_columns)


# Create a pivot table of the like_to_hike_alone column with Gender as rows and count as columns
print('\nlike_to_hike_alone')
pivot_table_like_to_hike_alone = df_gender_all_columns.pivot_table(index='Gender', columns='like_to_hike_alone', aggfunc='size', fill_value=0)
# Reorder the columns of the pivot table
pivot_table_like_to_hike_alone = pivot_table_like_to_hike_alone[['Strongly disagree', 'Somewhat disagree', 'Neither agree nor disagree', 'Somewhat agree', 'Strongly agree']]
display(pivot_table_like_to_hike_alone)
#calculate the total count
#total_count = pivot_table_like_to_hike_alone.sum().sum()
#calculate the percentage of each category
pivot_table_like_to_hike_alone_percentage = pivot_table_like_to_hike_alone.div(pivot_table_like_to_hike_alone.sum(axis=1), axis=0) * 100
# Create a bar graph for the pivot table like to hike alone
pivot_table_like_to_hike_alone_percentage.plot(kind='bar')
plt.xlabel('')
plt.ylabel('Percentage')
plt.title('Percentage of People by Like to Hike Alone')
legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
# Add data labels to the bars
# Add data labels to the bars
for p in plt.gca().patches:
    plt.text(p.get_x() * 1.005, p.get_height() * 1.005, '{:.1f}%'.format(p.get_height()), fontsize=8, color='black')
plt.show()


# Create a pivot table of the like_to_hike_in_group column with Gender as rows and count as columns
print('\n \nlike_to_hike_in_group')
pivot_table_like_to_hike_in_group = df_gender_all_columns.pivot_table(index='Gender', columns='like_to_hike_in_group', aggfunc='size', fill_value=0)
# Reorder the columns of the pivot table
pivot_table_like_to_hike_in_group = pivot_table_like_to_hike_in_group[['Strongly disagree', 'Somewhat disagree', 'Neither agree nor disagree', 'Somewhat agree', 'Strongly agree']]
display(pivot_table_like_to_hike_in_group)
# Create a bar graph for the pivot table like to hike in group
pivot_table_like_to_hike_in_group.plot(kind='bar')
plt.xlabel('Like to Hike in Group')
plt.ylabel('Count')
plt.title('Number of People by Like to Hike in Group')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
# Add data labels to the bars
ax = plt.gca()
for p in ax.patches:
    percentage = f"{p.get_height() / pivot_table_like_to_hike_in_group.sum().sum() * 100:.1f}%"
    ax.annotate(percentage, (p.get_x() + p.get_width() / 2, p.get_height() + 0.3), ha='center', va='bottom')

plt.show()


# Create a pivot table of the like_to_hike_near_home column with Gender as rows and count as columns
print('\nLike_to_hike_near_home')
pivot_table_like_to_hike_near_home = df_gender_all_columns.pivot_table(index='Gender', columns='like_to_hike_near_home', aggfunc='size', fill_value=0)
# Reorder the columns of the pivot table
pivot_table_like_to_hike_near_home = pivot_table_like_to_hike_near_home[['Strongly disagree', 'Somewhat disagree', 'Neither agree nor disagree', 'Somewhat agree', 'Strongly agree']]
display(pivot_table_like_to_hike_near_home)
# Create a bar graph for the pivot table like to hike near home
pivot_table_like_to_hike_near_home.plot(kind='bar')
plt.xlabel('Like to Hike Near Home')
plt.ylabel('Count')
plt.title('Number of People by Like to Hike Near Home')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
# Add data labels to the bars
ax = plt.gca()
for p in ax.patches:
    percentage = f"{p.get_height() / pivot_table_like_to_hike_near_home.sum().sum() * 100:.1f}%"
    ax.annotate(percentage, (p.get_x() + p.get_width() / 2, p.get_height() + 0.3), ha='center', va='bottom')
plt.show()


# Create a pivot table of the hike_while_traveling column with Gender as rows and count as columns
print('\n---------------Hike_while_traveling----------------')
pivot_table_hike_while_traveling = df_gender_all_columns.pivot_table(index='Gender', columns='hike_while_traveling', aggfunc='size', fill_value=0)
# Reorder the columns of the pivot table
pivot_table_hike_while_traveling = pivot_table_hike_while_traveling[['Strongly disagree', 'Somewhat disagree', 'Neither agree nor disagree', 'Somewhat agree', 'Strongly agree']]
display(pivot_table_hike_while_traveling)
#calculate the total count
#total_count = pivot_table_hike_while_traveling.sum().sum()
#calculate the percentage of each category
pivot_table_hike_while_traveling_percentage = pivot_table_hike_while_traveling.div(pivot_table_hike_while_traveling.sum(axis=1), axis=0) * 100
# Create a bar graph for the pivot table hike while traveling
pivot_table_hike_while_traveling_percentage.plot(kind='bar')
plt.xlabel('')
plt.ylabel('Percentage')
plt.title('Percentage of People by Hike While Traveling')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
# Add data labels to the bars
for p in plt.gca().patches:
    plt.text(p.get_x() * 1.005, p.get_height() * 1.005, '{:.1f}%'.format(p.get_height()), fontsize=8, color='black')
plt.show()


# Create a pivot table of the hike_for_health column with Gender as rows and count as columns
print('\nHike_for_health')
pivot_table_hike_for_health = df_gender_all_columns.pivot_table(index='Gender', columns='hike_for_health', aggfunc='size', fill_value=0)
# Reorder the columns of the pivot table
pivot_table_hike_for_health = pivot_table_hike_for_health[['Strongly disagree', 'Somewhat disagree', 'Neither agree nor disagree', 'Somewhat agree', 'Strongly agree']]
display(pivot_table_hike_for_health)
# Create a bar graph for the pivot table hike for health
pivot_table_hike_for_health.plot(kind='bar')
plt.xlabel('Hike for Health')
plt.ylabel('Count')
plt.title('Number of People by Hike for Health')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
# Add data labels to the bars
ax = plt.gca()
for p in ax.patches:
    percentage = f"{p.get_height() / pivot_table_hike_for_health.sum().sum() * 100:.1f}%"
    ax.annotate(percentage, (p.get_x() + p.get_width() / 2, p.get_height() + 0.3), ha='center', va='bottom')
plt.show()

#------------------------------------------------#

# Define a mapping from Gender column to one unique general column
mapping = {
    'Male': 'Gender',  # Gender for male
    'Female': 'Gender',  # Gender for female
    'Fill in the blank' : 'Gender', # Gender for fill in the blank
}
print ("\nreplacement of Gender to  General column...")
df['General'] = df['Gender'].replace(mapping)
df['General'] = df['General'].astype(str)

# Create a new dataframe that contains General column and all the columns that contain the word hike in the column name
df_general_hike = df[['General'] + [col for col in df.columns if 'hike' in col]]

# Create a dataframe for Gender row in General column
df_gender_general = df_general_hike

# Create a pivot table of the hike_for_health column with count as columns
print('\nHike_for_health "General"')
pivot_table_hike_for_health = df_gender_general['hike_for_health'].value_counts().to_frame('').T

# Define the desired column order
column_order = ['Strongly disagree', 'Somewhat disagree', 'Neither agree nor disagree', 'Somewhat agree', 'Strongly agree']

# Reindex the pivot table with the desired column order
pivot_table_hike_for_health = pivot_table_hike_for_health.reindex(columns=column_order, fill_value=0)
display(pivot_table_hike_for_health)

# Create a bar graph for the pivot table hike for health
pivot_table_hike_for_health.plot(kind='bar')
plt.xlabel('Level of agreeance')
plt.ylabel('Count')
plt.title('Preferences in Hiking for Health')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
# Add data labels to the bars
ax = plt.gca()
for p in ax.patches:
    percentage = f"{p.get_height() / pivot_table_hike_for_health.sum().sum() * 100:.1f}%"
    ax.annotate(percentage, (p.get_x() + p.get_width() / 2, p.get_height() + 0.3), ha='center', va='bottom')
plt.show()
#------------------------------------------------#
# Create a new column 'Smart Watch Found' based on the values in the 'combined' column
df['Smart Watch Found'] = df['combined'].str.contains('smart watch', case=False)
df['Smart Watch Found'] = df['Smart Watch Found'].astype(object)  # <-- Fix: cast to object before assigning strings

# Replace True with 'Smart watch' in the 'Smart Watch Found' column
df.loc[df['Smart Watch Found'] == True, 'Smart Watch Found'] = 'Smart watch'

# Replace False with 'Not smart watch' in the 'Smart Watch Found' column
df.loc[df['Smart Watch Found'] == False, 'Smart Watch Found'] = 'Not smart watch'

# Create a new dataframe 'watch' that contains rows with 'smart watch' in the 'combined' column
watch = df[df['combined'].str.contains('smart watch', case=False)]
#display(watch)

# Display the updated dataframe
display(df['Smart Watch Found'].head())





#------------------------------------------------#
# Define a mapping from Gender column to one unique general column
mapping = {
    'Male': 'Gender',  # Gender for male
    'Female': 'Gender',  # Gender for female
}
print ("\nreplacement of Gender to  General ciolumn...")
df['General'] = df['Gender'].replace(mapping)
df['General'] = df['General'].astype(str)
print ("Done\n")
# Exclude the rows with 'Fill in the blank' in the Gender column
df = df[df['Gender'] != 'Fill in the blank']

# Create a pivot table of the hike_for_fun column with Gender as rows and count as columns
print('\nHike_for_fun')
pivot_table_hike_for_fun = df_gender_all_columns.pivot_table(index='Gender', columns= 'hike_for_fun', aggfunc='size', fill_value=0)
# Reorder the columns of the pivot table
pivot_table_hike_for_fun = pivot_table_hike_for_fun[['Strongly disagree', 'Somewhat disagree', 'Neither agree nor disagree', 'Somewhat agree', 'Strongly agree']]
display(pivot_table_hike_for_fun)
# Create a bar graph for the pivot table hike for fun
pivot_table_hike_for_fun.plot(kind='bar')
plt.xlabel('Hike for Fun')
plt.ylabel('Count')
plt.title('Number of People by Hike for Fun')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
# Add data labels to the bars
ax = plt.gca()
for p in ax.patches:
    percentage = f"{p.get_height() / pivot_table_hike_for_fun.sum().sum() * 100:.1f}%"
    ax.annotate(percentage, (p.get_x() + p.get_width() / 2, p.get_height() + 0.3), ha='center', va='bottom')
plt.show()

#-----------General (Both men and women)----------------------#
# # Create a pivot table of the hike_for_fun column with Gender as rows and count as columns
# print('\nHike_for_fun "General"')
# pivot_table_hike_for_fun = df_gender_general.pivot_table(index='General', columns='hike_for_fun', aggfunc='size', fill_value=0)
# # Reorder the columns of the pivot table
# # Define the desired column order
# column_order = ['Strongly disagree', 'Somewhat disagree', 'Neither agree nor disagree', 'Somewhat agree', 'Strongly agree']

# # Reindex the pivot table with the desired column order
# pivot_table_hike_for_fun = pivot_table_hike_for_fun.reindex(columns=column_order, fill_value=0)
# display(pivot_table_hike_for_fun)
# # Create a bar graph for the pivot table hike for fun
# pivot_table_hike_for_fun.plot(kind='bar')
# plt.xlabel('Hike for fun')
# plt.ylabel('Count')
# plt.title('Number of People by Hike for Fun')
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
# # Add data labels to the bars
# ax = plt.gca()
# for p in ax.patches:
#     percentage = f"{p.get_height() / pivot_table_hike_for_fun.sum().sum() * 100:.1f}%"
#     ax.annotate(percentage, (p.get_x() + p.get_width() / 2, p.get_height() + 0.3), ha='center', va='bottom')
# plt.show()

#------------------------------------------------#


# Create a pivot table of the hike_for_social_interaction with Gender as rows and count as columns
print('\nHike_for_social_interaction')
pivot_table_hike_for_social_interaction = df_gender_all_columns.pivot_table(index='Gender', columns='hike_for_social_interaction', aggfunc='size', fill_value=0)
# Reorder the columns of the pivot table
pivot_table_hike_for_social_interaction = pivot_table_hike_for_social_interaction[['Strongly disagree', 'Somewhat disagree', 'Neither agree nor disagree', 'Somewhat agree', 'Strongly agree']]
display(pivot_table_hike_for_social_interaction)
# Create a bar graph for the pivot table hike for social interaction
pivot_table_hike_for_social_interaction.plot(kind='bar')
plt.xlabel('Hike for Social Interaction')
plt.ylabel('Count')
plt.title('Number of People by Hike for Social Interaction')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
# Add data labels to the bars
ax = plt.gca()
for p in ax.patches:
    percentage = f"{p.get_height() / pivot_table_hike_for_social_interaction.sum().sum() * 100:.1f}%"
    ax.annotate(percentage, (p.get_x() + p.get_width() / 2, p.get_height() + 0.3), ha='center', va='bottom')
plt.show()



# Create a pivot table of the hike_for_mediation column with Gender as rows and count as columns
print('\nHike_for_mediation')
pivot_table_hike_for_mediation = df_gender_all_columns.pivot_table(index='Gender', columns='hike_for_mediation', aggfunc='size', fill_value=0)
# Reorder the columns of the pivot table
pivot_table_hike_for_mediation = pivot_table_hike_for_mediation[['Strongly disagree', 'Somewhat disagree', 'Neither agree nor disagree', 'Somewhat agree', 'Strongly agree']]
display(pivot_table_hike_for_mediation)
# Create a bar graph for the pivot table hike for mediation
pivot_table_hike_for_mediation.plot(kind='bar')
plt.xlabel('Hike for Mediation')
plt.ylabel('Count')
plt.title('Number of People by Hike for Mediation')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
# Add data labels to the bars
ax = plt.gca()
for p in ax.patches:
    percentage = f"{p.get_height() / pivot_table_hike_for_mediation.sum().sum() * 100:.1f}%"
    ax.annotate(percentage, (p.get_x() + p.get_width() / 2, p.get_height() + 0.3), ha='center', va='bottom')
plt.show()


# Create a pivot table of the hike_for_less_than_1hour column with Gender as rows and count as columns
print('\n--------------Hike_for_less_than_1hour----------------')
pivot_table_hike_for_less_than_1hour = df_gender_all_columns.pivot_table(index='Gender', columns='hike_for_less_than_1hour', aggfunc='size', fill_value=0)
# Reorder the columns of the pivot table
pivot_table_hike_for_less_than_1hour = pivot_table_hike_for_less_than_1hour[['Strongly disagree', 'Somewhat disagree', 'Neither agree nor disagree', 'Somewhat agree', 'Strongly agree']]
display(pivot_table_hike_for_less_than_1hour)
#calculate the total count
#total_count = pivot_table_hike_for_less_than_1hour.sum().sum()
#calculate the percentage of each category
pivot_table_hike_for_less_than_1hour_percentage = pivot_table_hike_for_less_than_1hour.div(pivot_table_hike_for_less_than_1hour.sum(axis=1), axis=0) * 100
# Create a bar graph for the pivot table hike for less than 1 hour
pivot_table_hike_for_less_than_1hour_percentage.plot(kind='bar')
plt.xlabel('')
plt.ylabel('Percentage')
plt.title('Percentage of People by Hike for Less than 1 Hour')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2) 
# Add data labels to the bars
for p in plt.gca().patches:
    plt.text(p.get_x() * 1.005, p.get_height() * 1.005, '{:.1f}%'.format(p.get_height()), fontsize=8, color='black')
plt.show()


# Create a pivot table of the hike_for_half_a_day column with Gender as rows and count as columns
print('\nHike_for_half_a_day')
pivot_table_hike_for_half_a_day = df_gender_all_columns.pivot_table(index='Gender', columns='hike_for_half_a_day', aggfunc='size', fill_value=0)
# Reorder the columns of the pivot table
pivot_table_hike_for_half_a_day = pivot_table_hike_for_half_a_day[['Strongly disagree', 'Somewhat disagree', 'Neither agree nor disagree', 'Somewhat agree', 'Strongly agree']]
display(pivot_table_hike_for_half_a_day)
# Create a bar graph for the pivot table hike for half a day
pivot_table_hike_for_half_a_day.plot(kind='bar')
plt.xlabel('Hike for Half a Day')
plt.ylabel('Count')
plt.title('Number of People by Hike for Half a Day')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
# Add data labels to the bars
ax = plt.gca()
for p in ax.patches:
    percentage = f"{p.get_height() / pivot_table_hike_for_half_a_day.sum().sum() * 100:.1f}%"
    ax.annotate(percentage, (p.get_x() + p.get_width() / 2, p.get_height() + 0.3), ha='center', va='bottom')
plt.show()


# Create a pivot table of the hike_for_less_than_a_day column with Gender as rows and count as columns
print('\nHike_for_less_than_a_day')
pivot_table_hike_for_less_than_a_day = df_gender_all_columns.pivot_table(index='Gender', columns='hike_for_less_than_a_day', aggfunc='size', fill_value=0)
# Reorder the columns of the pivot table
pivot_table_hike_for_less_than_a_day = pivot_table_hike_for_less_than_a_day[['Strongly disagree', 'Somewhat disagree', 'Neither agree nor disagree', 'Somewhat agree', 'Strongly agree']]
display(pivot_table_hike_for_less_than_a_day)
# Create a bar graph for the pivot table hike for less than a day
pivot_table_hike_for_less_than_a_day.plot(kind='bar')
plt.xlabel('Hike for Less than a Day')
plt.ylabel('Count')
plt.title('Number of People by Hike for Less than a Day')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
# Add data labels to the bars
ax = plt.gca()
for p in ax.patches:
    percentage = f"{p.get_height() / pivot_table_hike_for_less_than_a_day.sum().sum() * 100:.1f}%"
    ax.annotate(percentage, (p.get_x() + p.get_width() / 2, p.get_height() + 0.3), ha='center', va='bottom')
plt.show()


# Create a pivot table of the hike_for_multiple_days column with Gender as rows and count as columns
print('\nHike_for_multiple_days')
pivot_table_hike_for_multiple_days = df_gender_all_columns.pivot_table(index='Gender', columns='hike_for_multiple_days', aggfunc='size', fill_value=0)
# Reorder the columns of the pivot table
pivot_table_hike_for_multiple_days = pivot_table_hike_for_multiple_days[['Strongly disagree', 'Somewhat disagree', 'Neither agree nor disagree', 'Somewhat agree', 'Strongly agree']]
display(pivot_table_hike_for_multiple_days)
# Create a bar graph for the pivot table hike for multiple days
pivot_table_hike_for_multiple_days.plot(kind='bar')
plt.xlabel('Hike for Multiple Days')
plt.ylabel('Count')
plt.title('Number of People by Hike for Multiple Days')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
# Add data labels to the bars
ax = plt.gca()
for p in ax.patches:
    percentage = f"{p.get_height() / pivot_table_hike_for_multiple_days.sum().sum() * 100:.1f}%"
    ax.annotate(percentage, (p.get_x() + p.get_width() / 2, p.get_height() + 0.3), ha='center', va='bottom')
plt.show()


# Create a pivot table of the easy_hike column with Gender as rows and count as columns
print('\n--------------Easy_hike----------------')
pivot_table_easy_hike = df_gender_all_columns.pivot_table(index='Gender', columns='easy_hike', aggfunc='size', fill_value=0)
# Reorder the columns of the pivot table
pivot_table_easy_hike = pivot_table_easy_hike[['Strongly disagree', 'Somewhat disagree', 'Neither agree nor disagree', 'Somewhat agree', 'Strongly agree']]
display(pivot_table_easy_hike)
#calculate the total count
#total_count = pivot_table_easy_hike.sum().sum()
#calculate the percentage of each category
pivot_table_easy_hike_percentage = pivot_table_easy_hike.div(pivot_table_easy_hike.sum(axis=1), axis=0) * 100
# Create a bar graph for the pivot table easy hike
pivot_table_easy_hike_percentage.plot(kind='bar')
plt.xlabel('')
plt.ylabel('Percentage')
plt.title('Percentage of People by Easy Hike')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
# Add data labels to the bars
for p in plt.gca().patches:
    plt.text(p.get_x() * 1.005, p.get_height() * 1.005, '{:.1f}%'.format(p.get_height()), fontsize=8, color='black')
plt.show()


# Create a pivot table of the difficult_hike column with Gender as rows and count as columns
print('\n-----------------Difficult_hike----------------')
pivot_table_difficult_hike = df_gender_all_columns.pivot_table(index='Gender', columns='difficult_hike', aggfunc='size', fill_value=0)
# Reorder the columns of the pivot table
pivot_table_difficult_hike = pivot_table_difficult_hike[['Strongly disagree', 'Somewhat disagree', 'Neither agree nor disagree', 'Somewhat agree', 'Strongly agree']]
display(pivot_table_difficult_hike)
#calculate the total count
#total_count = pivot_table_difficult_hike.sum().sum()
#calculate the percentage of each category
pivot_table_difficult_hike_percentage = pivot_table_difficult_hike.div(pivot_table_difficult_hike.sum(axis=1), axis=0) * 100
# Create a bar graph for the pivot table difficult hike
pivot_table_difficult_hike_percentage.plot(kind='bar')
plt.xlabel('')
plt.ylabel('Percentage')
plt.title('Percentage of People by Difficult Hike')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
# Add data labels to the bars
for p in plt.gca().patches:
    plt.text(p.get_x() * 1.005, p.get_height() * 1.005, '{:.1f}%'.format(p.get_height()), fontsize=8, color='black')
plt.show()





#ttest




# Define a mapping from categories to numerical values of like to hike alone column
mapping = {
    'Strongly disagree': 1,  # 1 for strongly disagree
    'Somewhat disagree': 2,  # 2 for somewhat disagree
    'Neither agree nor disagree': 3, # 3 for neither agree nor disagree
    'Somewhat agree': 4, # 4 for somewhat agree
    'Strongly agree': 5, # 5 for strongly agree
}

print ("\nreplacement of like to hike alone to numeric value...")
df['like_to_hike_alone'] = df['like_to_hike_alone'].replace({'Fill in the blank': np.nan})
# Replace non-finite values with a placeholder (e.g., 0)
df['like_to_hike_alone'] = df['like_to_hike_alone'].replace({'Fill in the blank': np.nan})
# Step 1: Print the non-finite values
print("Non-finite values in 'like_to_hike_alone':")
print(df['like_to_hike_alone'][~np.isfinite(pd.to_numeric(df['like_to_hike_alone'], errors='coerce'))])


df['like_to_hike_alone_value'] = df['like_to_hike_alone'].replace(mapping)
# Convert to numeric, forcing errors to NaN
df['like_to_hike_alone_value'] = pd.to_numeric(df['like_to_hike_alone_value'], errors='coerce')
# Step 1: Print the non-finite values
print("Non-finite values in 'like_to_hike_alone_value':")
print(df['like_to_hike_alone_value'][~np.isfinite(df['like_to_hike_alone_value'])])

# Step 2: Replace non-finite values with a placeholder (e.g., 0)
df['like_to_hike_alone_value'] = df['like_to_hike_alone_value'].replace([np.inf, -np.inf, np.nan], 0)

df['like_to_hike_alone_value'] = df['like_to_hike_alone_value'].astype(int)
print ("Done")
#exclude the rows with fill in the blank in Gender column
df_valid_gender = df[df['Gender'] == 'Fill in the blank']
# create a pivot table of the like_to_hike_alone_value
print('\nlike_to_hike_alone_value')
pivot_table_like_to_hike_alone_value = df_valid_gender.pivot_table(index='Gender', columns='like_to_hike_alone_value', aggfunc='size', fill_value=0)
print(pivot_table_like_to_hike_alone_value)

# Filter the data to only include male and female rows with a value in the like_to_hike_alone_value column
df_gender_like_to_hike_alone_value = df[df['Gender'].isin(['Male', 'Female']) & df['like_to_hike_alone_value'].notnull()][['Gender', 'like_to_hike_alone_value']]

# Perform the t-test
t_stat, p_value = ttest_ind(
    df_gender_like_to_hike_alone_value[df_gender_like_to_hike_alone_value['Gender'] == 'Male']['like_to_hike_alone_value'],
    df_gender_like_to_hike_alone_value[df_gender_like_to_hike_alone_value['Gender'] == 'Female']['like_to_hike_alone_value'],
    equal_var=False
)

# Print the t-statistic and p-value
print(f"t-statistic: {t_stat:.2f}")
print(f"p-value: {p_value:.4f}")
print('\n')

# Set significance level
alpha = 0.05

# Determine if the result is statistically significant
is_significant = p_value < alpha
print(f"Is it statistically significant at the {alpha} level? {is_significant}")
print('\n')

# Determine if the result is statistically significant
if p_value < alpha:
    print("**Statistically significant difference** between male and female participants' like_to_hike_alone_value values.")
    print(f"The mean like_to_hike_alone_value for male participants is {df_gender_like_to_hike_alone_value[df_gender_like_to_hike_alone_value['Gender'] == 'Male']['like_to_hike_alone_value'].mean():.2f}.")
    print(f"The mean like_to_hike_alone_value for female participants is {df_gender_like_to_hike_alone_value[df_gender_like_to_hike_alone_value['Gender'] == 'Female']['like_to_hike_alone_value'].mean():.2f}.")
else:
    print("No statistically significant difference between male and female participants' like_to_hike_alone_value values.")
print('\n')
# Group by Gender and calculate summary statistics
summary_stats = df_valid_gender.groupby('Gender')['like_to_hike_alone_value'].describe()
display(summary_stats)
# Create a bar graph of the like_to_hike_alone_value column for male and female participants
df_gender_like_to_hike_alone_value.boxplot(column='like_to_hike_alone_value', by='Gender')
plt.xlabel('Gender')
plt.ylabel('Like to Hike Alone (numeric)')
plt.title('Like to Hike Alone by Gender')
plt.show()
print('done\n')



# Define a mapping from categories to numerical values of like to hike in group column 
mapping = {
    'Strongly disagree': 1,  # 1 for strongly disagree
    'Somewhat disagree': 2,  # 2 for somewhat disagree
    'Neither agree nor disagree': 3, # 3 for neither agree nor disagree
    'Somewhat agree': 4, # 4 for somewhat agree
    'Strongly agree': 5, # 5 for strongly agree
}
print ("\nreplacement of like to hike in group to numeric value...")
df['like_to_hike_in_group_num'] = df['like_to_hike_in_group'].replace(mapping).infer_objects(copy=False)
df['like_to_hike_in_group_num'] = df['like_to_hike_in_group_num'].astype(int)
print ("Done\n")
#exclude the rows with fill in the blank
df_valid_gender = df[df['Gender'] != 'Fill in the blank']
# create a pivot table of the like_to_hike_in_group_num
print('\nlike_to_hike_in_group_num table\n')
pivot_table_like_to_hike_in_group_num = df_valid_gender.pivot_table(index='Gender', columns='like_to_hike_in_group_num', aggfunc='size', fill_value=0)
print(pivot_table_like_to_hike_in_group_num)

# Filter the data to only include male and female rows with a value in the like_to_hike_in_group_num column
df_gender_like_to_hike_in_group_num = df[df['Gender'].isin(['Male', 'Female']) & df['like_to_hike_in_group_num'].notnull()][['Gender', 'like_to_hike_in_group_num']]

# Perform the t-test
t_stat, p_value = ttest_ind(df_gender_like_to_hike_in_group_num[df_gender_like_to_hike_in_group_num['Gender'] == 'Male']['like_to_hike_in_group_num'],
                            df_gender_like_to_hike_in_group_num[df_gender_like_to_hike_in_group_num['Gender'] == 'Female']['like_to_hike_in_group_num'],
                            equal_var=False)

# Print the t-statistic and p-value
print(f"t-statistic: {t_stat:.2f}")
print(f"p-value: {p_value:.4f}")
print('\n')
# Set significance level
alpha = 0.05

# Determine if the result is statistically significant
is_significant = p_value < alpha
print(f"Is it statistically significant at the {alpha} level? {is_significant}")
print('\n')

# Determine if the result is statistically significant
if p_value < alpha:
    print("**Statistically significant difference** between male and female participants' like to hike in group.")
    print(f"The mean for male participants is {df_gender_like_to_hike_in_group_num[df_gender_like_to_hike_in_group_num['Gender'] == 'Male']['like_to_hike_in_group_numm'].mean():.2f}.")
    print(f"The mean for female participants is {df_gender_like_to_hike_in_group_num[df_gender_like_to_hike_in_group_num['Gender'] == 'Female']['like_to_hike_in_group_num'].mean():.2f}.")
else:
    print("No statistically significant difference between male and female participants' like to hike in group.")
print('\n')
# Group by Gender and calculate summary statistics
summary_stats = df_valid_gender.groupby('Gender')['like_to_hike_in_group_num'].describe()
display(summary_stats)
# Create a bar graph of the like_to_hike_in_group_num column for male and female participants
df_gender_like_to_hike_in_group_num.boxplot(column='like_to_hike_in_group_num', by='Gender')
plt.xlabel('Gender')
plt.ylabel('Like to Hike in Group (numeric)')
plt.title('Like to Hike in Group by Gender')
plt.show()
print('done\n')




# Define a mapping from categories to numerical values of like to hike near home column
mapping = {
    'Strongly disagree': 1,  # 1 for strongly disagree
    'Somewhat disagree': 2,  # 2 for somewhat disagree
    'Neither agree nor disagree': 3, # 3 for neither agree nor disagree
    'Somewhat agree': 4, # 4 for somewhat agree
    'Strongly agree': 5, # 5 for strongly agree
}
print ("\nreplacement of like to hike near home to numeric value...")
df['like_to_hike_near_home_num'] = df['like_to_hike_near_home'].replace(mapping).infer_objects(copy=False)
df['like_to_hike_near_home_num'] = df['like_to_hike_near_home_num'].astype(int)
#print (df['like_to_hike_near_home_num'].head(5))
print ("Done\n")
#exclude the rows with fill in the blank
df_valid_gender = df[df['Gender'] != 'Fill in the blank']
# create a pivot table of the like_to_hike_near_home_num
print('\nlike_to_hike_near_home_num table\n') 
pivot_table_like_to_hike_near_home_num = df_valid_gender.pivot_table(index='Gender', columns='like_to_hike_near_home_num', aggfunc='size', fill_value=0)
print(pivot_table_like_to_hike_near_home_num)
print('Done\n')

#  male and female rows with a value in the like_to_hike_near_home_num column
df_gender_like_to_hike_near_home_num = df[df['Gender'].isin(['Male', 'Female']) & df['like_to_hike_near_home_num'].notnull()][['Gender', 'like_to_hike_near_home_num']]

# Perform the t-test
t_stat, p_value = ttest_ind(df_gender_like_to_hike_near_home_num[df_gender_like_to_hike_near_home_num['Gender'] == 'Male']['like_to_hike_near_home_num'], 
                            df_gender_like_to_hike_near_home_num[df_gender_like_to_hike_near_home_num['Gender'] == 'Female']['like_to_hike_near_home_num'],
                            equal_var=False)

# Print the t-statistic and p-value
print(f"t-statistic: {t_stat:.2f}")
print(f"p-value: {p_value:.4f}")
print('\n')

# Set significance level
alpha = 0.05

# Determine if the result is statistically significant
is_significant = p_value < alpha
print(f"Is it statistically significant at the {alpha} level? {is_significant}")
print('\n')

# Determine if the result is statistically significant
if p_value < alpha:
    print("**Statistically significant difference** between male and female participants' like to hike near home.")
    print(f"The mean for male participants is {df_gender_like_to_hike_near_home_num[df_gender_like_to_hike_near_home_num['Gender'] == 'Male']['like_to_hike_near_home_num'].mean():.2f}.")
    print(f"The mean for female participants is {df_gender_like_to_hike_near_home_num[df_gender_like_to_hike_near_home_num['Gender'] == 'Female']['like_to_hike_near_home_num'].mean():.2f}.")
else:
    print("No statistically significant difference between male and female participants' like to hike near home.")
print('\n')
# Group by Gender and calculate summary statistics
summary_stats = df_valid_gender.groupby('Gender')['like_to_hike_near_home_num'].describe()
display(summary_stats)
# Create a bar graph of the like_to_hike_near_home_num column
df_gender_like_to_hike_near_home_num.boxplot(column='like_to_hike_near_home_num', by='Gender')
plt.xlabel('Gender')
plt.ylabel('Like to Hike Near Home (numeric)')
plt.title('Like to Hike Near Home by Gender')
plt.show()
print('done\n')
                            



# Define a mapping from categories to numerical values of hike while traveling column
mapping = {
    'Strongly disagree': 1,  # 1 for strongly disagree
    'Somewhat disagree': 2,  # 2 for somewhat disagree
    'Neither agree nor disagree': 3, # 3 for neither agree nor disagree
    'Somewhat agree': 4, # 4 for somewhat agree
    'Strongly agree': 5, # 5 for strongly agree
}
print ("\nreplacement of hike while traveling to numeric value...")
df['hike_while_traveling_num'] = df['hike_while_traveling'].replace(mapping).infer_objects(copy=False)
df['hike_while_traveling_num'] = df['hike_while_traveling_num'].astype(int)
print ("Done\n")
#exclude the rows with fill in the blank
df_valid_gender = df[df['Gender'] != 'Fill in the blank']
# create a pivot table of the hike_while_traveling_num
print('\nhike_while_traveling_num table\n')
pivot_table_hike_while_traveling_num = df_valid_gender.pivot_table(index='Gender', columns='hike_while_traveling_num', aggfunc='size', fill_value=0)
print(pivot_table_hike_while_traveling_num)
print('Done\n')


# Filter the data to only include male and female rows with a value in the hike_while_traveling_num column
df_gender_hike_while_traveling_num = df[df['Gender'].isin(['Male', 'Female']) & df['hike_while_traveling_num'].notnull()][['Gender', 'hike_while_traveling_num']]

# Perform the t-test
t_stat, p_value = ttest_ind(df_gender_hike_while_traveling_num[df_gender_hike_while_traveling_num['Gender'] == 'Male']['hike_while_traveling_num'],
                            df_gender_hike_while_traveling_num[df_gender_hike_while_traveling_num['Gender'] == 'Female']['hike_while_traveling_num'],
                            equal_var=False)

# Print the t-statistic and p-value
print(f"t-statistic: {t_stat:.2f}")
print(f"p-value: {p_value:.4f}")
# Set significance level
alpha = 0.05

# Determine if the result is statistically significant
is_significant = p_value < alpha
print(f"Is it statistically significant at the {alpha} level? {is_significant}")
print('\n')

# Determine if the result is statistically significant
if p_value < alpha:
    print("**Statistically significant difference** between male and female participants' hike while traveling.")
    print(f"The mean for male participants is {df_gender_hike_while_traveling_num[df_gender_hike_while_traveling_num['Gender'] == 'Male']['hike_while_traveling_num'].mean():.2f}.")
    print(f"The mean for female participants is {df_gender_hike_while_traveling_num[df_gender_hike_while_traveling_num['Gender'] == 'Female']['hike_while_traveling_num'].mean():.2f}.")
else:
    print("No statistically significant difference between male and female participants' hike while traveling.")
print('\n')
# Group by Gender and calculate summary statistics
summary_stats = df_valid_gender.groupby('Gender')['hike_while_traveling_num'].describe()
display(summary_stats)

# Create a bar graph of the hike_while_traveling_num column
# Create a bar plot of the like_to_hike_while_traveling column
df_gender_hike_while_traveling_num.boxplot(column='hike_while_traveling_num', by='Gender')
plt.xlabel('Gender')
plt.ylabel('Like to Hike While Traveling (numeric)')
plt.title('Hike While Traveling')
plt.show()
print('done\n')


# Define a mapping from categories to numerical values of hike for health column
mapping = {
    'Strongly disagree': 1,  # 1 for strongly disagree
    'Somewhat disagree': 2,  # 2 for somewhat disagree
    'Neither agree nor disagree': 3, # 3 for neither agree nor disagree
    'Somewhat agree': 4, # 4 for somewhat agree
    'Strongly agree': 5, # 5 for strongly agree
}
print ("\nreplacement of hike for health to numeric value...")
df['hike_for_health_num'] = df['hike_for_health'].replace(mapping).infer_objects(copy=False)
# Step 1: Identify non-finite values (e.g., NaN, inf, -inf)
non_finite_mask = ~np.isfinite(df['hike_for_health_num'])

# Step 2: Fill non-finite values with a default value (e.g., 0) or remove them
df['hike_for_health_num'] = df['hike_for_health_num'].fillna(0)  # Filling NaN with 0
# Alternatively, you can remove rows with non-finite values
# df = df[~non_finite_mask]
df['hike_for_health_num'] = df['hike_for_health_num'].astype(int)
print ("Done\n")
#exclude the rows with fill in the blank
df_valid_gender = df[df['Gender'] != 'Fill in the blank']
# create a pivot table of the hike_for_health_num
print('\nhike_for_health_num table\n')
pivot_table_hike_for_health_num = df_valid_gender.pivot_table(index='Gender', columns='hike_for_health_num', aggfunc='size', fill_value=0)
print(pivot_table_hike_for_health_num)
print('Done\n')

# Set significance level
alpha = 0.05

# Filter the data to only include male and female rows with a value in the hike_for_health_num column
df_gender_hike_for_health_num = df[df['Gender'].isin(['Male', 'Female']) & df['hike_for_health_num'].notnull()][['Gender', 'hike_for_health_num']]

# Perform the t-test
t_stat, p_value = ttest_ind(df_gender_hike_for_health_num[df_gender_hike_for_health_num['Gender'] == 'Male']['hike_for_health_num'],
                            df_gender_hike_for_health_num[df_gender_hike_for_health_num['Gender'] == 'Female']['hike_for_health_num'],
                            equal_var=False)
# Print the t-statistic and p-value
print(f"t-statistic: {t_stat:.2f}")
print(f"p-value: {p_value:.4f}")
print('\n')

# Determine if the result is statistically significant
is_significant = p_value < alpha
print(f"Is it statistically significant at the {alpha} level? {is_significant}")
print('\n')

# Determine if the result is statistically significant
if p_value < alpha:
    print("**Statistically significant difference** between male and female participants' hike for health.")
    print(f"The mean for male participants is {df_gender_hike_for_health_num[df_gender_hike_for_health_num['Gender'] == 'Male']['hike_for_health_num'].mean():.2f}.")
    print(f"The mean for female participants is {df_gender_hike_for_health_num[df_gender_hike_for_health_num['Gender'] == 'Female']['hike_for_health_num'].mean():.2f}.")
else:
    print("No statistically significant difference between male and female participants' hike for health.")
print('\n')


# Group by Gender and calculate summary statistics
summary_stats = df_valid_gender.groupby('Gender')['hike_for_health_num'].describe()
display(summary_stats)
# Create a bar graph of the hike_for_health_num column
df_gender_hike_for_health_num.boxplot(column='hike_for_health_num', by='Gender')
plt.xlabel('Gender')
plt.ylabel('Hike for Health (numeric)')
plt.title('Hike for Health')
plt.show()
print('done\n')
























# Define a mapping from categories to numerical values of hike for social interaction column
mapping = {
    'Strongly disagree': 1,  # 1 for strongly disagree
    'Somewhat disagree': 2,  # 2 for somewhat disagree
    'Neither agree nor disagree': 3, # 3 for neither agree nor disagree
    'Somewhat agree': 4, # 4 for somewhat agree
    'Strongly agree': 5, # 5 for strongly agree
}
print ("\nreplacement of hike for social interaction to numeric value...")
df['hike_for_social_interaction_num'] = df['hike_for_social_interaction'].replace(mapping).infer_objects(copy=False)
# Step 1: Identify non-finite values (e.g., NaN, inf, -inf)
non_finite_mask = ~np.isfinite(df['hike_for_social_interaction_num'])

# Step 2: Fill non-finite values with a default value (e.g., 0) or remove them
df['hike_for_social_interaction_num'] = df['hike_for_social_interaction_num'].fillna(0)  # Filling NaN with 0
# Alternatively, you can remove rows with non-finite values
# df = df[~non_finite_mask]
df['hike_for_social_interaction_num'] = df['hike_for_social_interaction_num'].astype(int)
print ("Done\n")
#exclude the rows with fill in the blank
df_valid_gender = df[df['Gender'] != 'Fill in the blank']
# create a pivot table of the hike_for_social_interaction_num
print('\nhike_for_social_interaction_num table\n')  
pivot_table_hike_for_social_interaction = df_valid_gender.pivot_table(index='Gender', columns='hike_for_social_interaction_num', aggfunc='size', fill_value=0)
print(pivot_table_hike_for_social_interaction)
print('Done\n')

# Set significance level
alpha = 0.05

# Filter the data to only Male and Female rows with a value in the hike_for_social_interaction_num column
df_gender_hike_for_social_interaction_num = df[df['Gender'].isin(['Male', 'Female']) & df['hike_for_social_interaction_num'].notnull()][['Gender', 'hike_for_social_interaction_num']]
# Perform the t-test
t_stat, p_value = ttest_ind(df_gender_hike_for_social_interaction_num[df_gender_hike_for_social_interaction_num['Gender'] == 'Male']['hike_for_social_interaction_num'],
                            df_gender_hike_for_social_interaction_num[df_gender_hike_for_social_interaction_num['Gender'] == 'Female']['hike_for_social_interaction_num'],
                            equal_var=False)
# Print the t-statistic and p-value
print(f"t-statistic: {t_stat:.2f}")
print(f"p-value: {p_value:.4f}")
print('\n')

# Determine if the result is statistically significant
is_significant = p_value < alpha
print(f"Is it statistically significant at the {alpha} level? {is_significant}")
print('\n')

# Determine if the result is statistically significant
if p_value < alpha:
    print("**Statistically significant difference** between male and female participants' hike for social interaction.")
    print(f"The mean for male participants is {df_gender_hike_for_social_interaction_num[df_gender_hike_for_social_interaction_num['Gender'] == 'Male']['hike_for_social_interaction_num'].mean():.2f}.")
    print(f"The mean for female participants is {df_gender_hike_for_social_interaction_num[df_gender_hike_for_social_interaction_num['Gender'] == 'Female']['hike_for_social_interaction_num'].mean():.2f}.")
else:
    print("No statistically significant difference between male and female participants' hike for social interaction.")
print('\n')
# Group by Gender and calculate summary statistics
summary_stats = df_valid_gender.groupby('Gender')['hike_for_social_interaction_num'].describe()
display(summary_stats)
# Create a bar graph of the hike_for_social_interaction_num column
df_gender_hike_for_social_interaction_num.boxplot(column='hike_for_social_interaction_num', by='Gender')
plt.xlabel('Gender')
plt.ylabel('Hike for Social Interaction (numeric)')
plt.title('Hike for Social Interaction')
plt.show()
print('done\n')



# Define a mapping from categories to numerical values of hike for Fun column
mapping = {
    'Strongly disagree': 1,  # 1 for strongly disagree
    'Somewhat disagree': 2,  # 2 for somewhat disagree
    'Neither agree nor disagree': 3, # 3 for neither agree nor disagree
    'Somewhat agree': 4, # 4 for somewhat agree
    'Strongly agree': 5, # 5 for strongly agree
}
print ("\nreplacement of hike for fun to numeric value...")
df['hike_for_fun_num'] = df['hike_for_fun'].replace(mapping).infer_objects(copy=False)
# Step 1: Identify non-finite values (e.g., NaN, inf, -inf)
non_finite_mask = ~np.isfinite(df['hike_for_fun_num'])

# Step 2: Fill non-finite values with a default value (e.g., 0) or remove them
df['hike_for_fun_num'].fillna(0, inplace=True)  # Filling NaN with 0
# Alternatively, you can remove rows with non-finite values
# df = df[~non_finite_mask]
df['hike_for_fun_num'] = df['hike_for_fun_num'].astype(int)
print ("Done\n")
#exclude the rows with fill in the blank
df_valid_gender = df[df['Gender'] != 'Fill in the blank']
# create a pivot table of the hike_for_fun_num
print('\nhike_for_fun_num table\n')
pivot_table_hike_for_fun_num = df_valid_gender.pivot_table(index='Gender', columns='hike_for_fun_num', aggfunc='size', fill_value=0)
print(pivot_table_hike_for_fun_num, '\n')
print('Done\n')
# Set significance level
alpha = 0.05

# Filter the male and famale rows with a value in the hike_for_fun_num column
df_gender_hike_for_fun_num = df[df['Gender'].isin(['Male', 'Female']) & df['hike_for_fun_num'].notnull()][['Gender', 'hike_for_fun_num']]
# Perform the t-test
t_stat, p_value = ttest_ind(df_gender_hike_for_fun_num[df_gender_hike_for_fun_num['Gender'] == 'Male']['hike_for_fun_num'],
                            df_gender_hike_for_fun_num[df_gender_hike_for_fun_num['Gender'] == 'Female']['hike_for_fun_num'],
                            equal_var=False)
# Print the t-statistic and p-value
print(f"t-statistic: {t_stat:.2f}")
print(f"p-value: {p_value:.4f}")
print('\n')
# Determine if the result is statistically significant
is_significant = p_value < alpha
print(f"Is it statistically significant at the {alpha} level? {is_significant}")
print('\n')

# Determine if the result is statistically significant
if p_value < alpha:
    print("**Statistically significant difference** between male and female participants' hike for fun.")
    print(f"The mean for male participants is {df_gender_hike_for_fun_num[df_gender_hike_for_fun_num['Gender'] == 'Male']['hike_for_fun_num'].mean():.2f}.")
    print(f"The mean for female participants is {df_gender_hike_for_fun_num[df_gender_hike_for_fun_num['Gender'] == 'Female']['hike_for_fun_num'].mean():.2f}.")
else:
    print("No statistically significant difference between male and female participants' hike for fun.")
print('\n')
# Group by Gender and calculate summary statistics
summary_stats = df_valid_gender.groupby('Gender')['hike_for_fun_num'].describe()
display(summary_stats)
# Create a bar graph of the hike_for_fun_num column
df_gender_hike_for_fun_num.boxplot(column='hike_for_fun_num', by = 'Gender')
plt.xlabel('Gender')
plt.ylabel('Hike for Fun (numeric)')
plt.title('Hike for Fun')
plt.show()
print('done\n')


#Define a mapping from categories to numerical values of hike for meditation column
mapping = {
    'Strongly disagree': 1,  # 1 for strongly disagree
    'Somewhat disagree': 2,  # 2 for somewhat disagree
    'Neither agree nor disagree': 3, # 3 for neither agree nor disagree
    'Somewhat agree': 4, # 4 for somewhat agree
    'Strongly agree': 5, # 5 for strongly agree
}
print ("\nreplacement of hike for meditation to numeric value...")
df['hike_for_meditation_num'] = df['hike_for_mediation'].replace(mapping).infer_objects(copy=False)
# Step 1: Identify non-finite values (e.g., NaN, inf, -inf)
non_finite_mask = ~np.isfinite(df['hike_for_meditation_num'])

# Step 2: Fill non-finite values with a default value (e.g., 0) or remove them
df['hike_for_meditation_num'].fillna(0, inplace=True)  # Filling NaN with 0  
# Alternatively, you can remove rows with non-finite values
# df = df[~non_finite_mask]

df['hike_for_meditation_num'] = df['hike_for_meditation_num'].astype(int)
print ("Done\n")
#exclude the rows with fill in the blank
df_valid_gender = df[df['Gender'] != 'Fill in the blank']
# create a pivot table of the hike_for_meditation_num
print('\nhike_for_meditation_num table\n')
pivot_table_hike_for_meditation_num = df_valid_gender.pivot_table(index='Gender', columns='hike_for_meditation_num', aggfunc='size', fill_value=0)
print(pivot_table_hike_for_meditation_num, '\n')
print('Done\n')
# Set significance level
alpha = 0.05
# Filter the male and female rows with a value in the hike_for_meditation_num column
df_gender_hike_for_meditation_num = df[df['Gender'].isin(['Male', 'Female']) & df['hike_for_meditation_num'].notnull()][['Gender', 'hike_for_meditation_num']]
# Perform the t-test
t_stat, p_value = ttest_ind(df_gender_hike_for_meditation_num[df_gender_hike_for_meditation_num['Gender'] == 'Male']['hike_for_meditation_num'],
                            df_gender_hike_for_meditation_num[df_gender_hike_for_meditation_num['Gender'] == 'Female']['hike_for_meditation_num'],
                            equal_var=False)
# Print the t-statistic and p-value
print(f"t-statistic: {t_stat:.2f}")
print(f"p-value: {p_value:.4f}")
print('\n')
# Determine if the result is statistically significant
is_significant = p_value < alpha
print(f"Is it statistically significant at the {alpha} level? {is_significant}")
print('\n')
# Determine if the result is statistically significant
if p_value < alpha:
    print("**Statistically significant difference** between Male and Female participants' hike for meditation.")
    print(f"The mean for male participants is {df_gender_hike_for_meditation_num[df_gender_hike_for_meditation_num['Gender'] == 'Male']['hike_for_meditation_num'].mean():.2f}.")
    print(f"The mean for female participants is {df_gender_hike_for_meditation_num[df_gender_hike_for_meditation_num['Gender'] == 'Female']['hike_for_meditation_num'].mean():.2f}.")
    print('\n')
else:
    print("No statistically significant difference between male and female participants' hike for meditation.")
print('\n')
# Group by Gender and calculate summary statistics
summary_stats = df_valid_gender.groupby('Gender')['hike_for_meditation_num'].describe()
display(summary_stats)
# Create a bar graph of the hike_for_meditation_num column
df_gender_hike_for_meditation_num.boxplot(column='hike_for_meditation_num', by='Gender')
plt.xlabel('Gender')
plt.ylabel('Hike for Meditation (numeric)')
plt.title('Hike for Meditation')
plt.show()
print('done\n')


# Define a mapping from categories to numerical values of hike for less than 1hour column
mapping = {
    'Strongly disagree': 1,  # 1 for strongly disagree
    'Somewhat disagree': 2,  # 2 for somewhat disagree
    'Neither agree nor disagree': 3, # 3 for neither agree nor disagree
    'Somewhat agree': 4, # 4 for somewhat agree
    'Strongly agree': 5, # 5 for strongly agree
}
print ("\nreplacement of hike for less than 1 hour to numeric value...")
df['hike_for_less_than_1hour_num'] = df['hike_for_less_than_1hour'].replace(mapping).infer_objects(copy=False)
# Step 1: Identify non-finite values (e.g., NaN, inf, -inf)
non_finite_mask = ~np.isfinite(df['hike_for_less_than_1hour_num'])

# Step 2: Fill non-finite values with a default value (e.g., 0) or remove them
df['hike_for_less_than_1hour_num'].fillna(0, inplace=True)  # Filling NaN with 0
# Alternatively, you can remove rows with non-finite values
# df = df[~non_finite_mask]
df['hike_for_less_than_1hour_num'] = df['hike_for_less_than_1hour_num'].astype(int)
print ("Done\n")
#exclude the rows with fill in the blank
df_valid_gender = df[df['Gender'] != 'Fill in the blank']
# create a pivot table of the hike_for_less_than_1hour_num
print('\nhike_for_less_than_1hour_num table\n')
pivot_table_hike_for_less_than_1hour_num = df_valid_gender.pivot_table(index='Gender', columns='hike_for_less_than_1hour_num', aggfunc='size', fill_value=0)
print(pivot_table_hike_for_less_than_1hour_num)
print('Done\n')
# Set significance level
alpha = 0.05
# Filter the male and female rows with a value in the hike_for_less_than_1hour_num column
df_gender_hike_for_less_than_1hour_num = df[df['Gender'].isin(['Male', 'Female']) & df['hike_for_less_than_1hour_num'].notnull()][['Gender', 'hike_for_less_than_1hour_num']]
# Perform the t-test
t_stat, p_value = ttest_ind(df_gender_hike_for_less_than_1hour_num[df_gender_hike_for_less_than_1hour_num['Gender'] == 'Male']['hike_for_less_than_1hour_num'],
                            df_gender_hike_for_less_than_1hour_num[df_gender_hike_for_less_than_1hour_num['Gender'] == 'Female']['hike_for_less_than_1hour_num'],
                            equal_var=False)
# Print the t-statistic and p-value
print(f"t-statistic: {t_stat:.2f}")
print(f"p-value: {p_value:.4f}")
print('\n')
# Determine if the result is statistically significant
is_significant = p_value < alpha
print(f"Is it statistically significant at the {alpha} level? {is_significant}")
print('\n')
# Determine if the result is statistically significant
if p_value < alpha:
    print("**Statistically significant difference** between Male and Female participants' hike for less than 1 hour values.")
    print(f"The mean for male participants is {df_gender_hike_for_less_than_1hour_num[df_gender_hike_for_less_than_1hour_num['Gender'] == 'Male']['hike_for_less_than_1hour_num'].mean():.2f}.")
    print(f"The mean for female participants is {df_gender_hike_for_less_than_1hour_num[df_gender_hike_for_less_than_1hour_num['Gender'] == 'Female']['hike_for_less_than_1hour_num'].mean():.2f}.")
    print('\n')
else:
    print("No statistically significant difference between male and female participants' for hike less than 1 hour.")
# Group by Gender and calculate summary statistics
summary_stats = df_valid_gender.groupby('Gender')['hike_for_less_than_1hour_num'].describe()
display(summary_stats)
# Create a bar graph of the hike_for_less_than_1hour_num column
df_gender_hike_for_less_than_1hour_num.boxplot(column='hike_for_less_than_1hour_num', by='Gender')
plt.xlabel('Gender')
plt.ylabel('Hike for Less than 1 Hour (numeric)')
plt.title('Hike for Less than 1 Hour')
plt.show()
print('done\n')


# Define a mapping from categories to numerical values of hike for half a day column
mapping = {
    'Strongly disagree': 1,  # 1 for strongly disagree
    'Somewhat disagree': 2,  # 2 for somewhat disagree
    'Neither agree nor disagree': 3, # 3 for neither agree nor disagree
    'Somewhat agree': 4, # 4 for somewhat agree
    'Strongly agree': 5, # 5 for strongly agree
}
print ("\nreplacement of hike for half a day to numeric value...")
df['hike_for_half_a_day_num'] = df['hike_for_half_a_day'].replace(mapping).infer_objects(copy=False)
# Step 1: Identify non-finite values (e.g., NaN, inf, -inf)
non_finite_mask = ~np.isfinite(df['hike_for_half_a_day_num'])

# Step 2: Fill non-finite values with a default value (e.g., 0) or remove them
df['hike_for_half_a_day_num'].fillna(0, inplace=True)  # Filling NaN with 0
# Alternatively, you can remove rows with non-finite values
# df = df[~non_finite_mask]
df['hike_for_half_a_day_num'] = df['hike_for_half_a_day_num'].astype(int)
print ("Done\n")
#exclude the rows with fill in the blank
df_valid_gender = df[df['Gender'] != 'Fill in the blank']
# create a pivot table of the hike_for_half_a_day_num
print('\nhike_for_half_a_day_num table\n')
pivot_table_hike_for_half_a_day_num = df_valid_gender.pivot_table(index='Gender', columns='hike_for_half_a_day_num', aggfunc='size', fill_value=0)
print(pivot_table_hike_for_half_a_day_num)
print('Done\n')
# Set significance level
alpha = 0.05
# Filter the male and female rows with a value in the hike_for_half_a_day_num column
df_gender_hike_for_half_a_day_num = df[df['Gender'].isin(['Male', 'Female']) & df['hike_for_half_a_day_num'].notnull()][['Gender', 'hike_for_half_a_day_num']]
# Perform the t-test
t_stat, p_value = ttest_ind(df_gender_hike_for_half_a_day_num[df_gender_hike_for_half_a_day_num['Gender'] == 'Male']['hike_for_half_a_day_num'],
                            df_gender_hike_for_half_a_day_num[df_gender_hike_for_half_a_day_num['Gender'] == 'Female']['hike_for_half_a_day_num'],
                            equal_var=False)
# Print the t-statistic and p-value
print(f"t-statistic: {t_stat:.2f}")
print(f"p-value: {p_value:.4f}")
print('\n')
# Determine if the result is statistically significant
is_significant = p_value < alpha
print(f"Is it statistically significant at the {alpha} level? {is_significant}")
print('\n')
# Determine if the result is statistically significant
if p_value < alpha:
    print("**Statistically significant difference** between male and female participants' hike for half a day values.")
    print(f"The mean for male participants is {df_gender_hike_for_half_a_day_num[df_gender_hike_for_half_a_day_num['Gender'] == 'Male']['hike_for_half_a_day_num'].mean():.2f}.")
    print(f"The mean for female participants is {df_gender_hike_for_half_a_day_num[df_gender_hike_for_half_a_day_num['Gender'] == 'Female']['hike_for_half_a_day_numm'].mean():.2f}.")
    print('\n')
else:
    print("No statistically significant difference between male and female participants' for hike for half a day.\n")
# Create a bar graph of the hike_for_half_a_day_num column
df_gender_hike_for_half_a_day_num.boxplot(column='hike_for_half_a_day_num', by='Gender')
plt.xlabel('Gender')
plt.ylabel('Hike for Half a Day (numeric)')
plt.title('Hike for Half a Day')
plt.axhline(y=df_gender_hike_for_half_a_day_num['hike_for_half_a_day_num'].median(), color='green', linestyle='--')
plt.show()
print('done\n')


#Define a mapping from categories to numerical values of hike for less than a day column
mapping = {
    'Strongly disagree': 1,  # 1 for strongly disagree
    'Somewhat disagree': 2,  # 2 for somewhat disagree
    'Neither agree nor disagree': 3, # 3 for neither agree nor disagree
    'Somewhat agree': 4, # 4 for somewhat agree
    'Strongly agree': 5, # 5 for strongly agree
}
print ("\nreplacement of hike for less than a day to numeric value...")
df['hike_for_less_than_a_day_num'] = df['hike_for_less_than_a_day'].replace(mapping).infer_objects(copy=False)
# Step 1: Identify non-finite values (e.g., NaN, inf, -inf)
non_finite_mask = ~np.isfinite(df['hike_for_less_than_a_day_num'])

# Step 2: Fill non-finite values with a default value (e.g., 0) or remove them
df['hike_for_less_than_a_day_num'].fillna(0, inplace=True)  # Filling NaN with 0
# Alternatively, you can remove rows with non-finite values
# df = df[~non_finite_mask]
df['hike_for_less_than_a_day_num'] = df['hike_for_less_than_a_day_num'].astype(int)
print ("\n")
#exclude the rows with fill in the blank
df_valid_gender = df[df['Gender'] != 'Fill in the blank']
# create a pivot table of the hike_for_less_than_a_day_num
print('\nhike_for_less_than_a_day_num table\n')
pivot_table_hike_for_less_than_a_day = df_valid_gender.pivot_table(index='Gender', columns='hike_for_less_than_a_day_num', aggfunc='size', fill_value=0)
print(pivot_table_hike_for_less_than_a_day)
print('Done\n')
# Set significance level
alpha = 0.05
# Filter the male and female rows with a value in the hike_for_less_than_a_day_num column
df_gender_hike_for_less_than_a_day_num = df[df['Gender'].isin(['Male', 'Female']) & df['hike_for_less_than_a_day_num'].notnull()][['Gender', 'hike_for_less_than_a_day_num']]
# Perform the t-test
t_stat, p_value = ttest_ind(df_gender_hike_for_less_than_a_day_num[df_gender_hike_for_less_than_a_day_num['Gender'] == 'Male']['hike_for_less_than_a_day_num'],
                            df_gender_hike_for_less_than_a_day_num[df_gender_hike_for_less_than_a_day_num['Gender'] == 'Female']['hike_for_less_than_a_day_num'],
                            equal_var=False)
# Print the t-statistic and p-value
print(f"t-statistic: {t_stat:.2f}")
print(f"p-value: {p_value:.4f}")
print('\n')
# Determine if the result is statistically significant
is_significant = p_value < alpha
print(f"Is it statistically significant at the {alpha} level? {is_significant}")
print('\n')
# Determine if the result is statistically significant
if p_value < alpha:
    print("**Statistically significant difference** between Male and Female participants' hike for less than a day values.")
    print(f"The mean for male participants is {df_gender_hike_for_less_than_a_day_num[df_gender_hike_for_less_than_a_day_num['Gender'] == 'Male']['hike_for_less_than_a_day_num'].mean():.2f}.")
    print(f"The mean for Female participants is {df_gender_hike_for_less_than_a_day_num[df_gender_hike_for_less_than_a_day_num['Gender'] == 'Female']['hike_for_less_than_a_day_num'].mean():.2f}.")
    print('\n')
else:
    print("No statistically significant between male and female participants' for hike for less than a day.\n")
# Group by Gender and calculate summary statistics
summary_stats = df_valid_gender.groupby('Gender')['hike_for_less_than_a_day_num'].describe()
display(summary_stats)
# Create a bar graph of the hike_for_less_than_a_day_num column
df_gender_hike_for_less_than_a_day_num.boxplot(column='hike_for_less_than_a_day_num', by='Gender')
plt.xlabel('Gender')
plt.ylabel('Hike for Less than a Day (numeric)')
plt.title('Hike for Less than a Day')
plt.show()
print('done\n')



# Define a mapping from categories to numerical values of hike for multiple days column
mapping = {
    'Strongly disagree': 1,  # 1 for strongly disagree
    'Somewhat disagree': 2,  # 2 for somewhat disagree
    'Neither agree nor disagree': 3, # 3 for neither agree nor disagree
    'Somewhat agree': 4, # 4 for somewhat agree
    'Strongly agree': 5, # 5 for strongly agree
}
print ("\nreplacement of hike for multiple days to numeric value...")
df['hike_for_multiple_days_num'] = df['hike_for_multiple_days'].replace(mapping).infer_objects(copy=False)
# Step 1: Identify non-finite values (e.g., NaN, inf, -inf)
non_finite_mask = ~np.isfinite(df['hike_for_multiple_days_num'])

# Step 2: Fill non-finite values with a default value (e.g., 0) or remove them
df['hike_for_multiple_days_num'].fillna(0, inplace=True)  # Filling NaN with 0
# Alternatively, you can remove rows with non-finite values
# df = df[~non_finite_mask]
df['hike_for_multiple_days_num'] = df['hike_for_multiple_days_num'].astype(int)
print ("\n")
#exclude the rows with fill in the blank
df_valid_gender = df[df['Gender'] != 'Fill in the blank']
# create a pivot table of the hike_for_multiple_days_num
print('\nhike_for_multiple_days_num table\n')
pivot_table_hike_for_multiple_days_num = df_valid_gender.pivot_table(index='Gender', columns='hike_for_multiple_days_num', aggfunc='size', fill_value=0)
print(pivot_table_hike_for_multiple_days_num)
print('Done\n')
# Set significance level
alpha = 0.05
# Filter the male and female rows with a value in the hike_for_multiple_days_num column
df_gender_hike_for_multiple_days_num = df[df['Gender'].isin(['Male', 'Female']) & df['hike_for_multiple_days_num'].notnull()][['Gender', 'hike_for_multiple_days_num']]
# Perform the t-test
t_stat, p_value = ttest_ind(df_gender_hike_for_multiple_days_num[df_gender_hike_for_multiple_days_num['Gender'] == 'Male']['hike_for_multiple_days_num'],
                            df_gender_hike_for_multiple_days_num[df_gender_hike_for_multiple_days_num['Gender'] == 'Female']['hike_for_multiple_days_num'],
                            equal_var=False)
# Print the t-statistic and p-value
print(f"t-statistic: {t_stat:.2f}")
print(f"p-value: {p_value:.4f}")
print('\n')
# Determine if the result is statistically significant
is_significant = p_value < alpha
print(f"Is it statistically significant at the {alpha} level? {is_significant}")
print('\n')
# Determine if the result is statistically significant
if p_value < alpha:
    print("**Statistically significant difference** between male and female participants' hike for multiple days values.")
    print(f"The mean for male participants is{df_gender_hike_for_multiple_days_num[df_gender_hike_for_multiple_days_num['Gender'] == 'Male']['hike_for_multiple_days_num'].mean():.2f}.")
    print(f"The mean for Female participants is {df_gender_hike_for_multiple_days_num[df_gender_hike_for_multiple_days_num['Gender'] == 'Female']['hike_for_multiple_days_num'].mean():.2f}.")
    print('\n')
else:
    print("No statistically significant between male and female participants' for hike for multiple days.\n")
# Group by Gender and calculate summary statistics
summary_stats = df_valid_gender.groupby('Gender')['hike_for_multiple_days_num'].describe()
display(summary_stats)
# Create a bar graph of the hike_for_multiple_days_num column
df_gender_hike_for_multiple_days_num.boxplot(column='hike_for_multiple_days_num', by='Gender')
plt.xlabel('Gender')
plt.ylabel('Hike for Multiple Days (numeric)')
plt.title('Hike for Multiple Days')
plt.show()
print('done\n')


# Define a mapping from categories to numerical values of easy hike column
mapping = {
    'Strongly disagree': 1,  # 1 for strongly disagree
    'Somewhat disagree': 2,  # 2 for somewhat disagree
    'Neither agree nor disagree': 3, # 3 for neither agree nor disagree
    'Somewhat agree': 4, # 4 for somewhat agree
    'Strongly agree': 5, # 5 for strongly agree
}
print ("\nreplacement of easy hike to numeric value...")
df['easy_hike_num'] = df['easy_hike'].replace(mapping).infer_objects(copy=False)  # suppress FutureWarning
# Step 1: Identify non-finite values (e.g., NaN, inf, -inf)
non_finite_mask = ~np.isfinite(df['easy_hike_num'])

# Step 2: Fill non-finite values with a default value (e.g., 0) or remove them
df['easy_hike_num'] = df['easy_hike_num'].fillna(0)  # Filling NaN with 0
# Alternatively, you can remove rows with non-finite values
# df = df[~non_finite_mask]
df['easy_hike_num'] = df['easy_hike_num'].astype(int)
print ("\n")
#exclude the rows with fill in the blank
df_valid_gender = df[df['Gender'] != 'Fill in the blank']
# create a pivot table of the easy_hike_num
#test pivot
df_valid_gender_pivot = df_valid_gender.pivot_table(index='Gender', columns='difficult_hike', aggfunc='size', fill_value=0)
print(df_valid_gender_pivot)
#ttest ends

# Create 'difficult_hike_n' column if it does not exist
mapping = {
    'Strongly disagree': 1,
    'Somewhat disagree': 2,
    'Neither agree nor disagree': 3,
    'Somewhat agree': 4,
    'Strongly agree': 5,
}
if 'difficult_hike_n' not in df.columns:
    df['difficult_hike_n'] = df['difficult_hike'].replace(mapping).infer_objects(copy=False)
    df['difficult_hike_n'] = df['difficult_hike_n'].fillna(0).astype(int)
    df_valid_gender = df[df['Gender'] != 'Fill in the blank']

print('\ndifficult_hike_n table\n')
pivot_table_difficult_hike_n = df_valid_gender.pivot_table(index='Gender', columns='difficult_hike_n', aggfunc='size', fill_value=0)
print(pivot_table_difficult_hike_n)
print('Done\n')

#Mann–Whitney U test
#from scipy.stats import mannwhitneyu
# Filter the male and female participants with a value in the difficult_hike_n column
#df_gender_difficult_hike_n = df[df['Gender'].isin(['Male', 'Female']) & df['difficult_hike_n'].notnull()][['Gender', 'difficult_hike_n']]

# Perform the Mann-Whitney U test
# u_stat, p_value = mannwhitneyu(df_gender_difficult_hike_n[df_gender_difficult_hike_n['Gender'] == 'Male']['difficult_hike_n'],
#                               df_gender_difficult_hike_n[df_gender_difficult_hike_n['Gender'] == 'Female']['difficult_hike_n'])

# display(f'Mann-Whitney U test statistic: {u_stat:.2f}')
# display(f'p-value: {p_value:.4f}')
# # Filter the male and female participants with a value in the difficult_hike_n column
# df_gender_difficult_hike_n = df[df['Gender'].isin(['Male', 'Female']) & df['difficult_hike_n'].notnull()][['Gender', 'difficult_hike_n']]

# Set significance level
alpha = 0.05
# Filter the male and female participants with a value in the difficult_hike_n column
df_gender_difficult_hike_n = df[df['Gender'].isin(['Male', 'Female']) & df['difficult_hike_n'].notnull()][['Gender', 'difficult_hike_n']]
# Perform the t-test
t_stat, p_value = ttest_ind(df_gender_difficult_hike_n[df_gender_difficult_hike_n['Gender'] == 'Male']['difficult_hike_n'],
                            df_gender_difficult_hike_n[df_gender_difficult_hike_n['Gender'] == 'Female']['difficult_hike_n'],
                            equal_var=False)
# Print the t-statistic and p-value
print(f"t-statistic: {t_stat:.2f}")
print(f"p-value: {p_value:.4f}")
print('\n')
# Determine if the result is statistically significant 
is_significant = p_value < alpha
print(f"Is it statistically significant at the {alpha} level? {is_significant}")
print('\n')
# Determine if the result is statistically significant
if p_value < alpha:
    print("**Statistically significant difference** between the genders and difficult hike values.")
    print(f"The mean for male participants is {df_gender_difficult_hike_n[df_gender_difficult_hike_n['Gender'] == 'Male']['difficult_hike_n'].mean():.2f}.")
    print(f"The mean for female participants is {df_gender_difficult_hike_n[df_gender_difficult_hike_n['Gender'] == 'Female']['difficult_hike_n'].mean():.2f}.")
    print('\n')
    
else:
    print("No statistically significant between male and female participants' for difficult hike.\n")
# Group by Gender and calculate summary statistics
summary_stats = df_valid_gender.groupby('Gender')['difficult_hike_n'].describe()
display(summary_stats)

# Create a boxplot to compare the distributions of difficult_hike_n between males and females
plt.boxplot([df_gender_difficult_hike_n[df_gender_difficult_hike_n['Gender'] == 'Male']['difficult_hike_n'],
             df_gender_difficult_hike_n[df_gender_difficult_hike_n['Gender'] == 'Female']['difficult_hike_n']],
            tick_labels=['Male', 'Female'])
plt.title('Distribution of difficult_hike_n by Gender')
plt.xlabel('Gender')
plt.ylabel('difficult Hike (numeric)') 
# Add a horizontal line to indicate the median
plt.axhline(y=df_gender_difficult_hike_n['difficult_hike_n'].median(), color='gray', linestyle='--')
plt.show()
print('done\n')



# A table of summary statistics for the variables of interest
variables = ['like_to_hike_alone_value', 'like_to_hike_in_group_num', 'like_to_hike_near_home_num', 'hike_while_traveling_num', 'hike_for_health_num', 'hike_for_social_interaction_num', 'hike_for_fun_num', 'hike_for_meditation_num', 'hike_for_less_than_1hour_num', 'hike_for_half_a_day_num', 'hike_for_less_than_a_day_num', 'hike_for_multiple_days_num', 'easy_hike_num', 'difficult_hike_n']

summary_stats = pd.DataFrame(columns=['Mean_Male', 'Mean_Female', 'Std_Male', 'Std_Female', 'p-value', 't-statistic', 'Significant'], index=variables)

for var in variables:
    mean_male = df.loc[df['Gender'] == 'Male', var].mean()
    mean_female = df.loc[df['Gender'] == 'Female', var].mean()
    std_male = df.loc[df['Gender'] == 'Male', var].std()
    std_female = df.loc[df['Gender'] == 'Female', var].std()
    
    t_stat, p_val = ttest_ind(df.loc[df['Gender'] == 'Male', var], df.loc[df['Gender'] == 'Female', var])
    
    summary_stats.loc[var] = [mean_male, mean_female, std_male, std_female, p_val, t_stat, p_val < alpha]

display(summary_stats)


# Benjamini-Hochberg procedure
# assume 'p_values' is an array of p-values from multiple tests

p_values = np.array([0.0065, 0.3133, 0.9578, 0.0013, 0.7590, 0.4947, 0.0904, 0.9602, 0.0023, 0.3437, 0.1143, 0.1277, 0.0002, 0.0016])  # replace with your p-values

rejected, corrected_p_values = fdrcorrection(p_values, alpha=0.05)

display("Rejected hypotheses:", rejected)
display("Corrected p-values:", corrected_p_values)

#Compare p-values to critical values: Compare each p-value to its corresponding critical value. If the p-value is less than or equal to the critical value, reject the null hypothesis.

# 
for var in variables:
    skewness_male = skew(df.loc[df['Gender'] == 'Male', var])
    kurt_male = kurtosis(df.loc[df['Gender'] == 'Male', var])
    
    skewness_female = skew(df.loc[df['Gender'] == 'Female', var])
    kurt_female = kurtosis(df.loc[df['Gender'] == 'Female', var])
    
    display(f"Variable: {var}")
    print(f"Male: Skewness: {skewness_male:.3f}, Kurtosis: {kurt_male:.3f}")
    print(f"Female: Skewness: {skewness_female:.3f}, Kurtosis: {kurt_female:.3f}")
    print()



































print('--------------- version on 2 devices starting-----------------------------')

print('\n two devices\n')

import pandas as pd
df = pd.read_excel('hiking_data_v2.xlsx')

# Create a dataframe of the percentage of 2 devices in the number_of_devices column with combined
percentage_2_devices = df[df['number _of_devices'] == 2]['combined'].value_counts(normalize=True) * 100
percentage_2_devices = percentage_2_devices.round(1).astype(str) + '%'

# Create a pivot table of the combined devices and the percentage as columns
pivot_table_combined = pd.pivot_table(df[df['number _of_devices'] == 2], values='combined', columns='number _of_devices', aggfunc=lambda x: str(round(len(x) / len(df[df['number _of_devices'] == 2]) * 100, 1)) + '%')

print("Percentage of 2 devices in the number_of_devices column with combined:")
print(percentage_2_devices)
print("\nPivot table of combined devices and percentage:")
print(pivot_table_combined)

#--------------------------------------------------------------
import matplotlib.patches as mpatches
# Count the occurrences of each category in the combined column for 2 devices
counts = df[df['number _of_devices'] == 2]['combined'].value_counts()

# Calculate the percentage of each category
percentages = (counts / counts.sum()) * 100

# Create a bar graph for the combined column
percentages.plot(kind='bar', color=plt.cm.tab20(range(len(percentages))))

plt.xlabel('2 Devices')
plt.ylabel('Count')
plt.title('Percentage of People by 2 Devices')

# Create a legend with custom colors and labels
labels = percentages.index
colors = plt.cm.tab20(range(len(percentages)))
legend_patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
plt.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)

# Add data labels to the bars
ax = plt.gca()
for p in ax.patches:
    percentage = f"{p.get_height() / percentages.sum().sum() * 100:.1f}%"
    ax.annotate(percentage, (p.get_x() + p.get_width() / 2, p.get_height() + 0.3), ha='center', va='bottom')
plt.xticks([])  # <--- Add this line to remove x-axis labels
plt.show()
###----------------------- Count -----------------------
" --------------------version on 2 devices -----------------"
import pandas as pd
df = pd.read_excel('hiking_data_v2.xlsx')
# Create a dataframe of 2 devices in the number_of_devices column with combined
df_2_devices_combined = df[df['number _of_devices'] == 2]['combined']
pivot_table_combined = df_2_devices_combined.value_counts().to_frame().rename(columns={'combined': 'Count'})
# display(pivot_table_combined)

# Create a list of colors
colors = plt.cm.tab20(range(len(pivot_table_combined)))

# Create a bar graph of the pivot table
bars = pivot_table_combined.plot(kind='bar')
for bar, color in zip(bars.patches, colors):
    bar.set_color(color)

plt.xlabel('Combined Devices')
plt.ylabel('Count')
plt.title('2 device combinations')

# Create a legend with custom colors and labels
labels = pivot_table_combined.index
legend_patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
plt.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)

# Add data labels to the bars
ax = plt.gca()
for p in ax.patches:
    ax.annotate(f"{p.get_height()}", (p.get_x() + p.get_width() / 2, p.get_height() + 0.3), ha='center', va='bottom')

plt.xticks([])  # Remove x-axis labels
plt.show()
print('done\n')


#create a dataframe to split number of devices in 2 devices row in the number_of_devices column and combined column and count each device
# print('\n2 devices\n')
# df_2_devices = df[df['number _of_devices'] == 2]['combined'].value_counts().to_frame().T
# print(df_2_devices)
# print('\n')

# Create a dataframe to split the devices in the combined column into separate columns of devices
device_columns = ['device0', 'device1', 'device2', 'device3', 'device4', 'device5', 'device6', 'device7', 'device8', 'device9']
df_devices_split = df['combined'].str.split(',', expand=True)
df_devices_split.columns = device_columns

# Concatenate the original dataframe with the new device columns
df = pd.concat([df, df_devices_split], axis=1)

# Display the first few rows of the updated dataframe
# display(df.head())
#save the updated dataframe to a new excel file
df.to_excel('hiking_data_v2_updated.xlsx', sheet_name='devices', index=False)

#df = pd.read_excel('hiking_data_v2.xlsx', sheet_name='2device_count')

#create a dataframe to count the number of devices in the 2devices column in the 2decice_count sheet
print('\n2 devices\n')
df_2_devices = df['2devices'].value_counts().to_frame().T
print(df_2_devices)
print('\n')

#create a graph of the 2devices dataframe as a bar graph with the devices as the x-axis and the count as the y-axis
df_2_devices.plot(kind='bar')
plt.xlabel('Devices')
plt.ylabel('Count')
plt.title('2 Devices')
legend_patches = [mpatches.Patch(color='blue', label='2 Devices')]
plt.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
#label the bars with the count
ax = plt.gca()
for p in ax.patches:
    ax.annotate(f"{p.get_height()}", (p.get_x() + p.get_width() / 2, p.get_height() + 0.3), ha='center', va='bottom')

plt.show()
print('done\n')


































df = pd.read_excel('hiking_data_v2.xlsx')



#perform a chi-square test on the pivot_table dataframe and print the p-value and residuals 
print('\nchi-square test on Gender and number of devices')
chi2, p, _, expected = stats.chi2_contingency(pivot_table_gender ) 
print("p-value:", p)
residuals = (pivot_table_gender - expected) / expected
print("Residuals:")
print(residuals)

#create a pivot table of Female and Male row in the Gender column with devices column and 2 devices row in the number _of_devices columns as rows and columns
print('\ngender with 2 devices')
pivot_table_gender_2_devices = df[(df['Gender'] != 'Fill in the blank') & (df['number _of_devices'] == 2)].pivot_table(index='Gender', columns='number _of_devices', aggfunc='size', fill_value=0)
print(pivot_table_gender_2_devices)
print('\n')

# Calculate the percentage of male and female who use 2 devices
print('\npercentage of gender with 2 devices')
percentage_male_2_devices = f"{pivot_table_gender_2_devices.loc['Male', 2] / gender_counts['Male'] * 100:.1f}%"
percentage_female_2_devices = f"{pivot_table_gender_2_devices.loc['Female', 2] / gender_counts['Female'] * 100:.1f}%"

print("Males:", percentage_male_2_devices)
print("Females:", percentage_female_2_devices)
print('\n')


























# a dataframe of the number of Male and Female row in Gender column
print('Gender counts')
pivot_table_gender = df['Gender'].value_counts().to_frame().T
print(pivot_table_gender)

# Count the number of male and female rows in the Gender column
print('\nGender counts')
gender_counts = df['Gender'].value_counts()
print(gender_counts)


#perform the two-proportion z-test
print('\nTwo-proportion z-test by 2 devices')

# Calculate the number of males and females
male_count = len(df[(df['Gender'] == 'Male') & (df['number _of_devices'] == 2)])
female_count = len(df[(df['Gender'] == 'Female') & (df['number _of_devices'] == 2)])

# Calculate the total number of observations
total_male = len(df[df['Gender'] == 'Male'])
total_female = len(df[df['Gender'] == 'Female'])

# Calculate the proportions
male_proportion = male_count / total_male
female_proportion = female_count / total_female

# Perform the two-proportion z-test
z, p = proportions_ztest([male_count, female_count], [total_male, total_female])

# Print the results
print("z-statistic:", z)
print("p-value:", p)


# Calculate the expected values
expected_values = pivot_table.values.sum(axis=0) * pivot_table.values.sum(axis=1)[:, np.newaxis] / pivot_table.values.sum()

# Display the expected values
print("Expected Values:")
print(expected_values)

#A z-statistic less than 0 indicates that the observed difference in proportions is less than the expected difference under the null hypothesis, suggesting that the proportion of one group (e.g., males) is lower than the proportion of the other group (e.g., females).
#A large absolute value of the z-statistic (e.g., greater than 2 or less than -2) indicates that the observed difference in proportions is statistically significant, suggesting that the null hypothesis can be rejected
#A p-value less than 0.05 indicates that the observed difference in proportions is statistically significant, suggesting that the null hypothesis can be rejected and the observed difference is unlikely to have occurred by chance. 
#A p-value greater than 0.05 indicates that the observed difference in proportions is not statistically significant, suggesting that the null hypothesis cannot be rejected and the observed difference is likely to have occurred by chance.

# create a graph of the pivot table as a bar graph with the number of devices as the x-axis and the count as the y-axis
pivot_table_gender_2_devices.plot(kind='bar')
plt.xlabel('Number of Devices')
plt.ylabel('Count')
plt.title('Number of Devices')
# Add data labels to the bars
ax = plt.gca()
for p in ax.patches:
    ax.annotate(f"{p.get_height():.0f}", (p.get_x() + p.get_width() / 2, p.get_height() + 0.5), ha='center', va='bottom')
plt.show()
print('done')

#create a heatmap of the pivot table as a heatmap with the number of devices as the x-axis and the count as the y-axis
plt.imshow (pivot_table_gender_2_devices, cmap='coolwarm', interpolation='nearest', extent=[-0.5, len (pivot_table_gender_2_devices.columns)-0.5, -0.5, len (pivot_table_gender_2_devices.index)-0.5])
plt.colorbar(label='Count')
plt.xlabel('Number of Devices')
plt.ylabel('Gender')
plt.xticks (range(len (pivot_table_gender_2_devices.columns)), pivot_table_gender_2_devices.columns)
plt.yticks (range(len (pivot_table_gender_2_devices.index)), pivot_table_gender_2_devices.index)
for i in range (len (pivot_table_gender_2_devices.index)):
    for j in range (len (pivot_table_gender_2_devices.columns)):
        plt.text (j, i, pivot_table_gender_2_devices.iloc[i, j], ha='center', va='center', color='black')
plt.title(' Number of Devices 2')
plt.show()
print('done')

# Create a pivot table of Male and Female row in Gender column with 2 devices in number _of_devices column
pivot_table_gender_2_devices = df[(df['Gender'].isin(['Male', 'Female'])) & (df['number _of_devices'] == 2)].pivot_table(index='Gender', columns='number _of_devices', aggfunc='size', fill_value=0)



# Create a pivot table of Female row in Gender column with 2 devices in combined column
print('\nFemale with 2 devices')
pivot_table_female_2_devices = df[(df['Gender'] == 'Female') & (df['number _of_devices'] == 2)].pivot_table(index='Gender', columns='combined', aggfunc='size', fill_value=0)
display(pivot_table_female_2_devices)

# Calculate the total count
total_count = pivot_table_female_2_devices.sum().sum()

# Calculate the percentage of each category
pivot_table_female_2_devices_percentage = (pivot_table_female_2_devices / total_count) * 100
# Create a bar chart of the pivot table as a bar graph with the number of devices as the x-axis and the count as the y-axis
pivot_table_female_2_devices_percentage.plot(kind='bar')
plt.xlabel('Number of Devices')
plt.ylabel('Count')
plt.title('2 device combination (Female)')
# Move the legend to the bottom of the table
legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
# Add data labels to the bars
ax = plt.gca()
for p in ax.patches:
    percentage = f"{p.get_height():.1f}%"
    ax.annotate(percentage, (p.get_x() + p.get_width() / 2, p.get_height() + 0.1), ha='center', va='bottom')
plt.show()


# Create a pivot table of Male row in Gender column with 2 devices in the combine column
print('\nMale with 2 devices')
pivot_table_male_2_devices = df[(df['Gender'] == 'Male') & (df['number _of_devices'] == 2)].pivot_table(index='Gender', columns='combined', aggfunc='size', fill_value=0)
display(pivot_table_male_2_devices)

#calculate the total count
total_count = pivot_table_male_2_devices.sum().sum()
#calculate the percentage of each category
pivot_table_male_2_devices_percentage = (pivot_table_male_2_devices / total_count) * 100

pivot_table_male_2_devices_percentage.plot(kind='bar')
plt.xlabel('Number of Devices')
plt.ylabel('Count')
plt.title('2 device combination (Male)')
# Move the legend to the bottom of the table
legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
# Add data labels to the bars
ax = plt.gca()
for p in ax.patches:
    percentage = f"{p.get_height():.1f}%"
    ax.annotate(percentage, (p.get_x() + p.get_width() * 0.6, p.get_height() + 0.1), ha='center', va='bottom')
plt.show()

print('------------------Heat map -----------------------\n')
#create a heatmap of male and female with 2 devices in the number _of_devices column
pivot_table_gender_2_devices = df[(df['Gender'].isin(['Male', 'Female'])) & (df['number _of_devices'] == 2)].pivot_table(index='Gender', columns='number _of_devices', aggfunc='size', fill_value=0)

# Create a heatmap of the pivot table
plt.imshow(pivot_table_gender_2_devices, cmap='coolwarm', interpolation='nearest', extent=[-0.5, len(pivot_table_gender_2_devices.columns)-0.5, -0.5, len(pivot_table_gender_2_devices.index)-0.5])
plt.colorbar(label='Count')
plt.xlabel('Number of Devices')
plt.ylabel('Gender')
plt.xticks(range(len(pivot_table_gender_2_devices.columns)), pivot_table_gender_2_devices.columns)
plt.yticks(range(len(pivot_table_gender_2_devices.index)), pivot_table_gender_2_devices.index)
for i in range(len(pivot_table_gender_2_devices.index)):
    for j in range(len(pivot_table_gender_2_devices.columns)):
        plt.text(j, i, pivot_table_gender_2_devices.iloc[i, j], ha='center', va='center', color='black')
plt.title('Number of Devices by Gender')
plt.show()

# Create a pivot table of Male and Female row in Gender column with 2 devices in the combine column
pivot_table_combined_2_devices = df[(df['Gender'].isin(['Male', 'Female'])) & (df['number _of_devices'] == 2)].pivot_table(index='Gender', columns='combined', aggfunc='size', fill_value=0)
display(pivot_table_combined_2_devices)

#calculate the percentage of each category
pivot_table_combined_2_devices_percentage = pivot_table_combined_2_devices.div(pivot_table_combined_2_devices.sum(axis=1), axis=0) * 100

# Create a bar chart of the pivot table as a bar graph with the number of devices as the x-axis and the count as the y-axis
pivot_table_combined_2_devices_percentage.plot(kind='bar')
#plt.xlabel('Number of Devices')
plt.ylabel('Percentage')
plt.title('2 Devices used by Male and Female')
legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
# Add data labels to the bars
# for p in plt.gca().patches:
#     plt.text(p.get_x() * 1.005, p.get_height() * 1.005, '{:.1f}%'.format(p.get_height()), fontsize=8, color='black')
plt.show()
print('done\n')
print('\nMale and Female with 2 devices in combined column')
pivot_table_combined_2_devices = df[(df['Gender'].isin(['Male', 'Female'])) & (df['number _of_devices'] == 2)].pivot_table(columns='Gender', index='combined', aggfunc='size', fill_value=0)
display(pivot_table_combined_2_devices)


# Calculate the percentage of men and women who use "Smartphone, Portable charger/battery"
percentage_male_smartphone = f"{pivot_table_combined_2_devices.loc['Smartphone, Portable charger/battery', 'Male'] / gender_counts['Male'] * 100:.1f}%"
percentage_female_smartphone = f"{pivot_table_combined_2_devices.loc['Smartphone, Portable charger/battery', 'Female'] / gender_counts['Female'] * 100:.1f}%"

print("Percentage of men who use 'Smartphone, Portable charger/battery':", percentage_male_smartphone)
print("Percentage of women who use 'Smartphone, Portable charger/battery':", percentage_female_smartphone)


percentage_male_smarthead = f"{pivot_table_combined_2_devices.loc['Smartphone, Headphones/earbuds', 'Male'] / gender_counts['Male'] * 100:.1f}%"
percentage_female_smarthead = f"{pivot_table_combined_2_devices.loc['Smartphone, Headphones/earbuds', 'Female'] / gender_counts['Female'] * 100:.1f}%"

print("Percentage of men who use 'Smartphone, Headphones/earbuds':", percentage_male_smarthead)
print("Percentage of women who use 'Smartphone, Headphones/earbuds':", percentage_female_smarthead)



# # Filter the dataframe to include only rows where the gender is 'Female' and the combined column is 'Smartphone, Portable charger/battery', and print the resulting dataframe
print('\nFemale with "Smartphone, Portable charger/battery"')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
female_smartphone = df[(df['Gender'] == 'Female') & (df['combined'] == 'Smartphone, Portable charger/battery')]
#display(female_smartphone)


# Create a pivot table of Female rows with 'Smartphone, Portable charger/battery' in the 'combined' column, 'easy_hike' and 'Difficult_hike' in the 'Safety_emergency' column
pivot_table_female_smartphone = female_smartphone.pivot_table(index='combined', columns='Safety_emergency', values=['easy_hike', 'difficult_hike'], aggfunc='size', fill_value=0)
display(pivot_table_female_smartphone)

# Create a pivot table of Female rows with 'Smartphone, Portable charger/battery' in the 'combined' column and 'hiking_duration' column
pivot_table_female_hiking_duration = female_smartphone.pivot_table(index='hiking_duration', columns='combined', aggfunc='size', fill_value=0)
display(pivot_table_female_hiking_duration)

# Create a pivot table of Female rows with 'Smartphone, Portable charger/battery' in the 'combined' column and 'like to hike alone' column
pivot_table_female_like_to_like_alone = female_smartphone.pivot_table(index='like_to_hike_alone', columns='combined', aggfunc='size', fill_value=0)
display(pivot_table_female_like_to_like_alone)

# Create a pivot table of Female rows with 'Smartphone, Portable charger/battery' in the 'combined' column and 'like to hike in group' column
pivot_table_female_in_group = female_smartphone.pivot_table(index= 'like_to_hike_in_group', columns='combined', aggfunc='size', fill_value=0)
display(pivot_table_female_in_group)

# Create a pivot table of Female rows with 'Smartphone, Portable charger/battery' in the 'combined' column and 'like to hike near home' column
pivot_table_female_near_home = female_smartphone.pivot_table(index= 'like_to_hike_near_home', columns='combined', aggfunc='size', fill_value=0)
display(pivot_table_female_near_home)

# Create a pivot table of Female rows with 'Smartphone, Portable charger/battery' in the 'combined' column and 'hike while traveling' column
pivot_table_female_while_traveling = female_smartphone.pivot_table(index= 'hike_while_traveling', columns='combined', aggfunc='size', fill_value=0)
display(pivot_table_female_while_traveling)

# Create a pivot table of Female rows with 'Smartphone, Portable charger/battery' in the 'combined' column and 'hike for health' column
pivot_table_female_health = female_smartphone.pivot_table(index= 'hike_for_health', columns='combined', aggfunc='size', fill_value=0)
display(pivot_table_female_health)

# Create a pivot table of Female rows with 'Smartphone, Portable charger/battery' in the 'combined' column and 'hike for fun' column
pivot_table_female_fun = female_smartphone.pivot_table(index= 'hike_for_fun', columns='combined', aggfunc='size', fill_value=0)
display(pivot_table_female_fun)

# Create a pivot table of Female rows with 'Smartphone, Portable charger/battery' in the 'combined' column and 'hike for less than 1hr' column
pivot_table_female_1hr = female_smartphone.pivot_table(index= 'hike_for_less_than_1hour', columns='combined', aggfunc='size', fill_value=0)
display(pivot_table_female_1hr)

# Create a pivot table of Female rows with 'Smartphone, Portable charger/battery' in the 'combined' column and 'hike for half a day' column
pivot_table_female_half_day = female_smartphone.pivot_table(index= 'hike_for_half_a_day', columns='combined', aggfunc='size', fill_value=0)
display(pivot_table_female_half_day)

# Create a pivot table of Female rows with 'Smartphone, Portable charger/battery' in the 'combined' column and 'hike for less than a day' column
pivot_table_female_less_day = female_smartphone.pivot_table(index= 'hike_for_less_than_a_day', columns='combined', aggfunc='size', fill_value=0)
display(pivot_table_female_less_day)


#Get the top 5 common 2 devices usage among men and women
print('\nTop 5 common 2 devices usage')
top_5_devices = df[df['number _of_devices'] == 2]['combined'].value_counts().head(5)
display(top_5_devices)

#create a pivot table of male and female rows with top 5 common 2 devices usage
print('\nMale and Female with top 5 common 2 devices usage')
pivot_table_top_5_devices = df[(df['number _of_devices'] == 2) & (df['combined'].isin(top_5_devices.index))].pivot_table(index='Gender', columns = 'combined', aggfunc='size', fill_value=0)
display(pivot_table_top_5_devices)

#calculate the percentage of each category
pivot_table_top_5_devices_percentage = pivot_table_top_5_devices.div(pivot_table_top_5_devices.sum(axis=1), axis=0) * 100
# Create a bar chart of the pivot table as a bar graph with the number of devices as the x-axis and the count as the y-axis
pivot_table_top_5_devices_percentage.plot(kind='bar', width=0.8)
plt.xlabel('Number of Devices')
plt.ylabel('Percentage')
plt.title('Top 5 common 2 devices usage')
legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
# Add data labels to the bars
for p in plt.gca().patches:
    plt.text(p.get_x() * 1.005, p.get_height() * 1.007, '{:.1f}%'.format(p.get_height()), fontsize=9, color='black')  
plt.show()
print('done\n')







































"-------------------Safety/emergency down-------------------"


# # Create a pivot table of Male and Female in Gender column with Safety/emergency column
# df = df.rename(columns={'Safety/emergency': 'Safety_emergency'})
# # Save the changes to the Excel file
# df.to_excel('Hiking_data_new.xlsx', index=False)

# #display(df.columns)

# print('\nMale and Female with Safety/emergency')
# pivot_table_gender_Safety_emergency = df[df['Gender'].isin(['Male', 'Female'])].pivot_table(columns='Gender', index='Safety_emergency', aggfunc='size', fill_value=0)
# display(pivot_table_gender_Safety_emergency)

# # Create a bar graph for the pivot table of Safety/emergency column
# pivot_table_gender_Safety_emergency.plot(kind='bar')
# plt.xlabel('Safety/emergency')
# plt.ylabel('Count')
# plt.title('Number of People by Safety/emergency')
# plt.show()


# # Create a pivot table of Female row in Gender column with Safety_emergency and combined columns
# print('\nFemale with Safety/emergency and combined')
# pivot_table_female_Safety_emergency_combined = df[(df['Gender'] == 'Female')].pivot_table(columns=['Gender', 'Safety_emergency'], index=[ 'combined'], aggfunc='size', fill_value=0)
# display(pivot_table_female_Safety_emergency_combined)
# print('\n')






# # Create a new dataframe that contains rows with Male and Female in the Gender column and the Safety_emergency column
# display(' A pivot table and p-value of the Safety_emergency column')
# df_gender_Safety_emergency = df[df['Gender'].isin(['Male', 'Female'])][['Gender', 'Safety_emergency']]

# # Create a pivot table of the Safe_emergency column with Gender as rows and count as columns
# pivot_table_gender_Safety_emergency = df_gender_Safety_emergency.pivot_table(columns='Gender', index='Safety_emergency', aggfunc='size', fill_value=0)
# display(pivot_table_gender_Safety_emergency)
# print('\n')
# # Perform the two-proportion z-test

# # Filter the data to only include male and female rows with 'emergency' in Safety_emergency
# display('Two-proportion z-test on emergency with male and female')
# male_emergency = df[(df['Gender'] == 'Male') & (df['Safety_emergency'] == 'emergency')].shape[0]
# female_emergency = df[(df['Gender'] == 'Female') & (df['Safety_emergency'] == 'emergency')].shape[0]

# total_male = df[df['Gender'] == 'Male'].shape[0]
# total_female = df[df['Gender'] == 'Female'].shape[0]

# # Perform the proportion test
# z, p = proportions_ztest([female_emergency, male_emergency], [total_female, total_male])

# display(["z-statistic:", z],["p-value:", p])
# #display("p-value:", p)
# print('\n')


# # Filter the data to only include male and female rows with 'safety' in Safety_emergency
# print('\n''Two-proportion z-test on Safety with male and female')
# male_safety_count = len(df[(df['Gender'] == 'Male') & (df['Safety_emergency'] == 'safety')])
# female_safety_count = len(df[(df['Gender'] == 'Female') & (df['Safety_emergency'] == 'safety')])

# total_male = len(df[df['Gender'] == 'Male'])
# total_female = len(df[df['Gender'] == 'Female'])

# # Perform the n-1 proportion test
# z, p = proportions_ztest([male_safety_count, female_safety_count], [total_male, total_female])

# display(["z-statistic:", z],["p-value:", p])
# #display("p-value:", p)
# print('\n')

"-------------------Safety/emergency up-------------------"


##------------------------------------## 3 Devices ##------------------------------------##
display('--------3 devices----------\n')

import pandas as pd
df = pd.read_excel('hiking_data_v2.xlsx')
df_3_devices_combined = df[(df['number _of_devices'] == 3)]['combined']
pivot_table_combined = df_3_devices_combined.value_counts().head(7).to_frame().rename(columns={'combined': 'Count'})
display(pivot_table_combined)

# Create a list of colors for the bar graph of the pivot table  of 3 devices in the number _of_devices column with combined column 
colors = plt.cm.tab20(range(len(pivot_table_combined)))

# Create a bar graph of the pivot table of 3 devices in the number _of_devices column with combined column 
bars = pivot_table_combined.plot(kind='bar')
for bar, color in zip(bars.patches, colors):
    bar.set_color(color)

plt.xlabel('Combined Devices')
plt.ylabel('Count')
plt.title('7 common 3 device combinations')

# Create a legend with custom colors and labels
labels = pivot_table_combined.index
legend_patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
plt.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)

# Add data labels to the bars
ax = plt.gca()
for p in ax.patches:
    ax.annotate(f"{p.get_height()}", (p.get_x() + p.get_width() / 2, p.get_height() + 0.3), ha='center', va='bottom')

plt.xticks([])  # Remove x-axis labels
plt.show()
print('done\n')









# Create a new dataframe that contains rows with 3 in the number _of_devices column and the combined column
df_3_devices_combined = df[df['number _of_devices'] == 3][['number _of_devices', 'combined']]

# Create a pivot table of 3 devices in number _of_devices and combined column
pivot_table_3_devices_combined = df_3_devices_combined.pivot_table(index='combined', columns='number _of_devices', aggfunc='size', fill_value=0)
display(pivot_table_3_devices_combined)

# Create a new dataframe that contains rows with Male and Female in the Gender column and 3 devices in the combined column
df_male_female_3_devices = df[(df['Gender'].isin(['Male', 'Female'])) & (df['combined'] == '3 devices')]

#--------------------------------------------------------------

# Create a new dataframe that contains rows with Male and Female in the Gender column and 3 devices in the number _of_devices column
df_male_female_3_devices = df[(df['Gender'].isin(['Male', 'Female'])) & (df['number _of_devices'] == 3)]

# Create a pivot table of Male and Female rows with 3 devices in the number _of_devices column and hiking_duration column
print('\nMale and female with 3 devices and hiking duration column')
pivot_table_male_female_3_devices_hiking_duration = df_male_female_3_devices.pivot_table(index='hiking_duration', columns='Gender', aggfunc='size', fill_value=0)
display(pivot_table_male_female_3_devices_hiking_duration)


# Create a pivot table Male and Female rows with 3 devices in the combined column and like_to_hike_alone column
print('\nMale and female with 3 devices in like to hike alone column')
pivot_table_male_female_3_devices_like_to_hike_alone = df_male_female_3_devices.pivot_table(index='like_to_hike_alone', columns='Gender', aggfunc='size', fill_value=0)
display(pivot_table_male_female_3_devices_like_to_hike_alone)

# Create a pivot table Male and Female rows with 3 devices in the combined column and like_to_hike_in_group column
print('\nMale and female with 3 devices in like to hike in group column')
pivot_table_male_female_3_devices_like_to_hike_in_group = df_male_female_3_devices.pivot_table(index='like_to_hike_in_group', columns='Gender', aggfunc='size', fill_value=0)
display(pivot_table_male_female_3_devices_like_to_hike_in_group)

# Create a pivot table Male and Female rows with 3 devices in the combined column and like_to_hike_near_home column
print('\nMale and female with 3 devices in like to hike near home column')
pivot_table_male_female_3_devices_like_to_hike_near_home = df_male_female_3_devices.pivot_table(index='like_to_hike_near_home', columns='Gender', aggfunc='size', fill_value=0)
display(pivot_table_male_female_3_devices_like_to_hike_near_home)

# Create a pivot table of Male and Female rows with 3 devices in the combined column and hike_while_traveling column
print('\nMale and Female with 3 devices in hike_while_traveling column')
pivot_table_male_female_3_devices_hike_while_traveling = df_male_female_3_devices.pivot_table(index='hike_while_traveling', columns='Gender', aggfunc='size', fill_value=0)
display(pivot_table_male_female_3_devices_hike_while_traveling)

# Create a pivot table of Male and Female rows with 3 devices in the combined column and hike_for_health column
print('\nMale and Female with 3 devices in hike_for_health column')
pivot_table_male_female_3_devices_hike_for_health = df_male_female_3_devices.pivot_table(index='hike_for_health', columns='Gender', aggfunc='size', fill_value=0)
display(pivot_table_male_female_3_devices_hike_for_health)

# Create a pivot table of Male and Female rows with 3 devices in the combined column and hike_for_social_interaction column
print('\nMale and Female with 3 devices in hike_for_social_interaction column')
pivot_table_male_female_3_devices_hike_for_social_interaction = df_male_female_3_devices.pivot_table(index='hike_for_social_interaction', columns='Gender', aggfunc='size', fill_value=0)
display(pivot_table_male_female_3_devices_hike_for_social_interaction)

# Create a pivot table of Male and Female rows with 3 devices in the combined column and hike_for_fun column
print('\nMale and Female with 3 devices in hike_for_fun column')
pivot_table_male_female_3_devices_hike_for_fun = df_male_female_3_devices.pivot_table(index='hike_for_fun', columns='Gender', aggfunc='size', fill_value=0)
display(pivot_table_male_female_3_devices_hike_for_fun)

# Create a pivot table of Male and Female rows with 3 devices in the combined column and hike_for_meditation column
print('\nMale and Female with 3 devices in hike_for_meditation column')
pivot_table_male_female_3_devices_hike_for_mediation = df_male_female_3_devices.pivot_table(index='hike_for_mediation', columns='Gender', aggfunc='size', fill_value=0)
display(pivot_table_male_female_3_devices_hike_for_mediation)

# Create a pivot table of Male and Female rows with 3 devices in the combined column and hike_for_less_than_1hour column
print('\nMale and Female with 3 devices in hike_for_less_than_1hour column')
pivot_table_male_female_3_devices_hike_for_less_than_1hour = df_male_female_3_devices.pivot_table(index='hike_for_less_than_1hour', columns='Gender', aggfunc='size', fill_value=0)
display(pivot_table_male_female_3_devices_hike_for_less_than_1hour)

# Create a pivot table of Male and Female rows with 3 devices in the combined column and hike_for_half_a_day column
print('\nMale and Female with 3 devices in hike_for_half_a_day column')
pivot_table_male_female_3_devices_hike_for_half_a_day = df_male_female_3_devices.pivot_table(index='hike_for_half_a_day', columns='Gender', aggfunc='size', fill_value=0)
display(pivot_table_male_female_3_devices_hike_for_half_a_day)

# Create a pivot table of Male and Female rows with 3 devices in the combined column and hike_for_less_than_a_day column
print('\nMale and Female with 3 devices in hike_for_less_than_a_day column')
pivot_table_male_female_3_devices_hike_for_less_than_a_day = df_male_female_3_devices.pivot_table(index='hike_for_less_than_a_day', columns='Gender', aggfunc='size', fill_value=0)
display(pivot_table_male_female_3_devices_hike_for_less_than_a_day)

# Create a pivot table of Male and Female rows with 3 devices in the combined column and hike_for_multple_days column
print('\nMale and Female with 3 devices in hike_for_multple_days column')
pivot_table_male_female_3_devices_hike_for_multiple_days = df_male_female_3_devices.pivot_table(index='hike_for_multiple_days', columns='Gender', aggfunc='size', fill_value=0)
display(pivot_table_male_female_3_devices_hike_for_multiple_days)


# Create a pivot table of Male and Female rows with 3 devices in the combined column and easy_hike column
print('\nMale and Female with 3 devices in easy_hike column')
pivot_table_male_female_3_devices_easy_hike = df_male_female_3_devices.pivot_table(index='easy_hike', columns='Gender', aggfunc='size', fill_value=0)
display(pivot_table_male_female_3_devices_easy_hike)

# Create a pivot table of Male and Female rows with 3 devices in the combined column and difficult_hike column
print('\nMale and Female with 3 devices in difficult_hike column')
pivot_table_male_female_3_devices_difficult_hike = df_male_female_3_devices.pivot_table(index='difficult_hike', columns='Gender', aggfunc='size', fill_value=0)
display(pivot_table_male_female_3_devices_difficult_hike)

# Create a bar graph of pivot_table_male_female_3_devices_difficult_hike
pivot_table_male_female_3_devices_difficult_hike.plot(kind='bar')

# Add labels and title
plt.xlabel('Difficulty of Hike')
plt.ylabel('Count')
plt.title('Gender with 3 Devices - Difficulty of Hike')
# Show the plot
plt.show()


# Create a pivot table of Male and Female rows with 3 devices in the combined column and Gender column
print('\nMale and Female with 3 devices in Gender column')
pivot_table_male_female_3_devices_gender = df_male_female_3_devices.pivot_table(index='Gender', columns='combined', aggfunc='size', fill_value=0)
display(pivot_table_male_female_3_devices_gender)

#calculate the total count
total_count = pivot_table_male_female_3_devices_gender.sum().sum()

# Calculate the percentage of each category
pivot_table_male_female_3_devices_gender_percentage =(pivot_table_male_female_3_devices_gender/total_count) * 100

# Create a bar graph of pivot_table_male_female_3_devices_gender_percentage
pivot_table_male_female_3_devices_gender_percentage.plot(kind='bar')

# Add labels and title
plt.xlabel('Gender')
plt.ylabel('Percentage')
plt.title('Gender with 3 Devices - Combined')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)

# Add data labels to the bars
for p in plt.gca().patches:
    plt.text(p.get_x() * 1.005, p.get_height() * 1.005, '{:.1f}%'.format(p.get_height()), fontsize=8, color='black')

# Show the plot
plt.show()


# Get the top 10 common device usage among men and women
top_4_common_devices = df_male_female_3_devices['combined'].value_counts().head(4)
display(top_4_common_devices)


## Create a pivot table of Male and Female rows with top 10 common devices
print('\nMale and Female with top 7 common devices')
pivot_table_male_female_top_10_common_devices = df_male_female_3_devices.pivot_table(index='combined', columns='Gender', aggfunc='size', fill_value=0)
pivot_table_male_female_top_10_common_devices = pivot_table_male_female_top_10_common_devices.loc[top_4_common_devices.index]
display(pivot_table_male_female_top_10_common_devices)

# Create a pivot table of Male and Female rows with top 10 common devices
print('\nMale and Female with top 10 common devices')
pivot_table_male_female_top_4_common_devices = df_male_female_3_devices.pivot_table(index='Gender', columns='combined', aggfunc='size', fill_value=0)

# Get the top 4 common devices
top_4_common_devices = pivot_table_male_female_top_4_common_devices.sum(axis=0).sort_values(ascending=False).head(4).index

# Filter the pivot table to only include the top 7 common devices
pivot_table_male_female_top_4_common_devices = pivot_table_male_female_top_4_common_devices[top_4_common_devices]


#calculate the percentage of each category
pivot_table_male_female_top_4_common_devices_percentage = pivot_table_male_female_top_4_common_devices.div(pivot_table_male_female_top_4_common_devices.sum(axis=1), axis=0) * 100
#create a bar graph of the pivot table as a bar graph with the number of devices as the x-axis and the count as the y-axis
pivot_table_male_female_top_4_common_devices_percentage.plot(kind='bar')
plt.xlabel('Number of Devices')
plt.ylabel('Percentage')
plt.title('Top 4 Common 3 Devices used')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=1)
# Add data labels to the bars
for p in plt.gca().patches:
    plt.text(p.get_x() * 1.005, p.get_height() * 1.005, '{:.1f}%'.format(p.get_height()), fontsize=8, color='black')
plt.show()
































































# Get the unique columns in the dataframe
columns = df.columns

# Define binary_cols
binary_cols = df.select_dtypes(include=[bool, int]).columns

# Create a dictionary to store the results
results = {}

for col in binary_cols:
    male_count = len(df[(df['Gender'] == 'Male') & (df[col] == 1)])
    female_count = len(df[(df['Gender'] == 'Female') & (df[col] == 1)])
    total_male = len(df[df['Gender'] == 'Male'])
    total_female = len(df[df['Gender'] == 'Female'])

    if male_count == 0 or female_count == 0:
        z, p = np.nan, np.nan
    else:
        z, p = proportions_ztest([male_count, female_count], [total_male, total_female])
    results[col] = {'z-statistic': z, 'p-value': p}
#print(results)
display(df.dtypes)













































# Create a new dataframe that contains rows with 1 in the number _of_devices column and the combined column
print('\n\n1 device and combined columns created')
df = pd.read_excel('Hiking_data_new.xlsx')
print('\n1 device and combined columns created')
df_1_devices_combined = df[df['number _of_devices'] == 1][['number _of_devices', 'combined']]
#print(df_1_devices_combined.head())

# create pivot table of the combined column with the number of devices as the rows and the count as the columns
print('\ncombined and 1 device in a pivot table')
pivot_table_combined = df_1_devices_combined.pivot_table(columns='number _of_devices', index='combined', aggfunc='size', fill_value=0)
display(pivot_table_combined)

# Calculate the percentage of each combined category for 1 device
pivot_table_combined_percentage = pivot_table_combined.div(pivot_table_combined.sum(axis=1), axis=0) * 100
pivot_table_combined_percentage = pivot_table_combined_percentage.round(1)
pivot_table_combined_percentage = pivot_table_combined_percentage.astype(str) + '%'
print(pivot_table_combined_percentage)







##------------------------------------##  ##------------------------------------##--------------------------------#
# Create a new column 'Smart Watch Found' based on the values in the 'combined' column
df['Smart Watch Found'] = df['combined'].str.contains('smart watch', case=False)
df['Smart Watch Found'] = df['Smart Watch Found'].astype(object)  # <-- Fix: cast to object before assigning strings

# Replace True with 'Smart watch' in the 'Smart Watch Found' column
df.loc[df['Smart Watch Found'] == True, 'Smart Watch Found'] = 'Smart watch'

# Replace False with 'Not smart watch' in the 'Smart Watch Found' column
df.loc[df['Smart Watch Found'] == False, 'Smart Watch Found'] = 'Not smart watch'

# Create a new dataframe 'watch' that contains rows with 'smart watch' in the 'combined' column
watch = df[df['combined'].str.contains('smart watch', case=False)]
#display(watch)

# Display the updated dataframe
display(df['Smart Watch Found'].head())


# Create a pivot table of the 'Smart Watch Found' column
print('\nSmart Watch Found')
pivot_table_watch = df['Smart Watch Found'].value_counts().to_frame().T
display(pivot_table_watch)
display(pivot_table)

# Create a bar graph for the pivot table watch
pivot_table_watch.plot(kind='bar')
plt.xlabel('Smart Watch Found')
plt.ylabel('Count')
plt.title('Smart Watch Found')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
# Add data labels to the bars
ax = plt.gca()
for p in ax.patches:
    percentage = f"{p.get_height() / pivot_table_watch.sum().sum() * 100:.1f}%"
    ax.annotate(percentage, (p.get_x() + p.get_width() / 2, p.get_height() + 0.3), ha='center', va='bottom')
plt.show()

# -------------------------------------------------#




display('-------- correlation b/n headphones and hike alone ----------\n')

import pandas as pd
df = pd.read_excel('hiking_data_v2.xlsx')

# Create a new column 'Headphones' based on the values in the 'combined' column that contain 'headphones/earbuds'
df['Headphones'] = df['combined'].str.contains('headphones/earbuds', case=False) 
df['Headphones'] = df['Headphones'].astype(object)  # <-- Fix: cast to object before assigning strings

# Replace True with 'Headphones' in the 'Headphones' column
df.loc[df['Headphones'] == True, 'Headphones'] = 'Headphones'  

# Replace False with 'No Headphones' in the 'Headphones' column
df.loc[df['Headphones'] == False, 'Headphones'] = 'No Headphones'

# Create a new dataframe 'headphones' that contains rows with 'headphones/earbuds' in the 'combined' column
headphones = df[df['combined'].str.contains('headphones/earbuds', case=False)]
#display(headphones)

# Create a pivot table of Headphones and like_to_hike_alone
print('\nHeadphones and like_to_hike_alone')
pivot_table_headphones_like_to_hike_alone = df.pivot_table(index='Headphones', columns='like_to_hike_alone', aggfunc='size', fill_value=0)
# Reorder the columns of the pivot table
pivot_table_headphones_like_to_hike_alone = pivot_table_headphones_like_to_hike_alone[['Strongly disagree', 'Somewhat disagree', 'Neither agree nor disagree', 'Somewhat agree', 'Strongly agree']]
display(pivot_table_headphones_like_to_hike_alone)

# Calculate the total count
total_count = pivot_table_headphones_like_to_hike_alone.sum().sum()

# Calculate the percentage of each category
pivot_table_headphones_like_to_hike_alone_percentage = pivot_table_headphones_like_to_hike_alone.div(pivot_table_headphones_like_to_hike_alone.sum(axis=1), axis=0) * 100


# Create a bar graph for the pivot table of Headphones and like_to_hike_alone
pivot_table_headphones_like_to_hike_alone_percentage.plot(kind='bar')
plt.xlabel('')
plt.ylabel('Percentage')
plt.title('Headphones and Like to Hike Alone')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
# Add data labels to the bars
for p in plt.gca().patches:
    plt.text(p.get_x() * 1.005, p.get_height() * 1.005, '{:.1f}%'.format(p.get_height()), fontsize=8, color='black')
plt.show()

import scipy.stats as stats

# Perform chi-square test of independence
chi2, p, dof, expected = stats.chi2_contingency(pivot_table_headphones_like_to_hike_alone)

# Print the results
print("Chi-square statistic:", chi2)
print("p-value:", p)
print("Degrees of freedom:", dof)
print("Expected frequencies:", expected)



display('-------- correlation b/n headphones and hike for meditation ----------\n')

# Create a pivot table of Headphones and hike_for_meditation
print('\nHeadphones and hike_for_meditation')
pivot_table_headphones_hike_for_meditation = df.pivot_table(index='Headphones', columns='hike_for_mediation', aggfunc='size', fill_value=0)
# Reorder the columns of the pivot table
pivot_table_headphones_hike_for_meditation = pivot_table_headphones_hike_for_meditation[['Strongly disagree', 'Somewhat disagree', 'Neither agree nor disagree', 'Somewhat agree', 'Strongly agree']]
display(pivot_table_headphones_hike_for_meditation)

#calculate the total count
total_count = pivot_table_headphones_hike_for_meditation.sum().sum()
#calculate the percentage of each category
pivot_table_headphones_hike_for_meditation_percentage = pivot_table_headphones_hike_for_meditation.div(pivot_table_headphones_hike_for_meditation.sum(axis=1), axis=0) * 100


# Create a bar graph for the pivot table of Headphones and hike_for_meditation percentage
pivot_table_headphones_hike_for_meditation_percentage.plot(kind='bar')
plt.xlabel('')
plt.ylabel('Percentage')
plt.title('Headphones and Hike for Meditation')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
# Add data labels to the bars
for p in plt.gca().patches:
    plt.text(p.get_x() * 1.005, p.get_height() * 1.005, '{:.1f}%'.format(p.get_height()), fontsize=8, color='black')
plt.show()

# Perform chi-square test of independence
chi2, p, dof, expected = stats.chi2_contingency(pivot_table_headphones_hike_for_meditation)

# Print the results
print("Chi-square statistic:", chi2)
print("p-value:", p)
print("Degrees of freedom:", dof)
print("Expected frequencies:", expected)







#-------------------------------------------------#
# Define a mapping from categories to numerical values of like to hike alone column
mapping = {
    'Strongly disagree': 1,  # 1 for strongly disagree
    'Somewhat disagree': 2,  # 2 for somewhat disagree
    'Neither agree nor disagree': 3, # 3 for neither agree nor disagree
    'Somewhat agree': 4, # 4 for somewhat agree
    'Strongly agree': 5, # 5 for strongly agree
}
print ("\nreplacement of like to hike alone to numeric value...")
df['like_to_hike_alone_value'] = df['like_to_hike_alone'].replace(mapping).infer_objects(copy=False)  # suppress FutureWarning
# Step 1: Print the non-finite values
print("Non-finite values in 'like_to_hike_alone_value':")
print(df['like_to_hike_alone_value'][~np.isfinite(df['like_to_hike_alone_value'])])

# Step 2: Replace non-finite values with a placeholder (e.g., 0)
df['like_to_hike_alone_value'] = df['like_to_hike_alone_value'].replace([np.inf, -np.inf, np.nan], 0)


df['like_to_hike_alone_value'] = df['like_to_hike_alone_value'].astype(int)
print ("Done")
print('\nlike_to_hike_alone_value')

# Create a pivot table of the like_to_hike_alone_value column
pivot_table_like_to_hike_alone_value = df.pivot_table(index='like_to_hike_alone_value', aggfunc='size', fill_value=0)
print(pivot_table_like_to_hike_alone_value)

pivot_table_headphones_like_to_hike_alone_value = df.pivot_table(index='Headphones', columns='like_to_hike_alone_value', aggfunc='size', fill_value=0)
display(pivot_table_headphones_like_to_hike_alone_value)
# Reorder the columns of the pivot table
# pivot_table_headphones_like_to_hike_alone_value = pivot_table_headphones_like_to_hike_alone_value[['Strongly disagree', 'Somewhat disagree', 'Neither agree nor disagree', 'Somewhat agree', 'Strongly agree']]












# Perform the t-test
t_stat, p_value = stats.ttest_ind(pivot_table_headphones_like_to_hike_alone_value.loc['Headphones'], pivot_table_headphones_like_to_hike_alone_value.loc['No Headphones'], equal_var=False)

# Print the t-statistic and p-value
print(f"t-statistic: {t_stat:.2f}")
print(f"p-value: {p_value:.4f}")
print('\n')

# Set significance level
alpha = 0.05

# Determine if the result is statistically significant
is_significant = p_value < alpha

# Determine if the result is statistically significant
if p_value < alpha:
    print("There is a statistically significant difference between headphones and like to hike alone value")
    print("**Statistically significant difference** between male and female participants' like_to_hike_alone_value values.")
else:
    print("There is no statistically significant difference between headphones and like to hike alone value")
print('\n')

# Group by Headphones and calculate summary statistics
summary_stats = pivot_table_headphones_like_to_hike_alone_value.T.describe()
display(summary_stats)

# Create a bar graph of the like_to_hike_alone_value column for headphones and all participants
pivot_table_headphones_like_to_hike_alone_value.T.boxplot()
plt.xlabel('Like to Hike Alone (numeric)')
plt.ylabel('Count')
plt.title('Like to Hike Alone by Headphones')
plt.show()
print('done\n')








#-------------------------------------------------#

df = df.infer_objects()
# Get the top 10 common device usage among men and women
top_4_common_devices = df_male_female_3_devices['combined'].value_counts().head(4)
display(top_4_common_devices)

## Create a pivot table of Male and Female rows with top 10 common devices
print('\nMale and Female with top 7 common devices')
pivot_table_male_female_top_10_common_devices = df_male_female_3_devices.pivot_table(index='combined', columns='Gender', aggfunc='size', fill_value=0)
pivot_table_male_female_top_10_common_devices = pivot_table_male_female_top_10_common_devices.loc[top_4_common_devices.index]
display(pivot_table_male_female_top_10_common_devices)

# Create a pivot table of Male and Female rows with top 10 common devices
print('\nMale and Female with top 10 common devices')
pivot_table_male_female_top_4_common_devices = df_male_female_3_devices.pivot_table(index='Gender', columns='combined', aggfunc='size', fill_value=0)

# Get the top 4 common devices
top_4_common_devices = pivot_table_male_female_top_4_common_devices.sum(axis=0).sort_values(ascending=False).head(4).index
# ...existing code...

