import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency
from scipy.stats import chi2
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.stats.weightstats import ztest
from statsmodels.stats.proportion import proportion_confint
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from IPython.display import display
from scipy.stats import skew, kurtosis
import matplotlib.patches as mpatches
import seaborn as sns
from skimpy import skim 

display("---------------- Apps ---------\n")
# Load only Sheet 2
df = pd.read_excel("hiking_data_v3.xlsx", sheet_name="AppUpdate")

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

df = pd.read_excel('hiking_data_v3.xlsx', sheet_name='DevicesUpdate')


# display(df.head())  # Display first few rows

# device_counts_per_column = df.astype(str).apply(pd.Series.value_counts).fillna(0).astype(int)
device_counts_per_column = df.drop(columns=['Count'], errors='ignore').astype(str).apply(pd.Series.value_counts).fillna(0).astype(int)
device_counts_per_column = device_counts_per_column.loc[~device_counts_per_column.index.str.lower().str.contains('nan')]

total_device_counts_per_row = device_counts_per_column.sum(axis=1)

#devices carried by the participants
top_devices = total_device_counts_per_row.nlargest(13)
display('top devices', top_devices)

#percentage of devices in the top_devices
top_devices_percentage = total_device_counts_per_row.nlargest(13) / len(df) * 100
top_devices_percentage = top_devices_percentage.round(1).astype(str) + '%'
display('top devices percentage', top_devices_percentage)

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

# print('\n-----------------device counts per column------------------')
# display(device_counts_per_column)

print('\n-----------------device 2 combination------------------')
df = pd.read_excel('hiking_data_v3.xlsx', sheet_name='2devices')

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

plt.figure(figsize=(8, 5))
device_2_colors = plt.cm.tab20(range(len(device_2_counts)))  # Use a colormap for consistent coloring
device_2_counts.plot(kind='bar', color=device_2_colors, width=0.90)  # Increase width to reduce distance between bars
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
df = pd.read_excel('hiking_data_v3.xlsx', sheet_name='2devCount')

# display(df.head())  # Display first few rows

# Count the number of devices in the '2_Devices' column
deviceTwo_counts = df['Two_Devices'].value_counts()
# display('device 2 count', deviceTwo_counts)

# percentage of devices in the 2devices column
deviceTwo_percentage = df['Two_Devices'].value_counts(normalize=True) * 100
deviceTwo_percentage = deviceTwo_percentage.round(1).astype(str) + '%'
display('devices in 2 device combination',deviceTwo_percentage.round(1))

plt.figure(figsize=(8, 5))
device2CountColors = plt.cm.tab20(range(len(device_2_counts)))  # Use a colormap for consistent coloring
deviceTwo_counts.plot(kind='bar', color=device2CountColors, width=0.90)  # Increase width to reduce distance between bars
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
df = pd.read_excel('hiking_data_v3.xlsx', sheet_name='3devices')

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

plt.figure(figsize=(8, 5))
device_3_colors = plt.cm.tab20(range(len(device_3_count)))  # Use a colormap for consistent coloring

device_3_count.plot(kind='bar', color=device_3_colors, width=0.90)  # Increase width to reduce distance between bars
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
df = pd.read_excel('hiking_data_v3.xlsx', sheet_name='3devCount')


# display(df.head())  # Display first few rows


# Count the number of devices in the '3_Devices' column
deviceThree_counts = df['Three_Devices'].value_counts()
# display('device 3 count', deviceThree_counts)

# percentage of devices in the 3devices column
deviceThree_percentage = df['Three_Devices'].value_counts(normalize=True) * 100
deviceThree_percentage = deviceThree_percentage.round(1).astype(str) + '%'
display('device 3 count percentage', deviceThree_percentage.round(1))

plt.figure(figsize=(8, 5))
device3CountColors = plt.cm.tab20(range(len(deviceThree_counts)))  # Use a colormap for consistent coloring

deviceThree_counts.plot(kind='bar', color=device3CountColors, width=0.90)  # Increase width to reduce distance between bars
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
df = pd.read_excel('hiking_data_v3.xlsx', sheet_name='Edit(2)')
#print(df.head()) 

print ("------these are the gender counts------")
# Count the number of respondants in the dataset by 
gender_counts = df['Gender'].value_counts()
print('\n-----------------participant counts------------------')
display(gender_counts)

# Calculate the percentage of each row in the Gender column
gender_percentage = df['Gender'].value_counts(normalize=True) * 100
gender_percentage = gender_percentage.round(1).astype(str) + '%'
display(gender_percentage)


print('---------------- end of participant counts ------------------\n\n')
#------------------------------------------------#

#------------------------------------------------#

# Split the devices in the name_of_devices column by comma and create new # Split the devices in the name_of_devices column by comma and create new columns
devices_df = df['combined'].str.split(',', expand=True).add_prefix('Device')

# Remove leading and trailing whitespace from the new columns
devices_df = devices_df.apply(lambda x: x.str.strip())

# Concatenate the original dataframe with the new devices dataframe
df = pd.concat([df, devices_df], axis=1)


#---------------testing--------------#
"under testing for consistency of the data. especially with the color coding of the devices"

# Create a table with the device names and their counts
table = pd.DataFrame({'Device': top_devices.index, 'Count': top_devices.values})




print('\n\n\n-----------------hiking preferences------------------\n\n\n')





print('-----------------hiking duration between 3 and 5 devices------------------')
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
pd.set_option('future.no_silent_downcasting', True)
#display (df['hiking_duration_numeric'].head(5))
print ("Done")

# Calculate the mean
avg_hiking_duration = df['hiking_duration_numeric'].mean()
print ("mean of all respondants: ")
display(avg_hiking_duration)


print('comparison between 3 and 5 devices')

# make a new dataframe that just contains the rows in df that have 3 in the number_of_devices column
df_3_devices = df[df['number _of_devices'] == 3]
#display (df_1_devices.head(10))
# and get a new dataframe with just the rows with 5 in number_of_devices
df_5_devices = df[df['number _of_devices'] == 5]
#display (df_5_devices.head(10))

# Calculate the mean of the hiking duration for both groups
avg_hiking_duration_3_devices = df_3_devices['hiking_duration_numeric'].mean()
avg_hiking_duration_5_devices = df_5_devices['hiking_duration_numeric'].mean()
print ("mean of 3 devices: ")
display(avg_hiking_duration_3_devices)
print ("mean of 5 devices: ")
display(avg_hiking_duration_5_devices)




print ("---------------grahp of the hiking duration for 1 and 5 devices-----------------")
# Graph histogram of 3 devices and 5 devices
plt.hist(df_3_devices['hiking_duration_numeric'], bins=10, alpha=0.5, label='3 devices')
plt.hist(df_5_devices['hiking_duration_numeric'], bins=10, alpha=0.5, label='5 devices')
plt.xlabel('Hiking Frequency')
plt.ylabel('Count')
plt.legend()
plt.show()
print('done')

# Step 3: Create a pivot table
pivot_table = pd.pivot_table(df, values='hiking_duration_numeric', index='number _of_devices', aggfunc='mean')
# display(pivot_table)
                   

# Create a pivot table with devices as columns and frequencies as rows
pivot_table = df[df['number _of_devices'].isin([3, 5])].pivot_table(index='hiking_duration_numeric', columns='number _of_devices', aggfunc='size', fill_value=0)

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
    plt.text(p.get_x() + p.get_width()/2, p.get_height() * 1.015, '{}'.format(p.get_height()),ha='center', fontsize=9, color='black')
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
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='number _of_devices', hue='number _of_devices', palette='tab10', legend=False)
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
plt.figure(figsize=(11, 5))
colors = plt.cm.tab10(range(len(pivot_table_gender_row_percent_numeric.columns)))
ax = pivot_table_gender_row_percent_numeric.T.plot(kind='bar', width=0.95, color=colors)
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





print('--------------- version on 2 devices starting-----------------------------')

print('\n two devices\n')

import pandas as pd
df = pd.read_excel('hiking_data_v3.xlsx', sheet_name='Edit(2)')

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
df = pd.read_excel('hiking_data_v3.xlsx', sheet_name='Edit(2)')
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
df.to_excel('hiking_data_v3_updated.xlsx', sheet_name='devices', index=False)

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


































df = pd.read_excel('hiking_data_v3.xlsx', sheet_name='Edit(2)')



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

# # Create a pivot table of Male and Female row in Gender column with 2 devices in the combine column
# pivot_table_combined_2_devices = df[(df['Gender'].isin(['Male', 'Female'])) & (df['number _of_devices'] == 2)].pivot_table(index='Gender', columns='combined', aggfunc='size', fill_value=0)
# display(pivot_table_combined_2_devices)

# #calculate the percentage of each category
# pivot_table_combined_2_devices_percentage = pivot_table_combined_2_devices.div(pivot_table_combined_2_devices.sum(axis=1), axis=0) * 100

# # Create a bar chart of the pivot table as a bar graph with the number of devices as the x-axis and the count as the y-axis
# pivot_table_combined_2_devices_percentage.plot(kind='bar')
# #plt.xlabel('Number of Devices')
# plt.ylabel('Percentage')
# plt.title('2 Devices used by Male and Female')
# legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
# # Add data labels to the bars
# # for p in plt.gca().patches:
# #     plt.text(p.get_x() * 1.005, p.get_height() * 1.005, '{:.1f}%'.format(p.get_height()), fontsize=8, color='black')
# plt.show()
# print('done\n')
# print('\nMale and Female with 2 devices in combined column')
# pivot_table_combined_2_devices = df[(df['Gender'].isin(['Male', 'Female'])) & (df['number _of_devices'] == 2)].pivot_table(columns='Gender', index='combined', aggfunc='size', fill_value=0)
# display(pivot_table_combined_2_devices)


# # Calculate the percentage of men and women who use "Smartphone, Portable charger/battery"
# percentage_male_smartphone = f"{pivot_table_combined_2_devices.loc['Smartphone, Portable charger/battery', 'Male'] / gender_counts['Male'] * 100:.1f}%"
# percentage_female_smartphone = f"{pivot_table_combined_2_devices.loc['Smartphone, Portable charger/battery', 'Female'] / gender_counts['Female'] * 100:.1f}%"

# print("Percentage of men who use 'Smartphone, Portable charger/battery':", percentage_male_smartphone)
# print("Percentage of women who use 'Smartphone, Portable charger/battery':", percentage_female_smartphone)


# percentage_male_smarthead = f"{pivot_table_combined_2_devices.loc['Smartphone, Headphones/earbuds', 'Male'] / gender_counts['Male'] * 100:.1f}%"
# percentage_female_smarthead = f"{pivot_table_combined_2_devices.loc['Smartphone, Headphones/earbuds', 'Female'] / gender_counts['Female'] * 100:.1f}%"

# print("Percentage of men who use 'Smartphone, Headphones/earbuds':", percentage_male_smarthead)
# print("Percentage of women who use 'Smartphone, Headphones/earbuds':", percentage_female_smarthead)



# # # Filter the dataframe to include only rows where the gender is 'Female' and the combined column is 'Smartphone, Portable charger/battery', and print the resulting dataframe
# print('\nFemale with "Smartphone, Portable charger/battery"')
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# female_smartphone = df[(df['Gender'] == 'Female') & (df['combined'] == 'Smartphone, Portable charger/battery')]
# #display(female_smartphone)


# # Create a pivot table of Female rows with 'Smartphone, Portable charger/battery' in the 'combined' column, 'easy_hike' and 'Difficult_hike' in the 'Safety_emergency' column
# pivot_table_female_smartphone = female_smartphone.pivot_table(index='combined', columns='Safety_emergency', values=['easy_hike', 'difficult_hike'], aggfunc='size', fill_value=0)
# display(pivot_table_female_smartphone)

# # Create a pivot table of Female rows with 'Smartphone, Portable charger/battery' in the 'combined' column and 'hiking_duration' column
# pivot_table_female_hiking_duration = female_smartphone.pivot_table(index='hiking_duration', columns='combined', aggfunc='size', fill_value=0)
# display(pivot_table_female_hiking_duration)

# # Create a pivot table of Female rows with 'Smartphone, Portable charger/battery' in the 'combined' column and 'like to hike alone' column
# pivot_table_female_like_to_like_alone = female_smartphone.pivot_table(index='like_to_hike_alone', columns='combined', aggfunc='size', fill_value=0)
# display(pivot_table_female_like_to_like_alone)

# # Create a pivot table of Female rows with 'Smartphone, Portable charger/battery' in the 'combined' column and 'like to hike in group' column
# pivot_table_female_in_group = female_smartphone.pivot_table(index= 'like_to_hike_in_group', columns='combined', aggfunc='size', fill_value=0)
# display(pivot_table_female_in_group)

# # Create a pivot table of Female rows with 'Smartphone, Portable charger/battery' in the 'combined' column and 'like to hike near home' column
# pivot_table_female_near_home = female_smartphone.pivot_table(index= 'like_to_hike_near_home', columns='combined', aggfunc='size', fill_value=0)
# display(pivot_table_female_near_home)

# # Create a pivot table of Female rows with 'Smartphone, Portable charger/battery' in the 'combined' column and 'hike while traveling' column
# pivot_table_female_while_traveling = female_smartphone.pivot_table(index= 'hike_while_traveling', columns='combined', aggfunc='size', fill_value=0)
# display(pivot_table_female_while_traveling)

# # Create a pivot table of Female rows with 'Smartphone, Portable charger/battery' in the 'combined' column and 'hike for health' column
# pivot_table_female_health = female_smartphone.pivot_table(index= 'hike_for_health', columns='combined', aggfunc='size', fill_value=0)
# display(pivot_table_female_health)

# # Create a pivot table of Female rows with 'Smartphone, Portable charger/battery' in the 'combined' column and 'hike for fun' column
# pivot_table_female_fun = female_smartphone.pivot_table(index= 'hike_for_fun', columns='combined', aggfunc='size', fill_value=0)
# display(pivot_table_female_fun)

# # Create a pivot table of Female rows with 'Smartphone, Portable charger/battery' in the 'combined' column and 'hike for less than 1hr' column
# pivot_table_female_1hr = female_smartphone.pivot_table(index= 'hike_for_less_than_1hour', columns='combined', aggfunc='size', fill_value=0)
# display(pivot_table_female_1hr)

# # Create a pivot table of Female rows with 'Smartphone, Portable charger/battery' in the 'combined' column and 'hike for half a day' column
# pivot_table_female_half_day = female_smartphone.pivot_table(index= 'hike_for_half_a_day', columns='combined', aggfunc='size', fill_value=0)
# display(pivot_table_female_half_day)

# # Create a pivot table of Female rows with 'Smartphone, Portable charger/battery' in the 'combined' column and 'hike for less than a day' column
# pivot_table_female_less_day = female_smartphone.pivot_table(index= 'hike_for_less_than_a_day', columns='combined', aggfunc='size', fill_value=0)
# display(pivot_table_female_less_day)


# #Get the top 5 common 2 devices usage among men and women
# print('\nTop 5 common 2 devices usage')
# top_5_devices = df[df['number _of_devices'] == 2]['combined'].value_counts().head(5)
# display(top_5_devices)

# #create a pivot table of male and female rows with top 5 common 2 devices usage
# print('\nMale and Female with top 5 common 2 devices usage')
# pivot_table_top_5_devices = df[(df['number _of_devices'] == 2) & (df['combined'].isin(top_5_devices.index))].pivot_table(index='Gender', columns = 'combined', aggfunc='size', fill_value=0)
# display(pivot_table_top_5_devices)

# #calculate the percentage of each category
# pivot_table_top_5_devices_percentage = pivot_table_top_5_devices.div(pivot_table_top_5_devices.sum(axis=1), axis=0) * 100
# # Create a bar chart of the pivot table as a bar graph with the number of devices as the x-axis and the count as the y-axis
# pivot_table_top_5_devices_percentage.plot(kind='bar', width=0.8)
# plt.xlabel('Number of Devices')
# plt.ylabel('Percentage')
# plt.title('Top 5 common 2 devices usage')
# legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
# # Add data labels to the bars
# for p in plt.gca().patches:
#     plt.text(p.get_x() * 1.005, p.get_height() * 1.007, '{:.1f}%'.format(p.get_height()), fontsize=9, color='black')  
# plt.show()
# print('done\n')







































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

df = pd.read_excel('hiking_data_v3.xlsx', sheet_name='Edit(2)')
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









# # Create a new dataframe that contains rows with 3 in the number _of_devices column and the combined column
# df_3_devices_combined = df[df['number _of_devices'] == 3][['number _of_devices', 'combined']]

# # Create a pivot table of 3 devices in number _of_devices and combined column
# pivot_table_3_devices_combined = df_3_devices_combined.pivot_table(index='combined', columns='number _of_devices', aggfunc='size', fill_value=0)
# display(pivot_table_3_devices_combined)

# # Create a new dataframe that contains rows with Male and Female in the Gender column and 3 devices in the combined column
# df_male_female_3_devices = df[(df['Gender'].isin(['Male', 'Female'])) & (df['combined'] == '3 devices')]

# #--------------------------------------------------------------

# # Create a new dataframe that contains rows with Male and Female in the Gender column and 3 devices in the number _of_devices column
# df_male_female_3_devices = df[(df['Gender'].isin(['Male', 'Female'])) & (df['number _of_devices'] == 3)]

# # Create a pivot table of Male and Female rows with 3 devices in the number _of_devices column and hiking_duration column
# print('\nMale and female with 3 devices and hiking duration column')
# pivot_table_male_female_3_devices_hiking_duration = df_male_female_3_devices.pivot_table(index='hiking_duration', columns='Gender', aggfunc='size', fill_value=0)
# display(pivot_table_male_female_3_devices_hiking_duration)


# # Create a pivot table Male and Female rows with 3 devices in the combined column and like_to_hike_alone column
# print('\nMale and female with 3 devices in like to hike alone column')
# pivot_table_male_female_3_devices_like_to_hike_alone = df_male_female_3_devices.pivot_table(index='like_to_hike_alone', columns='Gender', aggfunc='size', fill_value=0)
# display(pivot_table_male_female_3_devices_like_to_hike_alone)

# # Create a pivot table Male and Female rows with 3 devices in the combined column and like_to_hike_in_group column
# print('\nMale and female with 3 devices in like to hike in group column')
# pivot_table_male_female_3_devices_like_to_hike_in_group = df_male_female_3_devices.pivot_table(index='like_to_hike_in_group', columns='Gender', aggfunc='size', fill_value=0)
# display(pivot_table_male_female_3_devices_like_to_hike_in_group)

# # Create a pivot table Male and Female rows with 3 devices in the combined column and like_to_hike_near_home column
# print('\nMale and female with 3 devices in like to hike near home column')
# pivot_table_male_female_3_devices_like_to_hike_near_home = df_male_female_3_devices.pivot_table(index='like_to_hike_near_home', columns='Gender', aggfunc='size', fill_value=0)
# display(pivot_table_male_female_3_devices_like_to_hike_near_home)

# # Create a pivot table of Male and Female rows with 3 devices in the combined column and hike_while_traveling column
# print('\nMale and Female with 3 devices in hike_while_traveling column')
# pivot_table_male_female_3_devices_hike_while_traveling = df_male_female_3_devices.pivot_table(index='hike_while_traveling', columns='Gender', aggfunc='size', fill_value=0)
# display(pivot_table_male_female_3_devices_hike_while_traveling)

# # Create a pivot table of Male and Female rows with 3 devices in the combined column and hike_for_health column
# print('\nMale and Female with 3 devices in hike_for_health column')
# pivot_table_male_female_3_devices_hike_for_health = df_male_female_3_devices.pivot_table(index='hike_for_health', columns='Gender', aggfunc='size', fill_value=0)
# display(pivot_table_male_female_3_devices_hike_for_health)

# # Create a pivot table of Male and Female rows with 3 devices in the combined column and hike_for_social_interaction column
# print('\nMale and Female with 3 devices in hike_for_social_interaction column')
# pivot_table_male_female_3_devices_hike_for_social_interaction = df_male_female_3_devices.pivot_table(index='hike_for_social_interaction', columns='Gender', aggfunc='size', fill_value=0)
# display(pivot_table_male_female_3_devices_hike_for_social_interaction)

# # Create a pivot table of Male and Female rows with 3 devices in the combined column and hike_for_fun column
# print('\nMale and Female with 3 devices in hike_for_fun column')
# pivot_table_male_female_3_devices_hike_for_fun = df_male_female_3_devices.pivot_table(index='hike_for_fun', columns='Gender', aggfunc='size', fill_value=0)
# display(pivot_table_male_female_3_devices_hike_for_fun)

# # Create a pivot table of Male and Female rows with 3 devices in the combined column and hike_for_meditation column
# print('\nMale and Female with 3 devices in hike_for_meditation column')
# pivot_table_male_female_3_devices_hike_for_mediation = df_male_female_3_devices.pivot_table(index='hike_for_mediation', columns='Gender', aggfunc='size', fill_value=0)
# display(pivot_table_male_female_3_devices_hike_for_mediation)

# # Create a pivot table of Male and Female rows with 3 devices in the combined column and hike_for_less_than_1hour column
# print('\nMale and Female with 3 devices in hike_for_less_than_1hour column')
# pivot_table_male_female_3_devices_hike_for_less_than_1hour = df_male_female_3_devices.pivot_table(index='hike_for_less_than_1hour', columns='Gender', aggfunc='size', fill_value=0)
# display(pivot_table_male_female_3_devices_hike_for_less_than_1hour)

# # Create a pivot table of Male and Female rows with 3 devices in the combined column and hike_for_half_a_day column
# print('\nMale and Female with 3 devices in hike_for_half_a_day column')
# pivot_table_male_female_3_devices_hike_for_half_a_day = df_male_female_3_devices.pivot_table(index='hike_for_half_a_day', columns='Gender', aggfunc='size', fill_value=0)
# display(pivot_table_male_female_3_devices_hike_for_half_a_day)

# # Create a pivot table of Male and Female rows with 3 devices in the combined column and hike_for_less_than_a_day column
# print('\nMale and Female with 3 devices in hike_for_less_than_a_day column')
# pivot_table_male_female_3_devices_hike_for_less_than_a_day = df_male_female_3_devices.pivot_table(index='hike_for_less_than_a_day', columns='Gender', aggfunc='size', fill_value=0)
# display(pivot_table_male_female_3_devices_hike_for_less_than_a_day)

# # Create a pivot table of Male and Female rows with 3 devices in the combined column and hike_for_multple_days column
# print('\nMale and Female with 3 devices in hike_for_multple_days column')
# pivot_table_male_female_3_devices_hike_for_multiple_days = df_male_female_3_devices.pivot_table(index='hike_for_multiple_days', columns='Gender', aggfunc='size', fill_value=0)
# display(pivot_table_male_female_3_devices_hike_for_multiple_days)


# # Create a pivot table of Male and Female rows with 3 devices in the combined column and easy_hike column
# print('\nMale and Female with 3 devices in easy_hike column')
# pivot_table_male_female_3_devices_easy_hike = df_male_female_3_devices.pivot_table(index='easy_hike', columns='Gender', aggfunc='size', fill_value=0)
# display(pivot_table_male_female_3_devices_easy_hike)

# # Create a pivot table of Male and Female rows with 3 devices in the combined column and difficult_hike column
# print('\nMale and Female with 3 devices in difficult_hike column')
# pivot_table_male_female_3_devices_difficult_hike = df_male_female_3_devices.pivot_table(index='difficult_hike', columns='Gender', aggfunc='size', fill_value=0)
# display(pivot_table_male_female_3_devices_difficult_hike)

# # Create a bar graph of pivot_table_male_female_3_devices_difficult_hike
# pivot_table_male_female_3_devices_difficult_hike.plot(kind='bar')

# # Add labels and title
# plt.xlabel('Difficulty of Hike')
# plt.ylabel('Count')
# plt.title('Gender with 3 Devices - Difficulty of Hike')
# # Show the plot
# plt.show()





# Create a new dataframe that contains rows with 1 in the number _of_devices column and the combined column
print('\n\n1 device and combined columns created')
df = pd.read_excel('hiking_data_v3.xlsx', sheet_name='Edit(2)')
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





print('\n\n----------Chi-square tests-------------------\n')

display('-------- correlation b/n headphones and hike alone ----------\n')

df = pd.read_excel('hiking_data_v3.xlsx', sheet_name='Edit(2)')

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
# Merge categories for 'like_to_hike_alone'
like_to_hike_alone_map = {
    'Strongly disagree': 'Disagree',
    'Somewhat disagree': 'Disagree',
    'Neither agree nor disagree': 'Neither agree nor disagree',
    'Somewhat agree': 'Agree',
    'Strongly agree': 'Agree'
}
df['like_to_hike_alone_merged'] = df['like_to_hike_alone'].map(like_to_hike_alone_map)

pivot_table_headphones_like_to_hike_alone = df.pivot_table(
    index='Headphones',
    columns='like_to_hike_alone_merged',
    aggfunc='size',
    fill_value=0
)[['Disagree', 'Neither agree nor disagree', 'Agree']]
display(pivot_table_headphones_like_to_hike_alone)

# Calculate the total count
total_count = pivot_table_headphones_like_to_hike_alone.sum().sum()

# Calculate the percentage of each category
pivot_table_headphones_like_to_hike_alone_percentage = pivot_table_headphones_like_to_hike_alone.div(pivot_table_headphones_like_to_hike_alone.sum(axis=1), axis=0) * 100


# Create a bar graph for the pivot table of Headphones and like_to_hike_alone

sns.set(style="whitegrid")
pivot_table_headphones_like_to_hike_alone_percentage.plot(kind='bar')
plt.xlabel('')
plt.ylabel('Percentage')
plt.title('Headphones and Like to Hike Alone')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
# Add data labels to the bars
for p in plt.gca().patches:
    plt.text(p.get_x() * 1.005, p.get_height() * 1.015, '{:.1f}%'.format(p.get_height()), fontsize=9, color='black')
plt.xticks(rotation=0)
plt.show()


# Perform chi-square test of independence
chi2, p, dof, expected = stats.chi2_contingency(pivot_table_headphones_like_to_hike_alone)

# Print the results
print("Chi-square statistic:", chi2)
print("p-value:", p)
print("Degrees of freedom:", dof)
print("Expected frequencies:", expected)



print('-------- correlation b/n headphones and hike for meditation ----------\n')

# Create a pivot table of Headphones and hike_for_meditation
print('\nHeadphones and hike_for_meditation')
pivot_table_headphones_hike_for_meditation = df.pivot_table(index='Headphones', columns='hike_for_mediation', aggfunc='size', fill_value=0)
# Reorder the columns of the pivot table
# Merge categories for 'hike_for_meditation'
hike_for_meditation_map = {
    'Strongly disagree': 'Disagree',
    'Somewhat disagree': 'Disagree',
    'Neither agree nor disagree': 'Neither agree nor disagree',
    'Somewhat agree': 'Agree',
    'Strongly agree': 'Agree'
}
pivot_table_headphones_hike_for_meditation = pivot_table_headphones_hike_for_meditation.rename(columns=hike_for_meditation_map)
# Group by new categories and sum
pivot_table_headphones_hike_for_meditation = pivot_table_headphones_hike_for_meditation.groupby(axis=1, level=0).sum()
pivot_table_headphones_hike_for_meditation = pivot_table_headphones_hike_for_meditation[['Disagree', 'Neither agree nor disagree', 'Agree']]
display(pivot_table_headphones_hike_for_meditation)

#calculate the total count
total_count = pivot_table_headphones_hike_for_meditation.sum().sum()
#calculate the percentage of each category
pivot_table_headphones_hike_for_meditation_percentage = pivot_table_headphones_hike_for_meditation.div(pivot_table_headphones_hike_for_meditation.sum(axis=1), axis=0) * 100


# Create a bar graph for the pivot table of Headphones and hike_for_meditation percentage
sns.set(style="whitegrid")
pivot_table_headphones_hike_for_meditation_percentage.plot(kind='bar')
plt.xlabel('')
plt.ylabel('Percentage')
plt.title('Headphones and Hike for Meditation')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
# Add data labels to the bars
for p in plt.gca().patches:
    plt.text(p.get_x() * 1.005, p.get_height() * 1.005, '{:.1f}%'.format(p.get_height()), fontsize=9, color='black')
    plt.xticks(rotation=0)
plt.show()

# Perform chi-square test of independence
chi2, p, dof, expected = stats.chi2_contingency(pivot_table_headphones_hike_for_meditation)

# Print the results
print("Chi-square statistic:", chi2)
print("p-value:", p)
print("Degrees of freedom:", dof)
print("Expected frequencies:", expected)


print('\n -------- correlation b/n headphone and like to hike in group ----------\n')
# Create a pivot table of Headphones and like_to_hike_in_group
pivot_table_headphones_like_to_hike_in_group = df.pivot_table(index='Headphones', columns='like_to_hike_in_group', aggfunc='size', fill_value=0)
# Reorder the columns of the pivot table
# Merge categories for 'like_to_hike_in_group'
like_to_hike_in_group_map = {
    'Strongly disagree': 'Disagree',
    'Somewhat disagree': 'Disagree',
    'Neither agree nor disagree': 'Neither agree nor disagree',
    'Somewhat agree': 'Agree',
    'Strongly agree': 'Agree'
}
pivot_table_headphones_like_to_hike_in_group = pivot_table_headphones_like_to_hike_in_group.rename(columns=like_to_hike_in_group_map)
# Group by new categories and sum
pivot_table_headphones_like_to_hike_in_group = pivot_table_headphones_like_to_hike_in_group.groupby(axis=1, level=0).sum()
pivot_table_headphones_like_to_hike_in_group = pivot_table_headphones_like_to_hike_in_group[['Disagree', 'Neither agree nor disagree', 'Agree']]
display(pivot_table_headphones_like_to_hike_in_group)
# Calculate the total count
total_count = pivot_table_headphones_like_to_hike_in_group.sum().sum()
# Calculate the percentage of each category
pivot_table_headphones_like_to_hike_in_group_percentage = pivot_table_headphones_like_to_hike_in_group.div(pivot_table_headphones_like_to_hike_in_group.sum(axis=1), axis=0) * 100
# Create a bar graph for the pivot table of Headphones and like_to_hike_in_group percentage
sns.set(style="whitegrid")
pivot_table_headphones_like_to_hike_in_group_percentage.plot(kind='bar')
plt.xlabel('')
plt.ylabel('Percentage')
plt.title('Headphones and Like to Hike in Group')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
# Add data labels to the bars
for p in plt.gca().patches: 
    plt.text(p.get_x() * 1.005, p.get_height() * 1.005, '{:.1f}%'.format(p.get_height()), fontsize=9, color='black')
    plt.xticks(rotation=0)
plt.show()

# Perform chi-square test of independence
chi2, p, dof, expected = stats.chi2_contingency(pivot_table_headphones_like_to_hike_in_group)
# Print the results
print("Chi-square statistic:", chi2)
print("p-value:", p)
print("Degrees of freedom:", dof)
print("Expected frequencies:", expected)









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










print('\n\n\n ---- devices and preferences ----\n\n\n')


print('\n ---- headphones and hike for meditation correlation ----\n')
# Create a pivot table of Headphones and hike_for_meditation
pivot_table_headphones_hike_for_meditation = df.pivot_table(index='Headphones', columns='hike_for_mediation', aggfunc='size', fill_value=0)
# Reorder the columns of the pivot table
#create "hike_for_meditation_num" column if it doesn't exist
if 'hike_for_mediation_num' not in df.columns:
    mapping = {
        'Strongly disagree': 1,
        'Somewhat disagree': 2,
        'Neither agree nor disagree': 3,
        'Somewhat agree': 4,
        'Strongly agree': 5,
    }
    df['hike_for_mediation_num'] = df['hike_for_mediation'].replace(mapping).infer_objects(copy=False)
    pd.set_option('future.no_silent_downcasting', True)
    df['hike_for_mediation_num'] = df['hike_for_mediation_num'].fillna(0).astype(int)
    
    # Create a pivot table of Headphones and hike_for_meditation
pivot_table_headphones_hike_for_meditation = df.pivot_table(index='Headphones', columns='hike_for_mediation_num', aggfunc='size', fill_value=0)
print('Pivot table of Headphones and hike_for_meditation:')
print(pivot_table_headphones_hike_for_meditation)
#set significance level
alpha = 0.05
# perform the t-test
t_stat, p_value = ttest_ind(pivot_table_headphones_hike_for_meditation.loc['Headphones'],
                            pivot_table_headphones_hike_for_meditation.loc['No Headphones'],
                            equal_var=False)
# Print the t-statistic and p-value
print(f"t-statistic: {t_stat:.2f}")
print(f"p-value: {p_value:.4f}")
print("Significant difference:", p_value < alpha)

#Summary statistics for each group 
summary_stats = pivot_table_headphones_hike_for_meditation.T.describe()
# display(summary_stats)

print('\n\n ---- headphones and like to hike in group correlation ----\n')
#create a "like_to_hike_in_group_num" column if it doesn't exist
if 'like_to_hike_in_group_num' not in df.columns:
    mapping = {
        'Strongly disagree': 1,
        'Somewhat disagree': 2,
        'Neither agree nor disagree': 3,
        'Somewhat agree': 4,
        'Strongly agree': 5,
    }
    df['like_to_hike_in_group_num'] = df['like_to_hike_in_group'].replace(mapping).infer_objects(copy=False)
    pd.set_option('future.no_silent_downcasting', True)
    df['like_to_hike_in_group_num'] = df['like_to_hike_in_group_num'].fillna(0).astype(int)
# Create a pivot table of Headphones and like_to_hike_in_group
pivot_table_headphones_like_to_hike_in_group = df.pivot_table(index='Headphones', columns='like_to_hike_in_group_num', aggfunc='size', fill_value=0)
print('Pivot table of Headphones and like_to_hike_in_group:')
print(pivot_table_headphones_like_to_hike_in_group)
# Set significance level
alpha = 0.05
# Perform the t-test
t_stat, p_value = ttest_ind(pivot_table_headphones_like_to_hike_in_group.loc['Headphones'],
                            pivot_table_headphones_like_to_hike_in_group.loc['No Headphones'],
                            equal_var=False)
# Print the t-statistic, p-value, significance, and summary statistics
print(f"t-statistic: {t_stat:.2f}")
print(f"p-value: {p_value:.4f}")
print("Significant difference:", p_value < alpha)
# Summary statistics for each group
summary_stats = pivot_table_headphones_like_to_hike_in_group.T.describe()
# display(summary_stats)


print('\n\n ---- headphones and like to hike alone correlation ----\n')
#create a "like_to_hike_alone_num" column if it doesn't exist
if 'like_to_hike_alone_num' not in df.columns:
    mapping = {
        'Strongly disagree': 1,
        'Somewhat disagree': 2,
        'Neither agree nor disagree': 3,
        'Somewhat agree': 4,
        'Strongly agree': 5,
    }
    df['like_to_hike_alone_num'] = df['like_to_hike_alone'].replace(mapping).infer_objects(copy=False)
    pd.set_option('future.no_silent_downcasting', True)
    df['like_to_hike_alone_num'] = df['like_to_hike_alone_num'].fillna(0).astype(int)
# Create a pivot table of Headphones and like_to_hike_alone
pivot_table_headphones_like_to_hike_alone = df.pivot_table(index='Headphones', columns='like_to_hike_alone_num', aggfunc='size', fill_value=0)
print('Pivot table of Headphones and like_to_hike_alone:')
print(pivot_table_headphones_like_to_hike_alone)
# Set significance level
alpha = 0.05
# Perform the t-test
t_stat, p_value = ttest_ind(pivot_table_headphones_like_to_hike_alone.loc['Headphones'],
                            pivot_table_headphones_like_to_hike_alone.loc['No Headphones'],
                            equal_var=False)
# Print the t-statistic, p-value, significance, and summary statistics
print(f"t-statistic: {t_stat:.2f}")
print(f"p-value: {p_value:.4f}")
print("Significant difference:", p_value < alpha)
# Summary statistics for each group
summary_stats = pivot_table_headphones_like_to_hike_alone.T.describe()
# display(summary_stats)


print('\n\n ---- headphones and social interaction correlation ----\n')
#create a "hike_for_social_interaction_num" column if it doesn't exist
if 'hike_for_social_interaction_num' not in df.columns:
    mapping = {
        'Strongly disagree': 1,
        'Somewhat disagree': 2,
        'Neither agree nor disagree': 3,
        'Somewhat agree': 4,
        'Strongly agree': 5,
    }
    df['hike_for_social_interaction_num'] = df['hike_for_social_interaction'].replace(mapping).infer_objects(copy=False)
    pd.set_option('future.no_silent_downcasting', True)
    df['hike_for_social_interaction_num'] = df['hike_for_social_interaction_num'].fillna(0).astype(int)
# Create a pivot table of Headphones and hike_for_social_interaction
pivot_table_headphones_hike_for_social_interaction = df.pivot_table(index='Headphones', columns='hike_for_social_interaction_num', aggfunc='size', fill_value=0)
print('Pivot table of Headphones and hike_for_social_interaction:')
print(pivot_table_headphones_hike_for_social_interaction)
# Set significance level
alpha = 0.05
# Perform the t-test
t_stat, p_value = ttest_ind(pivot_table_headphones_hike_for_social_interaction.loc['Headphones'],
                            pivot_table_headphones_hike_for_social_interaction.loc['No Headphones'],
                            equal_var=False)
# Print the t-statistic, p-value, significance, and summary statistics
print(f"t-statistic: {t_stat:.2f}")
print(f"p-value: {p_value:.4f}")
print("Significant difference:", p_value < alpha)
# Summary statistics for each group
summary_stats = pivot_table_headphones_hike_for_social_interaction.T.describe()
# display(summary_stats)


print('\n\n ---- Portable charger/battery and hike for less than a day correlation ----\n')
# Create a pivot table of Portable charger/battery and hike_for_less_than_a_day
if 'hike_for_less_than_a_day_num' not in df.columns:
    mapping = {
        'Strongly disagree': 1,
        'Somewhat disagree': 2,
        'Neither agree nor disagree': 3,
        'Somewhat agree': 4,
        'Strongly agree': 5,
    }
    df['hike_for_less_than_a_day_num'] = df['hike_for_less_than_a_day'].replace(mapping).infer_objects(copy=False)
    pd.set_option('future.no_silent_downcasting', True)
    df['hike_for_less_than_a_day_num'] = df['hike_for_less_than_a_day_num'].fillna(0).astype(int)
# Create a new column 'Portable charger/battery' indicating presence in 'combined'
df['Portable charger/battery'] = df['combined'].str.contains('Portable charger/battery', case=False)
df['Portable charger/battery'] = df['Portable charger/battery'].map({True: 'Portable charger/battery', False: 'No Portable charger/battery'})

# Create a pivot table of Portable charger/battery and hike_for_less_than_a_day
pivot_table_portable_charger_hike_for_less_than_a_day = df.pivot_table(index='Portable charger/battery', columns='hike_for_less_than_a_day_num', aggfunc='size', fill_value=0)
print('Pivot table of Portable charger/battery and hike_for_less_than_a_day:')
print(pivot_table_portable_charger_hike_for_less_than_a_day)
# Set significance level
alpha = 0.05
# Perform the t-test
t_stat, p_value = ttest_ind(pivot_table_portable_charger_hike_for_less_than_a_day.loc['Portable charger/battery'],
                            pivot_table_portable_charger_hike_for_less_than_a_day.loc['No Portable charger/battery'],
                            equal_var=False)
# Print the t-statistic, p-value, significance, and summary statistics
print(f"t-statistic: {t_stat:.2f}")
print(f"p-value: {p_value:.4f}")
print("Significant difference:", p_value < alpha)
# Summary statistics for each group
summary_stats = pivot_table_portable_charger_hike_for_less_than_a_day.T.describe()
# display(summary_stats)


print('\n\n ---- Portable charger/battery and hike for multiple days correlation ----\n')
# Create a pivot table of Portable charger/battery and hike_for_multiple_days
if 'hike_for_multiple_days_num' not in df.columns:
    mapping = {
        'Strongly disagree': 1,
        'Somewhat disagree': 2,
        'Neither agree nor disagree': 3,
        'Somewhat agree': 4,
        'Strongly agree': 5,
    }
    df['hike_for_multiple_days_num'] = df['hike_for_multiple_days'].replace(mapping).infer_objects(copy=False)
    pd.set_option('future.no_silent_downcasting', True)
    df['hike_for_multiple_days_num'] = df['hike_for_multiple_days_num'].fillna(0).astype(int)
# Create a pivot table of Portable charger/battery and hike_for_multiple_days
pivot_table_portable_charger_hike_for_multiple_days = df.pivot_table(index='Portable charger/battery', columns='hike_for_multiple_days_num', aggfunc='size', fill_value=0)
print('Pivot table of Portable charger/battery and hike_for_multiple_days:')
print(pivot_table_portable_charger_hike_for_multiple_days)
# Set significance level
alpha = 0.05
# Perform the t-test
t_stat, p_value = ttest_ind(pivot_table_portable_charger_hike_for_multiple_days.loc['Portable charger/battery'],
                            pivot_table_portable_charger_hike_for_multiple_days.loc['No Portable charger/battery'],
                            equal_var=False)
# Print the t-statistic, p-value, significance, and summary statistics
print(f"t-statistic: {t_stat:.2f}")
print(f"p-value: {p_value:.4f}")
print("Significant difference:", p_value < alpha)
# Summary statistics for each group
summary_stats = pivot_table_portable_charger_hike_for_multiple_days.T.describe()
# display(summary_stats)

#display a table containing all the t_statistic, p-value, and significance for each of the above correlations
print('\n\n ---- devices and preferences section ----\n\n\n')
# Create a summary table for all the t-statistics, p-values, and significance

# Calculate and store t-statistics and p-values for each correlation
t_stat_headphones_meditation, p_value_headphones_meditation = ttest_ind(
    pivot_table_headphones_hike_for_meditation.loc['Headphones'],
    pivot_table_headphones_hike_for_meditation.loc['No Headphones'],
    equal_var=False
)
t_stat_headphones_group, p_value_headphones_group = ttest_ind(
    pivot_table_headphones_like_to_hike_in_group.loc['Headphones'],
    pivot_table_headphones_like_to_hike_in_group.loc['No Headphones'],
    equal_var=False
)
t_stat_headphones_alone, p_value_headphones_alone = ttest_ind(
    pivot_table_headphones_like_to_hike_alone.loc['Headphones'],
    pivot_table_headphones_like_to_hike_alone.loc['No Headphones'],
    equal_var=False
)
t_stat_headphones_social, p_value_headphones_social = ttest_ind(
    pivot_table_headphones_hike_for_social_interaction.loc['Headphones'],
    pivot_table_headphones_hike_for_social_interaction.loc['No Headphones'],
    equal_var=False
)
t_stat_portable_less_day, p_value_portable_less_day = ttest_ind(
    pivot_table_portable_charger_hike_for_less_than_a_day.loc['Portable charger/battery'],
    pivot_table_portable_charger_hike_for_less_than_a_day.loc['No Portable charger/battery'],
    equal_var=False
)
t_stat_portable_multiple_days, p_value_portable_multiple_days = ttest_ind(
    pivot_table_portable_charger_hike_for_multiple_days.loc['Portable charger/battery'],
    pivot_table_portable_charger_hike_for_multiple_days.loc['No Portable charger/battery'],
    equal_var=False
)

summary_table = pd.DataFrame({
    'Correlation': [
        'Headphones and hike for meditation',
        'Headphones and like to hike in group',
        'Headphones and like to hike alone',
        'Headphones and social interaction',
        'Portable charger/battery and hike for less than a day',
        'Portable charger/battery and hike for multiple days'
    ],
    't-statistic': [
        f"{t_stat_headphones_meditation:.2f}",
        f"{t_stat_headphones_group:.2f}",
        f"{t_stat_headphones_alone:.2f}",
        f"{t_stat_headphones_social:.2f}",
        f"{t_stat_portable_less_day:.2f}",
        f"{t_stat_portable_multiple_days:.2f}"
    ],
    'p-value': [
        f"{p_value_headphones_meditation:.4f}",
        f"{p_value_headphones_group:.4f}",
        f"{p_value_headphones_alone:.4f}",
        f"{p_value_headphones_social:.4f}",
        f"{p_value_portable_less_day:.4f}",
        f"{p_value_portable_multiple_days:.4f}"
    ],
    'Significant': [
        p_value_headphones_meditation < alpha,
        p_value_headphones_group < alpha,
        p_value_headphones_alone < alpha,
        p_value_headphones_social < alpha,
        p_value_portable_less_day < alpha,
        p_value_portable_multiple_days < alpha
    ]
})
display(summary_table)

print('\n\n ---- end of devices and preferences section ----\n\n\n')

print('\n\n\n ---- apps and preferences ----\n\n\n')

print('\n\n ---- maps and like to hike near home correlation ----\n')
# Create a pivot table of Maps and like_to_hike_near_home
if 'like_to_hike_near_home_num' not in df.columns:
    mapping = {
        'Strongly disagree': 1,
        'Somewhat disagree': 2,
        'Neither agree nor disagree': 3,
        'Somewhat agree': 4,
        'Strongly agree': 5,
    }
    df['like_to_hike_near_home_num'] = df['like_to_hike_near_home'].replace(mapping).infer_objects(copy=False)
    pd.set_option('future.no_silent_downcasting', True)
    df['like_to_hike_near_home_num'] = df['like_to_hike_near_home_num'].fillna(0).astype(int)
# Create a new column 'Maps' indicating presence in 'combined'
df['Maps'] = df['Apps'].str.contains('maps', case=False)
df['Maps'] = df['Maps'].map({True: 'Maps', False: 'No Maps'})
# Create a pivot table of Maps and like_to_hike_near_home
pivot_table_maps_like_to_hike_near_home = df.pivot_table(index='Maps', columns='like_to_hike_near_home_num', aggfunc='size', fill_value=0)
print('Pivot table of Maps and like_to_hike_near_home:')
print(pivot_table_maps_like_to_hike_near_home)
# Set significance level
alpha = 0.05
# Perform the t-test
t_stat, p_value = ttest_ind(pivot_table_maps_like_to_hike_near_home.loc['Maps'],
                            pivot_table_maps_like_to_hike_near_home.loc['No Maps'],
                            equal_var=False)
# Print the t-statistic, p-value, significance, and summary statistics
print(f"t-statistic: {t_stat:.2f}")
print(f"p-value: {p_value:.4f}")
print("Significant difference:", p_value < alpha)
# Summary statistics for each group
summary_stats = pivot_table_maps_like_to_hike_near_home.T.describe()
# display(summary_stats)


print('\n\n ---- maps and like to hike while traveling correlation ----\n')
# Create a pivot table of Maps and like_to_hike_while_traveling
if 'like_to_hike_while_traveling_num' not in df.columns:
    mapping = {
        'Strongly disagree': 1,
        'Somewhat disagree': 2,
        'Neither agree nor disagree': 3,
        'Somewhat agree': 4,
        'Strongly agree': 5,
    }
    df['like_to_hike_while_traveling_num'] = df['hike_while_traveling'].replace(mapping).infer_objects(copy=False)
    pd.set_option('future.no_silent_downcasting', True)
    df['like_to_hike_while_traveling_num'] = df['like_to_hike_while_traveling_num'].fillna(0).astype(int)
# Create a pivot table of Maps and like_to_hike_while_traveling
pivot_table_maps_like_to_hike_while_traveling = df.pivot_table(index='Maps', columns='like_to_hike_while_traveling_num', aggfunc='size', fill_value=0)
print('Pivot table of Maps and like_to_hike_while_traveling:')
print(pivot_table_maps_like_to_hike_while_traveling)
# Set significance level
alpha = 0.05
# Perform the t-test
t_stat, p_value = ttest_ind(pivot_table_maps_like_to_hike_while_traveling.loc['Maps'],
                            pivot_table_maps_like_to_hike_while_traveling.loc['No Maps'],
                            equal_var=False)
# Print the t-statistic, p-value, significance, and summary statistics
print(f"t-statistic: {t_stat:.2f}")
print(f"p-value: {p_value:.4f}")
print("Significant difference:", p_value < alpha)
# Summary statistics for each group
summary_stats = pivot_table_maps_like_to_hike_while_traveling.T.describe()
# display(summary_stats)


print('\n\n ---- camera and like to hike near home correlation ----\n')
# Create a pivot table of Camera and like_to_hike_near_home
if 'like_to_hike_near_home_num' not in df.columns:
    mapping = {
        'Strongly disagree': 1,
        'Somewhat disagree': 2,
        'Neither agree nor disagree': 3,
        'Somewhat agree': 4,
        'Strongly agree': 5,
    }
    df['like_to_hike_near_home_num'] = df['like_to_hike_near_home'].replace(mapping).infer_objects(copy=False)
    pd.set_option('future.no_silent_downcasting', True)
    df['like_to_hike_near_home_num'] = df['like_to_hike_near_home_num'].fillna(0).astype(int)
# Create a new column 'Camera' indicating presence in 'combined'
df['Camera'] = df['Apps'].str.contains('camera', case=False)
df['Camera'] = df['Camera'].map({True: 'Camera', False: 'No Camera'})
# Create a pivot table of Camera and like_to_hike_near_home
pivot_table_camera_like_to_hike_near_home = df.pivot_table(index='Camera', columns='like_to_hike_near_home_num', aggfunc='size', fill_value=0)
print('Pivot table of Camera and like_to_hike_near_home:')
print(pivot_table_camera_like_to_hike_near_home)
# Set significance level
alpha = 0.05
# Perform the t-test
t_stat, p_value = ttest_ind(pivot_table_camera_like_to_hike_near_home.loc['Camera'],
                            pivot_table_camera_like_to_hike_near_home.loc['No Camera'],
                            equal_var=False)
# Print the t-statistic, p-value, significance, and summary statistics
print(f"t-statistic: {t_stat:.2f}")
print(f"p-value: {p_value:.4f}")
print("Significant difference:", p_value < alpha)
# Summary statistics for each group
summary_stats = pivot_table_camera_like_to_hike_near_home.T.describe()
# display(summary_stats)

print('\n\n ---- camera and like to hike while traveling correlation ----\n')
# Create a pivot table of Camera and like_to_hike_while_traveling
if 'like_to_hike_while_traveling_num' not in df.columns:
    mapping = {
        'Strongly disagree': 1,
        'Somewhat disagree': 2,
        'Neither agree nor disagree': 3,
        'Somewhat agree': 4,
        'Strongly agree': 5,
    }
    df['like_to_hike_while_traveling_num'] = df['hike_while_traveling'].replace(mapping).infer_objects(copy=False)
    pd.set_option('future.no_silent_downcasting', True)
    df['like_to_hike_while_traveling_num'] = df['like_to_hike_while_traveling_num'].fillna(0).astype(int)
# Create a pivot table of Camera and like_to_hike_while_traveling
pivot_table_camera_like_to_hike_while_traveling = df.pivot_table(index='Camera', columns='like_to_hike_while_traveling_num', aggfunc='size', fill_value=0)
print('Pivot table of Camera and like_to_hike_while_traveling:')
print(pivot_table_camera_like_to_hike_while_traveling)
# Set significance level
alpha = 0.05
# Perform the t-test
t_stat, p_value = ttest_ind(pivot_table_camera_like_to_hike_while_traveling.loc['Camera'],
                            pivot_table_camera_like_to_hike_while_traveling.loc['No Camera'],
                            equal_var=False)
# Print the t-statistic, p-value, significance, and summary statistics
print(f"t-statistic: {t_stat:.2f}")
print(f"p-value: {p_value:.4f}")
print("Significant difference:", p_value < alpha)
# Summary statistics for each group 
summary_stats = pivot_table_camera_like_to_hike_while_traveling.T.describe()
# display(summary_stats)


print('\n\n ---- maps and multiple days correlation ----\n')
# Create a pivot table of Maps and hike_for_multiple_days
if 'hike_for_multiple_days_num' not in df.columns:
    mapping = {
        'Strongly disagree': 1,
        'Somewhat disagree': 2,
        'Neither agree nor disagree': 3,
        'Somewhat agree': 4,
        'Strongly agree': 5,
    }
    df['hike_for_multiple_days_num'] = df['hike_for_multiple_days'].replace(mapping).infer_objects(copy=False)
    pd.set_option('future.no_silent_downcasting', True)
    df['hike_for_multiple_days_num'] = df['hike_for_multiple_days_num'].fillna(0).astype(int)
# Create a pivot table of Maps and hike_for_multiple_days
pivot_table_maps_hike_for_multiple_days = df.pivot_table(index='Maps', columns='hike_for_multiple_days_num', aggfunc='size', fill_value=0)
print('Pivot table of Maps and hike_for_multiple_days:')
print(pivot_table_maps_hike_for_multiple_days)
# Set significance level
alpha = 0.05
# Perform the t-test
t_stat, p_value = ttest_ind(pivot_table_maps_hike_for_multiple_days.loc['Maps'],
                            pivot_table_maps_hike_for_multiple_days.loc['No Maps'],
                            equal_var=False)
# Print the t-statistic, p-value, significance, and summary statistics
print(f"t-statistic: {t_stat:.2f}")
print(f"p-value: {p_value:.4f}")
print("Significant difference:", p_value < alpha)
# Summary statistics for each group
summary_stats = pivot_table_maps_hike_for_multiple_days.T.describe()
# display(summary_stats)


print('\n\n ---- Text messaging and hike for less than an hour correlation ----\n')
# Create a pivot table of Text messaging and hike_for_less_than_an_hour
if 'hike_for_less_than_an_hour_num' not in df.columns:
    mapping = {
        'Strongly disagree': 1,
        'Somewhat disagree': 2,
        'Neither agree nor disagree': 3,
        'Somewhat agree': 4,
        'Strongly agree': 5,
    }
    df['hike_for_less_than_an_hour_num'] = df['hike_for_less_than_1hour'].replace(mapping).infer_objects(copy=False)
    pd.set_option('future.no_silent_downcasting', True)
    df['hike_for_less_than_an_hour_num'] = df['hike_for_less_than_an_hour_num'].fillna(0).astype(int)   
# Create a new column 'Text messaging' indicating presence in 'combined'
df['Text messaging'] = df['Apps'].str.contains('text messaging', case=False)
df['Text messaging'] = df['Text messaging'].map({True: 'Text messaging', False: 'No Text messaging'})
# Create a pivot table of Text messaging and hike_for_less_than_an_hour
pivot_table_text_messaging_hike_for_less_than_an_hour = df.pivot_table(index='Text messaging', columns='hike_for_less_than_an_hour_num', aggfunc='size', fill_value=0)
print('Pivot table of Text messaging and hike_for_less_than_an_hour:')
print(pivot_table_text_messaging_hike_for_less_than_an_hour)
# Set significance level
alpha = 0.05
# Perform the t-test
t_stat, p_value = ttest_ind(pivot_table_text_messaging_hike_for_less_than_an_hour.loc['Text messaging'],
                            pivot_table_text_messaging_hike_for_less_than_an_hour.loc['No Text messaging'],
                            equal_var=False)
# Print the t-statistic, p-value, significance, and summary statistics
print(f"t-statistic: {t_stat:.2f}")
print(f"p-value: {p_value:.4f}")
print("Significant difference:", p_value < alpha)
# Summary statistics for each group
summary_stats = pivot_table_text_messaging_hike_for_less_than_an_hour.T.describe()
# display(summary_stats)


#display a table containing all the t_statistic, p-value, and significance for each of the above correlations
print('\n\n ---- apps and preferences section ----\n\n\n')
# Create a summary table for all the t-statistics, p-values, and significance
# Calculate and store t-statistics and p-values for each correlation
t_stat_maps_near_home, p_value_maps_near_home = ttest_ind(
    pivot_table_maps_like_to_hike_near_home.loc['Maps'],
    pivot_table_maps_like_to_hike_near_home.loc['No Maps'],
    equal_var=False
)
t_stat_maps_traveling, p_value_maps_traveling = ttest_ind(
    pivot_table_maps_like_to_hike_while_traveling.loc['Maps'],
    pivot_table_maps_like_to_hike_while_traveling.loc['No Maps'],
    equal_var=False
)
t_stat_camera_near_home, p_value_camera_near_home = ttest_ind(
    pivot_table_camera_like_to_hike_near_home.loc['Camera'],
    pivot_table_camera_like_to_hike_near_home.loc['No Camera'],
    equal_var=False
)
t_stat_camera_traveling, p_value_camera_traveling = ttest_ind(
    pivot_table_camera_like_to_hike_while_traveling.loc['Camera'],
    pivot_table_camera_like_to_hike_while_traveling.loc['No Camera'],
    equal_var=False
)
t_stat_maps_multiple_days, p_value_maps_multiple_days = ttest_ind(
    pivot_table_maps_hike_for_multiple_days.loc['Maps'],
    pivot_table_maps_hike_for_multiple_days.loc['No Maps'],
    equal_var=False
)
t_stat_text_messaging_less_hour, p_value_text_messaging_less_hour = ttest_ind(
    pivot_table_text_messaging_hike_for_less_than_an_hour.loc['Text messaging'],
    pivot_table_text_messaging_hike_for_less_than_an_hour.loc['No Text messaging'],
    equal_var=False
)
summary_table_apps = pd.DataFrame({
    'Correlation': [
        'Maps and like to hike near home',
        'Maps and like to hike while traveling',
        'Camera and like to hike near home',
        'Camera and like to hike while traveling',
        'Maps and hike for multiple days',
        'Text messaging and hike for less than an hour'
    ],
    't-statistic': [
        f"{t_stat_maps_near_home:.2f}",
        f"{t_stat_maps_traveling:.2f}",
        f"{t_stat_camera_near_home:.2f}",
        f"{t_stat_camera_traveling:.2f}",
        f"{t_stat_maps_multiple_days:.2f}",
        f"{t_stat_text_messaging_less_hour:.2f}"
    ],
    'p-value': [
        f"{p_value_maps_near_home:.4f}",
        f"{p_value_maps_traveling:.4f}",
        f"{p_value_camera_near_home:.4f}",
        f"{p_value_camera_traveling:.4f}",
        f"{p_value_maps_multiple_days:.4f}",
        f"{p_value_text_messaging_less_hour:.4f}"
    ],
    'Significant': [
        p_value_maps_near_home < alpha,
        p_value_maps_traveling < alpha,
        p_value_camera_near_home < alpha,
        p_value_camera_traveling < alpha,
        p_value_maps_multiple_days < alpha,
        p_value_text_messaging_less_hour < alpha
    ]
})
display(summary_table_apps)
print('\n\n ---- end of apps and preferences section ----\n\n\n')

print('\n\n\n ---- devices and apps correlation ----\n\n\n')

print('\n\n ---- headphones and audio app correlation ----\n')
# Create a pivot table of Headphones and Audio
if 'Audio' not in df.columns:
    df['Audio'] = df['Apps'].str.contains('audio', case=False)
    df['Audio'] = df['Audio'].map({True: 'Audio', False: 'No Audio'})

# Create a pivot table of Headphones and Audio
# Fill missing values before creating the pivot table
df['Headphones'] = df['Headphones'].fillna('No Headphones')
df['Audio'] = df['Audio'].fillna('No Audio')

pivot_table_headphones_audio = df.pivot_table(index='Headphones', columns='Audio', aggfunc='size', fill_value=0)
print('Pivot table of Headphones and Audio:')
print(pivot_table_headphones_audio)

# Show the sum of all values in the pivot table
print("Total in pivot table:", pivot_table_headphones_audio.values.sum())
print("Expected total (number of rows in df):", len(df))

# Chi-square test of independence: Headphones/earbuds vs Audio app
alpha = 0.05
chi2, p_value, _, _ = chi2_contingency(pivot_table_headphones_audio)
print(f"Chi-squared statistic: {chi2:.2f}")
print(f"p-value: {p_value:.4f}")
print("Significant difference:", p_value < alpha)

# Display a heatmap of the contingency table
plt.figure(figsize=(5, 4))
sns.heatmap(pivot_table_headphones_audio, annot=True, fmt='d', cmap='Blues')
plt.title('Headphones vs Audio App Usage')
plt.xlabel('Audio App')
plt.ylabel('Headphones/Earbuds')
plt.tight_layout()
plt.show()

print("Total rows in df:", len(df))

#list of headphones and audio usage
headphones_audio_usage = df[df['Headphones'] == 'Headphones']['Audio'].value_counts().to_dict()
# print("Headphones and Audio usage counts:", headphones_audio_usage)

#display a table of headphones and audio and no audio usage
headphones_audio_table = pd.DataFrame({
    'Headphones/Earbuds': headphones_audio_usage.keys(),
    'Audio Usage': headphones_audio_usage.values()
})
total_headphones_usage = headphones_audio_table['Audio Usage'].sum()
headphones_audio_table['Percentage'] = (headphones_audio_table['Audio Usage'] / total_headphones_usage * 100).round(2)
headphones_audio_table = headphones_audio_table.set_index('Headphones/Earbuds')
display(headphones_audio_table)

print('\n\n ---- end of headphones and audio app correlation ----\n\n\n')

print('\n\n ---- Portable charger/battery and maps correlation ----\n')
# Create a pivot table of Portable charger/battery and Maps
if 'Maps' not in df.columns:
    df['Maps'] = df['Apps'].str.contains('maps', case=False)
    df['Maps'] = df['Maps'].map({True: 'Maps', False: 'No Maps'})
    
# Create a pivot table of Portable charger/battery and Maps
df['Portable charger/battery'] = df['combined'].str.contains('Portable charger/battery', case=False)
df['Portable charger/battery'] = df['Portable charger/battery'].map({True: 'Portable charger/battery', False: 'No Portable charger/battery'})
pivot_table_portable_charger_maps = df.pivot_table(index='Portable charger/battery', columns='Maps', aggfunc='size', fill_value=0)
print('Pivot table of Portable charger/battery and Maps:')
print(pivot_table_portable_charger_maps)

# Show the sum of all values in the pivot table
print("Total in pivot table:", pivot_table_portable_charger_maps.values.sum())
print("Expected total (number of rows in df):", len(df))

# Chi-square test of independence: Portable charger/battery vs Maps
alpha = 0.05
chi2, p_value, _, _ = chi2_contingency(pivot_table_portable_charger_maps)
print(f"Chi-squared statistic: {chi2:.2f}")
print(f"p-value: {p_value:.4f}")
print("Significant difference:", p_value < alpha)

# Display a heatmap of the contingency table
plt.figure(figsize=(5, 4))
sns.heatmap(pivot_table_portable_charger_maps, annot=True, fmt='d', cmap='Blues')
plt.title('Portable Charger/Battery vs Maps Usage')
# plt.xlabel('Maps')
# plt.ylabel('Portable Charger/Battery')
plt.tight_layout()
plt.show()

print('\n\n ---- end of Portable charger/battery and maps correlation ----\n\n\n')

print('\n\n ---- Smartwatch and maps correlation ----\n')
# Create a pivot table of Smartwatch and Maps
# Ensure 'Smart watch' column exists and is consistent
if 'Smart watch' not in df.columns:
    df['Smart watch'] = df['combined'].str.contains('smart watch', case=False)
    df['Smart watch'] = df['Smart watch'].map({True: 'Smart watch', False: 'No Smart watch'})
# Ensure 'Maps' column exists and is consistent
df['Maps'] = df['Apps'].str.contains('maps', case=False)
df['Maps'] = df['Maps'].map({True: 'Maps', False: 'No Maps'})
pivot_table_smartwatch_maps = df.pivot_table(index='Smart watch', columns='Maps', aggfunc='size', fill_value=0)
print('Pivot table of Smart watch and Maps:')
print(pivot_table_smartwatch_maps)
# Show the sum of all values in the pivot table
print("Total in pivot table:", pivot_table_smartwatch_maps.values.sum())
print("Expected total (number of rows in df):", len(df))
# Chi-square test of independence: Smartwatch vs Maps
alpha = 0.05
chi2, p_value, _, _ = chi2_contingency(pivot_table_smartwatch_maps)
print(f"Chi-squared statistic: {chi2:.2f}")
print(f"p-value: {p_value:.4f}")
print("Significant difference:", p_value < alpha)

# Display a heatmap of the contingency table
plt.figure(figsize=(5, 4))
sns.heatmap(pivot_table_smartwatch_maps, annot=True, fmt='d', cmap='Blues')
plt.title('Smartwatch vs Maps Usage')
plt.xlabel('Maps')
plt.ylabel('Smartwatch')
plt.tight_layout()
plt.show()

print('\n\n ---- end of Smartwatch and maps correlation ----\n\n\n')





print('\n\n\n ---- number of devices and preferences correlation ----\n')
print('\n\n ---- Number of Devices and easy hike preference correlation ----\n')

# Create a pivot table of Number of Devices (few=3 or less, many=4 or more) vs. easy hike preference

# Create a new column to categorize number of devices as 'Few' (<=3) or 'Many' (>=4)
df['Device_Group'] = np.where(df['number _of_devices'] <= 3, 'Few (≤3)', 'Many (≥4)')



print("\n\n ---- Device_Group and easy_hike_num correlation ----\n")
# Create a new column for easy_hike_num
if 'easy_hike_num' not in df.columns:
    mapping = {
        'Strongly disagree': 1,
        'Somewhat disagree': 2,
        'Neither agree nor disagree': 3,
        'Somewhat agree': 4,
        'Strongly agree': 5,
    }
    df['easy_hike_num'] = df['easy_hike'].replace(mapping).infer_objects(copy=False)
    pd.set_option('future.no_silent_downcasting', True)
    df['easy_hike_num'] = df['easy_hike_num'].fillna(0).astype(int)
# Create a pivot table of Device_Group and easy_hike_num
pivot_table_devicegroup_easyhike_num = df.pivot_table(index='Device_Group', columns='easy_hike_num', aggfunc='size', fill_value=0)
print('Pivot table of Device_Group and easy_hike_num:')
print(pivot_table_devicegroup_easyhike_num)
# Set significance level
alpha = 0.05
# Perform the t-test
t_stats, p_values = ttest_ind(
    pivot_table_devicegroup_easyhike_num.loc['Few (≤3)'],
    pivot_table_devicegroup_easyhike_num.loc['Many (≥4)'],
    equal_var=False
)
# Determine significance based on p-value
significance = p_values < alpha

# Print the t-statistic, p-value, significance, and summary statistics
print(f"t-statistic: {t_stats:.2f}")
print(f"p-value: {p_values:.4f}")
print("Significant difference:", significance)
# Summary statistics for each group
summary_stats = pivot_table_devicegroup_easyhike_num.T.describe()
# display(summary_stats)


print('\n\n ---- Device_Group and difficult hike preference correlation ----\n')
# Create a new column for difficult_hike_num
if 'difficult_hike_num' not in df.columns:
    mapping = {
        'Strongly disagree': 1,
        'Somewhat disagree': 2,
        'Neither agree nor disagree': 3,
        'Somewhat agree': 4,
        'Strongly agree': 5,
    }
    df['difficult_hike_num'] = df['difficult_hike'].replace(mapping).infer_objects(copy=False)
    pd.set_option('future.no_silent_downcasting', True)
    df['difficult_hike_num'] = df['difficult_hike_num'].fillna(0).astype(int)
# Create a pivot table of Device_Group and difficult_hike_num
pivot_table_devicegroup_difficult_hike_num = df.pivot_table(index='Device_Group', columns='difficult_hike_num', aggfunc='size', fill_value=0)
print('Pivot table of Device_Group and difficult_hike_num:')
print(pivot_table_devicegroup_difficult_hike_num)
# Set significance level
alpha = 0.05
# Perform the t-test
t_stats, p_values = ttest_ind(
    pivot_table_devicegroup_difficult_hike_num.loc['Few (≤3)'],
    pivot_table_devicegroup_difficult_hike_num.loc['Many (≥4)'],
    equal_var=False
)
# Determine significance based on p-value
significance = p_values < alpha
# Print the t-statistic, p-value, significance, and summary statistics
print(f"t-statistic: {t_stats:.2f}")
print(f"p-value: {p_values:.4f}")
print("Significant difference:", significance)
# Summary statistics for each group
summary_stats = pivot_table_devicegroup_difficult_hike_num.T.describe()
# display(summary_stats)

print('\n\n ---- Device_Group and hike for fun correlation ----\n')
# Create a new column for hike_for_fun_num
if 'hike_for_fun_num' not in df.columns:
    mapping = {
        'Strongly disagree': 1,
        'Somewhat disagree': 2,
        'Neither agree nor disagree': 3,
        'Somewhat agree': 4,
        'Strongly agree': 5,
    }
    df['hike_for_fun_num'] = df['hike_for_fun'].replace(mapping).infer_objects(copy=False)
    pd.set_option('future.no_silent_downcasting', True)
    df['hike_for_fun_num'] = df['hike_for_fun_num'].fillna(0).astype(int)
# Create a pivot table of Device_Group and hike_for_fun_num
pivot_table_devicegroup_hike_for_fun_num = df.pivot_table(index='Device_Group', columns='hike_for_fun_num', aggfunc='size', fill_value=0)
print('Pivot table of Device_Group and hike_for_fun_num:')
print(pivot_table_devicegroup_hike_for_fun_num)
# Set significance level
alpha = 0.05
# Perform the t-test
t_stats, p_values = ttest_ind(
    pivot_table_devicegroup_hike_for_fun_num.loc['Few (≤3)'],
    pivot_table_devicegroup_hike_for_fun_num.loc['Many (≥4)'],
    equal_var=False
)
# Determine significance based on p-value
significance = p_values < alpha
# Print the t-statistic, p-value, significance, and summary statistics
print(f"t-statistic: {t_stats:.2f}")
print(f"p-value: {p_values:.4f}")
print("Significant difference:", significance)
# Summary statistics for each group
summary_stats = pivot_table_devicegroup_hike_for_fun_num.T.describe()
# display(summary_stats)

print('\n\n ---- Device_Group and like_to_hike_alone correlation ----\n')
# Create a new column for like_to_hike_alone_num
if 'like_to_hike_alone_num' not in df.columns:
    mapping = {
        'Strongly disagree': 1,
        'Somewhat disagree': 2,
        'Neither agree nor disagree': 3,
        'Somewhat agree': 4,
        'Strongly agree': 5,
    }
    df['like_to_hike_alone_num'] = df['like_to_hike_alone'].replace(mapping).infer_objects(copy=False)
    pd.set_option('future.no_silent_downcasting', True)
    df['like_to_hike_alone_num'] = df['like_to_hike_alone_num'].fillna(0).astype(int)
# Create a pivot table of Device_Group and like_to_hike_alone_num
pivot_table_devicegroup_like_to_hike_alone_num = df.pivot_table(index='Device_Group', columns='like_to_hike_alone_num', aggfunc='size', fill_value=0)
print('Pivot table of Device_Group and like_to_hike_alone_num:')
print(pivot_table_devicegroup_like_to_hike_alone_num)
# Set significance level
alpha = 0.05
# Perform the t-test
t_stats, p_values = ttest_ind(
    pivot_table_devicegroup_like_to_hike_alone_num.loc['Few (≤3)'],
    pivot_table_devicegroup_like_to_hike_alone_num.loc['Many (≥4)'],
    equal_var=False
)
# Determine significance based on p-value
significance = p_values < alpha
# Print the t-statistic, p-value, significance, and summary statistics
print(f"t-statistic: {t_stats:.2f}")
print(f"p-value: {p_values:.4f}")
print("Significant difference:", significance)
# Summary statistics for each group
summary_stats = pivot_table_devicegroup_like_to_hike_alone_num.T.describe()
# display(summary_stats)

print('\n\n ---- Device_Group and like_to_hike_in_group correlation ----\n')
# Create a new column for like_to_hike_in_group_num
if 'like_to_hike_in_group_num' not in df.columns:
    mapping = {
        'Strongly disagree': 1,
        'Somewhat disagree': 2,
        'Neither agree nor disagree': 3,
        'Somewhat agree': 4,
        'Strongly agree': 5,
    }
    df['like_to_hike_in_group_num'] = df['like_to_hike_in_group'].replace(mapping).infer_objects(copy=False)
    pd.set_option('future.no_silent_downcasting', True)
    df['like_to_hike_in_group_num'] = df['like_to_hike_in_group_num'].fillna(0).astype(int)
# Create a pivot table of Device_Group and like_to_hike_in_group_num
pivot_table_devicegroup_like_to_hike_in_group_num = df.pivot_table(index='Device_Group', columns='like_to_hike_in_group_num', aggfunc='size', fill_value=0)
print('Pivot table of Device_Group and like_to_hike_in_group_num:')
print(pivot_table_devicegroup_like_to_hike_in_group_num)
# Set significance level
alpha = 0.05
# Perform the t-test
t_stats, p_values = ttest_ind(
    pivot_table_devicegroup_like_to_hike_in_group_num.loc['Few (≤3)'],
    pivot_table_devicegroup_like_to_hike_in_group_num.loc['Many (≥4)'],
    equal_var=False
)
# Determine significance based on p-value
significance = p_values < alpha
# Print the t-statistic, p-value, significance, and summary statistics
print(f"t-statistic: {t_stats:.2f}")
print(f"p-value: {p_values:.4f}")
print("Significant difference:", significance)
# Summary statistics for each group
summary_stats = pivot_table_devicegroup_like_to_hike_in_group_num.T.describe()
# display(summary_stats)

#display a table containing all the t_statistic, p-value, and significance for each of the above correlations

print('\n\n ---- number of devices and preferences section ----\n\n\n')
# Create a summary table for all the t-statistics, p-values, and significance
# Calculate and store t-statistics and p-values for each correlation
t_stat_easy_hike, p_value_easy_hike = ttest_ind(
    pivot_table_devicegroup_easyhike_num.loc['Few (≤3)'],
    pivot_table_devicegroup_easyhike_num.loc['Many (≥4)'],
    equal_var=False
)
t_stat_difficult_hike, p_value_difficult_hike = ttest_ind(
    pivot_table_devicegroup_difficult_hike_num.loc['Few (≤3)'],
    pivot_table_devicegroup_difficult_hike_num.loc['Many (≥4)'],
    equal_var=False
)
t_stat_hike_for_fun, p_value_hike_for_fun = ttest_ind(
    pivot_table_devicegroup_hike_for_fun_num.loc['Few (≤3)'],
    pivot_table_devicegroup_hike_for_fun_num.loc['Many (≥4)'],
    equal_var=False
)
t_stat_like_to_hike_alone, p_value_like_to_hike_alone = ttest_ind(
    pivot_table_devicegroup_like_to_hike_alone_num.loc['Few (≤3)'],
    pivot_table_devicegroup_like_to_hike_alone_num.loc['Many (≥4)'],
    equal_var=False
)
t_stat_like_to_hike_in_group, p_value_like_to_hike_in_group = ttest_ind(
    pivot_table_devicegroup_like_to_hike_in_group_num.loc['Few (≤3)'],
    pivot_table_devicegroup_like_to_hike_in_group_num.loc['Many (≥4)'],
    equal_var=False
)
summary_table_devices = pd.DataFrame({
    'Correlation': [
        'Device_Group and easy hike preference',
        'Device_Group and difficult hike preference',
        'Device_Group and hike for fun',
        'Device_Group and like to hike alone',
        'Device_Group and like to hike in group'
    ],
    't-statistic': [
        f"{t_stat_easy_hike:.2f}",
        f"{t_stat_difficult_hike:.2f}",
        f"{t_stat_hike_for_fun:.2f}",
        f"{t_stat_like_to_hike_alone:.2f}",
        f"{t_stat_like_to_hike_in_group:.2f}"
    ],
    'p-value': [
        f"{p_value_easy_hike:.4f}",
        f"{p_value_difficult_hike:.4f}",
        f"{p_value_hike_for_fun:.4f}",
        f"{p_value_like_to_hike_alone:.4f}",
        f"{p_value_like_to_hike_in_group:.4f}"
    ],
    'Significant': [
        p_value_easy_hike < alpha,
        p_value_difficult_hike < alpha,
        p_value_hike_for_fun < alpha,
        p_value_like_to_hike_alone < alpha,
        p_value_like_to_hike_in_group < alpha
    ]
})
display(summary_table_devices)



print('\n\n ---- ttest of Gender and number of devices correlation ----\n')
# Create a pivot table of Gender and number of devices 
# Create a pivot table of Gender and number of devices
pivot_table_gender_devices = df[df['Gender'].isin(['Male', 'Female'])].pivot_table(index='Gender', columns='number _of_devices', aggfunc='size', fill_value=0)
print(pivot_table_gender_devices)

# create a ttest for the above pivot table
alpha = 0.05

# Perform the t-test



# For t-test, we need two groups. Let's compare the number of devices for Male vs Female as arrays
male_counts = df[df['Gender'] == 'Male']['number _of_devices'].dropna()
female_counts = df[df['Gender'] == 'Female']['number _of_devices'].dropna()
# print("male count", male_counts)
# print("female count", female_counts)

t_stat, p_value = stats.ttest_ind(male_counts, female_counts, equal_var=False)
print(f"t-statistic: {t_stat:.2f}")
print(f"p-value: {p_value:.4f}")
print("Significant difference:", p_value < alpha)


print('\n\n ---- end of ttest of Gender and number of devices correlation ----\n\n\n')