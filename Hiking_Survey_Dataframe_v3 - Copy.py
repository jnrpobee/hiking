import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency
from statsmodels.stats.proportion import proportions_ztest
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from IPython.display import display
import matplotlib.patches as mpatches
import seaborn as sns
import textwrap

# Helper functions
def create_bar_plot(data, title, filename, figsize=(10, 6)):
    """Create standardized bar plot with labels and save"""
    plt.figure(figsize=figsize)
    colors = plt.cm.tab20(range(len(data)))
    ax = data.plot(kind='bar', color=colors, width=0.95)
    plt.xlabel('')
    plt.ylabel('Count')
    plt.title(title)
    
    #legend with custom colors and labels
    labels = data.index
    legend_patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
    plt.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

    # Add data labels
    for p in ax.patches:
        plt.text(p.get_x() + p.get_width()/2, p.get_height() * 1.005, 
                f'{p.get_height():.0f}', ha='center', fontsize=11, color='black')
    
    plt.xticks([])
    plt.tight_layout()
    plt.savefig(f'v3_copy/{filename}.png', bbox_inches='tight', dpi=300)
    plt.show()

def create_numeric_mapping(df, column, new_column):
    """Create numeric mapping for Likert scale responses"""
    mapping = {
        'Strongly disagree': 1, 'Somewhat disagree': 2,
        'Neither agree nor disagree': 3, 'Somewhat agree': 4, 'Strongly agree': 5
    }
    df[new_column] = df[column].replace(mapping).fillna(0).astype(int)
    return df

def perform_ttest_analysis(df, group_col, value_col, group1, group2):
    """Perform t-test between two groups and return results"""
    pivot = df.pivot_table(index=group_col, columns=value_col, aggfunc='size', fill_value=0)
    t_stat, p_value = ttest_ind(pivot.loc[group1], pivot.loc[group2], equal_var=False)
    return t_stat, p_value, p_value < 0.05

def save_summary_table(summary_df, filename):
    """Save summary table as PNG"""
    plt.figure(figsize=(12, len(summary_df) * 0.5))
    ax = plt.gca()
    ax.axis('off')
    table = plt.table(cellText=summary_df.values, colLabels=summary_df.columns,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    plt.tight_layout()
    plt.savefig(f'v3_copy/{filename}.png', bbox_inches='tight', dpi=300)
    plt.close()

# Apps Analysis
display("Apps Analysis")
df_apps = pd.read_excel("hiking_data_v3.xlsx", sheet_name="AppUpdate")
app_counts = df_apps.astype(str).apply(pd.Series.value_counts).fillna(0).astype(int)
app_counts = app_counts.loc[~app_counts.index.str.lower().str.contains('nan')]
total_app_counts = app_counts.sum(axis=1)
top_apps = total_app_counts.nlargest(13)

create_bar_plot(top_apps, 'Frequently Used Apps', 'top_apps')

# Device Analysis
display("Device Analysis")
df_devices = pd.read_excel('hiking_data_v3.xlsx', sheet_name='DevicesUpdate')
device_counts = df_devices.drop(columns=['Count'], errors='ignore').astype(str).apply(pd.Series.value_counts).fillna(0).astype(int)
device_counts = device_counts.loc[~device_counts.index.str.lower().str.contains('nan')]
total_device_counts = device_counts.sum(axis=1)
top_devices = total_device_counts.nlargest(13)

create_bar_plot(top_devices, 'Frequently Used Devices', 'top_devices')

# Device Combinations Analysis
for num_devices in [2, 3]:
    df_combo = pd.read_excel('hiking_data_v3.xlsx', sheet_name=f'{num_devices}devices.1')
    device_cols = [f'device{i}' for i in range(1, num_devices + 1)]
    df_combo[f'{num_devices}devices'] = df_combo[device_cols].apply(lambda x: ', '.join(x.dropna()), axis=1)
    combo_counts = df_combo[f'{num_devices}devices'].value_counts()
    create_bar_plot(combo_counts.head(7), f'{num_devices} Device Combinations', f'device_{num_devices}_combination')

# Main Survey Analysis
df = pd.read_excel('hiking_data_v3.xlsx', sheet_name='Edit(2)')

# Gender distribution
gender_counts = df['Gender'].value_counts()
display("Gender Distribution:", gender_counts)

# Hiking frequency analysis
frequency_mapping = {'Once a month': 12, '2-3 times per year': 2.5, 
                    'Once a week': 52, 'Once a year': 1, 'Never': 0}
df['hiking_duration_numeric'] = df['hiking_duration'].replace(frequency_mapping)

# Device count analysis
device_counts_dist = df['number _of_devices'].value_counts().sort_index()
create_bar_plot(device_counts_dist, 'Distribution of Devices', 'device_counts')

# Create device groups
df['Device_Group'] = np.where(df['number _of_devices'] <= 3, 'Few (≤3)', 'Many (≥4)')

# Create binary device/app indicators
device_apps = {
    'Headphones': 'headphones/earbuds',
    'Smart_Watch': 'smart watch',
    'Portable_Charger': 'Portable charger/battery',
    'Maps': 'maps',
    'Camera': 'camera',
    'Audio': 'audio',
    'Text_Messaging': 'text messaging'
}

for key, value in device_apps.items():
    if key in ['Maps', 'Camera', 'Audio', 'Text_Messaging']:
        df[key] = df['Apps'].str.contains(value, case=False).map({True: key, False: f'No {key}'})
    else:
        df[key] = df['combined'].str.contains(value, case=False).map({True: key, False: f'No {key}'})

# Preference mappings
preferences = ['hike_for_mediation', 'like_to_hike_alone', 'like_to_hike_in_group',
              'hike_for_social_interaction', 'hike_for_less_than_a_day', 'hike_for_multiple_days',
              'like_to_hike_near_home', 'hike_while_traveling', 'hike_for_less_than_1hour',
              'hike_for_health', 'easy_hike', 'difficult_hike', 'hike_for_fun']

for pref in preferences:
    if pref in df.columns:
        df = create_numeric_mapping(df, pref, f'{pref}_num')

# Statistical Analysis - Device vs Preferences
device_pref_tests = []
test_pairs = [
    ('Headphones', 'hike_for_mediation_num'),
    ('Headphones', 'like_to_hike_in_group_num'),
    ('Headphones', 'like_to_hike_alone_num'),
    ('Portable_Charger', 'hike_for_less_than_a_day_num'),
    ('Portable_Charger', 'hike_for_multiple_days_num')
]

for device, preference in test_pairs:
    t_stat, p_val, significant = perform_ttest_analysis(df, device, preference, device, f'No {device}')
    device_pref_tests.append({
        'Test': f'{device} vs {preference}',
        't-statistic': f'{t_stat:.2f}',
        'p-value': f'{p_val:.4f}',
        'Significant': significant
    })

device_summary = pd.DataFrame(device_pref_tests)
save_summary_table(device_summary, 'device_preferences_summary')
display(device_summary)

# Chi-square tests for device-app correlations
chi_square_tests = []
chi_pairs = [
    ('Headphones', 'Audio'),
    ('Portable_Charger', 'Maps'),
    ('Smart_Watch', 'Maps'),
    ('Smart_Watch', 'Audio')
]

for device, app in chi_pairs:
    contingency = df.pivot_table(index=device, columns=app, aggfunc='size', fill_value=0)
    chi2, p_val, _, _ = chi2_contingency(contingency)
    chi_square_tests.append({
        'Test': f'{device} vs {app}',
        'Chi-square': f'{chi2:.2f}',
        'p-value': f'{p_val:.4f}',
        'Significant': p_val < 0.05
    })

chi_summary = pd.DataFrame(chi_square_tests)
save_summary_table(chi_summary, 'device_app_correlations')
display(chi_summary)

# Gender analysis
gender_filtered = df[df['Gender'].isin(['Male', 'Female'])]
male_devices = gender_filtered[gender_filtered['Gender'] == 'Male']['number _of_devices'].dropna()
female_devices = gender_filtered[gender_filtered['Gender'] == 'Female']['number _of_devices'].dropna()

t_stat, p_value = ttest_ind(male_devices, female_devices, equal_var=False)
display(f"Gender vs Device Count - t-stat: {t_stat:.2f}, p-value: {p_value:.4f}, Significant: {p_value < 0.05}")

print("Analysis Complete")
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
plt.savefig('hsd_v3/device_2_count.png', bbox_inches='tight', dpi=300)
plt.show()





print ('\n-----------------device 3 combination------------------')
df = pd.read_excel('hiking_data_v3.xlsx', sheet_name='3devices.1')

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
plt.savefig('hsd_v3/device_3_combination.png', bbox_inches='tight', dpi=300)
plt.show()


print('\n-----------------devices in 3 device combination------------------')
df = pd.read_excel('hiking_data_v3.xlsx', sheet_name='3devCount.1')


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
plt.savefig('hsd_v3/device_3_count.png', bbox_inches='tight', dpi=300)
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
plt.savefig('hsd_v3/hiking_duration_3_5_devices.png', bbox_inches='tight', dpi=300)
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
plt.savefig('hsd_v3/hiking_duration_by_number_of_devices.png', bbox_inches='tight', dpi=300)
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
plt.savefig('hsd_v3/device_counts.png', bbox_inches='tight', dpi=300)
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
plt.savefig('hsd_v3/device_counts_palette_tab10.png', bbox_inches='tight', dpi=300)
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
plt.savefig('hsd_v3/device_counts_seaborn.png', bbox_inches='tight', dpi=300)
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
plt.figure(figsize=(14, 5))
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
plt.legend()
plt.tight_layout()
plt.savefig('hsd_v3/gender_device_counts.png', bbox_inches='tight', dpi=300)
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
# Count the occurrences of each category in the combined column for 2 devices
counts = df[df['number _of_devices'] == 2]['combined'].value_counts()

# Calculate the percentage of each category
percentages = (counts / counts.sum()) * 100

# Create a bar graph for the combined column
plt.figure(figsize=(10, 6))
percentages.plot(kind='bar', color=plt.cm.tab20(range(len(percentages))) , width=0.95)  # Increase width to reduce distance between bars
plt.xlabel('2 Devices')
plt.ylabel('Count')
plt.title('Percentage of People by 2 Devices')

# Create a legend with custom colors and labels
labels = percentages.index
colors = plt.cm.tab20(range(len(percentages)))
legend_patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
plt.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

# Add data labels to the bars
ax = plt.gca()
for p in ax.patches:
    percentage = f"{p.get_height() / percentages.sum().sum() * 100:.1f}%"
    ax.annotate(percentage, (p.get_x() + p.get_width() / 2, p.get_height() + 0.3), ha='center', va='bottom')
plt.xticks([])  # <--- Add this line to remove x-axis labels
plt.savefig('hsd_v3/2_devices_combined.png', bbox_inches='tight', dpi=300)
plt.show()
###----------------------- Count -----------------------
" --------------------version on 2 devices -----------------"

# df = pd.read_excel('hiking_data_v3.xlsx', sheet_name='Edit(2)')
# # Create a dataframe of 2 devices in the number_of_devices column with combined
# df_2_devices_combined = df[df['number _of_devices'] == 2]['combined']
# pivot_table_combined = df_2_devices_combined.value_counts().to_frame().rename(columns={'combined': 'Count'})
# # display(pivot_table_combined)

# # Create a list of colors
# colors = plt.cm.tab20(range(len(pivot_table_combined)))

# # Create a bar graph of the pivot table
# bars = pivot_table_combined.plot(kind='bar')
# for bar, color in zip(bars.patches, colors):
#     bar.set_color(color)

# plt.xlabel('Combined Devices')
# plt.ylabel('Count')
# plt.title('2 device combinations')

# # Create a legend with custom colors and labels
# labels = pivot_table_combined.index
# legend_patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
# plt.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

# # Add data labels to the bars
# ax = plt.gca()
# for p in ax.patches:
#     ax.annotate(f"{p.get_height()}", (p.get_x() + p.get_width() / 2, p.get_height() + 0.3), ha='center', va='bottom')

# plt.xticks([])  # Remove x-axis labels
# plt.savefig('hsd_v3/2_devices_combined_count.png', bbox_inches='tight', dpi=300)
# plt.show()
# print('done\n')


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
# df.to_excel('hiking_data_v3_updated.xlsx', sheet_name='devices', index=False)

# #df = pd.read_excel('hiking_data_v2.xlsx', sheet_name='2device_count')

# #create a dataframe to count the number of devices in the 2devices column in the 2decice_count sheet
# print('\n2 devices\n')
# df_2_devices = df['2devices'].value_counts().to_frame().T
# print(df_2_devices)
# print('\n')

# #create a graph of the 2devices dataframe as a bar graph with the devices as the x-axis and the count as the y-axis
# df_2_devices.plot(kind='bar')
# plt.xlabel('Devices')
# plt.ylabel('Count')
# plt.title('2 Devices')
# legend_patches = [mpatches.Patch(color='blue', label='2 Devices')]
# plt.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
# #label the bars with the count
# ax = plt.gca()
# for p in ax.patches:
#     ax.annotate(f"{p.get_height()}", (p.get_x() + p.get_width() / 2, p.get_height() + 0.3), ha='center', va='bottom')
# plt.xticks(rotation=0, ha='center')
# plt.tight_layout()
# plt.savefig('hsd_v3/2_devices_combined_count.png', bbox_inches='tight', dpi=300)
# plt.show()
# print('done\n')






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


# create a graph of the pivot table as a bar graph with the number of devices as the x-axis and the count as the y-axis
pivot_table_gender_2_devices.plot(kind='bar')
plt.xlabel('Number of Devices')
plt.ylabel('Count')
plt.title('Number of Devices')
# Add data labels to the bars
ax = plt.gca()
for p in ax.patches:
    ax.annotate(f"{p.get_height():.0f}", (p.get_x() + p.get_width() / 2, p.get_height() + 0.5), ha='center', va='bottom')
plt.xticks(rotation=0, ha='center')
plt.savefig('hsd_v3/number_of_devices', bbox_inches='tight', dpi=300)
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
plt.savefig('hsd_v3/heatmap_number_of_devices.png', bbox_inches='tight', dpi=300)
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
# Calculate the percentage of each category
pivot_table_female_2_devices_percentage = (pivot_table_female_2_devices / total_count) * 100

# Sort columns by percentage in descending order
sorted_cols = pivot_table_female_2_devices_percentage.loc['Female'].sort_values(ascending=False).index
pivot_table_female_2_devices_percentage = pivot_table_female_2_devices_percentage[sorted_cols]

# Create a bar chart of the percentages
fig, ax = plt.subplots(figsize=(10, 8))  # Use subplots for better control
colors = plt.cm.tab20(range(len(pivot_table_female_2_devices_percentage.columns)))
bars = pivot_table_female_2_devices_percentage.plot(
    kind='bar', color=colors, width=0.95, ax=ax, legend=True)
plt.xlabel('2 Devices')
plt.ylabel('Percentage')
plt.title('Percentage of People by 2 Devices (Female)')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
# Add data labels to the bars
for p in ax.patches:
    percentage = f"{p.get_height():.1f}%"
    ax.annotate(percentage, (p.get_x() + p.get_width() / 2, p.get_height() + 0.0), ha='center', va='bottom')
plt.xticks(rotation=0, ha='center')
plt.tight_layout()
plt.savefig('hsd_v3/female_2_devices_combination.png', bbox_inches='tight', dpi=300)
plt.show()


# Create a pivot table of Male row in Gender column with 2 devices in the combine column
print('\nMale with 2 devices')
pivot_table_male_2_devices = df[(df['Gender'] == 'Male') & (df['number _of_devices'] == 2)].pivot_table(index='Gender', columns='combined', aggfunc='size', fill_value=0)
display(pivot_table_male_2_devices)

#calculate the total count
total_count = pivot_table_male_2_devices.sum().sum()
#calculate the percentage of each category
pivot_table_male_2_devices_percentage = (pivot_table_male_2_devices / total_count) * 100

# Sort columns by percentage in descending order for males
sorted_cols_male = pivot_table_male_2_devices_percentage.loc['Male'].sort_values(ascending=False).index
pivot_table_male_2_devices_percentage = pivot_table_male_2_devices_percentage[sorted_cols_male]

# Create a bar chart of the percentages for males with 2 devices (add spaces between bars without changing width)
fig, ax = plt.subplots(figsize=(10, 8))
colors = plt.cm.tab20(range(len(pivot_table_male_2_devices_percentage.columns)))
bars = pivot_table_male_2_devices_percentage.plot(
    kind='bar', color=colors, width=0.95, ax=ax, legend=True)

# Add spaces between bars by adjusting bar positions
for i, bar in enumerate(ax.patches):
    bar.set_x(bar.get_x() + 0.03 * (i % len(pivot_table_male_2_devices_percentage.columns)))

plt.xlabel('2 Devices')
plt.ylabel('Percentage')
plt.title('Percentage of People by 2 Devices (Male)')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
# Add data labels to the bars
for p in ax.patches:
    percentage = f"{p.get_height():.1f}%"
    ax.annotate(percentage, (p.get_x() + p.get_width() / 2, p.get_height() + 0.0), ha='center', va='bottom')
plt.xticks(rotation=0, ha='center')
plt.tight_layout()
plt.savefig('hsd_v3/male_2_devices_combination.png', bbox_inches='tight', dpi=300)
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
plt.savefig('hsd_v3/heatmap_of_number_of_devices.png', bbox_inches='tight', dpi=300)
plt.show()



















"-------------------Safety/emergency down-------------------"




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
plt.figure(figsize=(10, 6))
bars = pivot_table_combined.plot(kind='bar', width=0.95)  # Increase width to reduce distance between bars
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
plt.tight_layout()
plt.savefig('hsd_v3/3_devices_combined_count.png', bbox_inches='tight', dpi=300)
plt.show()
print('done\n')











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
plt.xticks(rotation=0, ha='center')
plt.savefig('hsd_v3/smart_watch_found.png', bbox_inches='tight', dpi=300)
plt.tight_layout()
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
plt.savefig('hsd_v3/headphones_like_to_hike_alone.png', bbox_inches='tight', dpi=300)
plt.tight_layout()
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
plt.savefig('hsd_v3/headphones_hike_for_meditation.png', bbox_inches='tight', dpi=300)
plt.tight_layout()
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
plt.savefig('hsd_v3/headphones_like_to_hike_in_group.png', bbox_inches='tight', dpi=300)
plt.show()

# Perform chi-square test of independence
chi2, p, dof, expected = stats.chi2_contingency(pivot_table_headphones_like_to_hike_in_group)
# Print the results
print("Chi-square statistic:", chi2)
print("p-value:", p)
print("Degrees of freedom:", dof)
print("Expected frequencies:", expected)


print ('\n -------- correlation b/n smartwatch and like to hike for health -------\n')
# Create a pivot table of Smart watch and hike_for_health
# Ensure 'Smart watch' column exists and is consistent
if 'Smart watch' not in df.columns:
    df['Smart watch'] = df['combined'].str.contains('smart watch', case=False)
    df['Smart watch'] = df['Smart watch'].map({True: 'Smart watch', False: 'No Smart watch'})

pivot_table_smart_watch_hike_for_health = df.pivot_table(index='Smart watch', columns='hike_for_health', aggfunc='size', fill_value=0)
# Reorder the columns of the pivot table
# Merge categories for 'hike_for_health'
hike_for_health_map = {
    'Strongly disagree': 'Disagree',
    'Somewhat disagree': 'Disagree',
    'Neither agree nor disagree': 'Neither agree nor disagree',
    'Somewhat agree': 'Agree',
    'Strongly agree': 'Agree'
}
pivot_table_smart_watch_hike_for_health = pivot_table_smart_watch_hike_for_health.rename(columns=hike_for_health_map)
# Group by new categories and sum
pivot_table_smart_watch_hike_for_health = pivot_table_smart_watch_hike_for_health.groupby(axis=1, level=0).sum()
pivot_table_smart_watch_hike_for_health = pivot_table_smart_watch_hike_for_health[['Disagree', 'Neither agree nor disagree', 'Agree']]
display(pivot_table_smart_watch_hike_for_health)
# Calculate the total count
total_count = pivot_table_smart_watch_hike_for_health.sum().sum()
# Calculate the percentage of each category
pivot_table_smart_watch_hike_for_health_percentage = pivot_table_smart_watch_hike_for_health.div(pivot_table_smart_watch_hike_for_health.sum(axis=1), axis=0) * 100
# Create a bar graph for the pivot table of Smart Watch Found and hike_for_health percentage

sns.set(style="whitegrid")
pivot_table_smart_watch_hike_for_health_percentage.plot(kind='bar')
plt.xlabel('')
plt.ylabel('Percentage')
plt.title('Smart Watch Found and Hike for Health')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
# Add data labels to the bars
for p in plt.gca().patches:
    plt.text(p.get_x() * 1.005, p.get_height() * 1.005, '{:.1f}%'.format(p.get_height()), fontsize=9, color='black')
plt.xticks(rotation=0)
plt.savefig('hsd_v3/smart_watch_hike_for_health.png', bbox_inches='tight', dpi=300)
plt.tight_layout()
plt.show()
# Perform chi-square test of independence
chi2, p, dof, expected = stats.chi2_contingency(pivot_table_smart_watch_hike_for_health)
# Print the results
print("Chi-square statistic:", chi2)
print("p-value:", p)
print("Degrees of freedom:", dof)
print("Expected frequencies:", expected)
    






print('\n\n\n ---- devices and preferences (ttest) ----\n\n\n')


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
#save the summary table as a png file
# Save the summary table as an image using seaborn and matplotlib
plt.figure(figsize=(10, 2))
sns.set(font_scale=1.1)
ax = plt.gca()
ax.axis('off')
table = plt.table(
    cellText=summary_table.values,
    colLabels=summary_table.columns,
    cellLoc='center',
    loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(10)
for key, cell in table.get_celld().items():
    cell.set_height(0.4)
    cell.set_width(0.22)
    if isinstance(cell.get_text().get_text(), str) and len(cell.get_text().get_text()) > 25:
        wrapped = "\n".join(textwrap.wrap(cell.get_text().get_text(), 25))
        cell.get_text().set_text(wrapped)
table.auto_set_column_width(col=list(range(len(summary_table.columns))))
plt.tight_layout()
plt.savefig('hsd_v3/summary_table.png', bbox_inches='tight', dpi=300)
plt.close()

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
#save the summary statistics as a png file
# dfi.export(summary_stats, 'hsd_v3/maps_like_to_hike_while_traveling_summary.png')
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
#save the summary table as a png file
# Save the summary table as a png file using matplotlib instead of dfi.export to avoid Playwright/asyncio issues
plt.figure(figsize=(10, 2))
sns.set(font_scale=1.1)
ax = plt.gca()
ax.axis('off')
table = plt.table(
    cellText=summary_table_apps.values,
    colLabels=summary_table_apps.columns,
    cellLoc='center',
    loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(10)
# Adjust column widths and wrap text if it does not fit
for key, cell in table.get_celld().items():
    # cell.set_wrap(True)  # Removed because Cell has no set_wrap method
    cell.set_height(0.3)
    cell.set_width(0.22)
    # Wrap text for long content
    if isinstance(cell.get_text().get_text(), str) and len(cell.get_text().get_text()) > 25:
        wrapped = "\n".join(textwrap.wrap(cell.get_text().get_text(), 25))
        cell.get_text().set_text(wrapped)
table.auto_set_column_width(col=list(range(len(summary_table_apps.columns))))
plt.tight_layout()
plt.savefig('hsd_v3/summary_table_apps.png', bbox_inches='tight', dpi=300)
plt.close()
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
plt.savefig('hsd_v3/headphones_audio_heatmap.png')
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
plt.savefig('hsd_v3/portable_charger_maps_heatmap.png')
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
plt.savefig('hsd_v3/smartwatch_maps_heatmap.png')
plt.tight_layout()
plt.show()

print('\n\n ---- end of Smartwatch and maps correlation ----\n\n\n')

print('\n\n ---- Smartwatch and audio app correlation ----\n')

# Create a pivot table of Smartwatch and Audio App
pivot_table_smartwatch_audio = df.pivot_table(index='Smart watch', columns='Audio', aggfunc='size', fill_value=0)
# Reorder the columns of the pivot table
audio_order = ['No Audio', 'Audio']
pivot_table_smartwatch_audio = pivot_table_smartwatch_audio.reindex(columns=audio_order, fill_value=0)
display(pivot_table_smartwatch_audio)
# Show the sum of all values in the pivot table
print("Total in pivot table:", pivot_table_smartwatch_audio.values.sum())
print("Expected total (number of rows in df):", len(df))
# Chi-square test of independence: Smartwatch vs Audio App
alpha = 0.05
chi2, p_value, _, _ = chi2_contingency(pivot_table_smartwatch_audio)
print(f"Chi-squared statistic: {chi2:.2f}")
print(f"p-value: {p_value:.4f}")
print("Significant difference:", p_value < alpha)
# Display a heatmap of the contingency table
plt.figure(figsize=(5, 4))
sns.heatmap(pivot_table_smartwatch_audio, annot=True, fmt='d', cmap='Blues')
plt.title('Smartwatch vs Audio App Usage')
plt.xlabel('Audio App')
plt.ylabel('Smartwatch')
plt.tight_layout()
plt.savefig('hsd_v3/smartwatch_audio_heatmap.png')
plt.show()
print('\n\n ---- end of Smartwatch and audio app correlation ----\n\n\n')

print('\n\n ------- smartphone and camera correlation -------\n')
# Create a pivot table of Smartphone and Camera
if 'Camera' not in df.columns:
    df['Camera'] = df['Apps'].str.contains('camera', case=False)
    df['Camera'] = df['Camera'].map({True: 'Camera', False: 'No Camera'})
# Ensure 'Smartphone' column exists and is consistent
if 'smartphone' not in df.columns:
    df['smartphone'] = df['combined'].str.contains('smartphone', case=False)
    df['smartphone'] = df['smartphone'].map({True: 'Smartphone', False: 'No Smartphone'})
# Create a pivot table of Smartphone and Camera
df['Camera'] = df['Camera'].fillna('No Camera')
df['smartphone'] = df['smartphone'].fillna('No Smartphone')

pivot_table_smartphone_camera = df.pivot_table(index='smartphone', columns='Camera', aggfunc='size', fill_value=0)
print('Pivot table of Smartphone and Camera:')
print(pivot_table_smartphone_camera)
# Show the sum of all values in the pivot table
print("Total in pivot table:", pivot_table_smartphone_camera.values.sum())
print("Expected total (number of rows in df):", len(df))
# Chi-square test of independence: Smartphone vs Camera
alpha = 0.05
chi2, p_value, _, _ = chi2_contingency(pivot_table_smartphone_camera)
print(f"Chi-squared statistic: {chi2:.2f}")
print(f"p-value: {p_value:.4f}")
print("Significant difference:", p_value < alpha)

# Display a heatmap of the contingency table
plt.figure(figsize=(5, 4))
sns.heatmap(pivot_table_smartphone_camera, annot=True, fmt='d', cmap='Blues')
plt.title('Smartphone vs Camera Usage')
plt.xlabel('Camera')
plt.ylabel('Smartphone')
plt.tight_layout()
plt.savefig('hsd_v3/smartphone_camera_heatmap.png')
plt.show()
print('\n\n ---- end of smartphone and camera correlation ----\n\n\n')

print('\n\n ------ smartphone and audio correlation ------\n')
# Create a pivot table of Smartphone and Audio
if 'Audio' not in df.columns:
    df['Audio'] = df['Apps'].str.contains('audio', case=False)
    df['Audio'] = df['Audio'].map({True: 'Audio', False: 'No Audio'})
# Ensure 'Smartphone' column exists and is consistent
df['smartphone'] = df['smartphone'].fillna('No Smartphone')
pivot_table_smartphone_audio = df.pivot_table(index='smartphone', columns='Audio', aggfunc='size', fill_value=0)
print('Pivot table of Smartphone and Audio:')
print(pivot_table_smartphone_audio)
# Show the sum of all values in the pivot table
print("Total in pivot table:", pivot_table_smartphone_audio.values.sum())
print("Expected total (number of rows in df):", len(df))
# Chi-square test of independence: Smartphone vs Audio
alpha = 0.05
chi2, p_value, _, _ = chi2_contingency(pivot_table_smartphone_audio)
print(f"Chi-squared statistic: {chi2:.2f}")
print(f"p-value: {p_value:.4f}")
print("Significant difference:", p_value < alpha)
# Display a heatmap of the contingency table
plt.figure(figsize=(5, 4))
sns.heatmap(pivot_table_smartphone_audio, annot=True, fmt='d', cmap='Blues')
plt.title('Smartphone vs Audio Usage')
plt.xlabel('Audio')
plt.ylabel('Smartphone')
plt.tight_layout()
plt.savefig('hsd_v3/smartphone_audio_heatmap.png')
plt.show()

print('\n\n ---- end of smartphone and audio correlation ----\n\n\n')

print('\n\n ------- smartphone and maps correlation -------\n')
# Create a pivot table of Smartphone and Maps
if 'Maps' not in df.columns:
    df['Maps'] = df['Apps'].str.contains('maps', case=False)
    df['Maps'] = df['Maps'].map({True: 'Maps', False: 'No Maps'})
# Ensure 'Smartphone' column exists and is consistent
if 'smartphone' not in df.columns:
    df['smartphone'] = df['combined'].str.contains('smartphone', case=False)
    df['smartphone'] = df['smartphone'].map({True: 'Smartphone', False: 'No Smartphone'})   
# Create a pivot table of Smartphone and Maps
df['Maps'] = df['Maps'].fillna('No Maps')
df['smartphone'] = df['smartphone'].fillna('No Smartphone')
pivot_table_smartphone_maps = df.pivot_table(index='smartphone', columns='Maps', aggfunc='size', fill_value=0)
print('Pivot table of Smartphone and Maps:')
print(pivot_table_smartphone_maps)
# Show the sum of all values in the pivot table
print("Total in pivot table:", pivot_table_smartphone_maps.values.sum())
print("Expected total (number of rows in df):", len(df))
# Chi-square test of independence: Smartphone vs Maps
alpha = 0.05
chi2, p_value, _, _ = chi2_contingency(pivot_table_smartphone_maps)
print(f"Chi-squared statistic: {chi2:.2f}")
print(f"p-value: {p_value:.4f}")
print("Significant difference:", p_value < alpha)
# Display a heatmap of the contingency table
plt.figure(figsize=(5, 4))
sns.heatmap(pivot_table_smartphone_maps, annot=True, fmt='d', cmap='Blues')
plt.title('Smartphone vs Maps Usage')
plt.xlabel('Maps')
plt.ylabel('Smartphone')
plt.tight_layout()
plt.savefig('hsd_v3/smartphone_maps_heatmap.png')
plt.show()
print('\n\n ---- end of smartphone and maps correlation ----\n\n\n')
















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
# Save the summary table as a png file
# Save the summary table as a png file using matplotlib instead of dfi.export to avoid Playwright/asyncio issues
plt.figure(figsize=(10, 2))
sns.set(font_scale=1.1)
ax = plt.gca()
ax.axis('off')
table = plt.table(
    cellText=summary_table_devices.values,
    colLabels=summary_table_devices.columns,
    cellLoc='center',
    loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(10)
for key, cell in table.get_celld().items():
    cell.set_height(0.4)
    cell.set_width(0.22)
    if isinstance(cell.get_text().get_text(), str) and len(cell.get_text().get_text()) > 25:
        wrapped = "\n".join(textwrap.wrap(cell.get_text().get_text(), 25))
        cell.get_text().set_text(wrapped)
table.auto_set_column_width(col=list(range(len(summary_table_devices.columns))))
plt.tight_layout()
plt.savefig('hsd_v3/summary_table_devices.png', bbox_inches='tight', dpi=300)
plt.close()
display(summary_table_devices)



print('\n\n ---- ttest of Gender and number of devices correlation ----\n')
# Create a pivot table of Gender and number of devices 
# Create a pivot table of Gender and number of devices
pivot_table_gender_devices = df[df['Gender'].isin(['Male', 'Female'])].pivot_table(index='Gender', columns='number _of_devices', aggfunc='size', fill_value=0)
# Save the pivot_table_gender_devices as a PNG file using matplotlib (similar to summary_table_devices.png)
plt.figure(figsize=(10, 2))  # Increased width for better visibility
sns.set(font_scale=1.1)
ax = plt.gca()
ax.axis('off')
table = plt.table(
    cellText=pivot_table_gender_devices.values,
    colLabels=[''] + list(pivot_table_gender_devices.columns),
    rowLabels=pivot_table_gender_devices.index,
    cellLoc='center',
    loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(10)
for key, cell in table.get_celld().items():
    cell.set_height(0.25)
    cell.set_width(1)  # Increased width for each cell
    if isinstance(cell.get_text().get_text(), str) and len(cell.get_text().get_text()) > 25:
        wrapped = "\n".join(textwrap.wrap(cell.get_text().get_text(), 25))
        cell.get_text().set_text(wrapped)
table.auto_set_column_width(col=list(range(len(pivot_table_gender_devices.columns) + 1)))
plt.tight_layout()
plt.savefig('hsd_v3/pivot_table_gender_devices.png', bbox_inches='tight', dpi=300)
plt.close()
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

