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

# ============================================================================
# APPS AND DEVICES ANALYSIS SECTION
# ============================================================================

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

# ============================================================================
# PREFERENCE MAPPING AND STATISTICAL ANALYSIS SECTION
# ============================================================================

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

# ============================================================================
# DETAILED DEVICE ANALYSIS SECTION
# ============================================================================

print("Device 2 Count Analysis")
df = pd.read_excel('hiking_data_v3.xlsx', sheet_name='2devices.1')

# Combine device1 and device2 columns into a new column called 2devices
df['2devices'] = df['device1'] + ', ' + df['device2']

# Remove the leading and trailing spaces from the new column
df['2devices'] = df['2devices'].str.strip()

# Count the number of devices in 2devices column
device_2_counts = df['2devices'].value_counts()
# display(device_2_counts)

# Percentage of devices in the 2devices column
device_2_percentage = df['2devices'].value_counts(normalize=True) * 100
device_2_percentage = device_2_percentage.round(1).astype(str) + '%'
display('2 device combination ', device_2_percentage.round(1))

# Create a bar graph of the counts in the 2devices column
device_2_count = device_2_counts.nlargest(8)  # Get the top 13 device combinations

plt.figure(figsize=(8, 5))
device2CountColors = plt.cm.tab20(range(len(device_2_count)))  # Use a colormap for consistent coloring

device_2_count.plot(kind='bar', color=device2CountColors, width=0.90)  # Increase width to reduce distance between bars
plt.xlabel('Device Combinations')
plt.ylabel('Device Count')
# plt.title('Device Count in 2devices Column')
legend_patches = [plt.Rectangle((0,0),1,1,fc=color, edgecolor='none') for color in device2CountColors]
plt.legend(legend_patches, device_2_count.index, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)  # Adjusted bbox_to_anchor to remove white space

# Add data labels to the bars
for p in plt.gca().patches:
    plt.text(p.get_x() + p.get_width()/2, p.get_height() * 1.005, '{}'.format(p.get_height()), ha='center', fontsize=11, color='black')
plt.xticks([])
plt.savefig('hsd_v3/device_2_count.png', bbox_inches='tight', dpi=300)
plt.show()

print ('\n-----------------device 3 combination------------------')
df = pd.read_excel('hiking_data_v3.xlsx', sheet_name='3devices.1')

# Combine device1, device2 and device3 columns into a new column called 3devices
df['3devices'] = df['device1'] + ', ' + df['device2'] + ', ' + df['device3']

# Remove the leading and trailing spaces from the new column
df['3devices'] = df['3devices'].str.strip()

# Count the number of devices in 3devices column
deviceThree_counts = df['3devices'].value_counts()
# display(deviceThree_counts)

# Percentage of devices in the 3devices column
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


# Count the number of devices in the '3_Devices' column
deviceThree_counts = df['Three_Devices'].value_counts()
# display('device 3 count', deviceThree_counts)

# Percentage of devices in the 3devices column
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
plt.savefig('hsd_v3/heatmap_number_of_devices.png', bbox_inches='tight', dpi=300)
plt.show()



# Create a pivot table of Female and Male row in Gender column with devices column and number _of_devices columns as rows and columns
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
# legend_patches = [plt.Rectangle((0,0),1,1,fc=color, edgecolor='none') for color in colors]
# plt.legend(legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

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
plt.savefig('hsd_v3/heatmap_number_of_devices.png', bbox_inches='tight', dpi=300)
plt.show()



# Create a pivot table of Female and Male row in Gender column with devices column and number _of_devices columns as rows and columns
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
# legend_patches = [plt.Rectangle((0,0),1,1,fc=color, edgecolor='none') for color in colors]
# plt.legend(legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

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
plt.savefig('hsd_v3/heatmap_number_of_devices.png', bbox_inches='tight', dpi=300)
plt.show()


# Create a pivot table of Female and Male row in Gender column with devices column and number _of_devices columns as rows and columns
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
# legend_patches = [plt.Rectangle((0,0),1,1,fc=color, edgecolor='none') for color in colors]
# plt.legend(legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

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
plt.savefig('hsd_v3/heatmap_number_of_devices.png', bbox_inches='tight', dpi=300)
plt.show()


# Create a pivot table of Female and Male row in Gender column with devices column and number _of_devices columns as rows and columns
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
# legend_patches = [plt.Rectangle((0,0),1,1,fc=color, edgecolor='none') for color in colors]
# plt.legend(legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

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
plt.savefig('hsd_v3/heatmap_number_of_devices.png', bbox_inches='tight', dpi=300)
plt.show()


# Create a pivot table of Female and Male row in Gender column with devices column and number _of_devices columns as rows and columns
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
plt.savefig('hsd_v3/heatmap_number_of_devices.png', bbox_inches='tight', dpi=300)
plt.show()


# Create a pivot table of Female and Male row in Gender column with devices column and number _of_devices columns as rows and columns
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
# plt.legend(legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
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
legend_patches = [plt.Patch(color=c, label=l) for c, l in zip(colors, labels)]
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
# legend_patches = [plt.Rectangle((0,0),1,1,fc=color, edgecolor='none') for color in colors]
# plt.legend(legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

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
plt.savefig('hsd_v3/heatmap_number_of_devices.png', bbox_inches='tight', dpi=300)
plt.show()


# Create a pivot table of Female and Male row in Gender column with devices column and number _of_devices columns as rows and columns
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