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


# Create a DataFrame
data = {
    'Gender': ['Male', 'Female'],
    '1 Device': ['9%', '7%'],
    '2 Devices': ['19%', '23%'],
    '3 Devices': ['23%', '29%'],
    '4 Devices': ['20%', '19%'],
    '5 Devices': ['16%', '15%'],
    '6 Devices': ['5%', '3%'],
    '7 Devices': ['4%', '2%'],
    '8 Devices': ['3%', '1%'],
    '9 Devices': ['1%', '1%'],
    '10 Devices': ['1%', '0%']
}

# data = {
#     'Gender': ['Male', 'Female'],
#     '1 Device': ['16', '13'],
#     '2 Devices': ['36', '44'],
#     '3 Devices': ['43', '55'],
#     '4 Devices': ['38', '36'],
#     '5 Devices': ['30', '28'],
#     '6 Devices': ['10', '5'],
#     '7 Devices': ['7', '4'],
#     '8 Devices': ['5', '1'],
#     '9 Devices': ['2', '2'],
#     '10 Devices': ['1', '0']
# }

df = pd.DataFrame(data)

# Display the DataFrame
display(df)

										
# Corrected data with percentage symbols
Male = ["16", "36", "43", "38", "30", "10", "7", "5", "2", "1"]
Female = ["13", "44", "55", "36", "28", "5", "4", "1", "2", "0"]

# Convert percentages to decimals for t-test
Male_decimals = [float(x) for x in Male]
Female_decimals = [float(x) for x in Female]

# Perform t-test 
t_stat, p_value = stats.ttest_ind(Male_decimals, Female_decimals) 

print(f"T-statistic: {t_stat}") 
print(f"P-value: {p_value}")

# Print percentages
print("Male percentages: ", Male)
print("Female percentages: ", Female)

# Read the Excel file
df = pd.read_excel('Device_usage.xlsx')

# Display the DataFrame
display(df)

df = pd.read_excel('Device_usage.xlsx', sheet_name='Sheet1')

#display the head of the dataframe
display(df.head())

# Explicitly handle downcasting behavior
pd.set_option('future.no_silent_downcasting', True)

# Replace values with explicit downcasting
# Define the mapping for replacement
mapping = {
    'Yes': 1,
    'No': 0
}

df['like_to_hike_alone_num'] = df['like_to_hike_alone'].replace(mapping).infer_objects(copy=False)

# Avoid chained assignment by assigning directly to the column
df['like_to_hike_alone_num'] = df['like_to_hike_alone_num'].fillna(0)

#perform a t-test of the data in the dataframe of 1 devices in the Device_count column and Male and Female rows in the Gender column
# Filter the data for '1 Device' and Gender
df_1_device = df[df['Device_count'] == '1 Device']

# Separate the data by Gender
male_1_device = df_1_device[df_1_device['Gender'] == 'Male']['Device_count']
female_1_device = df_1_device[df_1_device['Gender'] == 'Female']['Device_count']

# Perform t-test
t_stat, p_value = stats.ttest_ind(male_1_device, female_1_device)

print(f"T-statistic for 1 Device: {t_stat}")
print(f"P-value for 1 Device: {p_value}")


# create a dataframe and count the number of participants in Gender column



