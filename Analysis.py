import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel('Project_TSL.xlsx', sheet_name='Sheet1')
df

duplicate_rows = df[df.duplicated()]
print("Duplicate Rows:")
print(duplicate_rows)

# Convert datetime columns to datetime format
df['Enrollment Time'] = pd.to_datetime(df['Enrollment Time'])
df['Completion Time'] = pd.to_datetime(df['Completion Time'])

# Convert 'Duration' to numeric (assuming it's a numerical value)
df['Duration(in hrs)'] = pd.to_numeric(df['Duration(in hrs)'], errors='coerce')

# Convert 'Learning Hours Spent' to numeric
df['Learning Hours Spent'] = pd.to_numeric(df['Learning Hours Spent'], errors='coerce')

# Convert NaN values in 'completed' column to False
df['Completed'].fillna(False, inplace=True)

# Replace 'Yes' with True and 'No' with False in 'completed' column
df['Completed'].replace({'Yes': True, 'No': False}, inplace=True)

# Calculate average learning hours spent per program
avg_learning_hours_by_program = df.groupby('Program Name')['Learning Hours Spent'].mean()

# Set option to display all rows
pd.set_option('display.max_rows', None)

print("Average Learning Hours Spent per Program:")
print(avg_learning_hours_by_program)

# Calculate completion rates by division
completion_by_division = df.groupby('Division')['Completed'].mean()

# Calculate completion rates by group
completion_by_group = df.groupby('Group')['Completed'].mean()

print("Completion Rates by Division:")
print(completion_by_division)
print("\nCompletion Rates by Group:")
print(completion_by_group)


# Filter the DataFrame to include only completed courses
completed_df = df[df['Completed'] == True]

# Group data by division and analyze unique skills learned for completed courses
division_unique_skills = completed_df.groupby('Division')['Skills Learned'].apply(lambda x: ', '.join(set('; '.join(x.dropna()).split('; '))))

# Set option to display the entire column without truncation
pd.set_option('display.max_colwidth', None)

print("Unique Skills Learned by Division (for Completed Courses Only):")
print(division_unique_skills)


# Calculate completion rates by division
completion_by_division = df.groupby('Division')['Completed'].mean()

# Create a bar plot for completion rates by division
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=completion_by_division.index, y=completion_by_division.values)
plt.title('Completion Rates by Division')
plt.xlabel('Division')
plt.ylabel('Completion Rate')
plt.xticks(rotation=45, ha='right')  # Rotate labels and align to the right
plt.tight_layout()

# Prevent label overlap
for label in ax.get_xticklabels():
    label.set_horizontalalignment('right')

plt.show()

# Calculate overall completion rate
overall_completion_rate = df['Completed'].mean()

# Create a dictionary to store division-wise information
division_stats = {}

# Iterate through each division
for division in df['Division'].unique():
    division_data = df[df['Division'] == division]
    
    # Calculate division-specific completion rate
    division_completion_rate = division_data['Completed'].mean()
    
    # Calculate average learning hours spent for completed courses in the division
    division_avg_learning_hours_completed = division_data[division_data['Completed']]['Learning Hours Spent'].mean()
    
    # Calculate the most common skills learned in the division (for completed courses)
    division_completed_skills = division_data[division_data['Completed']]['Skills Learned'].str.split('; ').explode()
    division_most_common_skills = division_completed_skills.value_counts().head(5)
    
    division_stats[division] = {
        'Completion Rate': division_completion_rate,
        'Avg Learning Hours (Completed)': division_avg_learning_hours_completed,
        'Most Common Skills': division_most_common_skills
    }

# Print overall completion rate
print("Overall Completion Rate:", overall_completion_rate)
print("=" * 50)

# Print division-wise statistics
for division, stats in division_stats.items():
    print(f"Division: {division}")
    print("Completion Rate:", stats['Completion Rate'])
    print("Average Learning Hours (Completed):", stats['Avg Learning Hours (Completed)'])
    print("Most Common Skills Learned (for Completed Courses):")
    for skill, count in stats['Most Common Skills'].items():
        print(f" - Skill: {skill} (Count: {count})")
    print("=" * 50)


import numpy as np

# Function to automate generating completion rate bar plot
def generate_completion_rate_bar_plot(dataframe, division_column, completed_column, threshold=0.5):
    # Calculate completion rates by division
    completion_by_division = dataframe.groupby(division_column)[completed_column].mean().reset_index()
    
    # Sort by completion rates in ascending order
    sorted_completion_by_division = completion_by_division.sort_values(by=completed_column)
    
    # Normalize completion rates between 0 and 1 and reverse the values
    normalized_completion_rates = 1 - (sorted_completion_by_division[completed_column] - sorted_completion_by_division[completed_column].min()) / (sorted_completion_by_division[completed_column].max() - sorted_completion_by_division[completed_column].min())
    
    # Create a gradient colormap ranging from blue to red
    cmap = plt.cm.get_cmap('RdBu_r')
    
    # Create a bar plot for completion rates with gradient colors
    plt.figure(figsize=(10, 8))  # Adjust the figsize as needed
    plot = sns.barplot(x=division_column, y=completed_column, data=sorted_completion_by_division, palette=cmap(normalized_completion_rates))
    plt.title('Completion Rates by Division')
    plt.xlabel('Division')
    plt.ylabel('Completion Rate')
    plt.xticks(rotation=45, ha='right')  # Rotate labels and align to the right
    plt.tight_layout()
    
    # Annotate with completion rates above the bars
    for i, (index, row) in enumerate(sorted_completion_by_division.iterrows()):
        plt.text(i, row[completed_column] + 0.001, f'{row[completed_column]:.2f}', ha='center', va='bottom', color='black')
    
    plt.show()

# Call the function to generate the completion rate bar plot with sorted data and fading colors
generate_completion_rate_bar_plot(df, division_column='Division', completed_column='Completed', threshold=0.6)


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random

# Shuffle the rows of the DataFrame
shuffled_df = df.sample(frac=1, random_state=42)  # You can change the random_state to get different shuffling

# Calculate the total number of courses allotted (non-null rows in 'Course Name' column)
total_courses = shuffled_df['Course Name'].count()

# Calculate the number of courses allotted to each division
division_courses = shuffled_df.groupby('Division')['Course Name'].count()

# Calculate the percentage weightage of courses for each division
division_courses_percentage = (division_courses / total_courses) * 100

# Create a DataFrame from the calculated percentages
division_courses_df = pd.DataFrame({'Division': division_courses_percentage.index, 'Percentage': division_courses_percentage.values})

# Calculate the total number of unique users (non-duplicated 'Name' values)
total_unique_users = shuffled_df['Name'].nunique()

# Calculate the number of unique users in each division
division_users = shuffled_df.groupby('Division')['Name'].nunique()

# Calculate the percentage weightage of unique users for each division
division_users_percentage = (division_users / total_unique_users) * 100

# Create a DataFrame from the calculated percentages
division_users_df = pd.DataFrame({'Division': division_users_percentage.index, 'Percentage': division_users_percentage.values})

# Calculate efficiency for each division and store it in a new DataFrame
efficiency_df = pd.DataFrame({'Division': division_courses_df['Division'], 'Efficiency': division_courses_df['Percentage'] / division_users_df['Percentage']})

# Display the efficiency DataFrame
print(efficiency_df)

# Sort the efficiency DataFrame by 'Efficiency' in ascending order
efficiency_df_sorted = efficiency_df.sort_values(by='Efficiency')

# Define a colormap from light green to dark yellow
cmap = mcolors.LinearSegmentedColormap.from_list(
    'light_green_to_dark_yellow', ['lightgreen', 'yellow', 'darkorange'], N=len(efficiency_df_sorted)
)

# Normalize the efficiency values for color mapping
normalize = mcolors.Normalize(vmin=efficiency_df_sorted['Efficiency'].min(), vmax=efficiency_df_sorted['Efficiency'].max())

# Create a figure and axis
fig, ax = plt.subplots(figsize=(13, 8))

# Create a bar chart with colors mapped to efficiency values
bars = ax.bar(efficiency_df_sorted['Division'], efficiency_df_sorted['Efficiency'], color=cmap(normalize(efficiency_df_sorted['Efficiency'])))
plt.xlabel('Division')
plt.ylabel('Current Efficiency')
plt.title('Current Efficiency of Divisions')
plt.xticks(rotation=45, ha='right')  # Rotate and align x-labels for better readability

# Create a colorbar to show the mapping
sm = plt.cm.ScalarMappable(cmap=cmap, norm=normalize)
sm.set_array([])  # Dummy array for colorbar
cbar = plt.colorbar(sm)

plt.tight_layout()  # Ensure proper spacing
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import random

# Calculate the total number of unique users (non-duplicated 'Name' values)
total_unique_users = shuffled_df['Name'].nunique()
print(total_unique_users)

# Calculate the number of unique users in each division
division_users = shuffled_df.groupby('Division')['Name'].nunique()

# Calculate the percentage weightage of unique users for each division
division_users_percentage = (division_users / total_unique_users) * 100

# Create a DataFrame from the calculated percentages
division_users_df = pd.DataFrame({'Division': division_users_percentage.index, 'Percentage': division_users_percentage.values})

# Shuffle the divisions (label names) and the corresponding data in a random order
shuffled_indices = list(division_users_df.index)
random.shuffle(shuffled_indices)
shuffled_data = division_users_df.loc[shuffled_indices]

# Plotting a pie chart
plt.figure(figsize=(10, 12))
colors = plt.cm.Set3.colors
plt.pie(shuffled_data['Percentage'], labels=shuffled_data['Division'], autopct=lambda p: '{:.1f}%'.format(p), pctdistance=0.75, startangle=140, colors=plt.cm.Set3.colors, textprops={'fontsize': 10})
plt.title('Number of Unique Users by Division')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import random

# Shuffle the rows of the DataFrame
shuffled_df = df.sample(frac=1, random_state=42)  # You can change the random_state to get different shuffling

# Calculate the total number of courses allotted (non-null rows in 'Course Name' column)
total_courses = shuffled_df['Course Name'].count()

# Calculate the number of courses allotted to each division
division_courses = shuffled_df.groupby('Division')['Course Name'].count()

# Calculate the percentage weightage of courses for each division
division_courses_percentage = (division_courses / total_courses) * 100

# Create a DataFrame from the calculated percentages
division_courses_df = pd.DataFrame({'Division': division_courses_percentage.index, 'Percentage': division_courses_percentage.values})

# Shuffle the divisions (label names) and the corresponding data in a random order
shuffled_indices = list(division_courses_df.index)
random.shuffle(shuffled_indices)
shuffled_data = division_courses_df.loc[shuffled_indices]

# Plotting a pie chart
plt.figure(figsize=(10, 12))
colors = plt.cm.Set3.colors
plt.pie(shuffled_data['Percentage'], labels=shuffled_data['Division'], autopct=lambda p: '{:.1f}%'.format(p), pctdistance=0.75, startangle=140, colors=plt.cm.Set3.colors, textprops={'fontsize': 10})
plt.title('Weightage of Courses by Division')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()
