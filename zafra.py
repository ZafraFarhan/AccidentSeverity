import requests
import pandas as pd

# Google Drive file ID
file_id = '1M4v3q73xpSVE18u6bvUAGrOIv937HnhW'
url = f'https://drive.google.com/uc?id={file_id}'

# Download the file using requests
response = requests.get(url)
output_file = 'road_accident.csv'

# Write the content to a file
with open(output_file, 'wb') as f:
    f.write(response.content)

# Read the downloaded CSV file into a pandas DataFrame
df = pd.read_csv(output_file)

from google.colab import drive
drive.mount('/content/drive')

!pip install scikit-learn scikit-learn-extra
!pip install prince

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
import prince

df.shape

df.info()



"""Duplicates"""

duplicates=df[df.duplicated()].shape[0]
print(duplicates)

df.drop_duplicates(inplace = True)
df.info()

df.head(5)

df.describe()

df.columns

df.shape

df.isnull().sum()

df['Carriageway_Hazards'].value_counts()

#replace NaN in Carriageway_Hazards as None
df['Carriageway_Hazards'].fillna('None', inplace=True)
df['Carriageway_Hazards'].value_counts()

# Drop Unbalanced columns
#df.drop(columns='Carriageway_Hazards', inplace=True)



#examining the categories
cols1 = ['Day_of_Week', 'Junction_Control',
       'Junction_Detail', 'Accident_Severity', 'Light_Conditions',
       'Local_Authority_(District)', 'Carriageway_Hazards', 'Police_Force',
       'Road_Surface_Conditions', 'Road_Type', 'Time', 'Urban_or_Rural_Area',
       'Weather_Conditions', 'Vehicle_Type']

for col in cols1:
    print(f"Unique values in column '{col}':")
    print(df[col].unique())
    print("\n")

# Drop columns
df.drop(columns='Accident_Index', inplace=True)

#replace vales in Accident_Severity columns
df['Accident_Severity'].unique()

df['Accident_Severity'].replace('Fetal', 'Fatal', inplace=True)
df['Accident_Severity'].unique()

df['Junction_Control'].value_counts()

df['Junction_Control'].replace('Auto traffic sigl', 'Auto traffic signal', inplace=True)
df['Junction_Control'].value_counts()



df['Vehicle_Type'].value_counts()

category_mapping = {
    'Car': 'Car',
    'Taxi/Private hire car': 'Car',
    'Motorcycle over 500cc': 'Motorcycle',
    'Van / Goods 3.5 tonnes mgw or under': 'Van',
    'Goods over 3.5t. and under 7.5t': 'Other',
    'Motorcycle 125cc and under': 'Motorcycle',
    'Motorcycle 50cc and under': 'Motorcycle',
    'Bus or coach (17 or more pass seats)': 'Bus',
    'Goods 7.5 tonnes mgw and over': 'Other',
    'Other vehicle': 'Other',
    'Motorcycle over 125cc and up to 500cc': 'Motorcycle',
    'Agricultural vehicle': 'Other',
    'Minibus (8 - 16 passenger seats)': 'Bus',
    'Pedal cycle': 'Pedal cycle',
    'Ridden horse': 'Other'
}

# Map values in 'Vehicle_Type' column to the specified categories
df['Vehicle_Type'] = df['Vehicle_Type'].map(category_mapping)

df['Vehicle_Type'].unique()

df['Vehicle_Type'].value_counts()

df['Weather_Conditions'].value_counts()

df['Road_Surface_Conditions'].value_counts()

# Mapping between Road Surface Conditions and corresponding Weather Conditions ttto fill the missing values
conditions_mapping = {
    'Dry': 'Fine no high winds',
    'Wet or damp': 'Raining no high winds',
    'Snow': 'Snowing no high winds',
    'Snow': 'Snowing + high winds'
}

df['Weather_Conditions'].fillna(df['Road_Surface_Conditions'].map(conditions_mapping), inplace=True)
df['Weather_Conditions'].fillna('Other', inplace=True)

df['Weather_Conditions'].value_counts()

df['Road_Type'].value_counts()

# Dropping rows where 'Time' is missing
df.dropna(subset=['Time'], inplace=True)

category_mapping_light = {
    'Daylight': 'Good',
    'Darkness - lights lit': 'Good',
    'Darkness - no lighting': 'Poor',
    'Darkness - lighting unknown': 'Unknown',
    'Darkness - lights unlit': 'Poor'
}

# Map values in 'Light_Conditions' column to the specified categories
df['Light_Conditions'] = df['Light_Conditions'].map(category_mapping_light)

df['Light_Conditions'].unique()

# Define rush hour periods (for example, 7-9 AM and 4-6 PM)
rush_hours_morning = range(7,10)  # 7, 8, 9
rush_hours_evening = range(16,20)  # 16, 17, 18,19
# Convert 'Time' column to hours
df['Hour'] = df['Time'].str.split(':', expand=True)[0].astype(int)

# Create a binary feature for rush hour
df['Rush_Hour'] = df['Hour'].apply(lambda x: 1 if x in rush_hours_morning or x in rush_hours_evening else 0)

# Drop the 'Hour' column if not needed
df = df.drop(columns=['Hour'])

# Check the new feature
print(df[['Day_of_Week', 'Time', 'Rush_Hour']].head())

df['Accident Date'] = pd.to_datetime(df['Accident Date'], format='%m/%d/%Y')
df['Month'] = df['Accident Date'].dt.strftime('%B')

df.head()

"""Droping Unnecessary variables"""

# Drop columns
df.drop(['Accident Date','Time','Longitude','Latitude'], axis=1, inplace=True)

df.head()

df.info()



import pandas as pd

# Sample dictionary to categorize by police authorities
police_region_categories = {
    "Greater London": ['Metropolitan Police', 'City of London'],
    "North West England": ['Cumbria', 'Lancashire', 'Merseyside', 'Greater Manchester', 'Cheshire'],
    "North East England": ['Northumbria', 'Durham', 'Cleveland'],
    "Yorkshire and the Humber": ['North Yorkshire', 'West Yorkshire', 'South Yorkshire', 'Humberside'],
    "West Midlands": ['West Midlands', 'Staffordshire', 'West Mercia', 'Warwickshire'],
    "East Midlands": ['Derbyshire', 'Nottinghamshire', 'Lincolnshire', 'Leicestershire', 'Northamptonshire'],
    "East of England": ['Cambridgeshire', 'Norfolk', 'Suffolk', 'Bedfordshire', 'Hertfordshire', 'Essex'],
    "South East England": ['Thames Valley', 'Hampshire', 'Surrey', 'Kent', 'Sussex'],
    "South West England": ['Devon and Cornwall', 'Avon and Somerset', 'Gloucestershire', 'Wiltshire', 'Dorset'],
    "Wales": ['North Wales', 'Gwent', 'South Wales', 'Dyfed-Powys'],
    "Scotland": ['Northern', 'Grampian', 'Tayside', 'Fife', 'Lothian and Borders', 'Central', 'Strathclyde', 'Dumfries and Galloway']
}

# Function to categorize police authorities by region
def categorize_region(authority):
    for region, authorities in police_region_categories.items():
        if authority in authorities:
            return region
    return 'Unknown'


# Apply categorization
df['Police_Region'] = df['Police_Force'].apply(categorize_region)

print(df)

df['Police_Region'].value_counts()

df.info()

df.drop(['Local_Authority_(District)','Police_Force'],axis=1,inplace=True)
df.info()

df['Rush_Hour'] = df['Rush_Hour'].astype(object)
df['Rush_Hour'].dtype
df.info()

import pandas as pd
from scipy.stats import chi2_contingency

# Assuming 'df' is your DataFrame
categorical_vars = [
    'Day_of_Week', 'Junction_Control', 'Junction_Detail', 'Light_Conditions',
    'Carriageway_Hazards', 'Road_Surface_Conditions', 'Road_Type',
    'Urban_or_Rural_Area', 'Weather_Conditions', 'Vehicle_Type',
    'Rush_Hour', 'Month', 'Police_Region'
]

# Initialize a dictionary to store the results
results = {}

# Iterate through categorical variables
for var in categorical_vars:
    # Create a contingency table
    contingency_table = pd.crosstab(df[var], df['Accident_Severity'])

    # Perform the Chi-Square test
    chi2, p, _, _ = chi2_contingency(contingency_table)

    # Store the result
    results[var] = p

# Display the results
results_df = pd.DataFrame(list(results.items()), columns=['Variable', 'p-value'])
print(results_df)

"""Data Split"""

y = df['Accident_Severity']
X = df.drop('Accident_Severity', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

X_train['Rush_Hour'] = X_train['Rush_Hour'].astype(object)
X_train['Rush_Hour'].dtype
X_train.info()

X_train.info()

y_train=pd.DataFrame(y_train,columns=['Accident_Severity'])
y_train

X_train



"""Missing Values"""

X_train.isnull().sum()



mode_values=X_train.mode().iloc[0]

X_train = X_train.fillna(mode_values)
X_test = X_test.fillna(mode_values)

X_train.isnull().sum()

X_test.isnull().sum()

descriptive_df = pd.concat([X_train, y_train], axis=1)
descriptive_df.head()



"""**Standardize**"""

from sklearn.preprocessing import StandardScaler

numerical_columns = ['Number_of_Casualties', 'Number_of_Vehicles', 'Speed_limit']

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler and transform the numerical columns
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])


# Check the standardized numerical columns
print(X_train[numerical_columns].head())



X_train.columns

X_train.info()

X_train.head()





"""**Target Encoding**"""

!pip install category_encoders

y_train.head()



import pandas as pd
import category_encoders as ce

# Load your DataFrame
# df = pd.read_csv('your_dataset.csv') # Uncomment and adjust if needed

# Define target variable
target = 'Accident_Severity'

# Define categorical columns
categorical_columns = X_train.select_dtypes(include=['object']).columns

# Initialize TargetEncoder
encoder = ce.TargetEncoder(cols=categorical_columns)

if y_train[target].dtype == 'object':
    y_train[target] = y_train[target].map({'Slight': 1, 'Serious': 2, 'Fatal': 3})


# Fit and transform the categorical columns
df_encoded = encoder.fit_transform(X_train[categorical_columns], y_train)

# Combine the encoded columns with the rest of the DataFrame
combined_data = pd.concat([X_train.drop(columns=categorical_columns), df_encoded], axis=1)

# Check the resulting DataFrame
print(combined_data.head())

from google.colab import files

# Save the DataFrame to a CSV file
combined_data.to_csv('combined_data.csv', index=False)

# Now you can download the file
files.download('combined_data.csv')

y_train.head()

from google.colab import files

# Save the DataFrame to a CSV file
y_train.to_csv('y_train.csv', index=False)

# Now you can download the file
files.download('y_train.csv')

combined_data.info()

combined_data.head()

X_train.head()



"""**Chu Square Test**"""

import pandas as pd
from scipy.stats import chi2_contingency

# Sample data: replace with your DataFrame
# X_train_encoded = pd.read_csv('your_data.csv')

# List of categorical columns

categorical_data =X_train.drop(columns= ['Number_of_Casualties', 'Number_of_Vehicles', 'Speed_limit'])
categorical_columns = X_train.columns


# Create a DataFrame to store the results
results = pd.DataFrame(columns=['Variable1', 'Variable2', 'Chi-Square', 'p-Value'])

# Perform Chi-Square tests for all pairs of categorical variables
for i in range(len(categorical_columns)):
    for j in range(i + 1, len(categorical_columns)):
        var1 = categorical_columns[i]
        var2 = categorical_columns[j]

        # Create a contingency table
        contingency_table = pd.crosstab(X_train[var1], X_train[var2])

        # Perform Chi-Square test
        chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)

         # Append results, use concat instead of append
        results = pd.concat([results, pd.DataFrame({
            'Variable1': [var1],
            'Variable2': [var2],
            'Chi-Square': [chi2_stat],
            'p-Value': [p_value]
        })], ignore_index=True)

# Display results
print(results)

# Define your significance level
alpha = 0.05

# Filter significant associations
significant_associations = results[results['p-Value'] < alpha]
print(significant_associations)

results.to_csv('significant_associations.csv', index=False)





"""**MCA**"""

X_train_encoded = combined_data.copy()

X_train_encoded.head()

categorical_data =X_train_encoded.drop(columns= ['Number_of_Casualties', 'Number_of_Vehicles', 'Speed_limit'])
categorical_columns = X_train_encoded.columns

# Perform MCA
mca = prince.MCA(n_components=min(len(categorical_columns), 50), random_state=42)
mca = mca.fit(categorical_data)

# Explained inertia (variance) by each component
explained_inertia = mca.eigenvalues_  # Use 'eigenvalues_' instead of 'explained_inertia_'

# Cumulative explained inertia
cumulative_inertia = explained_inertia.cumsum()

# Plot explained inertia
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_inertia) + 1), explained_inertia, marker='o', label='Individual')
plt.plot(range(1, len(explained_inertia) + 1), cumulative_inertia, marker='x', label='Cumulative')
plt.axhline(y=0.80, color='r', linestyle='--', label='80% explained')
plt.xlabel('Number of components')
plt.ylabel('Explained inertia')
plt.title('Explained inertia for MCA components')
plt.legend()
plt.grid(True)
plt.show()

import prince
import pandas as pd

# Assuming 'categorical_data' and 'categorical_columns' are already defined as in your provided code

# Perform MCA with the optimal number of components (5)
mca = prince.MCA(n_components=6, random_state=42)
mca = mca.fit(categorical_data)

# Transform the data
mca_transformed = mca.transform(categorical_data)

from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Define the range of clusters to test
range_n_clusters = range(2, 11)  # For example, from 2 to 10 clusters
silhouette_scores = []

# Iterate over the range of clusters
for n_clusters in range_n_clusters:
    kmedoids = KMedoids(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmedoids.fit_predict(mca_transformed)

    silhouette_avg = silhouette_score(mca_transformed, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f'For n_clusters = {n_clusters}, the average silhouette score is {silhouette_avg:.2f}')

# Plot the silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters, silhouette_scores, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores for different numbers of clusters')
plt.grid(True)
plt.show()



explained_inertia

"""**Clustering**"""



# Range of number of clusters to try
cluster_range = range(2, 5)
silhouette_scores = []

X_train_encoded_sub = combined_data.sample(frac=0.3, random_state=42)

# Iterate over the range of clusters and calculate silhouette scores
for n_clusters in cluster_range:
    kmedoids = KMedoids(n_clusters=n_clusters, random_state=42)
    kmedoids.fit(X_train_encoded_sub)
    labels = kmedoids.labels_
    score = silhouette_score(X_train_encoded_sub, labels)
    silhouette_scores.append(score)
    print(f'Number of clusters: {n_clusters}, Silhouette Score: {score:.3f}')

# Plot the silhouette scores for different numbers of clusters
plt.figure(figsize=(10, 6))
plt.plot(cluster_range, silhouette_scores, marker='o')
plt.title('Silhouette Scores for Different Numbers of Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()

# Find the optimal number of clusters
optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
print(f'Optimal number of clusters: {optimal_clusters}')



"""**DBSCAN outliers**"""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN

eps = 3
min_samples = 5

# Create the DBSCAN model
dbscan = DBSCAN(eps=eps, min_samples=min_samples)

# Fit the model to the data
dbscan.fit(X_train_encoded)

# Get the labels of the data points
labels = dbscan.labels_

# Identify the outliers
outliers = np.where(labels == -1)[0]

# Print the number of outliers
print("Number of outliers:", len(outliers))

# Plot the data with the outliers highlighted
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(X[outliers, 0], X[outliers, 1], c="red", marker="x")
plt.show()





"""**Descriptive Analysis**"""

descriptive_df = pd.concat([X_train, y_train], axis=1)
descriptive_df.head()

"""Target Variable"""

class_distribution = descriptive_df['Accident_Severity'].value_counts()
plt.figure(figsize=(8, 8))
pie_chart=plt.pie(class_distribution, labels=class_distribution.index, autopct='%1.1f%%')
plt.title('Accident_Severity')
plt.legend(pie_chart[0], class_distribution.index, title="Accident Severity", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
plt.ylabel('')  # Hide the y-label
plt.show()

"""Rush Hour"""

import matplotlib.pyplot as plt

# Sample data (replace with your actual data)
class_distribution = descriptive_df['Rush_Hour'].value_counts()

# Define the labels manually
labels = {0: 'Not a Rush Hour', 1: 'Rush Hour'}

# Plot the pie chart
plt.figure(figsize=(8, 8))
plt.pie(class_distribution, labels=[labels[i] for i in class_distribution.index], autopct='%1.1f%%')
plt.title('Rush Hour')

# Add the legend
plt.legend(title="Rush Hour", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
plt.ylabel('')  # Hide the y-label
plt.show()

"""Rush hours Vs Severity -stack bar"""

cross_tab_prop = pd.crosstab(index=descriptive_df['Rush_Hour'],
                             columns=descriptive_df['Accident_Severity'],
                             normalize="index")
cross_tab_prop

cross_tab = pd.crosstab(index=descriptive_df['Rush_Hour'],
                             columns=descriptive_df['Accident_Severity'])
cross_tab

cross_tab_prop.plot(kind='bar',
                    stacked=True,
                    colormap='tab10',
                    figsize=(10, 6))
custom_labels = [ 'No Rush', 'Rush']

plt.legend(loc="upper left", ncol=2)
plt.xticks(ticks=np.arange(len(custom_labels)), labels=custom_labels)
plt.xlabel("Rush_Hour")
plt.ylabel("Percentage")

for n, x in enumerate([*cross_tab.index.values]):
    for proportion in cross_tab_prop.loc[x]:

        plt.text(x=n,
                 y=proportion,
                 s=f'{np.round(proportion * 100, 1)}%',
                 color="black",
                 fontsize=12,
                 fontweight="normal")

plt.show()

"""Weather condiition"""

road_type_counts = descriptive_df['Weather_Conditions'].value_counts()

plt.figure(figsize=(10, 6))
road_type_counts.plot(kind='bar', color='skyblue')

plt.xlabel('Weather_Conditions')
plt.ylabel('Count')
plt.title('Distribution of Weather_Conditions')
plt.xticks(rotation=45)  # Rotate labels for better readability

plt.show()

cross_tab_prop = pd.crosstab(index=descriptive_df['Weather_Conditions'],
                             columns=descriptive_df['Accident_Severity'],
                             normalize="index")
cross_tab_prop

cross_tab = pd.crosstab(index=descriptive_df['Weather_Conditions'],
                             columns=descriptive_df['Accident_Severity'])
cross_tab

cross_tab_prop.plot(kind='bar',
                    stacked=True,
                    colormap='tab10',
                    figsize=(10, 6))

plt.legend(loc="upper left", ncol=2)
plt.xlabel("Weather_Conditions")
plt.ylabel("Percentage")

for n, x in enumerate([*cross_tab.index.values]):
    for proportion in cross_tab_prop.loc[x]:

        plt.text(x=n,
                 y=proportion,
                 s=f'{np.round(proportion * 100, 1)}%',
                 color="black",
                 fontsize=12,
                 fontweight="normal")

plt.show()

"""Speed limit"""

plt.hist(descriptive_df['Speed_limit'],bins=20,edgecolor='black')
plt.title('Speed_limit Distribution')
plt.xlabel('Speed_limit')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x='Accident_Severity', y='Speed_limit', data=descriptive_df)

# Add labels and title
plt.xlabel('Accident_Severity')
plt.ylabel('Speed_limit')
plt.title('Box Plot of Speed_limit by Accident_Severity')

# Show the plot
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Define data with central coordinates for the regions
data = {
    'region_category': ['Greater London', 'North West England', 'North East England', 'Yorkshire and the Humber', 'West Midlands'],
    'incident_count': [100, 150, 200, 250, 300],
    'longitude': [-0.1278, -2.7090, -1.6178, -1.0815, -1.8904],
    'latitude': [51.5074, 54.1125, 54.9010, 53.9590, 52.4862]
}
df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(df['longitude'], df['latitude'], c=df['incident_count'], cmap='OrRd', s=100)
plt.colorbar(label='Incident Count')
plt.title('Incident Distribution by Region')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)

# Annotate points with region names
for i, row in df.iterrows():
    plt.text(row['longitude'], row['latitude'], row['region_category'], fontsize=12)

plt.show()

"""Vehicle type"""

vehicle_type_counts = descriptive_df['Vehicle_Type'].value_counts()

plt.figure(figsize=(10, 6))
vehicle_type_counts.plot(kind='bar', color='skyblue')

plt.xlabel('Vehicle_Type')
plt.ylabel('Count')
plt.title('Distribution of Vehicle_Type')
plt.xticks(rotation=45)  # Rotate labels for better readability

plt.show()

cross_tab_prop = pd.crosstab(index=descriptive_df['Vehicle_Type'],
                             columns=descriptive_df['Accident_Severity'],
                             normalize="index")
cross_tab_prop

cross_tab = pd.crosstab(index=descriptive_df['Vehicle_Type'],
                             columns=descriptive_df['Accident_Severity'])
cross_tab

cross_tab_prop.plot(kind='bar',
                    stacked=True,
                    colormap='tab10',
                    figsize=(10, 6))

plt.legend(loc="upper left", ncol=2)
plt.xlabel("Vehicle_Type")
plt.ylabel("Percentage")

for n, x in enumerate([*cross_tab.index.values]):
    for proportion in cross_tab_prop.loc[x]:

        plt.text(x=n,
                 y=proportion,
                 s=f'{np.round(proportion * 100, 1)}%',
                 color="black",
                 fontsize=12,
                 fontweight="normal")

plt.show()

"""Day"""

class_distribution = descriptive_df['Day_of_Week'].value_counts()
plt.figure(figsize=(8, 8))
pie_chart=plt.pie(class_distribution, labels=class_distribution.index, autopct='%1.1f%%')
plt.title('Day_of_Week')
plt.legend(pie_chart[0], class_distribution.index, title="Accident Severity", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
plt.ylabel('')  # Hide the y-label
plt.show()

cross_tab_prop = pd.crosstab(index=descriptive_df['Day_of_Week'],
                             columns=descriptive_df['Accident_Severity'],
                             normalize="index")
cross_tab_prop

cross_tab = pd.crosstab(index=descriptive_df['Day_of_Week'],
                             columns=descriptive_df['Accident_Severity'])
cross_tab

cross_tab_prop.plot(kind='bar',
                    stacked=True,
                    colormap='tab10',
                    figsize=(10, 6))

plt.legend(loc="upper left", ncol=2)
plt.xlabel("Day_of_Week")
plt.ylabel("Percentage")

for n, x in enumerate([*cross_tab.index.values]):
    for proportion in cross_tab_prop.loc[x]:

        plt.text(x=n,
                 y=proportion,
                 s=f'{np.round(proportion * 100, 1)}%',
                 color="black",
                 fontsize=12,
                 fontweight="normal")

plt.show()

"""Month"""

vehicle_type_counts = descriptive_df['Month'].value_counts()

plt.figure(figsize=(10, 6))
vehicle_type_counts.plot(kind='bar', color='skyblue')

plt.xlabel('Month')
plt.ylabel('Count')
plt.title('Distribution of Month')
plt.xticks(rotation=45)  # Rotate labels for better readability

plt.show()

cross_tab_prop = pd.crosstab(index=descriptive_df['Month'],
                             columns=descriptive_df['Accident_Severity'],
                             normalize="index")
cross_tab_prop

cross_tab = pd.crosstab(index=descriptive_df['Month'],
                             columns=descriptive_df['Accident_Severity'])
cross_tab

cross_tab_prop.plot(kind='bar',
                    stacked=True,
                    colormap='tab10',
                    figsize=(10, 6))

plt.legend(loc="upper left", ncol=2)
plt.xlabel("Month")
plt.ylabel("Percentage")

for n, x in enumerate([*cross_tab.index.values]):
    for proportion in cross_tab_prop.loc[x]:

        plt.text(x=n,
                 y=proportion,
                 s=f'{np.round(proportion * 100, 1)}%',
                 color="black",
                 fontsize=12,
                 fontweight="normal")

plt.show()

sns.countplot(x='Police_Region', data=descriptive_df)
plt.show()

# Calculate counts
counts = descriptive_df.groupby(['Police_Region', 'Accident_Severity']).size().reset_index(name='count')

# Calculate proportions
total_counts = counts.groupby('Police_Region')['count'].transform('sum')
counts['proportions'] = counts['count'] / total_counts

sns.barplot(x='Police_Region', y='proportions', hue='Accident_Severity', data=counts)
plt.title('Proportional Bar Plot of Police_Region')
plt.show()

"""Urban_or_Rural_Area"""

import seaborn as sns
import matplotlib.pyplot as plt

# Mapping dictionary
label_mapping = {0: 'Rural', 1: 'Urban'}

# Replace values with labels
descriptive_df['Urban_or_Rural_Area'] = descriptive_df['Urban_or_Rural_Area'].map(label_mapping)

# Create counts DataFrame
counts = descriptive_df['Urban_or_Rural_Area'].value_counts().reset_index()
counts.columns = ['Urban_or_Rural_Area', 'Count']

# Plot
plt.figure(figsize=(10, 6))
barplot = sns.barplot(x='Urban_or_Rural_Area', y='Count', data=counts)

# Annotate bars with count values
for p in barplot.patches:
    barplot.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', xytext=(0, 5),
                     textcoords='offset points')

plt.title('Bar Plot with Custom Labels and Count Annotations')
plt.xlabel('Urban or Rural Area')
plt.ylabel('Count')
plt.show()

counts = descriptive_df.groupby(['Urban_or_Rural_Area', 'Accident_Severity']).size().reset_index(name='count')

# Calculate proportions
total_counts = counts.groupby('Urban_or_Rural_Area')['count'].transform('sum')
counts['proportions'] = counts['count'] / total_counts

sns.barplot(x='Urban_or_Rural_Area', y='proportions', hue='Accident_Severity', data=counts)
plt.title('Proportional Bar Plot of Urban_or_Rural_Area')
plt.show()



"""**Correlation**"""

numeric_data=X_train[['Number_of_Casualties', 'Number_of_Vehicles', 'Speed_limit']]

corr_matrix = numeric_data.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()



X_train=combined_data.copy()

combined_data.isnull().sum()



"""**Factor Analysis**"""

!pip install factor_analyzer

X_train.info()

"""**KAISER-MEYER-OLKIN (KMO) TEST**

KMO Test measures the proportion of variance that might be a common variance among the variables. Larger proportions are expected as it represents more correlation is present among the variables thereby giving way for the application of dimensionality reduction techniques such as Factor Analysis. KMO score is always between 0 to 1 and values more than 0.6 are much appreciated.

"""

from factor_analyzer.factor_analyzer import calculate_kmo
kmo_vars,kmo_model = calculate_kmo(combined_data)
print(kmo_model)



