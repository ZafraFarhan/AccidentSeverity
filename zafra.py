
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



# import pandas as pd

# # Sample dictionary to categorize by region
# region_categories = {
#     "Greater London": ['Kensington and Chelsea', 'Hammersmith and Fulham', 'Westminster', 'Hounslow',
#                        'City of London', 'Tower Hamlets', 'Hackney', 'Camden', 'Southwark', 'Brent',
#                        'Haringey', 'Islington', 'Barnet', 'Ealing', 'Newham', 'London Airport (Heathrow)',
#                        'Hillingdon', 'Waltham Forest', 'Redbridge', 'Barking and Dagenham', 'Havering',
#                        'Lambeth', 'Croydon', 'Wandsworth', 'Bromley', 'Lewisham', 'Greenwich', 'Bexley',
#                        'Harrow', 'Enfield', 'Sutton', 'Merton', 'Kingston upon Thames', 'Richmond upon Thames'],
#     "North West England": ['Eden', 'Copeland', 'South Lakeland', 'Barrow-in-Furness', 'Allerdale',
#                            'Carlisle', 'Fylde', 'Blackpool', 'Wyre', 'Lancaster', 'Chorley',
#                            'West Lancashire', 'South Ribble', 'Preston', 'Blackburn with Darwen',
#                            'Hyndburn', 'Ribble Valley', 'Burnley', 'Pendle', 'Rossendale', 'Wirral',
#                            'Sefton', 'St. Helens', 'Liverpool', 'Knowsley', 'Manchester', 'Salford',
#                            'Tameside', 'Stockport', 'Bolton', 'Wigan', 'Trafford', 'Bury', 'Rochdale',
#                            'Oldham'],
#     "North East England": ['Wansbeck', 'Blyth Valley', 'North Tyneside', 'Newcastle upon Tyne', 'Tynedale',
#                            'Alnwick', 'South Tyneside', 'Gateshead', 'Castle Morpeth', 'Sunderland',
#                            'Berwick-upon-Tweed', 'Northumberland', 'Durham', 'County Durham', 'Easington',
#                            'Chester-le-Street', 'Derwentside', 'Wear Valley', 'Teesdale', 'Darlington',
#                            'Sedgefield'],
#     "West Midlands": ['Birmingham', 'Wolverhampton', 'Walsall', 'Dudley', 'Sandwell', 'Solihull',
#                       'Coventry', 'Lichfield', 'Stafford', 'Stoke-on-Trent', 'East Staffordshire',
#                       'Newcastle-under-Lyme', 'Cannock Chase', 'South Staffordshire', 'Tamworth',
#                       'Staffordshire Moorlands', 'Wychavon', 'Malvern Hills', 'Worcester', 'Wyre Forest',
#                       'Herefordshire, County of', 'Shropshire', 'Redditch', 'Bromsgrove', 'South Shropshire',
#                       'North Shropshire', 'Shrewsbury and Atcham', 'Oswestry', 'Telford and Wrekin',
#                       'Bridgnorth', 'Stratford-upon-Avon', 'Warwick', 'North Warwickshire', 'Rugby',
#                       'Nuneaton and Bedworth'],
# }

# # Function to categorize regions
# def categorize_region(local_authority):
#     for region, areas in region_categories.items():
#         if local_authority in areas:
#             return region
#     return 'Other'


# # Apply the categorization function to the DataFrame
# df['Region'] = df['Local_Authority_(District)'].apply(categorize_region)

# # Display the DataFrame with the new 'Region' column
# print(df)

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

from google.colab import files
files.download('significant_associations.csv')



"""**Standardize**"""

from sklearn.preprocessing import StandardScaler

numerical_columns = ['Number_of_Casualties', 'Number_of_Vehicles', 'Speed_limit']

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler and transform the numerical columns
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])


# Check the standardized numerical columns
print(X_train[numerical_columns].head())







"""**One Hot Encoding**"""

from sklearn.preprocessing import OneHotEncoder

# Selecting categorical and numerical columns
categorical_data = X_train.select_dtypes(include=[object])
numerical_columns = X_train.select_dtypes(exclude=[object]).columns

# Initialize the OneHotEncoder
encoder = OneHotEncoder(sparse=False)

# Fit and transform the categorical data
encoded_data = encoder.fit_transform(categorical_data)

# Get the new column names
encoded_columns = encoder.get_feature_names_out(categorical_data.columns)

# Create a DataFrame with the encoded data
df_encoded = pd.DataFrame(encoded_data, columns=encoded_columns)

# Re-index the encoded DataFrame to match the original DataFrame
df_encoded.index = X_train.index

# Concatenate the encoded categorical columns to the numerical columns
X_train_encoded = pd.concat([df_encoded, X_train[numerical_columns]], axis=1)

# Verify the result
print(X_train_encoded)

print(X_train_encoded.isnull().sum())



"""**MCA**"""

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

explained_inertia

import prince

# Assume X_train_encoded is your mixed data DataFrame
# Initialize the FAMD object
famd = prince.FAMD(n_components=48, random_state=42)

# Fit and transform the data
X_reduced = famd.fit_transform(X_train_encoded)

# Create a DataFrame from the reduced data
X_reduced_df = pd.DataFrame(X_reduced)

# Sample the reduced data
X_train_encoded_sub = X_reduced_df.sample(frac=0.1, random_state=42)

# Range of number of clusters to try
cluster_range = range(2, 11)
silhouette_scores = []

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

"""**Clustering**"""



# Range of number of clusters to try
cluster_range = range(2, 5)
silhouette_scores = []

X_train_encoded_sub = X_train_encoded.sample(frac=0.3, random_state=42)

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



"""**Outliers**"""

from sklearn_extra.cluster import KMedoids
import numpy as np

# Example clustering
kmedoids = KMedoids(n_clusters=, random_state=42)
kmedoids.fit(X_train_encoded_sub)
labels = kmedoids.labels_
cluster_centers = kmedoids.cluster_centers_

# Calculate distances to cluster centers
distances = np.linalg.norm(X_train_encoded_sub.values - cluster_centers[labels], axis=1)

# Define a threshold for outliers
threshold = np.percentile(distances, 95)  # e.g., top 5% distance

# Identify outliers
outliers = distances > threshold
outlier_records = X_train_encoded_sub[outliers]
print(outlier_records)



from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt

# Define the range of k values
k_range = range(3, 10)  # Adjust the range as needed
mean_distances_list = []

# Iterate over the range of k values
for k in k_range:
    # Initialize and fit the KNN model
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(X_train_encoded)

    # Calculate distances to k-nearest neighbors
    distances, _ = knn.kneighbors(X_train_encoded)

    # Compute the mean distance to neighbors
    mean_distances = np.mean(distances, axis=1)
    mean_distances_list.append(np.mean(mean_distances))

# Plot the mean distances for different k values
plt.figure(figsize=(10, 6))
plt.plot(k_range, mean_distances_list, marker='o')
plt.title('Mean Distance to Neighbors vs. Number of Neighbors')
plt.xlabel('Number of Neighbors')
plt.ylabel('Mean Distance')
plt.show()

# Print out the k with the minimum mean distance
optimal_k = k_range[np.argmin(mean_distances_list)]
print(f'Optimal number of neighbors: {optimal_k}')

from sklearn.neighbors import NearestNeighbors
import numpy as np

# Fit KNN model
knn = NearestNeighbors(n_neighbors=6)  # Adjust the number of neighbors as needed
knn.fit(X_train_encoded)

# Calculate distances to k-nearest neighbors
distances, _ = knn.kneighbors(X_train_encoded)

# Compute the mean distance to neighbors
mean_distances = np.mean(distances, axis=1)

# Define an outlier threshold
threshold = np.percentile(mean_distances, 95)  # Top 5% as outliers

# Identify outliers
outliers = mean_distances > threshold
outlier_records = X_train_encoded[outliers]

print(outlier_records)

10772

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

class_distribution = descriptive_df['Rush_Hour'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(class_distribution, labels=class_distribution.index, autopct='%1.1f%%')
plt.title('Rush_Hour')
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





"""**Target Encoding**"""



!pip install category_encoders

X_train.info()

y_train



import category_encoders as ce

target = 'Accident_Severity'

# List of categorical columns
categorical_columns = [
    'Day_of_Week',
    'Junction_Control',
    'Junction_Detail',
    'Light_Conditions',
    'Carriageway_Hazards',
    'Police_Region',
    'Road_Surface_Conditions',
    'Road_Type',
    'Urban_or_Rural_Area',
    'Weather_Conditions',
    'Vehicle_Type',
    'Month',
    'Rush_Hour'
]

from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder for the target variable
severity_mapping = {'Slight': 1, 'Serious': 2, 'Fatal': 3}
df['Accident_Severity'] = df['Accident_Severity'].map(severity_mapping)


# Initialize TargetEncoder
encoder = ce.TargetEncoder(cols=categorical_columns)

# Fit and transform the categorical columns
X_train_encoded = encoder.fit_transform(X_train[categorical_columns], y_train_encoded)

# Combine the encoded columns with the rest of the DataFrame
combined_data = pd.concat([X_train.drop(columns=categorical_columns), X_train_encoded], axis=1)

# Check the resulting DataFrame
print(combined_data.head())

combined_data





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



from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt

fa = FactorAnalyzer(rotation = None,n_factors=X_train.shape[1])
fa.fit(combined_data)
ev,_ = fa.get_eigenvalues()
print(ev)

plt.scatter(range(1,X_train.shape[1]+1),ev)
plt.plot(range(1,X_train.shape[1]+1),ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigen Value')
plt.grid()

corr_matrix = X_train.corr()
print(corr_matrix)



fa = FactorAnalyzer(n_factors=4,rotation='varimax')
fa.fit(X_train)
print(pd.DataFrame(fa.loadings_,index=X_train.columns))

print(pd.DataFrame(fa.get_factor_variance(),index=['Variance','Proportional Var','Cumulative Var']))





print(pd.DataFrame(fa.get_communalities(),index=X_train.columns,columns=['Communalities']))

!pip install mpl-axes-aligner
import mpl_axes_aligner

scores = fa.fit_transform(X_train)

# Get the scores dataframe
dfScores = pd.DataFrame(fa.fit_transform(X_train), columns=['Factor'+str(i) for i in range(1, fa.n_factors+1)])

# Get the loadings dataframe
dfLoadings = pd.DataFrame(fa.loadings_, columns=['Factor'+str(i) for i in range(1, fa.n_factors+1)], index=X_train.columns)

scores.shape

dfScores

# Get the loadings dataframe
dfLoadings



df.head()





loadings=fa.loadings_
loadings

dfLoadings



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Assume loadings is a NumPy array with shape (n_features, n_factors)
# Extract loadings for the first three factors
xs = loadings[:, 0]  # Loadings for Factor 1
ys = loadings[:, 1]  # Loadings for Factor 2
zs = loadings[:, 2]  # Loadings for Factor 3

# Feature names
feature_names = ['Day_of_Week', 'Junction_Control', 'Junction_Detail',
                  'Light_Conditions', 'Local_Authority_(District)', 'Carriageway_Hazards',
                  'Police_Force', 'Road_Surface_Conditions', 'Road_Type', 'Urban_or_Rural_Area',
                  'Weather_Conditions', 'Vehicle_Type', 'Time Segment', 'Month',
                  'Number_of_Casualties', 'Number_of_Vehicles', 'Speed_limit']

# Create 3D scatter plot
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(xs, ys, zs, s=100, c='b', marker='o')

# Add arrows and labels
for i, feature in enumerate(feature_names):
    ax.quiver(0, 0, 0, xs[i], ys[i], zs[i], color='r', arrow_length_ratio=0.1)
    ax.text(xs[i], ys[i], zs[i], feature, fontsize=8)

# Define the axes
ax.set_xlabel('Factor 1')
ax.set_ylabel('Factor 2')
ax.set_zlabel('Factor 3')

plt.title('3D Loading Plot')
plt.show()

import numpy as np
import seaborn as sns
sns.set()

xs = loadings[:,0]
ys = loadings[:,1]

feature_names = ['Day_of_Week', 'Junction_Control',
       'Junction_Detail',  'Light_Conditions',
       'Local_Authority_(District)', 'Carriageway_Hazards', 'Police_Force',
       'Road_Surface_Conditions', 'Road_Type', 'Urban_or_Rural_Area',
       'Weather_Conditions', 'Vehicle_Type','Time Segment', 'Month','Number_of_Casualties','Number_of_Vehicles','Speed_limit']

plt.figure(figsize=(15, 8))
# Plot the loadings on a scatterplot
for i, varnames in enumerate(feature_names):
    plt.scatter(xs[i], ys[i], s=200)
    plt.arrow(
        0, 0, # coordinates of arrow base
        xs[i], # length of the arrow along x
        ys[i], # length of the arrow along y
        color='r',
        head_width=0.01
        )
    plt.text(xs[i], ys[i], varnames, fontsize=8)

# Define the axes

xticks = np.linspace(-0.8, 0.8, num=5)
yticks = np.linspace(-0.8, 0.8, num=5)

plt.xticks(xticks)
plt.yticks(yticks)
plt.xlabel('Factor1')
plt.ylabel('Factor2')


# Show plot

plt.title('2D Loading plot')
plt.show()







from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

kmeans_kwargs = {"init": "random","n_init": 10,"max_iter": 10,"random_state": 42}

sse = []
for k in range(1, 18):
  kmeans = KMeans(n_clusters=k)
  kmeans.fit(dfScores)
  sse.append(kmeans.inertia_)

plt.plot(range(1, 18), sse)
plt.xticks(range(1, 18))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()

df_selected = dfScores.sample(n=70000, random_state=1)

from sklearn.cluster import KMeans
silhouette_coefficients = []
for k in range(2, 15):
  kmeans = KMeans(n_clusters=k)
  kmeans.fit(df_selected)
  score = silhouette_score(df_selected, kmeans.labels_, random_state=200)
  silhouette_coefficients.append(score)

silhouette_coefficients



plt.plot(range(2, 15), silhouette_coefficients)
plt.xticks(range(2, 15))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.show()



