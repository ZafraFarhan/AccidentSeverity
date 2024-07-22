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

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as ss
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

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

# Drop columns
df.drop(['Accident_Index','Longitude','Latitude'], axis=1, inplace=True)

#replace vales in Accident_Severity columns
df['Accident_Severity'].unique()

df['Accident_Severity'].replace('Fetal', 'Fatal', inplace=True)

df['Accident_Severity'].unique()



object_columns = df.select_dtypes(include='object').columns
print(object_columns)

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

df['Junction_Control'].value_counts()

df['Junction_Control'].replace('Auto traffic sigl', 'Auto traffic signal', inplace=True)
df['Junction_Control'].value_counts()



contingency_table = pd.crosstab(df['Junction_Control'], df['Local_Authority_(District)'])
data_missing_row = contingency_table.loc['Data missing or out of range'].sort_values(ascending=False)
print(data_missing_row)

contingency_table = pd.crosstab(df['Local_Authority_(District)'],df['Junction_Control'])
data_missing_row = contingency_table.loc['Northumberland']
print(data_missing_row)

df['Junction_Control'].replace('Data missing or out of range', 'Give way or uncontrolled', inplace=True)
df['Junction_Control'].replace('Auto traffic sigl', 'Auto traffic signal', inplace=True)

df['Junction_Control'].value_counts()



df['Light_Conditions'].value_counts()





"""Encoding"""

df.info()

len(df['Police_Force'].unique())

df1=df.copy()

df



def categorize_time(hour):
    if 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    elif 18 <= hour < 24:
        return 'Evening'
    else:
        return 'Night'

# Extract hour from the time string
df['hour'] = pd.to_numeric(df['Time'].str.split(':').str[0], errors='coerce')

# Apply the function to create a new categorical column
df['Time_segment'] = df['hour'].apply(lambda x: categorize_time(x) if pd.notna(x) else np.nan)

print(df)

df['Accident Date'] = pd.to_datetime(df['Accident Date'], format='%m/%d/%Y')
df['Month'] = df['Accident Date'].dt.strftime('%B')

df

df.drop(['Time','Accident Date','hour'], axis=1, inplace=True)
df.info()





"""Label Encoding"""

numeric_data = df.select_dtypes(include=[np.number])
categorical_data = df.select_dtypes(include=[object])

#Replace words with numbers so that analyses can be performed.
for i in categorical_data.columns:
    categorical_data[i] = LabelEncoder().fit_transform(categorical_data[i])

df = pd.concat([categorical_data, numeric_data], axis = 1)
df.head()









"""Data Split"""

#X = combined_data
X = df.drop('Accident_Severity', axis=1)
y = df['Accident_Severity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

"""Missing Values"""

df.isnull().sum()



mode_values=X_train.mode().iloc[0]

X_train = X_train.fillna(mode_values)
X_test = X_test.fillna(mode_values)

X_train.isnull().sum()

X_train.shape

"""Feature Engineering"""

X_train



"""Correlation"""

X_train.info()

X_train



corr_matrix = numeric_data.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()





from sklearn.feature_selection import SelectKBest, chi2
from sklearn.datasets import load_iris

# Apply feature selection
selector = SelectKBest(score_func=chi2, k=2)
X_new = selector.fit_transform(X_train, y_train)

# Get the selected feature names
feature_names = X_train.columns[selector.get_support()]
print("Selected features:", feature_names)

# Get selected feature indices
selected_features = selector.get_support(indices=True)
print("Selected features:", selected_features)





"""**Factor Analysis**"""

!pip install factor_analyzer

"""**KAISER-MEYER-OLKIN (KMO) TEST**

KMO Test measures the proportion of variance that might be a common variance among the variables. Larger proportions are expected as it represents more correlation is present among the variables thereby giving way for the application of dimensionality reduction techniques such as Factor Analysis. KMO score is always between 0 to 1 and values more than 0.6 are much appreciated.

"""

from factor_analyzer.factor_analyzer import calculate_kmo
kmo_vars,kmo_model = calculate_kmo(X_train)
print(kmo_model)



from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt

fa = FactorAnalyzer(rotation = None,n_factors=X_train.shape[1])
fa.fit(X_train)
ev,_ = fa.get_eigenvalues()
print(ev)

plt.scatter(range(1,X_train.shape[1]+1),ev)
plt.plot(range(1,X_train.shape[1]+1),ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigen Value')
plt.grid()

fa = FactorAnalyzer(n_factors=5,rotation='varimax')
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
