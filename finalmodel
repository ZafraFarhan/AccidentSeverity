


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


from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from pathos.multiprocessing import Pool
from catboost import Pool, CatBoostClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score, accuracy_score



"""Duplicates"""

duplicates=df[df.duplicated()].shape[0]
print(duplicates)

df.drop_duplicates(inplace = True)

#replace NaN in Carriageway_Hazards as None
df['Carriageway_Hazards'].fillna('None', inplace=True)

# Drop Unbalanced columns
#df.drop(columns='Carriageway_Hazards', inplace=True)



# Drop columns
df.drop(columns='Accident_Index', inplace=True)

df['Accident_Severity'].replace('Fetal', 'Fatal', inplace=True)

df['Junction_Control'].replace('Auto traffic sigl', 'Auto traffic signal', inplace=True)



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

# Mapping between Road Surface Conditions and corresponding Weather Conditions ttto fill the missing values
conditions_mapping = {
    'Dry': 'Fine no high winds',
    'Wet or damp': 'Raining no high winds',
    'Snow': 'Snowing no high winds',
    'Snow': 'Snowing + high winds'
}

df['Weather_Conditions'].fillna(df['Road_Surface_Conditions'].map(conditions_mapping), inplace=True)
df['Weather_Conditions'].fillna('Other', inplace=True)

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

# Define rush hour periods (for example, 7-9 AM and 4-6 PM)
rush_hours_morning = range(7,10)  # 7, 8, 9
rush_hours_evening = range(16,20)  # 16, 17, 18,19
# Convert 'Time' column to hours
df['Hour'] = df['Time'].str.split(':', expand=True)[0].astype(int)

# Create a binary feature for rush hour
df['Rush_Hour'] = df['Hour'].apply(lambda x: 1 if x in rush_hours_morning or x in rush_hours_evening else 0)

# Drop the 'Hour' column if not needed
df = df.drop(columns=['Hour'])

df['Accident Date'] = pd.to_datetime(df['Accident Date'], format='%m/%d/%Y')
df['Month'] = df['Accident Date'].dt.strftime('%B')

"""Droping Unnecessary variables"""

# Drop columns
df.drop(['Accident Date','Time','Longitude','Latitude'], axis=1, inplace=True)

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

df.drop(['Local_Authority_(District)','Police_Force'],axis=1,inplace=True)

df['Rush_Hour'] = df['Rush_Hour'].astype(object)



from feature_engine.encoding import RareLabelEncoder

# select label
main_label = 'Accident_Severity'

from tqdm import tqdm

# set up the rare label encoder limiting number of categories to max_n_categories
for col in tqdm(df.columns):
    if col != main_label:
        df[col] = df[col].fillna('None').astype(str)
        try:
            encoder = RareLabelEncoder(n_categories=1, max_n_categories=5, replace_with='Other', tol=20/df.shape[0])
            df[col] = encoder.fit_transform(df[[col]])
        except:
            print(f'Passed encoding for column {col}')

df[main_label] = df[main_label].map({'Slight': 0, 'Serious': 1, 'Fatal': 1})

# initialize data
y = df[main_label].values.reshape(-1,)
X = df.drop([main_label], axis=1)
cat_cols = X.select_dtypes(include=['object']).columns
cat_cols_idx = [list(X.columns).index(c) for c in cat_cols]
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.5, stratify=y)

from sklearn.utils.class_weight import compute_class_weight
# add class weights to handle class imbalance
classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = dict(zip(classes, weights))
print(class_weights)



!pip install category_encoders

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
from sklearn.metrics import confusion_matrix # Import the missing function
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
# Create a pipeline with preprocessing and model
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline # Import the Pipeline class
from sklearn.metrics import confusion_matrix

# Define preprocessing for categorical and numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X.select_dtypes(exclude=['object']).columns),
        ('cat', TargetEncoder(cols=cat_cols, smoothing=1), cat_cols) # Added cat_cols here
    ])

y_train_encoded = y_train.astype(int)
y_test_encoded = y_test.astype(int)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier',  XGBClassifier(random_state=0))
])

# Train the model
pipeline.fit(X_train, y_train_encoded)

# Predict on the test set
y_predt = pipeline.predict(X_train)
# Calculate accuracy
accuracy = accuracy_score(y_train_encoded, y_predt)

# Print accuracy
print("Accuracy:", accuracy)

# Evaluation
print("F1 Score:", f1_score(y_train_encoded, y_predt, average='weighted'))
print("ROC AUC Score:", roc_auc_score(y_train_encoded, pipeline.predict_proba(X_train)[:,1], multi_class='ovr'))
print("Classification Report:\n", classification_report(y_train_encoded, y_predt))
print("Confusion Matrix:\n", confusion_matrix(y_train_encoded, y_predt))


# Predict on the test set
y_pred = pipeline.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test_encoded, y_pred)

# Print accuracy
print("Accuracy:", accuracy)

# Evaluation
print("F1 Score:", f1_score(y_test_encoded, y_pred, average='weighted'))
print("ROC AUC Score:", roc_auc_score(y_test_encoded, pipeline.predict_proba(X_test)[:,1], multi_class='ovr'))
print("Classification Report:\n", classification_report(y_test_encoded, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test_encoded, y_pred))

# prompt: how to genrate feature importance plot above?

# Get feature importances from the trained model
importances = pipeline.named_steps['classifier'].feature_importances_

# Get feature names from the preprocessor
feature_names = list(X.select_dtypes(exclude=['object']).columns) + list(cat_cols)

# Create a DataFrame for feature importances
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# Sort the DataFrame by importance in descending order
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)







# prompt: how to fit the model again with the variables which have more than 0.05?

# Filter features based on importance threshold
important_features = feature_importance_df[feature_importance_df['Importance'] > 0.05]['Feature'].tolist()

# Select the important features from the original data
X_train_important = X_train[important_features]
X_test_important = X_test[important_features]

# Update the preprocessor to use only the important features
preprocessor_important = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X_train_important.select_dtypes(exclude=['object']).columns),
        ('cat', TargetEncoder(cols=X_train_important.select_dtypes(include=['object']).columns, smoothing=1),
         X_train_important.select_dtypes(include=['object']).columns)
    ])

# Create a new pipeline with the updated preprocessor and the same classifier
pipeline_important = Pipeline(steps=[
    ('preprocessor', preprocessor_important),
    ('classifier', XGBClassifier(random_state=0))
])

# Fit the new pipeline on the training data with important features
pipeline_important.fit(X_train_important, y_train_encoded)

# Predict on the test set
y_predt = pipeline_important.predict(X_train_important)
# Calculate accuracy
accuracy = accuracy_score(y_train_encoded, y_predt)

# Print accuracy
print("Accuracy:", accuracy)

# Evaluation
print("F1 Score:", f1_score(y_train_encoded, y_predt, average='weighted'))
print("ROC AUC Score:", roc_auc_score(y_train_encoded, pipeline_important.predict_proba(X_train_important)[:,1], multi_class='ovr'))
print("Classification Report:\n", classification_report(y_train_encoded, y_predt))
print("Confusion Matrix:\n", confusion_matrix(y_train_encoded, y_predt))


# Predict on the test set
y_pred = pipeline_important.predict(X_test_important)
# Calculate accuracy
accuracy = accuracy_score(y_test_encoded, y_pred)

# Print accuracy
print("Accuracy:", accuracy)

# Evaluation
print("F1 Score:", f1_score(y_test_encoded, y_pred, average='weighted'))
print("ROC AUC Score:", roc_auc_score(y_test_encoded, pipeline_important.predict_proba(X_test_important)[:,1], multi_class='ovr'))
print("Classification Report:\n", classification_report(y_test_encoded, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test_encoded, y_pred))

# Optional: Plot feature importances
importances = pipeline_important.named_steps['classifier'].feature_importances_
sorted_indices = np.argsort(importances)[::-1]
sorted_features = np.array(important_features)[sorted_indices]
sorted_importances = importances[sorted_indices]

plt.figure(figsize=(12, 8))
plt.barh(sorted_features, sorted_importances, color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances from XGBoost Classifier')
plt.gca().invert_yaxis()
plt.show()

import joblib

joblib.dump(pipeline_important, 'pipeline_important.pkl')

import shutil

# If you're using a notebook and want to download the file in the browser:
from IPython.display import FileLink

# Display a download link
FileLink('pipeline_important.pkl')



# Combine the preprocessor with a classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Create a pipeline with preprocessing and model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=0))
])

# Train the model
pipeline.fit(X_train, y_train_encoded)

# Predict on the test set
y_predt = pipeline.predict(X_train)
# Calculate accuracy
accuracy = accuracy_score(y_train_encoded, y_predt)

# Print accuracy
print("Accuracy:", accuracy)

# Evaluation
print("F1 Score:", f1_score(y_train_encoded, y_predt, average='weighted'))
print("ROC AUC Score:", roc_auc_score(y_train_encoded, pipeline.predict_proba(X_train)[:,1], multi_class='ovr'))
print("Classification Report:\n", classification_report(y_train_encoded, y_predt))
print("Confusion Matrix:\n", confusion_matrix(y_train_encoded, y_predt))


# Predict on the test set
y_pred = pipeline.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test_encoded, y_pred)

# Print accuracy
print("Accuracy:", accuracy)

# Evaluation
print("F1 Score:", f1_score(y_test_encoded, y_pred, average='weighted'))
print("ROC AUC Score:", roc_auc_score(y_test_encoded, pipeline.predict_proba(X_test)[:,1], multi_class='ovr'))
print("Classification Report:\n", classification_report(y_test_encoded, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test_encoded, y_pred))





!pip install catboost

from catboost import CatBoostClassifier

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', CatBoostClassifier(random_state=0, verbose=0))
])

# Train the model
pipeline.fit(X_train, y_train_encoded)


## Predict on the test set
y_predt = pipeline.predict(X_train)
# Calculate accuracy
accuracy = accuracy_score(y_train_encoded, y_predt)

# Print accuracy
print("Accuracy:", accuracy)

# Evaluation
print("F1 Score:", f1_score(y_train_encoded, y_predt, average='weighted'))
print("ROC AUC Score:", roc_auc_score(y_train_encoded, pipeline.predict_proba(X_train)[:,1], multi_class='ovr'))
print("Classification Report:\n", classification_report(y_train_encoded, y_predt))
print("Confusion Matrix:\n", confusion_matrix(y_train_encoded, y_predt))


# Predict on the test set
y_pred = pipeline.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test_encoded, y_pred)

# Print accuracy
print("Accuracy:", accuracy)

# Evaluation
print("F1 Score:", f1_score(y_test_encoded, y_pred, average='weighted'))
print("ROC AUC Score:", roc_auc_score(y_test_encoded, pipeline.predict_proba(X_test)[:,1], multi_class='ovr'))
print("Classification Report:\n", classification_report(y_test_encoded, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test_encoded, y_pred))



from sklearn.ensemble import AdaBoostClassifier

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', AdaBoostClassifier(random_state=0))
])

# Train the model
pipeline.fit(X_train, y_train_encoded)


# Predict on the test set
y_predt = pipeline.predict(X_train)
# Calculate accuracy
accuracy = accuracy_score(y_train_encoded, y_predt)

# Print accuracy
print("Accuracy:", accuracy)

# Evaluation
print("F1 Score:", f1_score(y_train_encoded, y_predt, average='weighted'))
print("ROC AUC Score:", roc_auc_score(y_train_encoded, pipeline.predict_proba(X_train)[:,1], multi_class='ovr'))
print("Classification Report:\n", classification_report(y_train_encoded, y_predt))
print("Confusion Matrix:\n", confusion_matrix(y_train_encoded, y_predt))


# Predict on the test set
y_pred = pipeline.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test_encoded, y_pred)

# Print accuracy
print("Accuracy:", accuracy)

# Evaluation
print("F1 Score:", f1_score(y_test_encoded, y_pred, average='weighted'))
print("ROC AUC Score:", roc_auc_score(y_test_encoded, pipeline.predict_proba(X_test)[:,1], multi_class='ovr'))
print("Classification Report:\n", classification_report(y_test_encoded, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test_encoded, y_pred))



from sklearn.neighbors import KNeighborsClassifier

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier())
])

# Train the model
pipeline.fit(X_train, y_train_encoded)

# Predict on the test set
y_predt = pipeline.predict(X_train)
# Calculate accuracy
accuracy = accuracy_score(y_train_encoded, y_predt)

# Print accuracy
print("Accuracy:", accuracy)

# Evaluation
print("F1 Score:", f1_score(y_train_encoded, y_predt, average='weighted'))
print("ROC AUC Score:", roc_auc_score(y_train_encoded, pipeline.predict_proba(X_train)[:,1], multi_class='ovr'))
print("Classification Report:\n", classification_report(y_train_encoded, y_predt))
print("Confusion Matrix:\n", confusion_matrix(y_train_encoded, y_predt))


# Predict on the test set
y_pred = pipeline.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test_encoded, y_pred)

# Print accuracy
print("Accuracy:", accuracy)

# Evaluation
print("F1 Score:", f1_score(y_test_encoded, y_pred, average='weighted'))
print("ROC AUC Score:", roc_auc_score(y_test_encoded, pipeline.predict_proba(X_test)[:,1], multi_class='ovr'))
print("Classification Report:\n", classification_report(y_test_encoded, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test_encoded, y_pred))



from sklearn.linear_model import LogisticRegression

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(penalty='l1', solver='liblinear', class_weight=class_weights, random_state=0))
])

# Train the model
pipeline.fit(X_train, y_train_encoded)

# Predict on the test set
y_predt = pipeline.predict(X_train)
# Calculate accuracy
accuracy = accuracy_score(y_train_encoded, y_predt)

# Print accuracy
print("Accuracy:", accuracy)

# Evaluation
print("F1 Score:", f1_score(y_train_encoded, y_predt, average='weighted'))
print("ROC AUC Score:", roc_auc_score(y_train_encoded, pipeline.predict_proba(X_train)[:,1], multi_class='ovr'))
print("Classification Report:\n", classification_report(y_train_encoded, y_predt))
print("Confusion Matrix:\n", confusion_matrix(y_train_encoded, y_predt))


# Predict on the test set
y_pred = pipeline.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test_encoded, y_pred)

# Print accuracy
print("Accuracy:", accuracy)

# Evaluation
print("F1 Score:", f1_score(y_test_encoded, y_pred, average='weighted'))
print("ROC AUC Score:", roc_auc_score(y_test_encoded, pipeline.predict_proba(X_test)[:,1], multi_class='ovr'))
print("Classification Report:\n", classification_report(y_test_encoded, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test_encoded, y_pred))





# prompt: how to plot roc curve for all the model in one grapgh

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))

# Iterate through models and plot ROC curves
models = {
    'XGBoost': XGBClassifier(random_state=0),
    'Random Forest': RandomForestClassifier(random_state=0),
    'CatBoost': CatBoostClassifier(random_state=0, verbose=0),
    'AdaBoost': AdaBoostClassifier(random_state=0),
    'KNN': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(penalty='l1', solver='liblinear', class_weight=class_weights, random_state=0)
}

for model_name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train_encoded)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test_encoded, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

# Plot diagonal line (random guessing)
plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc="lower right")
plt.show()



import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

# Set the figure size with more width and height for better readability
plt.figure(figsize=(24, 80))  # Adjust width and height for better readability

# Ensure that all features are numeric after preprocessing
# (assuming your pipeline handles encoding)
X_train_important_transformed = pipeline_important.named_steps['preprocessor'].transform(X_train_important)

# Select features to plot PDP
features_to_plot = range(X_train_important_transformed.shape[1])  # Plot all features; adjust as needed

# Generate Partial Dependence Plot
pdp = PartialDependenceDisplay.from_estimator(
    pipeline_important.named_steps['classifier'],  # The estimator within the pipeline
    X_train_important_transformed,  # The transformed numerical data
    features_to_plot,  # Indexes of features or specific columns
    kind='average'  # Default is average, can also use 'individual' for ICE plots
)



# Show the plot with named titles
for i, ax in enumerate(pdp.axes_.ravel()):
    feature_name = X_train_important.columns[features_to_plot[i]]  # Get feature name
    ax.set_title(f'{feature_name}', fontsize=8)  # Set the title with larger font size
    ax.set_xlabel('')  # Remove x-axis label
    ax.set_ylabel('')  # Remove y-axis label
    ax.tick_params(axis='both', which='major', labelsize=10)  # Increase tick label size

for ax in pdp.axes_.ravel():
    ax.set_xticklabels([])  # Remove x-axis tick labels
    ax.set_xticks([])

# Add a main title and adjust spacing
pdp.figure_.suptitle('Partial Dependence Plots', fontsize=18)
plt.subplots_adjust(hspace=0.8, wspace=0.8)  # Increase spacing between subplots

# Display the plot
plt.show()  # Use plt.show() instead of figure_.show()

!pip install shap

import shap

# Calculate SHAP values
explainer = shap.Explainer(pipeline_important.named_steps['classifier'])
shap_values = explainer(pipeline_important.named_steps['preprocessor'].transform(X_train_important))

# Summary plot
shap.summary_plot(shap_values, feature_names=important_features)

# Dependence plot for a specific feature
shap.dependence_plot(0, shap_values.values, X_train_important, feature_names=important_features)

shap_values

