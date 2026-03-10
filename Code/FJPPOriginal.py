# Import Required Libraries

import pandas as pd
import numpy as np
import re
import pickle
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# Load dataset
dataset = pd.read_csv("/Users/digreelal/Docs/Fake Job Posting Project/Dataset/fake_job_postings 2.csv")

# Display first 5 rows
print(dataset.head())

# Number of rows and columns
print(dataset.shape)

# Information about dataset
print(dataset.info())

# Show all column names
print(dataset.columns)

# Statistical summary
print(dataset.describe())

# Check dataset type
print(type(dataset))

# Check data types
print(dataset.dtypes)

# Check missing values
print(dataset.isnull().sum())


# Check unique values
# 0 is Real, 1 is Fake
print(dataset['fraudulent'].unique())

# Visualize the Graph of real vs fake jobs
'''print(dataset['fraudulent'].value_counts())

sns.countplot(x='fraudulent', data=dataset)

plt.title("Fake vs Real Job Postings")
plt.xlabel("Fraudulent (0 = Real, 1 = Fake)")
plt.ylabel("Count")
plt.show()'''

# Select only Numeric Column
numeric_data = dataset.select_dtypes(include=['int64', 'float64'])

# Complete Correlation Matrix
correlation_matrix = numeric_data.corr()

# visualize code of correlation matrix to study the relationship between the numeric data
'''plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')

plt.title("Correlation Matrix")
plt.show()'''


# Extract State from Location
dataset['state'] = dataset['location'].str.split(',').str[1]
dataset['state'] = dataset['state'].str.strip()

# Count Jobs by state
state_counts = dataset['state'].value_counts().head(15)

# Visualize the Count by States
'''plt.figure(figsize=(12,6))
sns.barplot(x=state_counts.index, y=state_counts.values, palette="viridis")

plt.title("Job Count by State")
plt.xlabel("State")
plt.ylabel("Number of Job Postings")
plt.xticks(rotation=45)

plt.show()'''


# Select Top Location
top_locations = dataset['location'].value_counts().head(10).index

filtered_data = dataset[dataset['location'].isin(top_locations)]

# Visualize the Distribution of fake and real jobs based on location
'''plt.figure(figsize=(12,6))

sns.countplot(
    data=filtered_data,
    x='location',
    hue='fraudulent',
    order=top_locations,
    palette={0: "green", 1: "red"}
)

plt.title("Distribution of Fake and Real Jobs by Location", fontsize=14)
plt.xlabel("Location", fontsize=12)
plt.ylabel("Number of Job Postings", fontsize=12)

plt.xticks(rotation=45)

plt.legend(title="Job Type", labels=["Real Jobs", "Fake Jobs"])

plt.tight_layout()
plt.show()'''



# Clean Location Column
dataset['location'] = dataset['location'].fillna('Unknown')

# Extract State and city
location_split = dataset['location'].str.split(',', expand=True)

dataset['state'] = location_split[1].str.strip()
dataset['city'] = location_split[2].str.strip()

# Calculate fake to real job ratio by city and state
city_state_ratio = dataset.groupby(['state','city','fraudulent']).size().unstack(fill_value=0)

city_state_ratio['fake_real_ratio'] = city_state_ratio[1] / (city_state_ratio[0] + 1)

# Select Top Location
top_locations = city_state_ratio.sort_values(
    by='fake_real_ratio', ascending=False
).head(10).reset_index()

# Visualize the Ratio of Fake to Real Jobs based on City and state
'''plt.figure(figsize=(12,6))

sns.barplot(data=top_locations, x='city', y='fake_real_ratio', palette='Reds')

plt.title("Ratio of Fake to Real Jobs by City")
plt.xlabel("City")
plt.ylabel("Fake / Real Job Ratio")

plt.xticks(rotation=45)

plt.show()'''

# Count employment type
print(dataset['employment_type'].value_counts())

# Visualize
'''plt.figure(figsize=(8,5))

sns.countplot(y='employment_type', data=dataset)

plt.title("Employment Type Distribution")
plt.xlabel("Count")
plt.ylabel("Employment Type")

plt.show()'''

# Required experience analysis
print(dataset['required_experience'].value_counts())

# Visualization
'''plt.figure(figsize=(10,5))

sns.countplot(y='required_experience', data=dataset)

plt.title("Required Experience Distribution")
plt.show()'''

# Top industries
top_industries = dataset['industry'].value_counts().head(10)

print(top_industries)

# Visualization
'''plt.figure(figsize=(10,6))

top_industries.plot(kind='bar')

plt.title("Top 10 Industries")
plt.xlabel("Industry")
plt.ylabel("Number of Jobs")

plt.show()'''

# telecommuting Jobs
print(dataset['telecommuting'].value_counts())

# Visualization
'''sns.countplot(x='telecommuting', data=dataset)

plt.title("Telecommuting Jobs")
plt.xlabel("0 = No, 1 = Yes")

plt.show()'''

# Comapany Logo analysis
print(dataset['has_company_logo'].value_counts())

# Visualization
'''sns.countplot(x='has_company_logo', data=dataset)

plt.title("Company Logo Availability")
plt.show()'''

# Job Description Length
dataset['description'] = dataset['description'].fillna('')
dataset['description_length'] = dataset['description'].apply(len)

print(dataset['description_length'].head())

# Visualization
'''plt.figure(figsize=(8,5))

sns.histplot(dataset['description_length'], bins=50)

plt.title("Distribution of Job Description Length")
plt.xlabel("Length of Description")

plt.show()'''

# Handling Missing Values
dataset['description'] = dataset['description'].fillna('')

# Calculate Character Count
dataset['char_count'] = dataset['description'].apply(len)

# Visualize Character Count Distribution
'''plt.figure(figsize=(10,6))

sns.histplot(
    data=dataset,
    x='char_count',
    hue='fraudulent',
    bins=50,
    palette={0: "green", 1: "red"},
    kde=True
)

plt.title("Character Count Distribution of Fake vs Real Job Descriptions")
plt.xlabel("Character Count")
plt.ylabel("Frequency")

plt.legend(title="Job Type", labels=["Real Jobs", "Fake Jobs"])

plt.show()'''

# Check missing values
print(dataset.isnull().sum())

# Filling Missing Values
print(dataset.fillna('', inplace=True))

# Check missing values again
print(dataset.isnull().sum())

# Drop unnecessary columns
print(dataset.drop(['job_id'], axis=1, inplace=True))

# Convert Target columns
print(dataset['fraudulent'].value_counts())


# Combine Text Columns
dataset['text'] = dataset['title'] + " " + \
                  dataset['company_profile'] + " " + \
                  dataset['description'] + " " + \
                  dataset['requirements'] + " " + \
                  dataset['benefits']

# Check
print(dataset[['text']].head())


# Text Cleaning Function

# fill missing values
dataset['text'] = dataset['text'].fillna('')

def clean_text(text):
    text = str(text)
    text = text.lower()
    return text

dataset['text'] = dataset['text'].apply(clean_text)

print(dataset['text'].head())


# Feature and target
X = dataset['text']
y = dataset['fraudulent']

# Check
print(X.head())
print(y.head())


# TF-IDF Feature Extraction
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')

X_features = tfidf.fit_transform(X)

# Check Frature Shape
print(X_features.shape)

# Convert Features to Array
X_features = X_features.toarray()


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_features,
    y,
    test_size=0.2,
    random_state=42
)


# Train Model
model = LogisticRegression()

model.fit(X_train, y_train)


# Check
print(X_train.shape)
print(X_test.shape)

# Predictions
y_pred = model.predict(X_test)

# Check Predictions
print(y_pred[:10])


# Model Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred))


'''# Visualize the Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()'''


# Generate Classification Report
report = classification_report(y_test, y_pred, output_dict=True)

# Convert Reoprt to Dataframe
report_df = pd.DataFrame(report).transpose()

# Visualize the Classifiaction Report
'''plt.figure(figsize=(8,6))

sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap="Blues", fmt=".2f")

plt.title("Classification Report Visualization")

plt.show()'''


# Save Model
pickle.dump(model, open("fake_job_model.pkl","wb"))
pickle.dump(tfidf, open("tfidf_vectorizer.pkl","wb"))


# Prediction Function
def predict_job_posting(job_text):

    job_text = clean_text(job_text)

    vector = tfidf.transform([job_text])

    prediction = model.predict(vector)

    if prediction[0] == 1:
        print("Fake Job Posting")
    else:
        print("Real Job Posting")


        
        

# Test Prediction Function
job_description = "Work from home job earn $5000 weekly no experience required"

predict_job_posting(job_description)


