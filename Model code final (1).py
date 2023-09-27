#!/usr/bin/env python
# coding: utf-8

# In[1]:


#adding the dependencies in the start of the notebook
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV


# In[2]:


df = pd.read_csv("C:/Users/gouds/Downloads/instagram.csv")
df.head()


# In[3]:


#addressing the outliers
features_with_outliers = ['#posts', '#followers', '#posts']

for feature in features_with_outliers:
    df[feature] = np.log1p(df[feature])


# In[4]:


#adding features in X and the target variable in y
X = df.drop('fake', axis=1)
y = df['fake']


# In[5]:


#adding the train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[6]:


#Standardizing the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# In[7]:


#Define the models in a for loop
models = {
    "Gradient Boosting": GradientBoostingClassifier(random_state = 42),
    "Random Forest" : RandomForestClassifier(random_state = 42),
    "Support Vector Machine" : SVC(random_state = 42)
}


# In[8]:


# Initiating dictionary to hold the results for each model
results = {
    "Model" : [],
    "Accuracy" : [],
    "Precision" : [],
    "Recall" : [],
    "F1 Score" : []
}


# In[9]:


#For each model, running a for loop
for model_name, model in models.items():
    #train the model
    model.fit(X_train, y_train)

    #Make predictions using the test set
    y_pred = model.predict(X_test)

    #Calculating the performance metrics:
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    #results of the model
    results["Model"].append(model_name)
    results["Accuracy"].append(accuracy)
    results["Precision"].append(precision)
    results["Recall"].append(recall)
    results["F1 Score"].append(f1)


# In[10]:


#converting the results into dataframe for easier viewing
results_df = pd.DataFrame(results)
print(results_df)


# In[11]:


for model_name, model in models.items():
    # Calculate the accuracy on the training set
    train_accuracy = accuracy_score(y_train, model.predict(X_train))

    # Calculate the accuracy on the test set
    test_accuracy = accuracy_score(y_test, y_pred)

    print(f"{model_name}:")
    print(f"Training set accuracy: {train_accuracy}")
    print(f"Test set accuracy: {test_accuracy}\n")


# In[12]:


X1 = df.drop(['#posts', '#followers', '#follows', 'name==username', 'nums/length fullname', 'fake'], axis=1)


# In[13]:


# Split the data into a training set and a test set
X1_train, X1_test, y_train, y_test = train_test_split(X1, y, test_size=0.2, random_state=42)


# In[14]:


# Standardize the features
scaler = StandardScaler()
X1_train = scaler.fit_transform(X1_train)
X1_test = scaler.transform(X1_test)


# In[15]:


#For each model, running a for loop
for model_name, model in models.items():
    #train the model
    model.fit(X1_train, y_train)

    #Make predictions using the test set
    y_pred = model.predict(X1_test)

    #Calculating the performance metrics:
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    #results of the model
    results["Model"].append(model_name)
    results["Accuracy"].append(accuracy)
    results["Precision"].append(precision)
    results["Recall"].append(recall)
    results["F1 Score"].append(f1)


# In[16]:


#converting the results into dataframe for easier viewing
results_df = pd.DataFrame(results)
print(results_df)


# In[17]:


# Define the parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}


# In[18]:


# Create a GridSearchCV object
grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=5, scoring='accuracy')


# In[19]:


# Perform the grid search
grid_search.fit(X_train, y_train)


# In[20]:


# Get the best parameters
best_params = grid_search.best_params_

print(f"Best parameters: {best_params}")


# In[21]:


# Create a new SVM model with the optimal parameters
svm_model_optimized = SVC(C=0.1, gamma='scale', kernel='linear', random_state=42)


# In[22]:


# Train the model
svm_model_optimized.fit(X_train, y_train)


# In[23]:


# Evaluate the model
y_pred = svm_model_optimized.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
train_accuracy = accuracy_score(y_train, svm_model_optimized.predict(X_train))
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


# In[24]:


print(f"Test Accuracy: {test_accuracy}")
print(f"Train Accuracy: {train_accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")


# In[25]:


# Create a GradientBoostingClassifier model
gb_model = GradientBoostingClassifier(random_state=42)


# In[26]:


# Train the model
gb_model.fit(X_train, y_train)


# In[27]:


# Get feature importance
feature_importances = gb_model.feature_importances_


# In[28]:


feature_names = df.drop('fake', axis=1).columns


# In[29]:


# Create a DataFrame to display the feature importances
features_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})


# In[30]:


# Sort the DataFrame by importance in descending order
features_df = features_df.sort_values(by='Importance', ascending=False)

print(features_df)


# In[31]:


# Select the top 5 features
top_5_features = features_df.iloc[:5, 0].values

# Get the integer indices of the top 5 features from the original DataFrame
top_5_indices = [df.columns.get_loc(feature) for feature in top_5_features]

# Select only the top 5 features from X_train and X_test
X_train_selected = X_train[:, top_5_indices]
X_test_selected = X_test[:, top_5_indices]


# In[32]:


# Create a GradientBoostingClassifier model
gb_model_new = GradientBoostingClassifier(random_state=42)

# Train the model
gb_model_new.fit(X_train_selected, y_train)

# Evaluate the model
y_pred = gb_model_new.predict(X_test_selected)
test_accuracy = accuracy_score(y_test, y_pred)
train_accuracy = accuracy_score(y_train, gb_model_new.predict(X_train_selected))
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Test Accuracy: {test_accuracy}")
print(f"Train Accuracy: {train_accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")


# In[3]:


import pickle
your_model=gb_model_new
file_path="C:/Users/gouds/Downloads/model.pickle"
with open(file_path,'wb') as file:
    pickle.dump(your_model,file)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




