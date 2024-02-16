"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset
dataset=pd.read_csv("heart.csv")
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataset[:,:-1].values, dataset[:,-1].values, test_size=0.2)

# Create an SVM model
svm = SVC(kernel='rbf', C=1)

# Train the SVM model
svm.fit(X_train, y_train)
# Test the SVM model
y_pred = svm.predict(X_test)
# Evaluate the performance of the SVM model
accuracy = accuracy_score(y_test, y_pred)*100
print('Accuracy:', accuracy)
"""

# Importing necessary libraries
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Loading the dataset
heart_data = pd.read_csv('heart.csv')

# Splitting the dataset into features and target variable
X = heart_data.loc[:,].values
y = heart_data.iloc[:, -1].values

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Creating a non-linear SVM classifier with RBF kernel
classifier = SVC(kernel='rbf', random_state=0)

# Training the classifier
classifier.fit(X_train, y_train)

# Predicting the target variable for the test set
y_pred = classifier.predict(X_test)
print(y_pred)
"""
"""
# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Loading the dataset
data = pd.read_csv('heart.csv')

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop("HeartDisease", axis=1), data[:,:-1], test_size=0.2, random_state=42)

# Creating the SVM model with a non-linear kernel
svm_model = SVC(kernel='rbf')

# Training the model on the training set
svm_model.fit(X_train, y_train)

# Predicting the labels of the test set
y_pred = svm_model.predict(X_test)

# Calculating the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

# Printing the accuracy of the model
print('Accuracy:', accuracy)"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import seaborn as sns
df=pd.read_csv("heart.csv")
df.head()
df.isnull().sum()
print(df)

print(df.info())
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
sns.heatmap(df.corr(),annot=True,cmap="terrain")
sns.pairplot(data=df)
df.hist(figsize=(12,12),layout=(5,3))
print(df)
from  sklearn.preprocessing import StandardScaler
ss=StandardScaler()
col=["age","trestbps","chol","thalach","oldpeak"]
df[col]=ss.fit_transform(df[col])
x=df.drop(["target"],axis=1)
y=df["target"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
svm_model = SVC(kernel='rbf')

# Training the model on the training set
svm_model.fit(x_train, y_train)

# Predicting the labels of the test set
y_pred = svm_model.predict(x_test)
print(y_pred)
