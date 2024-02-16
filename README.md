# -chronic-heart-disease-prediction-using-machine-learning
using machine learning  i was predicted the heart failure using the support vector machine from the unsupervised learning 

Import Libraries:

    pandas as pd: Used for data manipulation and analysis (DataFrame creation, reading CSV files, etc.).
    sklearn.model_selection import train_test_split: For splitting data into training and testing sets.
    sklearn.svm import SVC: For building the Support Vector Machine (SVM) model.
    sklearn.metrics import accuracy_score: To evaluate the model's performance.
    seaborn as sns: For creating visual data representations (heatmaps, pair plots, histograms).
    matplotlib.pyplot as plt: For generating plots.

Read and Explore Data:

    df=pd.read_csv("heart.csv"): Reads heart disease data from the "heart.csv" file, creating a DataFrame df.
    df.head(): Displays the first few rows of df to get a glimpse of the data.
    df.isnull().sum(): Checks for missing values in each column.
    print(df): Prints the entire DataFrame (if not too large).
    print(df.info()): Provides detailed information about the DataFrame's data types, memory usage, etc.

Visualization:

    sns.heatmap(df.corr(),annot=True,cmap="terrain"): Creates a heatmap to visualize correlations between the numerical features (columns).
    sns.pairplot(data=df): Shows pairwise relationships between features with scatter plots and histograms.
    df.hist(figsize=(12,12),layout=(5,3)): Visualizes distributions of each feature using histograms.

Data Preprocessing:

    from sklearn.preprocessing import StandardScaler: Imports the StandardScaler class for normalization.
    ss=StandardScaler(): Creates an instance of StandardScaler.
    col=["age","trestbps","chol","thalach","oldpeak"]: Selects numerical features for normalization.
    df[col]=ss.fit_transform(df[col]): Normalizes the selected features using the fitted StandardScaler.

Train-Test Split:

    x=df.drop(["target"],axis=1): Creates a DataFrame x containing all features except the target variable.
    y=df["target"]: Extracts the target variable into a NumPy array y.
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3): Splits x and y into training and testing sets, using 30% for testing.

Model Training and Prediction:

    svm_model = SVC(kernel='rbf'): Creates an SVM model with the radial basis function (RBF) kernel.
    svm_model.fit(x_train, y_train): Trains the model on the training data.
    y_pred = svm_model.predict(x_test): Uses the trained model to predict labels for the test data.
    print(y_pred): Prints the predicted labels.

Additional Considerations:

    Missing value handling: While not explicitly shown, it's important to address missing values before training the model (e.g., imputation, deletion).
    Feature engineering: Additional feature creation or selection might improve model performance.
    Hyperparameter tuning: Optimizing hyperparameters (e.g., C, gamma) using grid search or cross-validation could further improve results.
    Evaluation metrics: While accuracy is used, consider employing other evaluation metrics like precision, recall, F1-score, and AUC-ROC depending on the problem and dataset characteristics.
