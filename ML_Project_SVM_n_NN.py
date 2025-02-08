# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 21:28:01 2024

@author: Gilbert Hernandez
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import time


#====================================
# Loading dataset
#====================================
## Make sure to download the data from the link in the Readme file and save it into the same directory as the code
data = pd.read_csv('./loan_dataset.csv')



#======================================
#Performing a basic data exploration
#======================================

print(f"{data.columns}\n")
print(f"{data.info()}\n")
print(f"{data.describe()}\n")


if data.isnull().values.any():
  print('There are null values')
else:
  print('There are no null values')
  

for col in data.select_dtypes(include=['object']).columns:
    print(f"Value counts for column: '{col}':")
    print(data[col].value_counts())
    print("\n")
    

#======================================================================
# Droping categorical features since were mapped into numerical values
#======================================================================
data = data.drop(columns=['id','year','issue_d','final_d','home_ownership','income_category','term','application_type','purpose','interest_payments','loan_condition','grade','region'])


#======================================================================
# Checking for class distribution
#======================================================================

counts = data['loan_condition_cat'].value_counts()
ratio = counts[0] / counts[1]
print(f"Target variable distribution: \n{counts}")
print(f"Imbalance rati0: \n{ratio}\n")


#===========================================================================================
# Computing feature correlation to optimize training runtime and Recursive Feature Selection
#===========================================================================================

# Feature Correlation
correlation_matrix = data.corr()

# Plot the heatmap
plt.figure(figsize=(16, 14))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()

# Get the correlations with the target variable and sort descendingly
target_correlation = correlation_matrix["loan_condition_cat"].sort_values(ascending=False)

# Display the ranked correlations
print("Features ranked by correlation with the target:")
print(target_correlation)


"""Feature selection based on Correlation to target variable"""
# data = data.drop(columns=['application_type_cat','emp_length_int','annual_inc','home_ownership_cat','income_cat','total_pymnt','total_rec_prncp'])


# Computing Recursive Feature Selection to compare against feature correlation method RF Based Classifier
X = data.drop("loan_condition_cat", axis=1)  # Features
y = data["loan_condition_cat"]  # Target variable

# # Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Initialize Random Forest model and RFE
# Initialize a Random Forest Classifier
model = RandomForestClassifier(random_state=42)

# Recursive Feature Elimination (RFE) to select the top 10 features
rfe = RFE(estimator=model, n_features_to_select=10)
rfe.fit(X_train, y_train)

# Step 4: Get selected features
selected_features = X_train.columns[rfe.support_]
print("Selected Features:")
print(selected_features)


""" Selected features based on Recursive Feature Selection """

data = data[['emp_length_int', 'annual_inc', 'loan_amount', 'purpose_cat',
       'interest_rate', 'dti', 'total_pymnt', 'total_rec_prncp', 'recoveries',
       'installment','loan_condition_cat']]


# Defining features and target variable
X = data.drop("loan_condition_cat", axis=1)  # Features
y = data["loan_condition_cat"]  # Target variable

# Convert to numeric arrays for optimal computation
X = X.astype(float)
y = y.astype(int)


#===========================
# Handling Class Imbalance
#===========================
""" Uncomment this code to try the models with SMOTE Technique"""
 
# # Apply SMOTE to the training set
# smote = SMOTE(random_state=42)
# Xresampled, yresampled = smote.fit_resample(X, y)
# X_train, X_test, y_train, y_test = train_test_split(Xresampled, yresampled, test_size=0.7, random_state=42)


# ============================
# Normalization
# ============================
def normalize(x, mode='mean'):
    x = np.asarray(x, dtype=float)
    mean_val = np.mean(x, axis=0)
    std_val = np.std(x, axis=0)
    std_val[std_val == 0] = 1  # Avoid division by zero
    if mode == 'mean':
        return (x - mean_val) / std_val
    else:
        raise ValueError("Only 'mean' normalization is supported")
        
X_train_normalized = pd.DataFrame(normalize(X_train.values, mode='mean'), columns=X_train.columns)

X_test_normalized = pd.DataFrame(normalize(X_test.values, mode='mean'), columns=X_test.columns)


#===============================================
# Classifier 1: Support Vector Machine (SVM)
#===============================================

class LinearSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        # Map 0 -> -1 and 1 -> 1 for SVM training
        y_ = np.where(y == 0, -1, 1)

        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.learning_rate * y_[idx]

    def predict(self, X):
        # Compute decision boundary
        approx = np.dot(X, self.w) - self.b
        # Map predictions back to original labels (0 and 1)
        return np.where(np.sign(approx) == -1, 0, 1)
    

# Train the Linear SVM model
svm = LinearSVM(learning_rate=0.001, lambda_param=0.01, epochs=1000)

# Measure training time
start_time = time.time()
svm.fit(X_train_normalized.values, y_train.values)
end_time = time.time()

# Report training time
training_time = end_time - start_time
print(f"Training Time: {training_time:.2f} seconds")

# Evaluate the model
y_pred = svm.predict(X_test_normalized)
print("Linear SVM Classification Report:")
print(classification_report(y_test,y_pred))


#===================================================
# Enhanced Linear SVM with Hyperparameter Tunning
#===================================================

class LinearSVM_v2(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=0.001, lambda_param=0.01, epochs=1000, C=1.0,
                 early_stopping=False, patience=5, validation_fraction=0.1, class_weight=None):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.C = C
        self.early_stopping = early_stopping
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.class_weight = class_weight
        self.w = None
        self.b = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        self.classes_ = np.unique(y)

        # Convert labels {0,1} to {-1,1}
        y_ = np.where(y == 0, -1, 1)

        # If class_weight is specified, create a weight array for each sample
        if self.class_weight is not None:
            # For example, if class_weight={0:1, 1:12}:
            # For y_ = -1 (original=0), weight = self.class_weight[0]
            # For y_ = +1 (original=1), weight = self.class_weight[1]
            class_weights_array = np.zeros_like(y_, dtype=float)
            for i in range(len(y_)):
                if y_[i] == -1:
                    class_weights_array[i] = self.class_weight[0]
                else:
                    class_weights_array[i] = self.class_weight[1]
        else:
            class_weights_array = np.ones_like(y_, dtype=float)

        if self.early_stopping and self.validation_fraction > 0.0:
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_, test_size=self.validation_fraction, random_state=42, stratify=y_
            )
            # Also split class weights
            w_train, w_val = train_test_split(class_weights_array, test_size=self.validation_fraction,
                                              random_state=42, stratify=y_)
        else:
            X_train, y_train = X, y_
            w_train = class_weights_array
            X_val, y_val, w_val = None, None, None

        n_samples, n_features = X_train.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        best_score = -np.inf
        epochs_no_improve = 0

        for epoch in range(self.epochs):
            for idx, x_i in enumerate(X_train):
                condition = y_train[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    # No violation
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    # Violation occurs, incorporate class weight
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - self.C * w_train[idx] * np.dot(x_i, y_train[idx]))
                    self.b -= self.learning_rate * (self.C * w_train[idx] * y_train[idx])

            if self.early_stopping and X_val is not None:
                val_preds = self.predict(X_val)
                # Convert back from {-1,1} to {0,1} for evaluation
                current_score = f1_score(np.where(y_val == -1, 0, 1), val_preds, average='macro')
                if current_score > best_score:
                    best_score = current_score
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}. Best validation score: {best_score}")
                    break

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        approx = np.dot(X, self.w) - self.b
        return np.where(approx < 0, 0, 1)


"""" After performing Gridsearch with the following hyperparameter, we got this enhanced model """
# Define hyperparameters
hyperparams = {
    'learning_rate': [0.1, 0.01, 0.001, 0.0001, 0.00001],
    'epochs': [100, 300, 500, 700, 1000],
    'C': [0.1, 0.5, 1.0, 2.0, 10.0]
}


# Now specify the class weights to handle class imbalance:
class_weights = {0:1, 1:7}


svm_model = LinearSVM_v2(
    learning_rate = 0.00001,
    lambda_param = 0.01,
    epochs = 100,
    C= 0.5,
    early_stopping = False,
    patience = 5,
    validation_fraction = 0.1,
    class_weight=class_weights
)


start_time = time.time()
svm_model.fit(X_train_normalized, y_train)
end_time = time.time()

training_time = end_time - start_time
print(f"Training Time: {training_time:.2f} seconds")

y_pred = svm_model.predict(X_test_normalized)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", classification_report(y_test, y_pred))


#==========================================================
# Defining function for Dimensionality Reduction using PCA
#==========================================================

# ============================
#  Defining function for Dimensionality Reduction using PCA
# ============================
class PCAFromScratch:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        covariance_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.components_ = eigenvectors[:, :self.n_components]

    def transform(self, X):
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# ============================
# Kernel Functions
# ============================
def linear_kernel(X, Y):
    return np.dot(X, Y.T)

def rbf_kernel(X, Y, sigma=1.0):
    pairwise_sq_dists = (
        np.sum(X**2, axis=1).reshape(-1, 1) - 2 * np.dot(X, Y.T) + np.sum(Y**2, axis=1)
    )
    return np.exp(-pairwise_sq_dists / (2 * sigma**2))

# =================================
# Optimized / Sophisticated Dual SVM
# =================================
class DualSVM:
    def __init__(self, C=1.0, kernel='linear', sigma=1.0, epochs=1000, learning_rate=0.001, tol=1e-4):
        self.C = C
        self.kernel_type = kernel
        self.sigma = sigma
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.tol = tol
        self.alpha_ = None
        self.X_ = None
        self.y_ = None
        self.kernel_matrix_ = None

    def _compute_kernel_matrix(self, X):
        if self.kernel_type == 'linear':
            return linear_kernel(X, X)
        elif self.kernel_type == 'rbf':
            return rbf_kernel(X, X, self.sigma)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.X_ = X
        self.y_ = np.where(y == 0, -1, 1)  # Convert {0,1} to {-1,1}
        n_samples = X.shape[0]

        # Precompute the kernel matrix
        self.kernel_matrix_ = self._compute_kernel_matrix(X) * np.outer(self.y_, self.y_)

        # Initialize alpha
        self.alpha_ = np.zeros(n_samples)

        for epoch in range(self.epochs):
            grad = 1 - np.dot(self.kernel_matrix_, self.alpha_)
            self.alpha_ += self.learning_rate * grad
            self.alpha_ = np.clip(self.alpha_, 0, self.C)

            # Check for convergence
            if np.linalg.norm(grad) < self.tol:
                print(f"Converged at epoch {epoch + 1}")
                break

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        kernel = (
            linear_kernel(X, self.X_)
            if self.kernel_type == 'linear'
            else rbf_kernel(X, self.X_, self.sigma)
        )
        decision = np.dot(kernel, self.alpha_ * self.y_)
        return np.where(decision >= 0, 1, 0)

# ============================
# Data Preparation
# ============================
data = data[['emp_length_int', 'annual_inc', 'loan_amount', 'purpose_cat',
             'interest_rate', 'dti', 'total_pymnt', 'total_rec_prncp', 'recoveries',
             'installment', 'loan_condition_cat']]

X = data.drop("loan_condition_cat", axis=1)
y = data["loan_condition_cat"]

X = X.astype(float)
y = y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

fraction = 0.1
X_train_sub = X_train.sample(frac=fraction, random_state=42)
y_train_sub = y_train.loc[X_train_sub.index]



X_train_normalized = normalize(X_train_sub.values, mode='mean')
X_test_normalized = normalize(X_test.values, mode='mean')

# ============================
# PCA for Dimensionality Reduction
# ============================
n_components = 10
pca = PCAFromScratch(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_normalized)
X_test_pca = pca.transform(X_test_normalized)

# ============================
# Dual SVM Training
# ============================
svm_model = DualSVM(C=0.5, kernel='linear', epochs=100, learning_rate=1e-5, tol=1e-4)

start_time = time.time()
svm_model.fit(X_train_pca, y_train_sub)
end_time = time.time()

# Predictions
y_pred = svm_model.predict(X_test_pca)

# ============================
# Evaluation
# ============================
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", classification_report(y_test, y_pred))

training_time = end_time - start_time
print(f"\nTraining Time: {training_time:.2f} seconds")



