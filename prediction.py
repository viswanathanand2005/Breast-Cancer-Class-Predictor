from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
import pandas as pd
import numpy as np

# Importing the data set
dataset = pd.read_csv('breast_cancer.csv')
print('The given dataset is: \n',dataset)

# Creating the feature matrix and dependent variable vector
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

print('The feature matrix is: \n',X)
print('The dependent variable vector is: \n',y)

# Splitting them into training and test sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# Scaling the training and the test sets
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Creating the Logistic regression object and training it with the training set
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

# Predicting the values of test set and comparing it
y_pred = classifier.predict(X_test)
print(np.concatenate((y_test.reshape(len(y_test),1),y_pred.reshape(len(y_pred),1)),axis=1))

# Predicting the class for a single input
ct = int(input("Enter Clump Thickness: "))
uc = int(input("Enter Uniformity of Cell Size: "))
us = int(input("Enter Uniformity of Cell Shape: "))
ma = int(input("Enter Marginal Adhesion: "))
se = int(input("Enter Single Epithelial Cell Size: "))
bn = int(input("Enter Bare Nuclei: "))
bc = int(input("Enter Bland Chromatin: "))
nn = int(input("Enter Normal Nucleoli: "))
m  = int(input("Enter Mitoses: "))

# Formatting input
input_data = [[ct, uc, us, ma, se, bn, bc, nn, m]]

# Scaling the input
input_scaled = sc.transform(input_data)

# Making the prediction
prediction = classifier.predict(input_scaled)

# Interpreting the result
if prediction[0] == 2:
    print("\n🟢 Prediction: **Benign Tumor** (Class 2) \n")
elif prediction[0] == 4:
    print("\n🔴 Prediction: **Malignant Tumor** (Class 4) \n")
else:
    print("\n⚠️ Prediction: Unknown Class →", prediction[0])

# Creating the confusion matrix and accuraccy score
cm = confusion_matrix(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)

print('The confusion matrix is: \n',cm)
print('The accuracy score is: ',accuracy)