import torch
from imblearn.over_sampling import BorderlineSMOTE, SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl

# 1.Load data
clincal_train = pd.read_csv(".\data/filled_train.csv",encoding='gbk')
clincal_test = pd.read_csv(".\data/filled_test.csv",encoding='gbk')


X_train = clincal_train.iloc[:, 1:20]
Y_train = clincal_train["转移"]
# np.random.seed(51)
# Training set balance processing
sm = BorderlineSMOTE(kind='borderline-1',k_neighbors = 10,m_neighbors=5)
# sm = SMOTE()
X_train, Y_train = sm.fit_resample(X_train, Y_train)


X_test = clincal_test.iloc[:,1:20]
# print(X_test)
Y_test= clincal_test["转移"]


# standardization
transfer = StandardScaler()
x_train = transfer.fit_transform(X_train)
x_test = transfer.transform(X_test)
#
# 3.Model training
# Instantiation predictor
# estimator = RandomForestClassifier()
estimator = GradientBoostingClassifier(n_estimators=300)
# estimator.fit(x_train, Y_train)

# Softmax regression was used for classification.
# estimator = LogisticRegression()
# estimator = LogisticRegression(C=1.0, tol=1e-6, multi_class='multinomial', solver='newton-cg')
estimator.fit(x_train, Y_train)
y_predict = estimator.predict(x_test)
print(y_predict)
# print("Compare the real value with the predicted value：\n", y_predict == Y_test)
score = estimator.score(x_test, Y_test)
print("The accuracy of SVM is：\n", score)
auc = skl.metrics.roc_auc_score(Y_test, y_predict)
matrix = skl.metrics.confusion_matrix(Y_test, y_predict, labels=[0, 1])
tn, fp, fn, tp = matrix.ravel()
Accuracy = estimator.score(x_test, Y_test)

sensitivity = tp/(tp+fn)
specificity = tn / (tn + fp)
PPV = tp / (tp + fp)
NPV = tn / (tn + fn)
outputs3 = estimator.predict_proba(x_test)
outputs3 = torch.tensor(outputs3)
# print(outputs3)
output = outputs3[:,1]

y_predict_1 = output.unsqueeze(-1)

# print(y_predict_1)
y_train_1 = Y_train
print(y_train_1.shape)
y_test_1 = Y_test
# print('y_test_1:',y_test_1)



