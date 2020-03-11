#  importing the libraries #
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

#  loading the raw data #

mydata = pd.read_csv("uci_adult.csv")
incomedata = (mydata.dropna())  # Removing the null values

# Encoding Categorical features #

le = LabelEncoder()
incomedata.workclass = le.fit_transform(incomedata.workclass)
incomedata.education = le.fit_transform(incomedata.education)
incomedata.maritalstatus = le.fit_transform(incomedata.maritalstatus)
incomedata.occupation = le.fit_transform(incomedata.occupation)
incomedata.relationship = le.fit_transform(incomedata.relationship)
incomedata.race = le.fit_transform(incomedata.race)
incomedata.sex = le.fit_transform(incomedata.sex)
incomedata.nativecountry = le.fit_transform(incomedata.nativecountry)
incomedata.Income = le.fit_transform(incomedata.Income)

#  Checking the correlation #
#
numeric_features = incomedata.select_dtypes(include=[np.number])
corr = abs(numeric_features.corr())
print("The top 10 related features are below:")
print(corr['Income'].sort_values(ascending=False)[1:11])  # top 5 co-related features
#
# # Heat mapping different feature co-relation #
#
g = sns.heatmap(corr, annot=True)
plt.show()

# pair plotting top 8 co-related features #

features_eight = incomedata[['Income', 'educationnum', 'relationship', 'age', 'hoursperweek', 'capitalgain',
                             'sex', 'maritalstatus', 'capitalloss']]
sns.pairplot(features_eight, hue='Income')
plt.show()

# preparing features and target and reshaping #

target = (incomedata['Income'])
target = (pd.DataFrame(target))
features = incomedata.drop('Income', axis=1)
print("Please enter your choice: 1 for top 8 co-related features, any other integer for all features")
p = int(input())

if p == 1:
    features = incomedata[['educationnum', 'relationship', 'age', 'hoursperweek', 'capitalgain',
                           'sex', 'maritalstatus', 'capitalloss']]
    features_train, features_test, target_train, target_test = train_test_split(features, target,
                                                                                test_size=0.3, random_state=123)
    # building Naive Bays classifications model for top 8 co-related features #

    gnb = GaussianNB()
    target_pred_gnb = gnb.fit(features_train, target_train).predict(target_test)
    accuracy_gnb = accuracy_score(target_test, target_pred_gnb)
    print("Naive Bays Accuracy % using top 8 co-related features is:")
    print(str(round(accuracy_gnb * 100)) + ' %')

    # building K nearest neighbour model for top 8 co-related features#

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(features_train, target_train)
    Y_pred = knn.predict(features_train)
    acc_knn = round(knn.score(features_train, target_train) * 100, 2)
    print("K nearest neighbor Accuracy % using top 8 co-related features is:")
    print(str(acc_knn) + ' %')

    # building the SVM model #

    svc = SVC()
    target_pred = svc.fit(features_train, target_train).predict(features_test)
    acc_svc = accuracy_score(target_test, target_pred)
    print("SVM Accuracy % using top 8 co-related features is:")
    print(str(acc_svc * 100) + ' %')
    exit()
else:
    # Splitting training and test data set #
    features_train, features_test, target_train, target_test = train_test_split(features, target,
                                                                                test_size=0.3, random_state=123)
    # building Naive Bays classifications model for all features #
    gnb = GaussianNB()
    target_pred_gnb = gnb.fit(features_train, target_train).predict(target_test)
    accuracy_gnb = accuracy_score(target_test, target_pred_gnb)
    print("Naive Bays Accuracy % using all features is:")
    print(str(round(accuracy_gnb * 100)) + ' %')

    # building K nearest neighbour model #

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(features_train, target_train)
    Y_pred = knn.predict(features_train)
    acc_knn = round(knn.score(features_train, target_train) * 100, 2)
    print("K nearest neighbour Accuracy % using all features is:")
    print(str(acc_knn) + ' %')

    # building the SVM model #

    svc = SVC()
    target_pred = svc.fit(features_train, target_train).predict(features_test)
    acc_svc = accuracy_score(target_test, target_pred)
    print("SVM Accuracy % using all features is:")
    print(str(acc_svc * 100) + ' %')
    exit()
