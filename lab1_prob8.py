#  importing the libraries #
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#  loading the raw data #
mydata = pd.read_csv("myreg.csv")
cardata = pd.DataFrame(mydata.dropna())  # Removing the null values

# Encoding Categorical features #
le = LabelEncoder()
cardata.make = le.fit_transform(cardata.make)
cardata.fueltype = le.fit_transform(cardata.fueltype)
cardata.aspiration = le.fit_transform(cardata.aspiration)
cardata.numofdoors = le.fit_transform(cardata.numofdoors)
cardata.bodystyle = le.fit_transform(cardata.bodystyle)
cardata.drivewheels = le.fit_transform(cardata.drivewheels)
cardata.enginelocation = le.fit_transform(cardata.enginelocation)
cardata.enginetype = le.fit_transform(cardata.enginetype)
cardata.numofcylinders = le.fit_transform(cardata.numofcylinders)
cardata.fuelsystem = le.fit_transform(cardata.fuelsystem)

# #  Checking the correlation #
numeric_features = cardata.select_dtypes(include=[np.number])
corr = (cardata.corr())
print("top 10 co-related features are below:")
print(corr['price'].sort_values(ascending=False)[1:11])  # top 10 co-related features

# # # building the regression model #
Y = cardata['price']  # selecting the target

# #
# experiment with different number of features #

print("please input the number off parameter you want to check:")
print("*** 5 for 5 features, 10 for 10 features, any other integer for all features***")
p = int(input())

# #
if p == 5:
    # Building the model performance  for 5 most co-related features#
    X_five = cardata[['curbweight', 'width', 'enginesize', 'length', 'horsepower']]

    # Normalizing the data #
    scaler = StandardScaler()
    scaler.fit(X_five)
    X_five = pd.DataFrame(scaler.transform(X_five))
    x_train, x_test, y_train, y_test = train_test_split(X_five, Y, test_size=0.2, random_state=123)
    lm = linear_model.LinearRegression()
    model = lm.fit(x_train, y_train)
    prediction = model.predict(x_test)

    # Getting the model performance#
    print('MSE for 5 predictors:', mean_squared_error(y_test, prediction))
    print('R2 score for 5 predictors:', r2_score(y_test, prediction))
    exit()

if p == 10:
    # Building the model performance  for 10 most co-related features#
    X_ten = cardata[
        ['curbweight', 'width', 'enginesize', 'length', 'horsepower', 'wheelbase', 'drivewheels', 'bore', 'fuelsystem',
         'aspiration']]

    # Normalizing the data #
    scaler = StandardScaler()
    scaler.fit(X_ten)
    X_ten = pd.DataFrame(scaler.transform(X_ten))
    x_train, x_test, y_train, y_test = train_test_split(X_ten, Y, test_size=0.2, random_state=123)
    lm = linear_model.LinearRegression()
    model = lm.fit(x_train, y_train)
    prediction = model.predict(x_test)

    # Getting the model performance#
    print('MSE for 10 features:', mean_squared_error(y_test, prediction))
    print('R2 score for 10 features:', r2_score(y_test, prediction))
    exit()

else:
    # Building the model performance  for all features#
    X_all = cardata.drop('price', axis=1)

    # Normalizing the data #
    scaler = StandardScaler()
    scaler.fit(X_all)
    X_all = pd.DataFrame(scaler.transform(X_all))
    x_train, x_test, y_train, y_test = train_test_split(X_all, Y, test_size=0.2, random_state=123)
    lm = linear_model.LinearRegression()
    model = lm.fit(x_train, y_train)
    prediction = model.predict(x_test)

    # Getting the model performance#
    print('MSE for all features:', mean_squared_error(y_test, prediction))
    print('R2 score for all features:', r2_score(y_test, prediction))
    exit()
