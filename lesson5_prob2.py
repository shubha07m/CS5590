#### importing all the required libraries ####

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import seaborn as sns

sns.set()

data = pd.read_csv('winequality-red.csv')
winedata = data.dropna()

### dividing into training and test data set ###

X = winedata[['alcohol', 'volatile acidity', 'sulphates']]

Y = np.log(winedata['quality'])

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=123)

#### building the model ####

lm = linear_model.LinearRegression()
model = lm.fit(x_train, y_train)

#### Getting the model performance ####
print(model.score(x_test, y_test))
prediction = model.predict(x_test)
print('MSE', mean_squared_error(y_test, prediction))
print('R2 score', r2_score(y_test, prediction))

##### correlation part #####

numeric_features = winedata.select_dtypes(include=[np.number])
corr = abs(numeric_features.corr())
print(corr['quality'].sort_values(ascending=False)[1:4])

g = sns.heatmap(corr, annot=True)
plt.show()
