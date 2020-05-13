import pandas
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from sklearn.preprocessing import LabelEncoder

# load dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv("breastcancer.csv")
le = LabelEncoder()
sc = StandardScaler()

target = (np.array(dataset))[:, -1]
target = le.fit_transform(target)
feature = (np.array(dataset))[:, :-1]
sc = sc.fit_transform(feature)
X_train, X_test, Y_train, Y_test = train_test_split(feature, target, test_size=0.25, random_state=87)

np.random.seed(155)
my_first_nn = Sequential()  # create model
my_first_nn.add(Dense(50, input_dim=31, activation='relu'))  # hidden layer
my_first_nn.add(Dense(20, activation='relu'))  # hidden layer
my_first_nn.add(Dense(1, activation='sigmoid'))  # output layer
my_first_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=100,
                                     initial_epoch=0)

print(my_first_nn.summary())
acc = ((my_first_nn.evaluate(X_test, Y_test))[1])*100
loss = ((my_first_nn.evaluate(X_test, Y_test))[0])
print('accuracy of the model is: ' + str(acc)[:5] + '%')
print('loss of the model is: ' + str(loss)[:5])
