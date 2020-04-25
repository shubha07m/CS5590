import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.wrappers.scikit_learn import KerasClassifier
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('Sentiment.csv')
# Keeping only the neccessary columns
data = data[['text', 'sentiment']]

data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)

X = pad_sequences(X)

# embed_dim = 128
# lstm_out = 196
#
#
# def createmodel():
#     model = Sequential()
#     model.add(Embedding(max_fatures, embed_dim, input_length=X.shape[1]))
#     model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
#     model.add(Dense(3, activation='softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer='RMSProp', metrics=['accuracy'])
#     return model
#
#
# model = KerasClassifier(build_fn=createmodel, verbose=0)

labelencoder = LabelEncoder()
integer_encoded = labelencoder.fit_transform(data['sentiment'])
y = to_categorical(integer_encoded)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)

batch_size = 10
# model = createmodel()
epochs = 2
# param_grid = dict(batch_size=batch_size, epochs=epochs)
# from sklearn.model_selection import GridSearchCV
# grid = GridSearchCV(estimator=model, param_grid=param_grid)
# grid_result = grid.fit(X_train, Y_train)
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

model = load_model('sentiment_model.h5')
model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=2)
score, acc = model.evaluate(X_test, Y_test, verbose=2, batch_size=batch_size)
# model.save('sentiment_model.h5')
print('The ' + model.metrics_names[1] + ' of this model is ' + str(acc * 100)[:5] + ' %')
print('The ' + model.metrics_names[0] + ' of this model is ' + str(score)[:7])
