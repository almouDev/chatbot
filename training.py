import pickle

from keras import Sequential, Input
from keras.datasets import mnist
import nltk
from keras.optimizers import SGD
from matplotlib import pyplot as plt
from nltk.stem import WordNetLemmatizer
from keras.layers import TextVectorization, Dense, Dropout, LSTM, Embedding, Flatten
import numpy as np

documents = []
x_train = list()
y_train = list()
classes = list()
lemmatizer = WordNetLemmatizer()
ignored_words = ["?", ".", "!"]
import json

# Example training data, of dtype `string`.
intents = json.loads(open("intents.json").read())
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        documents.append((pattern, intent["tag"]))
        classes.append(intent["tag"])
classes = sorted(list(set(classes)))
pickle.dump(classes, open('classes.pkl', 'wb'))
for doc in documents:
    pattern_words = str()
    pattern_words
    for w in nltk.word_tokenize(doc[0]):
        if w not in ignored_words: pattern_words += " " + lemmatizer.lemmatize(w)
    x_train.append([pattern_words])
    y_train.append([classes.index(doc[1])])
x_train = np.array(x_train)
y_train = np.array(y_train)
vectorizer = TextVectorization(output_mode="binary", ngrams=2)
vectorizer.adapt(x_train)
x_train = vectorizer(x_train)
print(y_train.shape)
pickle.dump({'config': vectorizer.get_config(),
             'weights': vectorizer.get_weights()}
            , open("vectorizer.pkl", "wb"))

model = Sequential()
model.add(Input(shape=(x_train.shape[1],)))
model.add(Dense(64,activation="relu"))
model.add(Dense(len(classes), activation='softmax'))
model.summary()
model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
inputs=vectorizer(np.array(["See you later"]))
hist = model.fit(x_train, y_train, epochs=200,verbose=1)
plt.plot(hist.history['accuracy'], label='Accurary')
plt.plot(hist.history['loss'], label='Loss')
plt.legend()
plt.title("Training our chatbot")
plt.show()
model.save('chatbot_model.h5', hist)
