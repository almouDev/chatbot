from keras.layers import TextVectorization
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import nltk
import numpy as np
from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('intents.json',encoding="utf-8").read())
classes = pickle.load(open('classes.pkl','rb'))
def clean_up_sentence(sentence):
    sentence_words=str()
    for word in nltk.word_tokenize(sentence):
        sentence_words+=" "+lemmatizer.lemmatize(word.lower())
    return sentence_words

def bow(sentence):
    sentence_words = clean_up_sentence(sentence)
    vectorizer_congfig = pickle.load(open("vectorizer.pkl","rb"))
    vectorizer=TextVectorization.from_config(vectorizer_congfig["config"])
    vectorizer.set_weights(vectorizer_congfig["weights"])
    vectorized_sentence=vectorizer(np.array([sentence_words]))
    return vectorized_sentence

def predict_class(sentence, model):
    sentence=clean_up_sentence(sentence)
    to_predict=bow(sentence)
    res=model.predict(to_predict)[0]
    ERROR_THRESHOLD = 0.2
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    if len(ints)==0:tag="noanswer"
    else:tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg,modele):
    ints = predict_class(msg, modele)
    res = getResponse(ints, intents)
    return res