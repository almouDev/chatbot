import string
import numpy as np
import tensorflow as tf
import pandas as pd
import json
import nltk
from tkinter import *
import sklearn
import random
import matplotlib.pyplot as plt
from keras.models import load_model

import prediction
tags = []
inputs = []
responses = {}
responses = {}
input_shape = 0
model=model = load_model('chatbot_model.h5')
class ChatBot(Tk):
    def __init__(self, model):
        Tk.__init__(self)
        self.model = model
        self.title("ChatBot")
        self.geometry("500x700")
        self.ChatBox = Text(self, bd=0, bg="white", font="Arial", pady=8)
        self.ChatBox.config(state=DISABLED)
        self.send_button = Button(self, font=("Verdana", 12, 'bold'), text="Send", width=12,
                                  bd=0, bg="#0073cf", activebackground="#3c9d9b", fg='#ffffff',
                                  command=self.sendMessage)
        self.entryBox = Text(self, bd=0, bg="white", font="Arial", padx=8, pady=8, width=40)
        self.label = Label(self, text="Chat Bot", font=("Verdana", 12, 'bold'), background="white", padx=8, pady=8)
        self.label.pack(fill=BOTH)
        self.ChatBox.pack(pady=8)
        self.entryBox.pack(side=LEFT, fill=Y)
        self.send_button.pack(side=LEFT, fill=BOTH)

    def sendMessage(self):
        msg = self.entryBox.get("1.0", 'end-1c').strip()
        self.entryBox.delete("0.0", END)
        if msg != '':
            self.ChatBox.config(state=NORMAL)
            self.ChatBox.insert(END, "You: " + msg + '\n\n')
            self.ChatBox.config(foreground="#442265", font=("Verdana", 12))

            res = self.preditct(msg)
            self.ChatBox.insert(END, "Bot: " + res + '\n\n')

            self.ChatBox.config(state=DISABLED)
            self.ChatBox.yview(END)

    def preditct(self, message):
        return prediction.chatbot_response(message,self.model)

chatBot = ChatBot(model)
chatBot.mainloop()
