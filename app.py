import streamlit as st
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pickle
import numpy as np

#Load the model and the tokenizer
my_model = load_model("model.h5")
my_tokenizer = pickle.load(open('mytoken.pk1', 'rb'))

# Function to generate next word predictions
def generate_predictions(seed_text, my_model, my_tokenizer, max_sequence_len=8, next_words=5):
    for _ in range(next_words):
        token_list = my_tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted_probs = my_model.predict(token_list)
        predicted_word = my_tokenizer.index_word[np.argmax(predicted_probs)]
        seed_text += " " + predicted_word
    return seed_text

# Streamlit UI
st.title("R204433P Shona Word Prediction")

# Input box for user to enter up to 5 words
user_input = st.text_input("Enter up to 5 words:")

# Check if input is not empty and contains up to 5 words
if user_input:
    input_words = user_input.split()
    if len(input_words) <= 5:
        # Generate predictions
        predicted_text = generate_predictions(user_input, my_model, my_tokenizer)
        st.write("The next 5 shona predicted words:", predicted_text)
    else:
        st.warning("Please enter up to 5 words.")
