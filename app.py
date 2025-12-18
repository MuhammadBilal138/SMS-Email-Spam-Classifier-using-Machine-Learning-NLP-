import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

stop_words = set(stopwords.words('english'))
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stop_words and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


st.title("Email/Sms Spam CLassifier")

input_sms = st.text_input("Enter the message")

if st.button("Predict"):
    # 1. Preprocess
    transform_sms = transform_text(input_sms)

    # 2. Vectorize
    vector_input = tfidf.transform([transform_sms])

    # 3. Predict
    result = model.predict(vector_input)[0]

# 4. Display Result
    if result == 1:
        st.error("Spam Message Detected")
        st.write("This message appears to be **spam**. Please avoid clicking suspicious links or sharing personal information.")
    else:
        st.success("Safe Message (Not Spam)")
    st.write("This message looks **safe** and does not appear to be spam.")

