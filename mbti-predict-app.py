######################
# Import libraries
######################
import numpy as np
import pandas as pd
import regex as re
import contractions
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st
import pickle
from PIL import Image
# import  data from another file
import mbti_data as data

######################
# Custom function
######################
## Clean the typing input
def clean(text):

    expanded_words = []
    for word in text.split():
        # using contractions.fix to expand the shortened words
        expanded_words.append(contractions.fix(word))

    expanded_text = ' '.join(expanded_words)

    text = expanded_text.lower()
    text = re.sub(r'[^\w\d\s]+', '', text)
    text = re.sub("[\_]+", '', text)

    return text

######################
# Page Title
######################

image = Image.open('mbti-logo.jpg')

st.image(image, use_column_width=True)

st.write("""
# MBTI Prediction Web App
This app predicts the **MBTI type** of a person from their _typing_!\n
Data obtained from Kaggle: [(MBTI) Myers-Briggs Personality Type Dataset](https://www.kaggle.com/datasets/datasnaek/mbti-type).
""")


######################
# Input typing (Side Panel)
######################

st.sidebar.header('User Input Features')

# typing default display
typing_input = """Well, in order to answer your question I would have to—— Wait. Wait a second. Is that… A BUTTERFLY over there?! Holy crap! That is so pretty! I’m curious about butterflies now. I am going to spend the next 4 hours researching all about the anatomy, mating habits, and migration patterns of the butterfly. Wow, I’m halfway done with this article about their wing patterns. So cool! And it says here that—— Wait. Wait a SECOND. Is that a rabbit over there? Did I just see a rabbit?! I LOVE rabbits. I’m gonna close this tab about butterflies and now open 5 tabs about the taxonomy of rabbits in the Americas. Frick yea."""

# Read typing input
typing = st.sidebar.text_area("Typing input", typing_input, height=400)

# display typing input on main page
st.header('Your Typing')
typing

######################
# Pre-built model
######################
# clean the typing
cleaned_text = clean(typing)

# Reads in saved model
load_model = pickle.load(open('mbti_model.pkl', 'rb'))

# Apply model to make predictions
prediction = load_model.predict([cleaned_text])
st.write('***')

######################
# applied MBTI data
######################
# MBTI type name
st.header('Predicted MBTI descriptors')
st.title('\n' + prediction[0] + " - " + data.mbti[prediction[0]]['alias'] + '\n\n')

# icon of the MBTI type
img = Image.open('mbti_ava/' + prediction[0] + '.png')
st.image(img, width=250)

# description of MBTI type separated every new line
for text in data.mbti[prediction[0]]['desc'].split('\n'):
    st.markdown(text)

# quote of MBTI type's famous people
st.write('#### ' +data.mbti[prediction[0]]['quote'][0])
st.caption('**_'+data.mbti[prediction[0]]['quote'][1] +'_**')
st.write('***')

# lists famous people with same type
st.header('Famous People of **_' + prediction[0] + '_**')
for ppl in data.mbti[prediction[0]]['people'].split(', '):
    st.write('- ' + ppl)

# link to know more the MBTI type
st.write('***\n\n###### For further information: ' +'[' + prediction[0] +
        ' Personality Type](' + data.mbti[prediction[0]]['link'] +').')
