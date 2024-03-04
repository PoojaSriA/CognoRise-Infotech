import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

data = pd.read_csv("mail_data.csv")
data.drop_duplicates(inplace=True)
data['Category'] = data['Category'].replace(['ham','spam'],['Not Spam','Spam'])

mess = data['Message']
cat = data['Category']

(mess_train,mess_test,cat_train,cat_test) = train_test_split(mess, cat, test_size=0.2)

cv = CountVectorizer(stop_words='english')
features = cv.fit_transform(mess_train)

model = MultinomialNB()
model.fit(features, cat_train)

features_test = cv.transform(mess_test)

def predict(message):
    input_message = cv.transform([message]).toarray()
    result = model.predict(input_message)
    return result

st.header('Spam Detection')


input_mess = st.text_input('Enter Message Here')

if st.button('Vaildate'):
    output = predict(input_mess)
    st.markdown(output)