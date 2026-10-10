import streamlit as st
import joblib
model=joblib.load("sentiment.pkl")
st.set_page_config(layout='wide')
st.title("Sentiment Analysis Project")
st.sidebar.image("926015_passport_photo.PNG")
st.sidebar.title("About us")
st.sidebar.text("we are developing ml projects based on NLP ")
st.sidebar.title("About Projects")
st.sidebar.text("Sentiment Analysis is an NLP and machine learning project that analyzes text and classify as Positive or Negative")
st.sidebar.title("Contact us")
st.sidebar.text("+916283008506")
sample_review=st.selectbox("Sample reviews",options=['good food','quality was not good','awesome food'])
if st.button("Predict",key="b1"):
    pred=model.predict([sample_review])
    prob=model.predict_proba([sample_review])
    if pred[0]==0:
        st.error(f"Negative {prob[0][0]:.2f}")
    else:
        st.success(f"Posiive{prob[0][1]:.2f}")
        st.balloons()
sample_review2=st.text_input("Review")
if st.button("Predict",key="b2"):
    pred=model.predict([sample_review2])
    prob=model.predict_proba([sample_review2])
    if pred[0]==0:
        st.error(f"Negative {prob[0][0]:.2f}")
    else:
        st.success(f"Posiive{prob[0][1]:.2f}")
        st.balloons()
