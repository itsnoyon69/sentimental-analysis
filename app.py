import streamlit as st
import tensorflow as tf
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# =========================
# PAGE TITLE
# =========================

st.title("Sentiment Analysis")

st.write("Type a review and AI will predict sentiment")


# =========================
# LOAD DATASET
# =========================

data = pd.read_csv("IMDB Dataset.csv")

data["sentiment"] = data["sentiment"].str.strip()


# =========================
# TOKENIZER
# =========================

texts = data["review"].astype(str).to_numpy()

tokenizer = Tokenizer(
    num_words=10000,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(texts)


# =========================
# LOAD MODEL
# =========================

model = tf.keras.models.load_model(
    "sentiment_ai.h5"
)


# =========================
# USER INPUT
# =========================

user_text = st.text_area(
    "Enter your review"
)


# =========================
# PREDICTION
# =========================

if user_text:

    # Convert text → sequence

    seq = tokenizer.texts_to_sequences(
        [user_text]
    )

    # Padding

    pad = pad_sequences(
        seq,
        maxlen=200,
        padding='post',
        truncating='post'
    )

    # Prediction

    prediction = model.predict(pad)

    # Score

    score = prediction[0][0]

    # Show confidence

    st.write(f"Confidence Score: {score:.4f}")

    # Final result

    if score > 0.5:

        st.success("Positive 😊")

    else:

        st.error("Negative 😡")