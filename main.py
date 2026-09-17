import os
import re
import pickle
import numpy as np
import streamlit as st
import tensorflow as tf


# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="Twitter Sentiment Analyzer",
    page_icon="💬",
    layout="centered"
)

st.title("💬 Twitter Sentiment Analyzer")
st.write("Choose a model and enter a tweet to analyze its sentiment.")


# ==========================================
# FILE PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SVM_PATH = os.path.join(
    BASE_DIR,
    "sentiment_svm_pipeline.pkl"
)

ENCODER_PATH = os.path.join(
    BASE_DIR,
    "label_encoder.pkl"
)

RNN_PATH = os.path.join(
    BASE_DIR,
    "sentiment_rnn_model.keras"
)


# ==========================================
# CHECK MODEL FILES
# ==========================================

if not os.path.exists(SVM_PATH):
    st.error("❌ sentiment_svm_pipeline.pkl not found.")

if not os.path.exists(ENCODER_PATH):
    st.error("❌ label_encoder.pkl not found.")

if not os.path.exists(RNN_PATH):
    st.error("❌ sentiment_rnn_model.keras not found.")


# ==========================================
# LOAD SVM MODEL
# ==========================================

@st.cache_resource
def load_svm_model():

    with open(SVM_PATH, "rb") as file:
        model = pickle.load(file)

    return model


# ==========================================
# LOAD LABEL ENCODER
# ==========================================

@st.cache_resource
def load_label_encoder():

    with open(ENCODER_PATH, "rb") as file:
        encoder = pickle.load(file)

    return encoder


# ==========================================
# LOAD RNN MODEL
# ==========================================

@st.cache_resource
def load_rnn_model():

    model = tf.keras.models.load_model(
        RNN_PATH,
        compile=False
    )

    return model


# ==========================================
# TEXT PREPROCESSING
# ==========================================

def preprocess_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\.\S+",
        " ",
        text
    )

    # Remove mentions
    text = re.sub(
        r"@\w+",
        " ",
        text
    )

    # Remove # symbol
    text = re.sub(
        r"#",
        "",
        text
    )

    # Keep letters and basic punctuation
    text = re.sub(
        r"[^a-z\s!?]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ==========================================
# MODEL SELECTION
# ==========================================

st.subheader("Select Model")

model_choice = st.radio(
    "Choose one:",
    [
        "SVM",
        "SimpleRNN"
    ],
    horizontal=True
)


# ==========================================
# USER INPUT
# ==========================================

st.subheader("Enter Tweet")

tweet = st.text_area(
    "Write your tweet here:",
    placeholder="Example: I really love this product!",
    height=130
)


# ==========================================
# ANALYZE BUTTON
# ==========================================

if st.button(
    "🔍 Analyze Sentiment",
    use_container_width=True
):

    if not tweet.strip():

        st.warning(
            "⚠️ Please enter a tweet first."
        )

    else:

        # Clean text
        cleaned_tweet = preprocess_text(tweet)

        # ==================================
        # SVM PREDICTION
        # ==================================

        if model_choice == "SVM":

            try:

                svm_model = load_svm_model()
                label_encoder = load_label_encoder()

                prediction = svm_model.predict(
                    [cleaned_tweet]
                )

                predicted_class = prediction[0]

                sentiment = label_encoder.inverse_transform(
                    [predicted_class]
                )[0]

                st.success(
                    f"### Sentiment: {sentiment}"
                )

                st.write(
                    f"**Model Used:** SVM"
                )

                st.write(
                    f"**Input:** {tweet}"
                )

            except Exception as e:

                st.error(
                    f"SVM Error: {e}"
                )


        # ==================================
        # SIMPLE RNN PREDICTION
        # ==================================

        elif model_choice == "SimpleRNN":

            try:

                rnn_model = load_rnn_model()
                label_encoder = load_label_encoder()

                # IMPORTANT:
                # Send real TensorFlow string tensor
                # instead of NumPy string array

                input_text = tf.constant(
                    [cleaned_tweet],
                    dtype=tf.string
                )

                # Prediction
                prediction = rnn_model(
                    input_text,
                    training=False
                )

                prediction = prediction.numpy()

                # Get class with highest probability
                predicted_class = np.argmax(
                    prediction,
                    axis=1
                )[0]

                # Convert number back to sentiment
                sentiment = label_encoder.inverse_transform(
                    [predicted_class]
                )[0]

                # Confidence
                confidence = (
                    float(
                        np.max(prediction)
                    ) * 100
                )

                st.success(
                    f"### Sentiment: {sentiment}"
                )

                st.write(
                    f"**Model Used:** SimpleRNN"
                )

                st.write(
                    f"**Input:** {tweet}"
                )

                st.info(
                    f"Confidence: {confidence:.2f}%"
                )

            except Exception as e:

                st.error(
                    f"SimpleRNN Error: {e}"
                )