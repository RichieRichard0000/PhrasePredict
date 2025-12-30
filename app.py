import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ----------------------------
# Load artifacts
# ----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("artifacts/idiom_lstm.h5")

@st.cache_resource
def load_tokenizer():
    with open("artifacts/tokenizer.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
tokenizer = load_tokenizer()

with open("artifacts/max_len.txt") as f:
    max_len = int(f.read())

# ----------------------------
# Sampling logic
# ----------------------------
def sample_top_k(preds, k=2, temperature=0.3):
    preds = np.log(preds + 1e-8) / temperature
    top_k = preds.argsort()[-k:]
    probs = np.exp(preds[top_k])
    probs /= probs.sum()
    return np.random.choice(top_k, p=probs)

def predict_next_word(seed_text, temperature, k):
    token_list = tokenizer.texts_to_sequences([seed_text.lower()])[0]
    token_list = pad_sequences(
        [token_list],
        maxlen=max_len - 1,
        padding="pre"
    )

    preds = model.predict(token_list, verbose=0)[0]
    idx = sample_top_k(preds, k=k, temperature=temperature)

    for word, index in tokenizer.word_index.items():
        if index == idx:
            return word
    return "<unknown>"

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Idiom Predictor", layout="centered")

st.title("🧠 Idiom Next-Word Predictor")
st.write("LSTM model trained on English idioms (Kaggle).")

seed = st.text_input(
    "Enter a partial idiom:",
    "fortune favors the"
)

temperature = st.slider(
    "Temperature (lower = safer)",
    0.1, 1.0, 0.3, 0.1
)

top_k = st.slider(
    "Top-K",
    1, 5, 2
)

if st.button("Predict"):
    next_word = predict_next_word(seed, temperature, top_k)
    st.success(f"{seed} **{next_word}**")
