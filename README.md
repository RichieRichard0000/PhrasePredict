# 🧠 Idiom Next-Word Predictor (LSTM)

From *"I know the theory"* to *"I know how it behaves"* 🚀

This project is a **hands-on NLP application** that uses an **LSTM-based language model** to predict the **next word in English idioms and sayings**.

The model is trained on a **custom dataset scraped from Wiktionary** and deployed locally using a **Streamlit frontend**.

---

## ✨ Project Motivation

I wanted a practical way to deeply understand **LSTMs and sequence modeling**, beyond just theory.

At the same time, I realized I personally struggle with remembering famous idioms exactly when I need them — so I decided to build something fun and educational to solve that.

---

## 📊 Dataset

- Source: Wiktionary – Category: English idioms
- Collected by **web scraping**
- Cleaned to remove noise and non-idiomatic phrases
- ~1500–2000 idioms after filtering
- One idiom per line

Example:
```
fortune favors the bold
spill the beans
once in a blue moon
break the ice
```

---

## 🧠 Model Overview

**Architecture**
```
Embedding Layer
↓
LSTM (256 units)
↓
Dense + Softmax (next-word probabilities)
```

- Word-level tokenization
- N-gram sequence generation
- Vocabulary size capped to reduce noise
- Trained on Kaggle (CPU)

---

## 🎛️ Decoding Strategy

To improve output quality during inference:

- **Temperature sampling** controls randomness
- **Top-K sampling** restricts predictions to most probable words

This balances correctness and creativity.

---

## 🔮 Sample Predictions

```
Input:  fortune favors the
Output: bold

Input:  spill the
Output: beans

Input:  break the
Output: ice
```

Some near-misses occur, which reflects the probabilistic nature of language models.

---

## 🖥️ Streamlit Frontend

The trained model is connected to a local Streamlit app that allows:

- Interactive text input
- Temperature & Top-K adjustment
- Real-time predictions

---

## 📁 Project Structure

```
idiom-lstm/
├── app.py
├── artifacts/
│   ├── idiom_lstm.h5
│   ├── tokenizer.pkl
│   └── max_len.txt
├── README.md
├── requirements.txt
└── venv/
```

---

## ▶️ How to Run Locally

### 1. Clone the repo
```bash
git clone <repo-url>
cd idiom-lstm
```

### 2. Create & activate virtual environment
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Streamlit app
```bash
streamlit run app.py
```

Open: http://localhost:8501

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- Streamlit
- NumPy, Pandas
- BeautifulSoup (scraping)

---

## ⚠️ Notes & Limitations

- Model is not perfectly accurate (small dataset)
- Idioms can have multiple valid continuations
- Focus is on **learning behavior**, not state-of-the-art accuracy

---

## 🚀 Future Improvements

- Bi-LSTM comparison
- Multi-word generation
- Streamlit Cloud deployment
- Transformer-based baseline

---

## 🎯 Learning Outcome

This project helped me move from understanding LSTM theory to **observing how sequence models behave in practice**, including their strengths and limitations.

---

⭐ If you found this project interesting, feel free to star the repository!
