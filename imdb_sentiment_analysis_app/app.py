from flask import Flask, render_template, request
import torch
import numpy as np
import re
import pickle

from model import SentimentLSTM


app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

MAX_SEQ_LEN = 12


# Load vocabulary
with open("vocabulary.pkl", "rb") as f:
    word_to_id = pickle.load(f)


# Load model
model = SentimentLSTM(
    vocab_size=len(word_to_id) + 1,
    output_size=1,
    embedding_dim=128,
    hidden_dim=128,
    num_layers=2,
    dropout_prob=0.3
)

model.load_state_dict(
    torch.load(
        "checkpoints/model_state_dict.pt",
        map_location=device
    )
)

model.to(device)
model.eval()


def preprocess_string(s):

    s = re.sub(r"[^\w\s]", "", s)

    s = re.sub(r"\d", "", s)

    return s


def add_padding(sequences, seq_len):

    padded = np.zeros(
        (len(sequences), seq_len),
        dtype=int
    )

    for i, review in enumerate(sequences):

        truncated_review = np.array(review)[:seq_len]

        if len(review) <= 0:
            continue

        padded[i, -len(review):] = truncated_review

    return padded


def predict_sentiment(text):

    word_seq = np.array([
        word_to_id[preprocess_string(word)]
        for word in text.split()
        if preprocess_string(word) in word_to_id
    ])

    word_seq = np.expand_dims(word_seq, axis=0)

    padded = add_padding(
        word_seq,
        MAX_SEQ_LEN
    )

    padded = torch.from_numpy(padded).to(device)

    hidden, cell = model.init_hidden(
        batch_size=1,
        device=device
    )

    with torch.inference_mode():

        y_logit, _ = model(
            padded,
            hidden,
            cell
        )

        y_prob = torch.sigmoid(y_logit)

    probability = y_prob.item()

    if probability > 0.5:
        sentiment = "positive"
        confidence = probability
    else:
        sentiment = "negative"
        confidence = 1 - probability

    return sentiment, confidence


@app.route("/", methods=["GET", "POST"])
def home():

    sentiment = None
    confidence = None
    review = ""

    if request.method == "POST":

        review = request.form.get("review", "")

        if review.strip():

            sentiment, confidence = predict_sentiment(review)

    return render_template(
        "index.html",
        sentiment=sentiment,
        confidence=confidence,
        review=review
    )


if __name__ == "__main__":
    app.run(debug=True)

