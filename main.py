from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import joblib

app = FastAPI(
    title="NLP Text Classification API",
    description="FastAPI service for Spam/Ham text classification",
    version="1.0"
)

model = tf.keras.models.load_model("nlp_text_classifier.keras")
vectorizer = joblib.load("tfidf_vectorizer.joblib")


class TextRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "NLP Text Classification API is running"}


@app.post("/predict")
def predict(request: TextRequest):
    vector = vectorizer.transform([request.text]).toarray().astype("float32")

    spam_probability = float(
        model.predict(vector, verbose=0)[0][0]
    )

    ham_probability = 1.0 - spam_probability

    prediction = (
        "Spam"
        if spam_probability >= 0.5
        else "Ham"
    )

    return {
        "text": request.text,
        "prediction": prediction,
        "spam_probability": round(spam_probability, 4),
        "ham_probability": round(ham_probability, 4)
    }
