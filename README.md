
# NLP FastAPI Capstone

## Project Overview

This project implements an end-to-end NLP Text Classification system using TensorFlow, TF-IDF, FastAPI, Docker, and automated unit testing.

The trained model classifies text messages as Spam or Ham and exposes predictions through a REST API.

## System Architecture

Client Request  
↓  
FastAPI REST API  
↓  
TF-IDF Vectorizer  
↓  
TensorFlow Neural Network  
↓  
Spam / Ham Prediction  
↓  
Prediction Probability Response

## Technologies Used

- Python
- TensorFlow
- Scikit-learn
- TF-IDF
- FastAPI
- Uvicorn
- Docker
- Pytest
- Joblib

## Repository Files

- `main.py` - FastAPI REST API
- `nlp_text_classifier.keras` - Trained TensorFlow model
- `tfidf_vectorizer.joblib` - Saved TF-IDF vectorizer
- `requirements.txt` - Python dependencies
- `Dockerfile` - Docker configuration
- `test_api.py` - API unit tests
- `README.md` - Project documentation

## API Endpoint

### POST /predict

Example request:

```json
{
  "text": "claim your free cash prize now"
}
```

Example response:

```json
{
  "text": "claim your free cash prize now",
  "prediction": "Spam",
  "spam_probability": 0.99,
  "ham_probability": 0.01
}
```

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

API documentation:

```text
http://localhost:8000/docs
```

## Docker

Build Docker image:

```bash
docker build -t nlp-fastapi-capstone .
```

Run Docker container:

```bash
docker run -p 8000:8000 nlp-fastapi-capstone
```

## Unit Testing

Run tests:

```bash
pytest test_api.py
```

The tests validate API status codes, prediction response schema, probability ranges, and invalid request handling.

## Model Pipeline

Text Input → TF-IDF Vectorization → TensorFlow Neural Network → Spam/Ham Classification → Prediction Probabilities → REST API Response

## Conclusion

This capstone demonstrates a complete machine learning production workflow from a trained NLP model to a containerized REST API with automated tests and documentation.
