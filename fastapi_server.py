"""
FASTAPI SERVER FOR LSTM SENTIMENT ANALYSIS

This FastAPI application serves an LSTM neural network that predicts whether movie reviews
are positive or negative. We'll explain each FastAPI concept as we go!
"""

# ============================================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ============================================================================
# FastAPI: A modern Python web framework for building APIs
from fastapi import FastAPI
# Pydantic: Used for data validation (ensuring correct data types)
from pydantic import BaseModel
# For loading our pre-trained LSTM model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence
# For pre-processing text
from tensorflow.keras.datasets import imdb
import pickle
import numpy as np

# ============================================================================
# STEP 2: CREATE A FASTAPI APPLICATION INSTANCE
# ============================================================================
# This is the main FastAPI app object that handles all HTTP requests
# Think of it as the "server" that listens for incoming requests
app = FastAPI(
    title="LSTM Sentiment Analysis API",  # Name of your API (shows in documentation)
    description="Predict sentiment of movie reviews using LSTM",
    version="1.0.0"
)

# ============================================================================
# STEP 3: LOAD THE TRAINED LSTM MODEL AND CONFIGURATION
# ============================================================================
# Load the pre-trained LSTM model we saved earlier
model = load_model('imdb_lstm_model.h5')

# Load the vocabulary size (number of unique words the model knows)
with open('imdb_vocab_size.pkl', 'rb') as f:
    vocab_size = pickle.load(f)

# Load the max sequence length (how long reviews should be padded to)
with open('imdb_max_length.pkl', 'rb') as f:
    max_length = pickle.load(f)

# Load the word index (mapping of words to numbers used by IMDB dataset)
word_index = imdb.get_word_index()

# Create a reverse mapping: number -> word (useful for understanding predictions)
reverse_word_index = {v: k for k, v in word_index.items()}

print("Model loaded successfully!")

# ============================================================================
# DUMMY DATA / DEFAULTS
# ============================================================================
# Provide a default negative review example. This can be used as a fallback
# when no review is provided in the request body, and is useful for testing
# and documentation. We'll also expose this example in the health check.
DUMMY_NEGATIVE_REVIEW = (
    "I disliked this movie a lot. Terrible pacing, poor acting, and the plot made no sense."
)

# A small list of negative examples (could be expanded later)
DUMMY_NEGATIVE_EXAMPLES = [
    DUMMY_NEGATIVE_REVIEW,
    "Boring and slow. I fell asleep halfway through.",
    "Worst movie I've seen this year. Don't waste your time."
]

# ============================================================================
# STEP 4: DEFINE REQUEST DATA MODEL USING PYDANTIC
# ============================================================================
# Pydantic models define the structure of data coming INTO the API
# This ensures that clients send valid data in the correct format
class ReviewRequest(BaseModel):
    """
    This class defines what data we expect when someone sends a review to our API.

    We assign a default negative review so that the API can be called without
    a request body during quick tests; FastAPI will use this default value
    when the client omits the body.

    Attributes:
        review (str): The movie review text that we want to classify as positive/negative
    """
    review: str = DUMMY_NEGATIVE_REVIEW

# Define the response data model
class PredictionResponse(BaseModel):
    """
    This class defines what data we send BACK to the client after making a prediction.
    
    Attributes:
        sentiment (str): Either "positive" or "negative"
        confidence (float): A score between 0 and 1 (0=negative, 1=positive)
        review_preview (str): First 50 characters of the review
    """
    sentiment: str
    confidence: float
    review_preview: str

# ============================================================================
# STEP 5: CREATE HELPER FUNCTION TO CONVERT TEXT TO NUMBERS
# ============================================================================
def decode_review(encoded_review):
    """
    The IMDB dataset uses numbers instead of words.
    This function converts encoded numbers back to text (for understanding).
    """
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def encode_review(text):
    """
    This function converts text (words) into numbers that the model understands.
    It uses the IMDB word index to map words to their numeric codes.
    """
    # Use Keras' text_to_word_sequence so tokenization matches IMDB preprocessing
    tokens = text_to_word_sequence(text)

    # Convert each word to its numeric code
    encoded = []
    for word in tokens:
        if word in word_index:
            # Add 3 because IMDB reserves 0-2 for special purposes
            idx = word_index[word] + 3
            # Only include words within the vocab size used during training
            if idx < vocab_size:
                encoded.append(idx)
            else:
                # Word exists but is outside top `vocab_size` -> unknown
                encoded.append(2)
        else:
            # If word not in vocabulary, use code 2 (unknown word)
            encoded.append(2)

    return encoded

# ============================================================================
# STEP 6: DEFINE THE FIRST API ENDPOINT - HEALTH CHECK
# ============================================================================
# An endpoint is a URL path that clients can request data from

@app.get("/health")
def health_check():
    """
    GET endpoint: http://localhost:8000/health
    
    This is a simple health check endpoint.
    It allows clients to verify that the API server is running and responsive.
    
    Returns:
        dict: A simple status message
    
    Example response:
        {"status": "API is running!", "model": "LSTM Sentiment Analysis"}
    """
    # Include a sample negative review in the health response so users can see
    # an example payload and the default that will be used when no body is sent.
    return {
        "status": "API is running!",
        "model": "LSTM Sentiment Analysis",
        "model_loaded": True,
        "dummy_negative_example": DUMMY_NEGATIVE_REVIEW
    }

# ============================================================================
# STEP 7: DEFINE THE SECOND API ENDPOINT - SENTIMENT PREDICTION
# ============================================================================

@app.post("/predict")
def predict_sentiment(request: ReviewRequest = ReviewRequest()) -> PredictionResponse:
    """
    POST endpoint: http://localhost:8000/predict
    
    This endpoint accepts a movie review and returns a sentiment prediction.
    
    How it works:
    1. Client sends a review in the request body (JSON format)
    2. FastAPI automatically validates the data using our ReviewRequest model
    3. We convert the text to numbers that the LSTM model understands
    4. The LSTM model makes a prediction (0-1 score)
    5. We convert the score to "positive" or "negative"
    6. We send back the result as JSON
    
    Args:
        request (ReviewRequest): Contains the review text from the client
    
    Returns:
        PredictionResponse: Contains sentiment, confidence score, and review preview
    
    Example request (JSON):
        {
            "review": "This movie was absolutely fantastic! I loved every minute of it."
        }
    
    Example response:
        {
            "sentiment": "positive",
            "confidence": 0.95,
            "review_preview": "This movie was absolutely fantastic!..."
        }
    """
    
    # STEP 7a: Extract the review text from the request. If the client did not
    # provide a body, FastAPI will use the default `ReviewRequest()` value
    # which contains `DUMMY_NEGATIVE_REVIEW`.
    review_text = request.review
    
    # STEP 7b: Convert the text to numbers (encode it)
    encoded_review = encode_review(review_text)
    
    # STEP 7c: Pad the encoded review to match the model's expected input length
    # The model was trained on reviews padded to max_length
    # Use 'pre' padding/truncating to match how the model was trained.
    # The training script used the default pad_sequences behavior (padding='pre').
    padded_review = pad_sequences(
        [encoded_review],  # Must be a list
        maxlen=max_length,  # Pad/truncate to this length
        padding='pre',      # Add padding at the start (match training)
        truncating='pre'    # Truncate long sequences at the start (match training)
    )
    
    # STEP 7d: Make a prediction using the LSTM model
    # The model returns a probability between 0 and 1
    prediction_score = float(model.predict(padded_review, verbose=0)[0][0])
    
    # STEP 7e: Convert the numeric prediction to a sentiment label
    # Scores > 0.5 are considered positive, < 0.5 are considered negative
    sentiment = "positive" if prediction_score > 0.5 else "negative"
    
    # STEP 7f: Create a preview of the review (first 50 characters)
    review_preview = review_text[:50] + "..." if len(review_text) > 50 else review_text
    
    # STEP 7g: Return the prediction as a response object
    # FastAPI automatically converts this to JSON and sends it to the client
    return PredictionResponse(
        sentiment=sentiment,
        confidence=round(prediction_score, 4),
        review_preview=review_preview
    )

# ============================================================================
# STEP 8: RUNNING THE SERVER
# ============================================================================
# To run this server, use the following command in terminal:
# 
# uvicorn fastapi_server:app --reload --host 0.0.0.0 --port 8000
#
# Breakdown:
# - uvicorn: The ASGI server that runs FastAPI apps
# - fastapi_server:app: The module (fastapi_server.py) and app instance to run
# - --reload: Automatically restart server when code changes (development only)
# - --host 0.0.0.0: Listen on all network interfaces (allows external connections)
# - --port 8000: Run on port 8000 (default FastAPI port)
#
# Once running, you can test the API at:
# - Health check: http://localhost:8000/health
# - Prediction: http://localhost:8000/predict (POST request with JSON body)
# - Auto-generated docs: http://localhost:8000/docs (Swagger UI)
# - Alternative docs: http://localhost:8000/redoc (ReDoc)

if __name__ == "__main__":
    # This allows the script to be run directly with:
    # python fastapi_server.py
    # But for production, use uvicorn as shown above
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
