"""
LSTM Model Training on IMDB Dataset
This script trains an LSTM neural network on IMDB movie review data for sentiment analysis.
"""

import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.optimizers import Adam
import pickle

# Set parameters
vocab_size = 10000  # Use top 10,000 most common words
max_length = 200    # Pad/truncate reviews to 200 words
embedding_dim = 128  # Size of word embeddings
lstm_units = 64     # Number of LSTM units
batch_size = 32
epochs = 5

print("Loading IMDB dataset...")
# Load IMDB dataset - this gives us the word indices (not words, but numbers)
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Pad sequences to same length
print("Padding sequences...")
X_train = pad_sequences(X_train, maxlen=max_length)
X_test = pad_sequences(X_test, maxlen=max_length)

# Build LSTM model
print("Building LSTM model...")
model = Sequential([
    # Embedding layer: converts integer indices to dense vectors
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    
    # LSTM layer: captures sequential patterns and dependencies
    LSTM(units=lstm_units, return_sequences=False),
    
    # Dropout: prevents overfitting by randomly dropping neurons
    Dropout(0.5),
    
    # Dense layer: fully connected layer for classification
    Dense(units=1, activation='sigmoid')  # sigmoid for binary classification (positive/negative)
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),  # Adam optimizer with learning rate
    loss='binary_crossentropy',             # Binary crossentropy for binary classification
    metrics=['accuracy']                    # Track accuracy during training
)

# Display model architecture
model.summary()

# Train the model
print("Training model...")
history = model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_test, y_test),
    verbose=1
)

# Evaluate the model
print("\nEvaluating model...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Save the model and vocabulary size
print("Saving model...")
model.save('imdb_lstm_model.h5')

# Save vocab size for later use in API
with open('imdb_vocab_size.pkl', 'wb') as f:
    pickle.dump(vocab_size, f)

with open('imdb_max_length.pkl', 'wb') as f:
    pickle.dump(max_length, f)

print("Model training complete! Saved as 'imdb_lstm_model.h5'")
