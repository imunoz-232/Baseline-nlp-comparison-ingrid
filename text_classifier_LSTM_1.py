"""
Text Classification with LSTM Networks (PyTorch)

This code demonstrates an LSTM-based binary classification model for sentiment analysis.
It tokenizes the input documents, converts them to sequences, and pads sequences to 
ensure equal length. The LSTM layer captures sequential information, and the final 
dense layer produces binary classification (positive/negative sentiment).

Key concepts:
- Tokenization: Converting text to numerical sequences
- Padding: Making all sequences the same length for neural networks
- Embedding: Learning vector representations of words
- LSTM: Capturing sequential patterns and context
- Binary Classification: Predicting 0 (negative) or 1 (positive)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


# ===== Simple Tokenizer Class (No TensorFlow Required) =====
class SimpleTokenizer:
    """Custom tokenizer - converts text to word IDs and pads sequences."""
    
    def __init__(self):
        self.word_index = {}  # Maps words to IDs
        self.word_count = 0
    
    def fit_on_texts(self, texts):
        """Learn vocabulary from documents."""
        for text in texts:
            words = text.lower().split()
            for word in words:
                if word not in self.word_index:
                    self.word_count += 1
                    self.word_index[word] = self.word_count
    
    def texts_to_sequences(self, texts):
        """Convert texts to sequences of word IDs."""
        sequences = []
        for text in texts:
            words = text.lower().split()
            seq = [self.word_index[word] for word in words if word in self.word_index]
            sequences.append(seq)
        return sequences
    
    def pad_sequences(self, sequences, maxlen=None):
        """Pad sequences to fixed length."""
        if maxlen is None:
            maxlen = max(len(seq) for seq in sequences)
        padded = []
        for seq in sequences:
            if len(seq) < maxlen:
                seq = seq + [0] * (maxlen - len(seq))  # Pad with zeros
            else:
                seq = seq[:maxlen]  # Truncate if too long
            padded.append(seq)
        return np.array(padded)


# ===== Sample documents and labels =====
documents = [
    "I love programming in Python",
    "Python is a great language",
    "I enjoy machine learning",
    "I dislike bugs in my code",
    "Debugging can be frustrating"
]
labels = [1, 1, 1, 0, 0]  # 1 for positive sentiment, 0 for negative sentiment

print("=" * 70)
print("SENTIMENT ANALYSIS WITH LSTM")
print("=" * 70)
print(f"\nTraining on {len(documents)} documents")
print(f"Positive samples: {sum(labels)}, Negative samples: {len(labels) - sum(labels)}\n")

# ===== Tokenize the documents and convert them to sequences =====
# Tokenization turns words into numbers so neural networks can process them
tokenizer = SimpleTokenizer()
tokenizer.fit_on_texts(documents)
sequences = tokenizer.texts_to_sequences(documents)
vocab_size = len(tokenizer.word_index) + 1  # +1 for padding token

print(f"Vocabulary size: {vocab_size} unique words")
print(f"Vocabulary: {tokenizer.word_index}")
print(f"Sample: '{documents[0]}' → {sequences[0]}\n")

# ===== Pad the sequences to ensure equal length =====
# Neural networks need fixed-size inputs. Padding adds zeros to shorter sequences.
padded_sequences = tokenizer.pad_sequences(sequences)
max_length = padded_sequences.shape[1]

print(f"Max sequence length: {max_length}")
print(f"After padding shape: {padded_sequences.shape}\n") # will output (5, max_length) where 5 is the number of documents

# ===== Convert to PyTorch tensors =====
X = torch.LongTensor(padded_sequences)
y = torch.FloatTensor(labels).unsqueeze(1)
dataset = TensorDataset(X, y) # Pairs of (input, label) for training
dataloader = DataLoader(dataset, batch_size=2, shuffle=True) # Split into batches of 2 for training


# ===== Build the LSTM model =====
class LSTMClassifier(nn.Module):
    """LSTM-based sentiment classifier."""
    
    def __init__(self, vocab_size, embedding_dim=50, hidden_dim=64):
        super(LSTMClassifier, self).__init__()
        
        # Embedding layer: converts word IDs to dense vectors
        # Learns to represent similar words with similar vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM layer: captures sequential patterns
        # Takes embedding_dim input and outputs hidden_dim values
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
        # Dense layer: makes final binary classification decision
        self.fc = nn.Linear(hidden_dim, 1)
        
        # Sigmoid activation: outputs probability between 0 and 1
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """Forward pass through the network."""
        # Embedding: converts word IDs to vectors
        embedded = self.embedding(x)
        
        # LSTM: processes sequence, returns hidden state
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use last hidden state
        last_hidden = hidden.squeeze(0)
        
        # Dense layer for classification
        logits = self.fc(last_hidden)
        
        # Sigmoid: outputs probability
        output = self.sigmoid(logits)
        
        return output


# Instantiate the model
model = LSTMClassifier(vocab_size, embedding_dim=50, hidden_dim=64)

print("Model Architecture:")
print("  Embedding Layer: word_id → 50-dim vector")
print("  LSTM Layer: 50-dim → 64-dim hidden state")
print("  Dense Layer: 64-dim → 1 output (binary classification)")
print("  Activation: Sigmoid (outputs probability 0-1)\n")

# ===== Compile the model (set loss and optimizer) =====
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training Configuration:")
print("  Loss Function: Binary Cross-Entropy (BCE)")
print("  Optimizer: Adam")
print("  Learning Rate: 0.001\n")

# ===== Train the model =====
print("=" * 70)
print("TRAINING")
print("=" * 70 + "\n")

epochs = 10
for epoch in range(epochs):
    total_loss = 0
    batch_count = 0
    
    for batch_X, batch_y in dataloader:
        # Forward pass: compute predictions
        predictions = model(batch_X)
        
        # Compute loss
        loss = criterion(predictions, batch_y)
        
        # Backward pass: compute gradients
        optimizer.zero_grad()
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1
    
    avg_loss = total_loss / batch_count
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

print("\n" + "=" * 70)
print("PREDICTIONS")
print("=" * 70 + "\n")

# ===== Make predictions =====
model.eval()  # Set to evaluation mode
with torch.no_grad():
    predictions = model(X)

print("Predictions (probability of positive sentiment):\n")
for i, (doc, true_label, pred) in enumerate(zip(documents, labels, predictions)):
    predicted_class = 1 if pred.item() > 0.5 else 0
    correct = "✓" if predicted_class == true_label else "✗"
    print(f"{correct} Doc {i+1}: {pred.item():.3f}")
    print(f"   Text: '{doc}'")
    print(f"   True: {true_label} | Predicted: {predicted_class}\n")
