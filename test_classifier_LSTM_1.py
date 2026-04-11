# Text classification tasks which are traditional and use long short-tern memory or lstm networks.
# This code defines an LSTM-based binary classification model for sentiment analysis. It toeknizes
# the input documents, converts them to sequences, and pads the sequences to ensure equal length. 
# The LSTM layer captures sequential information and the final dense layer produces
# binary cla
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample documents and corresponding labels 
documents = [
    "I love programming in Python",
    "Python is a great language",
    "I enjoy machine learning",
    "I dislike bugs in my code",
    "Debugging can be frustrating"
]
labels = [1, 1, 1, 0, 0]  # 1 for positive sentiment, 0 for negative sentiment

# Tokenize the documents and convert them to sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts(documents)
sequences = tokenizer.texts_to_sequences(documents)
# Pad the sequences to ensure equal length

max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
# Build the LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
vocab_size = len(tokenizer.word_index) + 1  # +1 for padding token
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=50, input_length=max_length))
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Train the model
model.fit(padded_sequences, labels, epochs=10, batch_size=2)
