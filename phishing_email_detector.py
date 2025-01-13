import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from collections import Counter

# The RNN used to predict whether or not an email is phishing
class PhishingRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(PhishingRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (ht, ct) = self.lstm(embedded)
        output = self.fc(ht[-1])  # Use the final hidden state for prediction
        return self.sigmoid(output)

# Read in dataset of emails for training and tokenize
# each email to feed into model
def read_and_tokenize_data(csv_path):
    # Phishing Dataset
    data = pd.read_csv(csv_path)

    # Email contents and labels of phishing emails
    texts = data['Email Text'].values
    labels = data['Email Type'].values
    labels = [0 if label=="Safe Email" else 1 for label in labels]

    # # Getting statistics on lengths of emails to get a good max for later
    # email_lengths = [len(email.split()) for email in texts]
    # plt.hist(email_lengths, bins=500, range=(0, 5000))
    # plt.xlabel("Number of Words")
    # plt.ylabel("Number of Emails")
    # plt.title("Distribution of Email Lengths")
    # plt.show()

    # # Check summary statistics
    # print(f"Mean: {np.mean(email_lengths):.2f}, Median: {np.median(email_lengths)}, Max: {max(email_lengths)}")

    # Get rid of empty strings (bad data) and long emails (unusable)
    cur = 0
    while cur < len(texts):
        if type(texts[cur]) != str or len(texts[cur].split()) > 500:
            texts = np.delete(texts, cur)
            labels = np.delete(labels, cur)
        else:
            cur += 1

    # Tokenize each email (email string -> list of words)
    tokenized_texts = [nltk.word_tokenize(text) for text in texts]

    # Build vocabulary (each unique word gets an index, 0 for unknowns)
    all_tokens = [token for tokens in tokenized_texts for token in tokens]
    vocab = Counter(all_tokens)
    vocab_size = len(vocab)
    word_to_idx = {word: idx + 1 for idx, (word, _) in enumerate(vocab.most_common())}
    word_to_idx['<UNK>'] = 0  # Add <UNK> token for unknown words

    # Turn each email into a sequence of the corresponding indices of each word and pad up to 500 words
    max_length = 500
    sequences = [[word_to_idx.get(word, 0) for word in text] for text in tokenized_texts]
    padded_sequences = [seq[:max_length] + [0] * (max_length - len(seq)) if len(seq) < max_length else seq[:max_length] for seq in sequences]
    return padded_sequences, labels, vocab_size

# Prepare data to be passed into the model for training and/or prediction
def prep_data(emails, labels):
    sequences_tensor = torch.tensor(emails, dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
    X_train, X_val, y_train, y_val = train_test_split(sequences_tensor, labels_tensor, test_size=0.2, random_state=42)
    train_data = TensorDataset(X_train, y_train)
    val_data = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    for inputs, targets in train_loader:
        print(f"Inputs shape: {inputs.shape}")  # Should be (batch_size, max_length)
        print(f"Targets shape: {targets.shape}")  # Should be (batch_size, 1)
    return train_loader, val_loader

# Train the model
def train(model, epochs, optimizer, loss_fn, train_loader):
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        correct_preds = 0
        total_preds = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = (outputs > 0.5).float()  # Binary classification threshold
            correct_preds += (preds == targets).sum().item()
            total_preds += targets.size(0)

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_preds / total_preds
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")

# Evaluate the accuracy of hte model
def eval(model, val_loader):
    model.eval()
    with torch.no_grad():
        correct_preds = 0
        total_preds = 0

        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = (outputs > 0.5).float()  # Binary classification threshold
            correct_preds += (preds == targets).sum().item()
            total_preds += targets.size(0)

        val_accuracy = correct_preds / total_preds
        print(f"Validation Accuracy: {val_accuracy:.4f}")


if __name__ == "__main__":
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Read and tokenize the dataset
    print("Reading dataset")
    emails, labels, vocab_size = read_and_tokenize_data("\\Users\\rudy_\OneDrive\Desktop\Phishing Detector\Phishing_Email.csv")

    # Initialize the model
    embedding_dim = 64
    hidden_dim = 128
    output_dim = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = PhishingRNN(vocab_size + 1, embedding_dim, hidden_dim, output_dim).to(device)

    # Prepare data for training
    print("Preparing data for training")
    train_loader, val_loader = prep_data(emails, labels)
    
    # Training the model
    epochs = 5
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("Training")
    train(model, epochs, optimizer, loss_fn, train_loader)
    
    # Evaluate the model
    print("Evaluating")
    eval(model, val_loader)