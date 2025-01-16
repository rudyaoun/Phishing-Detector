import json
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
    emails = data['Email'].values
    labels = data['Label'].values

    # Get rid of empty strings (bad data)
    cur = 0
    while cur < len(emails):
        if type(emails[cur]) != str or len(emails[cur].split()) > 1000:
            emails = np.delete(emails, cur)
            labels = np.delete(labels, cur)
        else:
            cur += 1

    # Tokenize each email (email string -> list of words)
    tokenized_emails = [nltk.word_tokenize(email) for email in emails]

    # Build vocabulary (each unique word gets an index, 0 for unknowns)
    all_tokens = [token for tokens in tokenized_emails for token in tokens]
    vocab = Counter(all_tokens)
    vocab_size = len(vocab)
    word_to_idx = {word: idx + 1 for idx, (word, _) in enumerate(vocab.most_common())}
    word_to_idx['<UNK>'] = 0  # Add <UNK> token for unknown words

    # Turn each email into a sequence of the corresponding indices of each word and pad up to 1000 words
    max_length = 1000
    sequences = [[word_to_idx.get(word, 0) for word in text] for text in tokenized_emails]
    padded_sequences = [seq[:max_length] + [0] * (max_length - len(seq)) if len(seq) < max_length else seq[:max_length] for seq in sequences]
    return padded_sequences, labels, vocab_size, word_to_idx

# Prepare data to be passed into the model for training and/or prediction
def prep_data(emails, labels):
    sequences_tensor = torch.tensor(emails, dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
    X_train, X_val, y_train, y_val = train_test_split(sequences_tensor, labels_tensor, test_size=0.2, random_state=42)
    train_data = TensorDataset(X_train, y_train)
    val_data = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
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

# Make a prediction on a single email
def predict_email(model, email_text, word_to_idx, max_length=1000, threshold=0.5):
    # Tokenize and preprocess
    tokenized_email = nltk.word_tokenize(email_text)
    email_indices = [word_to_idx.get(word, 0) for word in tokenized_email]
    padded_email = email_indices[:max_length] + [0] * (max_length - len(email_indices))
    
    # Convert to tensor
    input_tensor = torch.tensor([padded_email], dtype=torch.long).to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        phishing_probability = output.item()
    
    # Interpret result
    prediction = "PHISHING" if phishing_probability > threshold else "SAFE"
    return phishing_probability, prediction

if __name__ == "__main__":
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if input("Train new model or load latest model? (train/load): ") == "train":
        # Read and tokenize the dataset
        print("Reading dataset")
        emails, labels, vocab_size, word_to_idx = read_and_tokenize_data("Datasets/full_dataset.csv")
        embedding_dim = 64
        hidden_dim = 128
        output_dim = 1
        
        # Initialize the model
        print("Creating new model")
        model = PhishingRNN(vocab_size + 1, embedding_dim, hidden_dim, output_dim).to(device)

        # Prepare data for training
        print("Preparing data for training")
        train_loader, val_loader = prep_data(emails, labels)
    
        # Training the model
        epochs = 5
        loss_fn = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        print("Training model")
        train(model, epochs, optimizer, loss_fn, train_loader)
    
        # Evaluate the model
        print("Evaluating model")
        eval(model, val_loader)

        if input("Save model? (y/n): ") == "y":
            # Save model state dictionary
            metadata = {"vocab_size": vocab_size, "embedding_dim": embedding_dim, 
                        "hidden_dim": hidden_dim, "output_dim": output_dim, "word_to_idx": word_to_idx}
            file = open("phishing_detector_metadata.json", "w")
            json.dump(metadata, file)
            file.close()
            torch.save(model.state_dict(), "phishing_detector.pth")
            print("Model saved as phishing_detector.pth\nMetadata saved as phishing_detector_metadata.json")        

    else:
        # Load the latest model
        print("Loading model")
        file = open("phishing_detector_metadata.json", "r")
        metadata = json.load(file)
        file.close()
        model = PhishingRNN(metadata["vocab_size"] + 1, metadata["embedding_dim"], metadata["hidden_dim"], metadata["output_dim"])
        model.load_state_dict(torch.load("phishing_detector.pth"))
        model = model.to(device)
        print("Model loaded")

        if input("Evaluate on datasets or use model on new input? (eval/use): ") == "eval":
            files = ["phishing_emails.csv", "CEAS_08.csv", "Enron.csv", "Ling.csv", "Nazario.csv", "nigerian_fraud.csv", "spam_assassin.csv"]
            for file in files:
                # Load the dataset
                file_path = "Datasets/" + file
                print("Reading dataset " + file)
                emails, labels, _, _ = read_and_tokenize_data(file_path)

                # Preparing data to test model
                print("Preparing data for evaluation")
                sequences_tensor = torch.tensor(emails, dtype=torch.long)
                labels_tensor = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
                data = TensorDataset(sequences_tensor, labels_tensor)
                data_loader = DataLoader(data, batch_size=32, shuffle=False)

                # Evaluate the model
                print("Evaluating model on dataset " + file)
                eval(model, data_loader)

                print("\n")
        else:
            email = input("Enter email for phishing prediction: ")
            phish_prob, pred = predict_email(model, email, metadata["word_to_idx"])
            print(f"There's a {phish_prob*100}% chance that this is a phishing email.\nPrediction: {pred}")
