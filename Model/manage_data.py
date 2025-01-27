import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


# Combine the listed file names together into a merged dataset.
#
# This function should be used to merge together some choice of
# datasets together in order to train the RNN
#
# files: a list of the names of the files to be merged
# verbose: print out information about each dataset and the final dataset
def merge_datasets(files, verbose=False):
    # Combined data to be used for training, evaluation, etc.
    combined_data = {"Email": [], "Label": []}

    # Get data for each individual dataset
    # Also combine the data into full dataset
    for file in files:
        # Load the dataset
        file_path = "Datasets/" + file
        data = pd.read_csv(file_path)

        # Load the emails and the labels
        emails = [email for email in data["Email"].values]
        labels = [label for label in data["Label"].values]

        # Get rid of empty strings (bad data)
        cur = 0
        total = len(emails)
        counter = 0
        while cur < len(emails):
            counter += 1
            if type(emails[cur]) != str or len(emails[cur]) == 0 or not (labels[cur] == 0 or labels[cur] == 1):
                emails.pop(cur)
                labels.pop(cur)
            else:
                cur += 1
        
        # Get stats for this individual dataset
        if verbose:
            print(file + " data:")
            total = len(labels)
            phishes = sum(labels)
            print(f"Phishing Emails: {phishes}")
            print(f"Valid Emails: {total - phishes}")
            print(f"Skipped over: {counter - total}")
            print(f"Invalid Rows: {len(labels) - len(emails)}\n")

        # Add to the combined dataset
        combined_data["Email"] += emails
        combined_data["Label"] += labels

    # Write the combined dataset to a csv
    df = pd.DataFrame(combined_data)
    df.to_csv("Datasets/full_dataset.csv", index=False)
    
    if verbose:
    # Get combined data stats
        print("Combined data:")
        total = len(combined_data["Label"])
        phishes = sum(combined_data["Label"])
        print(f"Phishing Emails: {phishes}")
        print(f"Valid Emails: {total - phishes}")
        email_lengths = [len(email.split()) for email in combined_data["Email"]]
        print(f"Mean: {np.mean(email_lengths):.2f}, Median: {np.median(email_lengths)}, Max: {max(email_lengths)}")

        # Plot the distribution of word lengths
        plt.hist(email_lengths, bins=100, range=(0, 5000))
        plt.xlabel("Number of Words")
        plt.ylabel("Number of Emails")
        plt.title("Distribution of Email Lengths")
        plt.show()


if __name__ == "__main__":
    files = ["phishing_emails.csv", "CEAS_08.csv", "Enron.csv", "Ling.csv", "nigerian_fraud.csv", "Nazario.csv", "spam_assassin.csv"]
    merge_datasets(files, True)
