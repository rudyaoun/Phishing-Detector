import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # file name: (emails column, labels column)
    files = {"phishing_emails.csv": ("Email Text", "Email Type"), "CEAS_08.csv": ("body", "label"), "Enron.csv": ("body", "label"), "Ling.csv": ("body", "label"), "Nazario.csv": ("body", "label"), "nigerian_fraud.csv": ("body", "label"), "spam_assassin.csv": ("body", "label")}
    file_names = list(files.keys())

    # Combined data to be used for training, evaluation, etc.
    combined_data = {"Email": [], "Label": []}

    # Get data for each individual dataset
    # Also combine the data into full dataset
    for i in range(len(file_names)):
        # Load the dataset
        file_path = "Datasets/" + file_names[i]
        data = pd.read_csv(file_path)

        # Load the emails and the labels
        email_col = files[file_names[i]][0]
        label_col = files[file_names[i]][1]
        emails = data[email_col].values
        emails = [email for email in emails]
        labels = data[label_col].values
        if label_col == "Email Type":
            labels = [0 if label=="Safe Email" else 1 for label in labels]
        else:
            labels = [label for label in labels]

        # Get rid of empty strings (bad data)
        cur = 0
        while cur < len(emails):
            if type(emails[cur]) != str or len(emails[cur]) == 0:
                emails = np.delete(emails, cur)
                labels = np.delete(labels, cur)
            else:
                cur += 1
        
        # Get stats for this individual datasets
        print(file_names[i] + " data:")
        phishes = sum(labels)
        print(f"Phishing Emails: {phishes}")
        print(f"Valid Emails: {len(labels) - phishes}\n")

        # Add to the combined dataset
        combined_data["Email"] += emails
        combined_data["Label"] += labels

    # Write the combined dataset to a csv
    df = pd.DataFrame(combined_data)
    df.to_csv("Datasets/full_dataset.csv", index=False)
    
    # Get combined data stats
    email_lengths = [len(email) for email in data["Email"]]
    print(f"Mean: {np.mean(email_lengths):.2f}, Median: {np.median(email_lengths)}, Max: {max(email_lengths)}")

    plt.show()