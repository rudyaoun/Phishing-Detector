import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # file name: (emails column, labels column)
    files = {"phishing_emails.csv": ("Email Text", "Email Type"), "CEAS_08.csv": ("body", "label"), "Enron.csv": ("body", "label"), "Ling.csv": ("body", "label"), "Nazario.csv": ("body", "label"), "nigerian_fraud.csv": ("body", "label"), "spam_assassin.csv": ("body", "label")}
    file_names = files.keys()
    fig, axs = plt.subplots(2, 4)

    # Combined data to be used for training, evaluation, etc.
    data = {"Email": [], "Label": []}

    # Plot number of phishing vs safe emails per data plot
    # Also construct combined data set for training
    for i in range(len(file_names)):
        # Load the dataset
        file_path = "Datasets/" + file_names[i]
        data = pd.read_csv(file_path)

        # Load the emails and the labels
        email_col = files[file_names[i]][0]
        label_col = files[file_names[i]][1]
        emails = data[email_col].values
        labels = data[label_col].values
        if label_col == "Email Type":
            labels = [0 if label=="Safe Email" else 1 for label in labels]
        else:
            labels = [label for label in labels]
        
        # Plot this dataset
        axs[i // 4, i % 4].bar(labels)
        axs[i // 4, i % 4].set(xlabel="Valid vs. Phishing", ylabel="Number of Emails")
        axs[i // 4, i % 4].set_title(file_names[i])

        # Add to the combined dataset
        data["Email"] += emails
        data["Label"] += labels

    # Write the combined dataset to a csv
    df = pd.DataFrame(data)
    df.to_csv("full_dataset.csv", index=False)
    
    # Plot the combined data
    axs[3, 3].bar(labels)
    axs[3, 3].xlabel(xlabel="Valid vs. Phishing", ylabel="Number of Emails")
    axs[3, 3].set_title(file_names[i])

    plt.show()