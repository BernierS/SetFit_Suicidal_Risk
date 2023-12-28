"""
Project Name: MACHINE LEARNING TECHNIQUES FOR ESTIMATING SUICIDAL RISK ON SOCIAL MEDIA
Author: Samuel Bernier
Thesis Paper (French): https://uqo.on.worldcat.org/oclc/1415207814
GitHub repository: https://github.com/BernierS/SetFit_Suicidal_Risk
Huggin Face repository: https://huggingface.co/BernierS/SetFit_Suicidal_Risk
File Description:
    This file is used to make predictions using custom SetFit model on data from Reddit.
--------------------------------------------------------------------------------
This file is part of the MACHINE LEARNING TECHNIQUES FOR ESTIMATING SUICIDAL RISK SUICIDAL RISK ON SOCIAL NETWORKS project, 
developed as a part of Samuel Bernier's thesis. For more information, visit https://uqo.on.worldcat.org/oclc/1415207814.
--------------------------------------------------------------------------------
"""

import re
import pandas as pd
import os

from setfit import SetFitModel
from tqdm import tqdm
from dotenv import load_dotenv

# Loads the variables from .env
load_dotenv()

# Split a text into sentences
def split_into_sentences(text):
    # We split on period, exclamation mark and question mark.
    # NOTE: This is a simple approach and will not work well with some edge cases (e.g. abbreviations, ellipsis).
    sentences = re.split(r'[.!?]', text)
    
    # Remove leading/trailing whitespace from each sentence
    sentences = [s.strip() for s in sentences]
    
    # Remove strings with 50 characters or less
    sentences = [s for s in sentences if s and len(s) > 50]
    
    return sentences
    
def process_dataframe(df, file_name, model, label_map):
    # Split the Body or Selftext column into sentences to process them individually
    df['Sentences'] = df['Body or Selftext'].apply(split_into_sentences)
    df = df.explode('Sentences').reset_index(drop=True)

    #Remove empty sentences
    df = df[df['Sentences'].str.len() > 0]

    # Create a batch process if there is more than 100,000 sentences to prevent memory overload
    batch_size = 100000
    num_batches = len(df) // batch_size + (len(df) % batch_size != 0)

    # For loop to process the dataframe in batches
    for i in range(num_batches):
        print(f"Processing batch {i + 1} of {num_batches}...")
        start = i * batch_size
        end = start + batch_size
        batch_df = df.iloc[start:end].copy()

        # Apply the SetFit model to the sentences
        # progress_apply() is a tqdm wrapper around apply() to give a progress bar
        # lambda transforms the sentence into a list to match the model's input format
        batch_df['Label'] = batch_df['Sentences'].progress_apply(lambda x: model([x]).item())

        # Map the label number to the label text
        batch_df['Label Text'] = batch_df['Label'].map(label_map)
        
        # If it's the first batch, write a new file with headers
        if i == 0:
            batch_df.to_csv(f'{file_name}.csv', index=False)
        # Otherwise, append to the existing file without headers
        else:
            batch_df.to_csv(f'{file_name}.csv', mode='a', header=False, index=False)


def main():
    # Load the SetFit model and label map
    model = SetFitModel.from_pretrained("BernierS/SetFit_Suicidal_Risk")

    # Label map for the CSV file
    label_map = {
        0: "Suicidal planning",
        1: "Previous attempt",
        2: "Ability to hope for change",
        3: "Consumption",
        4: "Ability to control oneself",
        5: "Presence of a loved one",
        6: "Ability to take care of oneself",
        7: "Other"
    }

    # Initialize tqdm to use progress_apply()
    tqdm.pandas()

    # Import CSV file
    reddit_data = pd.read_csv('Data/reddit_data.csv')

    # Full dataset
    process_dataframe(reddit_data, 'Data/reddit_sentences', model, label_map)

if __name__ == "__main__":
    main()
