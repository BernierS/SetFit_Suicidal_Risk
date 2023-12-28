"""
Project Name: MACHINE LEARNING TECHNIQUES FOR ESTIMATING SUICIDAL RISK ON SOCIAL MEDIA
Author: Samuel Bernier
Thesis Paper (French): https://uqo.on.worldcat.org/oclc/1415207814
GitHub repository: 
Huggin Face repository: https://huggingface.co/BernierS/SetFit_Suicidal_Risk
File Description:
    This file is used to train the SetFit model using the training data from Suicide_Data_E16.csv.
--------------------------------------------------------------------------------
This file is part of the MACHINE LEARNING TECHNIQUES FOR ESTIMATING SUICIDAL RISK SUICIDAL RISK ON SOCIAL NETWORKS project, 
developed as a part of Samuel Bernier's thesis. For more information, visit https://uqo.on.worldcat.org/oclc/1415207814.
--------------------------------------------------------------------------------
"""

import re
import evaluate
import pandas as pd
import os

from setfit import SetFitModel
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitTrainer
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sklearn.neural_network import MLPClassifier
from dotenv import load_dotenv

# Loads the variables from .env
load_dotenv()  

# Function to HTML tags and URLs
def process_text(text):
    # Remove HTML tags
    text = re.sub('<[^<]+?>', ' ', text)

    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    return text

#Load the proper sentence transformer model
model_body = SentenceTransformer('paraphrase-mpnet-base-v2')
model_head = MLPClassifier()

# Load your dataset using pandas
dataset = pd.read_csv("Data/Suicide_Data_E16.csv")

# #Pre-processe the text
dataset['sentence'] = dataset['sentence'].apply(process_text)

# Split the dataset into training and validation datasets
train_df, test_df = train_test_split(dataset, test_size=0.33, stratify=dataset['label'])
train_ds = Dataset.from_pandas(train_df)
test_ds = Dataset.from_pandas(test_df)

# Define the model
model = SetFitModel(
    model_body=model_body,
    model_head=model_head,
    multi_target_strategy=None,
    l2_weight=1e-2,
    normalize_embeddings=False
)

# Determine which accuracy metric to use
multilabel_f1_metric = evaluate.load("f1", "multiclass")
multilabel_accuracy_metric = evaluate.load("accuracy", "multiclass")

def compute_metrics(y_pred, y_test):
    return {
        "f1": multilabel_f1_metric.compute(predictions=y_pred, references=y_test, average="micro")["f1"],
        "accuracy": multilabel_accuracy_metric.compute(predictions=y_pred, references=y_test)["accuracy"],
    }

# Associating the right column with the right label 
train_dataset = Dataset.from_dict({"text": train_ds['sentence'], "label": train_ds['label']})
eval_dataset = Dataset.from_dict({"text": test_ds['sentence'], "label": test_ds['label']})

# Train the model
trainer = SetFitTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss_class=CosineSimilarityLoss,
    metric=compute_metrics,
    column_mapping={"text": "text", "label": "label"},
)

trainer.train()
# Display the results of the accuracy and f1 metrics
metrics = trainer.evaluate()
print(metrics)

# Push model to HuggingFace
trainer.push_to_hub(
    repo_id = "BernierS/SetFit_Suicidal_Risk",
    token = os.environ.get("HF_TOKEN"))
