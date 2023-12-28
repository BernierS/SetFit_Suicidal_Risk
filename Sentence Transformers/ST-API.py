"""
Project Name: MACHINE LEARNING TECHNIQUES FOR ESTIMATING SUICIDAL RISK ON SOCIAL MEDIA
Author: Samuel Bernier
Thesis Paper (French): https://uqo.on.worldcat.org/oclc/1415207814
GitHub repository: 
Huggin Face repository: https://huggingface.co/BernierS/SetFit_Suicidal_Risk
File Description:
    This file is used to compare different Sentence Transformers found on HuggingFace.
--------------------------------------------------------------------------------
This file is part of the MACHINE LEARNING TECHNIQUES FOR ESTIMATING SUICIDAL RISK SUICIDAL RISK ON SOCIAL NETWORKS project, 
developed as a part of Samuel Bernier's thesis. For more information, visit https://uqo.on.worldcat.org/oclc/1415207814.
--------------------------------------------------------------------------------
"""

import requests
import json
import time
import os

from dotenv import load_dotenv
load_dotenv()  # This loads the variables from .env

#The following code is to test different Sentence Transformer found on HuggingFace. 
# It is mandatory in order to find the most suitable Sentence Transformer for SetFit. 
headers = {"Authorization": os.environ.get("HF_BEARER")}
results = {}

with open('Sentence Transformers\Sentences.json', 'r') as f:
    sentences_list = json.load(f)

def query(API_URL, payload):
	params = {'wait_for_model': 'True'}
	response = requests.post(API_URL, headers=headers, json=payload, params=params)
	return response.json()

def request(API_URL, sentences):
	output = query(API_URL,sentences)
	return output

models = {
	"paraphrase_mpnet_base_v2": "https://api-inference.huggingface.co/models/sentence-transformers/paraphrase-mpnet-base-v2",
	"all_MiniLM_L6_v2": "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2",
	"paraphrase_MiniLM_L6_v2": "https://api-inference.huggingface.co/models/sentence-transformers/paraphrase-MiniLM-L6-v2",
    "roberta-base-nli-stsb-mean-tokens": "https://api-inference.huggingface.co/models/sentence-transformers/roberta-base-nli-stsb-mean-tokens",
    "all-mpnet-base-v2": "https://api-inference.huggingface.co/models/sentence-transformers/all-mpnet-base-v2",
    "all-distilroberta-v1": "https://api-inference.huggingface.co/models/sentence-transformers/all-distilroberta-v1",
    "all-MiniLM-L12-v2": "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L12-v2",
    "multi-qa-distilbert-cos-v1": "https://api-inference.huggingface.co/models/sentence-transformers/multi-qa-distilbert-cos-v1"
}

# Iterate through models and print the output of request()
for x in models:
    print(x)
    general_average = 0
    # If the model isn't loaded yet,  wait for it to load and try again
    for sign_name, sign_value in sentences_list.items():
        while True:
            try:
                accuracy = request(models[x], sign_value)
                print(sign_name)
                print(accuracy)
                general_average += sum(accuracy)/len(accuracy)
                print("Average:", sum(accuracy)/len(accuracy))
                break  # if no error, break the loop
            except Exception as e:
                print(f"An error occurred: {e}")
                print("Waiting 10 seconds before trying again...")
                time.sleep(10)
    print("General Average: ",general_average/len(sentences_list))
    print("\n")
    results[x] = general_average/len(sentences_list)
    
print("Final results: ", results)
