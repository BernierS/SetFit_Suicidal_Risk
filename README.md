# SetFit_Suicidal_Risk
This repository is the code portion of my master's thesis named Machine learning techniques for estimating suicidal risk on social networks. It is meant as a proof of concept on how to use reduced dataset to train AI on text classification such as suicidal risk estimation with the collaboration of a social worker. For more detailed, the complete research is available here: https://uqo.on.worldcat.org/oclc/1415207814. 

## Abstract
In recent years, the growing influence of social media platforms has revolutionized communication, connecting millions of individuals worldwide. Among these platforms, Reddit, as a prominent social forum, has become a hotbed for users to express their thoughts and emotions freely. However, this unprecedented level of connectivity also brings to light concerning mental health challenges, including potential suicide risk signs embedded within user posts. On the flip side of the coin, Natural Language Processing (NLP) is a subject at the heart of today’s discussion and, as such, has been the source of numerous studies over the past decades. With its great expansion, this field of study offers remarkable technological advancements, such as BERT, GPT, or T0. These advancements allow the creation of solutions to significant societal problems.

With the alarming rise in mental health problems among online communities, it is vital to have mechanisms for early identification of individuals at risk of suicide. The aim of this M. Sc thesis is precisely to harness the power of NLP to provide rapid and accurate support to individuals in distress. The aim is to tackle the crucial challenge of detecting suicidal signs and present an innovative solution using the SetFit model, a sentence-BERT refinement technique SBERT, in the context of social media Reddit. The SetFit model has the ability to analyze unstructured text while displaying impressive classification accuracy, even with limited training data, making it a powerful tool for suicide risk assessment.

After ensuring the relevance and authenticity of posts, a curated dataset of posts encompassing diverse linguistic patterns and emotive expressions for potential suicidal cases based on the Reddit platform was created. This dataset serves as the cornerstone for initially validating the promising SetFit model's efficacy through preliminary experiments and analysis.

## Installation and set up
This project is meant for research purposes and is separated as so. Every python file is built to execute individually as needed, which means they all contain a main function. For all library requirements, please verify with the file named "requirements.txt". 

## Usage
The scripts are separated by a folder as how they are meant to be used.
- SetFit
    - The first file is "SetFit-Train.py", which is used for training SetFit on the Dataset built by the collaboration of a social worker and myself. The training dataset can be found under the "Data" folder and is named "Suicide_Data_E16.csv.” 
    The second file is "SetFit-Pred.py", which is used for making prediction using the trained SetFit algorithm. The results are compiled in the file named "reddit_sentences.csv", which can be found in the "Data" folder. 
- Sentence Transformers
    - This folder contains the script "ST-API.py" to compare multiple sentence transformers to find a best fit for my use case. The details on application and results can be found in the thesis. 
- Reddit
    - This folder contains the script "Reddit_API.py" to fetch data from Reddit using the PRAW API. The data is compiled into a dataset named "reddit_data.csv" located in the "Data" folder. 
- Compare-algo
    - This folder contains the script "Multi-algo.py" used to compare different types of algorithm used for text classification mentioned in the thesis. The results can all be found in the thesis. 
- Application
    - This folder contains the script "application.py", which is used for compiling the results from the prediction script "SetFit-Pred.py". The dataset used for the application is under "Data" as "reddit_sentences". 
