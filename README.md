# Baseline NLP Comparison

This project is part of my MSBA 265 course.

The goal of this project is to compare different baseline NLP models using the 20 Newsgroups dataset.

## Project Description
In this project, I applied basic Natural Language Processing (NLP) techniques, including:
- Text cleaning and preprocessing
- Feature extraction using Bag of Words and TF-IDF
- Text classification using Logistic Regression
- Document clustering using K-Means
- Basic information retrieval using cosine similarity

## Files
- newsgroups_cleaning.py → data preprocessing
- newsgroups_baseline_route.py → baseline model
- newsgroups_classification_route.py → classification model
- newsgroups_clustering_route.py → clustering model
- newsgroups_end_to_end_practice.py → full pipeline execution

## Objective
The objective of this project is to understand how NLP models work and compare their performance across different approaches.

## Business Value
This type of analysis can help companies understand large amounts of text data, such as customer reviews, emails, or support tickets, and make better decisions based on patterns found in the data.
This project shows how companies can analyze large amounts of text data, such as customer reviews or emails, to support better decision-making.

## One-line Reproducible Run

git clone https://github.com/imunoz-232/Baseline-nlp-comparison-ingrid.git
cd Baseline-nlp-comparison-ingrid
pip install -r requirements.txt
python newsgroups_end_to_end_practice.py

## Alternative (with virtual environment)

python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python newsgroups_end_to_end_practice.py

## Author
Ingrid Munoz
