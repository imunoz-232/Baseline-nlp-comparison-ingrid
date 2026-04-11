# NLP and LLM Learning

Hands-on learning repository for classical NLP and neural text modeling.

Hands-on NLP and LLM learning lab with modular text pipelines: preprocessing, Bag-of-Words and TF-IDF feature engineering, classification, clustering, retrieval, baseline comparison, and beginner-friendly architecture notes.

## Quick Start (Beginner-Friendly)

These steps assume no prior virtual environment setup.

### 1) Install Python

- Install Python 3.11+ from https://www.python.org/downloads/
- During install on Windows, check: "Add Python to PATH"

Verify install:

```bash
python --version
```

If `python` does not work on macOS, try:

```bash
python3 --version
```

### 2) Open a command line

Windows (choose one):
- Start menu -> search `PowerShell` -> open it
- Or in VS Code: Terminal -> New Terminal

macOS (choose one):
- Applications -> Utilities -> Terminal
- Or in VS Code: Terminal -> New Terminal

### 3) Go to your project folder

If you already have the folder locally:

```bash
cd path/to/nlp-llm-learning-lab
```

If you want to clone from GitHub first:

```bash
git clone https://github.com/shylasolis/nlp-llm-learning-lab.git
cd nlp-llm-learning-lab
```

### 4) Create a virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
```

macOS:

```bash
python3 -m venv .venv
```

### 5) Activate the virtual environment

Windows (PowerShell):

```powershell
& .\.venv\Scripts\Activate.ps1
```

If you get an execution policy error, run once in PowerShell, then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

macOS:

```bash
source .venv/bin/activate
```

You should now see `(.venv)` at the start of your terminal line.

### 6) Install dependencies

```bash
pip install -r requirements.txt
```

### 7) Run the end-to-end NLP practice pipeline

```bash
python newsgroups_end_to_end_practice.py
```

### 8) Run the simple Bag-of-Words learning script

```bash
python bag_of_words_MSBA_265.py
```

### 9) Deactivate when done

```bash
deactivate
```

## First Class Exercise (5 Minutes)

Use this mini activity to help students confirm everything is working.

### Goal

- Run one script end-to-end.
- Recognize key output sections.
- Explain in plain language what each section means.

### Student steps

1. Activate the virtual environment.
2. Run:

```bash
python newsgroups_end_to_end_practice.py
```

### Expected output checkpoints

Students should see these section headers in order:

1. `20 NEWSGROUPS: DATA ORIGIN AND COLLECTION CONTEXT`
2. `STEP 1: LOAD DATA`
3. `PREPROCESSING GUIDE`
4. `STEP 2: CLEAN TEXT`
5. `ROUTE 1: TEXT CLASSIFICATION (BoW + Logistic Regression)`
6. `ROUTE 2: DOCUMENT CLUSTERING (TF-IDF + KMeans)`
7. `ROUTE 3: SEARCH AND RETRIEVAL (TF-IDF + COSINE SIMILARITY)`
8. `ROUTE 4: BASELINE MODEL COMPARISON`
9. `FINAL CROSS-ROUTE SUMMARY`

If internet is unavailable, students may see a fallback message. That is expected, and the pipeline should still continue.

### 3 quick interpretation questions

1. Classification route: What does accuracy tell us?
2. Clustering route: Why is silhouette not the same thing as accuracy?
3. Retrieval route: What does Precision@5 mean in plain English?

### One beginner tuning experiment

In `newsgroups_end_to_end_practice.py`, change:

- `min_token_length=2` to `min_token_length=3`

Run again and compare summary metrics.

Discuss:

- Did performance improve or drop?
- Why might removing short tokens help or hurt?

## New end-to-end practice pipeline
Use the 20 Newsgroups practice script to learn full text-ML workflows:

- Data context and collection notes
- Preprocessing and cleaning as a separate module
- Classification route
- Clustering route
- Retrieval route
- Baseline model comparison route
- Cross-route interpretation summary

Run (while virtual environment is active):

```bash
python newsgroups_end_to_end_practice.py
```

## Offline fallback
If dataset download is unavailable, the script automatically switches to a local fallback corpus so all routes still run for practice.

## Architecture guide
See NLP_ROUTE_ARCHITECTURE_GUIDE.md for route purpose, math architecture, tuning, and hard-coding intuition.

## By-hand math walkthrough
See WORKED_NLP_MATH_EXAMPLE.md for a tiny 3-sentence corpus where students can manually compute:

- Cleaning outputs
- Bag-of-Words matrix
- TF-IDF values
- Cosine retrieval ranking
- One linear classifier score
- One KMeans assignment/update iteration

## Printable worksheet for class
Use these files for a quick in-class hand-calculation activity:

- NLP_BY_HAND_WORKSHEET.md
- NLP_BY_HAND_WORKSHEET_ANSWER_KEY.md
