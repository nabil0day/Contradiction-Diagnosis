# Contradiction-Diagnosis

## Project Overview
This project aims to classify pairs of sentences into one of three categories:  
- **Contradiction**: Sentences have opposite meanings.  
- **Entailment**: One sentence logically follows from the other.  
- **Neutral**: Sentences are related but do not imply each other.  

Multiple models are implemented and evaluated to achieve optimal classification performance.

## Dataset Description
The dataset consists of labeled sentence pairs:
- **Columns**:
  - `id`: Unique identifier for each sentence pair.
  - `premise`: The first sentence in the pair.
  - `hypothesis`: The second sentence in the pair.
  - `label`: Relationship classification (0: Contradiction, 1: Neutral, 2: Entailment).

The dataset is split into training and testing sets to evaluate model performance.

## Model Implementation Details
We implemented the following models:
- **Random Forest**: Baseline model using TF-IDF features.
- **Decision Tree**: Simple tree-based classifier.
- **XGBoost**: Gradient boosting for improved performance.
- **Artificial Neural Network (ANN)**: Fully connected neural network trained with TF-IDF features.
- **LSTM**: Sequence-based model trained on tokenized sentence pairs.
- **BERT (Bidirectional Encoder Representations from Transformers)**: Fine-tuned for sentence classification.



## Steps to Run the Code

### 1. Setup the Environment
Ensure you have the required dependencies installed:
```bash
pip install numpy pandas matplotlib seaborn nltk scikit-learn xgboost tensorflow torch transformers
```

### 2. Download the Dataset
Place the `train.csv` file in your working directory or update the path accordingly.

### 3. Run the Code
Execute the Jupyter Notebook `Task.ipynb`:
```bash
jupyter notebook Task.ipynb
```
Alternatively, if using Google Colab, mount Google Drive and update the dataset path accordingly.

# Performance Evaluation Report

## Model Performance Summary

Based on the evaluation metrics obtained from our contradiction-diagnosis models, I've prepared the following comprehensive analysis of their performance.

| Model | Accuracy | Precision | Recall | F1-score |
|-------|----------|-----------|--------|----------|
| Random Forest | 0.28 | 0.27 | 0.28 | 0.27 |
| Decision Tree | 0.33 | 0.33 | 0.33 | 0.33 |
| XGBoost | 0.34 | 0.34 | 0.34 | 0.33 |
| ANN | 0.23* | - | - | - |
| BERT | 0.32 | - | - | - |
| Optimized RF | 0.36 | 0.36 | 0.36 | 0.29 |

*ANN validation accuracy

