# Disaster Tweets Classification

## Overview
This project implements various machine learning models to classify tweets as either disaster-related (1) or non-disaster-related (0). The system uses natural language processing techniques and compares the performance of multiple classification algorithms, with Multinomial Naive Bayes achieving the best performance at 80% accuracy.

## Features
- Text preprocessing pipeline including:
  - Special character removal
  - Lowercase conversion
  - Stop word removal
  - Stemming and lemmatization
- TF-IDF vectorization
- Multiple model implementations:
  - Multinomial Naive Bayes
  - Logistic Regression
  - K-Nearest Neighbors

## Model Performance
### Multinomial Naive Bayes (Best Performer)
- Accuracy: 80%
- Precision (Disaster): 81%
- Recall (Disaster): 70%
- F1-Score (Disaster): 75%

### Logistic Regression
- Accuracy: 79%
- Precision (Disaster): 80%
- Recall (Disaster): 69%
- F1-Score (Disaster): 74%

### KNN
- Accuracy: 67%
- Precision (Disaster): 96%
- Recall (Disaster): 23%
- F1-Score (Disaster): 38%

## Requirements
```
pandas
numpy
scikit-learn
nltk
re
```

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/disaster-tweets-classification.git
cd disaster-tweets-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required NLTK data:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

## Usage
1. Prepare your data in CSV format with a 'tweets' column and 'target' column
2. Run the preprocessing:
```python
python disaster_tweets_data.py
```

## Data Preprocessing Steps
1. Null value handling
2. Text cleaning:
   - Special character removal
   - Lowercase conversion
   - Tokenization
   - Stop word removal
   - Stemming and lemmatization
3. TF-IDF vectorization (max_features=5000)

## Project Structure
```
disaster-tweets-classification/
├── disaster_tweets_data.py
├── requirements.txt
├── README.md
└── data/
    └── disaster_tweets_data(DS).csv
```

## Model Details
- Text Vectorization: TF-IDF with 5000 features
- Train-Test Split: 80-20
- Random State: 42
- Models Implemented:
  - Multinomial Naive Bayes (default parameters)
  - Logistic Regression (max_iter=1000)
  - KNN (n_neighbors=5)

## Future Improvements
1. Implement advanced text preprocessing techniques
2. Try other classification algorithms (SVM, Random Forest)
3. Hyperparameter tuning
4. Add cross-validation
5. Implement real-time tweet classification
6. Add word embeddings (Word2Vec, BERT)

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
