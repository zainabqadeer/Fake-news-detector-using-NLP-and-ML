# Fake News Detection Using NLP and ML

## Overview of the project
This project classifies news articles as **Fake** or **Real** using Natural Language Processing and Machine Learning.

The dataset contains two files:

- `Fake.csv` for fake news articles
- `True.csv` for real news articles

The project combines both datasets, prepares the text data, converts text into numerical features using TF-IDF, and compares two machine learning models:

- Logistic Regression
- Multinomial Naive Bayes

The workflow also includes model evaluation, confusion matrix analysis, custom text prediction, and model saving.

## Project Workflow

1. Upload and extract the dataset ZIP file
2. Load `Fake.csv` and `True.csv`
3. Add class labels:
   - `0` = Fake
   - `1` = Real
4. Combine the two datasets
5. Check missing values
6. Perform exploratory data analysis
7. Combine the news title and article text
8. Shuffle the dataset
9. Split the data into training and testing sets
10. Convert text into numerical features using TF-IDF
11. Train Logistic Regression
12. Train Multinomial Naive Bayes
13. Compare model performance
14. Evaluate the models
15. Test the model on custom news text
16. Save the trained model and TF-IDF vectorizer

## Dataset

The dataset consists of fake and real news articles.

Main columns include:

- `title`
- `text`
- `subject`
- `date`

A new `label` column is added during preprocessing:

| Label | Class |
|---|---|
| 0 | Fake News |
| 1 | Real News |

## Exploratory Data Analysis

The project includes:

- Dataset shape analysis
- Missing value checking
- Fake vs. real news class distribution
- Article length analysis
- Article length distribution visualization

## Text Preprocessing

The `title` and `text` columns are combined into one feature called `content`.

The dataset is shuffled and split into:

- 80% training data
- 20% testing data

Stratified splitting is used to maintain a similar class distribution in both sets.

## Feature Extraction

The project uses `TfidfVectorizer` to convert text into numerical features.

Configuration used:

- English stop words
- Maximum document frequency: `0.7`
- Maximum features: `50,000`
- N-grams: `(1, 2)`

TF-IDF helps convert important words and word combinations into numerical features for the machine learning models.

## Models

### Logistic Regression

Logistic Regression is trained on the TF-IDF features to classify articles as fake or real.

### Multinomial Naive Bayes

Multinomial Naive Bayes is trained as a second model and compared with Logistic Regression.

## Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

The final model should be selected based on overall performance, not accuracy alone.

## Custom Prediction

The project includes a function that accepts custom news text and predicts:

- Fake News
- Real News

The function also displays the model's confidence.

## Important Limitation

This model is **not a fact-checking system**.

It does not independently verify whether a real-world claim is true or false. It learns patterns from the training dataset and classifies new text based on those learned patterns. A high confidence score should not be treated as proof that a news article is factually correct or incorrect.

## Technologies Used

- Python
- Google Colab
- Pandas
- Matplotlib
- Scikit-learn
- TF-IDF
- Logistic Regression
- Multinomial Naive Bayes
- Joblib

## Project Structure

```text
fake-news-detection/
│
├── Fake_News_Detection_with_Interpretations.ipynb
├── README.md
├── Fake.csv
└── True.csv
```

## How to Run

1. Open the notebook in Google Colab.
2. Run the upload cell and upload the dataset ZIP file.
3. Run all cells in order.
4. Review the exploratory analysis and model evaluation results.
5. Use the custom prediction function to test new text.
6. Save the trained model and TF-IDF vectorizer.

## Future Improvements

Possible improvements include:

- Testing additional machine learning models
- Using cross-validation
- Applying hyperparameter tuning
- Testing the model on a more diverse and recent dataset
- Building a simple web interface for predictions
- Using transformer-based NLP models for comparison

## Author

Zainab Rana
