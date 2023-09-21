"""
DATA PIPELINE

Script Execution via:
--> python data_pipeline.py disaster_response.db

:param file name of sqlite database
"""

import pickle
import os
import re
import sys

from lightgbm import LGBMClassifier
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline

sys.path.append('../')
from config.definitions import root_dir
from models.ml_smote import resample as ml_smote_resample


def tokenize(text: str, url_placeholder='url_placeholder'):
    """
    Replace:
         - URLs - with url_spaceholder,
         - consecutive identical punctuation - with a single instance,
         - non-alphanumeric - with space
    Tokenize, lemmatize and drop stop words

    Input: text: <class 'str'>
           url_placeholder: str
    Output: list of clean_tokens
    """

    # Replace URLs with url_placeholder
    url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(url_regex, url_placeholder, str(text))
    # Replace consecutive identical punctuation with a single instance
    text = re.sub(r'([!?.,])\1+', r'\1', text)
    # Replace non-alphanumeric with space
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.strip().lower())
    # Tokenize and lemmatize
    tokens = word_tokenize(text)
    stop_words = stopwords.words('english')
    # drop stop words from tokens
    clean_tokens = [WordNetLemmatizer().lemmatize(word) for word in tokens if word not in stop_words]

    return clean_tokens


def load_data(data_file):
    """
    Load data from sql database
    Input:  data_file: name of the database to load (*.db)
    Output:      X, y: disasters messages, labels
    """
    database_path = os.path.join(root_dir, 'data', data_file)
    df = pd.read_sql_table(table_name=data_file, con=f'sqlite:///{database_path}')

    cols_message = ['id', 'message', 'original', 'genre']
    X = df.message
    y = df.drop(columns=cols_message)

    return X, y


def build_model():
    """
    Build the model,
    resample using multi-label SMOTE and
    fit the model with the resampled data set

    Output:     model: tuple of a) grid search object of LGBMClassifier (cv) and
                                b) pipeline for vectorization and tfidf (vec_tfidf)
    """
    # text processing and model pipeline
    vec_tfidf = make_pipeline(
        CountVectorizer(tokenizer=tokenize,
                        token_pattern=None,
                        lowercase=False),
        TfidfTransformer())

    clf = MultiOutputClassifier(LGBMClassifier(verbose=-1))

    # define parameters and create grid search object with cross-validation
    parameters = {
        'estimator__num_leaves': [5, 7, 9],
        'estimator__min_child_samples': [20, 50, 100],
        'estimator__max_depth': [30, 50, 70]
    }
    cv = GridSearchCV(clf, param_grid=parameters, cv=5)
    model = (cv, vec_tfidf)

    return model


def resample(X_train, y_train, model, n_sample=500):
    """
    Resample the data using multi-label SMOTE
    Input:
         X_train: disaster messages as training data
         y_train: labels
           model: tuple of a) grid search object of LGBMClassifier (cv) and
                           b) pipeline for vectorization and tfidf (vec_tfidf)
        n_sample: number of samples to create for each label using multi-label SMOTE

    Output:
    X_res, y_res: resampled data and labels
           model: model tuple
    """

    # unpack model, perform vectorization, tfidf and resample
    cv, vec_tfidf = model
    X_vec_tfidf = vec_tfidf.fit_transform(X_train)
    X_res, y_res = ml_smote_resample(X_vec_tfidf, y_train, n_sample=n_sample)

    # repack model tuple
    model = (cv, vec_tfidf)

    return X_res, y_res, model


def train(X, y, model):
    """
    Perform train-test-split, resample and fit GridSearchCV object
    Input:
        X, y: disaster messages, labels
       model: tuple of a) grid search object of LGBMClassifier (cv) and
                       b) pipeline for vectorization and tfidf (vec_tfidf)
    Output:
       model: model tuple
    """
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # resampling using multi-label smote
    X_res, y_res, model = resample(X_train, y_train, model=model, n_sample=500)
    # unpack cross-fold GridSearch object and fitted vectorization/tfidf-pipeline
    cv, vec_tfidf = model

    # perform transformation on X_test
    X_test_vec_tfidf = vec_tfidf.transform(X_test)

    # fit with resampled data (X_res, y_res) and predict with transformed (X_test_vec_tfidf)
    cv.fit(X_res, y_res)
    y_pred = cv.predict(X_test_vec_tfidf)

    # print classification report and save as csv
    clf_report = classification_report(y_test, y_pred, target_names=y_test.columns, output_dict=True)
    clf_report = pd.DataFrame.from_dict(clf_report).T
    print(clf_report)
    clf_report_path = os.path.join(root_dir, 'models', 'data_pipeline_classification_report.csv')
    clf_report.to_csv(clf_report_path)

    # pack model tuple
    model = (cv, vec_tfidf)
    return model


def export_model(model):
    """
    Serialize the model and save it as a binary using the Pickle format.
    Input:
         model: tuple of a) grid search object of LGBMClassifier (cv) and
                         b) pipeline for vectorization and tfidf (vec_tfidf)
    """
    model_path = os.path.join(root_dir, 'models', 'model.pkl')
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)


def run_pipeline(data_file):
    print('Loading data...')
    X, y = load_data(data_file)  # run ETL pipeline
    print('Building model...')
    model = build_model()  # build model pipeline
    print('Training model...')
    model = train(X, y, model)  # train model pipeline
    print('Saving model...')
    export_model(model)  # save model
    print('Trained model saved!')


if __name__ == '__main__':
    data_file = sys.argv[1]  # get filename of dataset
    run_pipeline(data_file)  # run data pipeline
