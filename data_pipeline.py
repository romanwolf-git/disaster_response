import pickle
import os
import re
import sys

from lightgbm import LGBMClassifier
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sqlalchemy import create_engine

from config import root_dir
from ml_smote_resampler import MLSMOTE_resampler


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

    # Create the database engine and read the table
    engine = create_engine(f'sqlite:///{database_path}')
    df = pd.read_sql_table(table_name=data_file, con=engine)

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

    clf = MultiOutputClassifier(LGBMClassifier())

    # define parameters and create grid search object
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
        n_sample: number of samples to create using multi-label SMOTE

    Output:
    X_res, y_res: resampled data and labels
           model: model tuple
    """
    # instantiate MLSMOTE_resampler
    ml_smote = MLSMOTE_resampler(n_sample)

    # unpack model, perform vectorization and tfidf and resample
    cv, vec_tfidf = model
    X_vec_tfidf = vec_tfidf.fit_transform(X_train)
    X_res, y_res = ml_smote.fit_resample(X_vec_tfidf, y_train)

    # save resampled data as pickle-files
    with open('X_res.pkl', 'wb') as file1, open('y_res.pkl', 'wb') as file2:
        pickle.dump(X_res, file1)
        pickle.dump(y_res, file2)

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
    cv, vec_tfidf = model

    # perform transformation on X_test
    X_test_vec_tfidf = vec_tfidf.transform(X_test)

    # fit with resampled data (X_res, y_res) and predict with transformed (X_test_vec_tfidf)
    cv.fit(X_res, y_res)
    y_pred = cv.predict(X_test_vec_tfidf)

    # output model test results
    clf_report = classification_report(y_test, y_pred, target_names=y_test.columns)
    acc = round(accuracy_score(y_test, y_pred), 4)

    # print classification report for each feature and accuracy
    print(clf_report)
    print(f'accuracy : {acc}')

    model = (cv, vec_tfidf)
    return model


def export_model(model):
    """
    Serialize the model and save it as a binary using the Pickle format.
    Input:
         model: tuple of a) grid search object of LGBMClassifier (cv) and
                         b) pipeline for vectorization and tfidf (vec_tfidf)
    """
    with open('models/model.pkl', 'wb') as file:
        pickle.dump(model, file)


def run_pipeline(data_file):
    X, y = load_data(data_file)  # run ETL pipeline
    model = build_model()  # build model pipeline
    model = train(X, y, model)  # train model pipeline
    export_model(model)  # save model


if __name__ == '__main__':
    data_file = sys.argv[1]  # get filename of dataset
    run_pipeline(data_file)  # run data pipeline
