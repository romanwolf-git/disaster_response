import os
import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sqlalchemy import create_engine
from config import root_dir


def load_data():
    """ load data from sql database """

    database_path = os.path.join(root_dir, 'data', 'BaseData.db')

    # Create the database engine and read the table
    engine = create_engine(f'sqlite:///{database_path}')
    df = pd.read_sql_table(table_name='BaseData.db', con=engine)

    X = df.message
    cols_message = ['id', 'message', 'original', 'genre']
    y = df.drop(columns=cols_message)

    return X, y


def tokenize(text: str, url_placeholder='url_placeholder'):
    """ Check, clean and tokenize text
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
    # Convert all items of tokens as str
    # tokens = [str(token) for token in tokens]
    # tokens = [item for item in corpus if not isinstance(item, int)]
    tokens = [token for token in tokens if not isinstance(token, int)]
    stop_words = stopwords.words('english')
    clean_tokens = [WordNetLemmatizer().lemmatize(word) for word in tokens if word not in stop_words]

    return clean_tokens
