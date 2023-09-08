# Project: Disaster Response Pipeline
#### Classification of disaster messages
## Table of Content
1. [Overview](#overview)
2. [Installation](#installation)
3. [About the Data](#about-the-data)
4. [About the Methods](#about-the-methods)
5. [Web Application](#webpage)
6. [Acknowledgements & Licensing](#acknowledgements--licensing)

## Overview <a name="overview"/>

Climate change has led to an increase in environmental disasters, making the role of emergency services more crucial than ever before. One of the challenges emergency services face during these disasters is the influx of text messages that need to be classified into specific categories to streamline the response process. This is particularly challenging during disasters when resources are stretched to their limits.

As part of my capstone project for the Udacity Data Science course, I have developed a machine learning model using Python to address this problem. The goal of this project is to create a tool that can automatically classify text messages sent during disasters into predefined categories. This tool can assist emergency services in efficiently allocating resources and responding to critical situations during disasters.

## Installation <a name="installation"/>
1. Clone repository:
```
git clone [https://github.com/romanwolf-git/](https://github.com/romanwolf-git/disaster_response)
```

2. Install requirements.txt
```
pip install requirements.txt
```

3. Run process_data.py to clean the data and store it in a database:
```
python data/process_data.py data/messages.csv data/categories.csv data/disaster_response.db
```
4. Run the 'data_pipeline.py' to train and save the model
```
python models/data_pipeline.py data/disaster_response.db
```
5. Run flask app 'run.py' which opens an adress http://0.0.0.0:3001/ which you open in your browser to view the app
```
python app/run.py
```
6. After running the command, you should see output indicating that the Flask app is running. By default, it should be accessible at http://0.0.0.0:3001/.
Open your web browser and enter http://0.0.0.0:3001/ in the address bar. Your web browser should load the Flask app. You can now interact with the app to view its content and functionality.

## About the Data <a name="about-the-data"/>
In this project, I utilized two primary datasets to develop and evaluate a machine learning model:

    categories.csv: This dataset provides multilabel categories for each message in our corpus. Each row in this dataset corresponds to a specific message and includes labels that categorize the message according to the types of assistance or information it may require during a disaster.
    messages.csv: This dataset contains the actual text messages. It includes the messages in their original language as well as their English translations. These text messages serve as the core data upon which our model is trained and tested.

One critical aspect of this dataset is its significant class imbalance. Imbalanced data means that some categories have far fewer instances than others. In fact, the category 'child_alone' doesn't have any representation. This imbalance can pose challenges during model training and evaluation, as it can lead to skewed model performance metrics and difficulties in effectively capturing minority classes.
To handle this imbalance, the data is resampled using [multi-label SMOTE](https://www.kaggle.com/code/tolgadincer/upsampling-multilabel-data-with-mlsmote).

## About the Methods <a name="about-the-methods"/>
The python script 'process_data.py' is used to clean the data.
Categories are split into single columns. Column names for each category are given. Duplicates are removed as well as one category which has no counts. 
Both datasets are concatenated to one dataset. And finally it is saved to a sql database.

The python script 'train_classifier.py' generates the machine learning model.
The english text messages are tokenized, lemmatized and stop words are being removed. The model is build by using the CountVectorizer combined with the custom 
tokenizer and a tfidf transformer. MLSMOTE (Multilabel Synthetic Minority Over-sampling TEchnique) is used to oversample the train data. Random Forest Classifier combined with the MultiOutputClassifier from scikit is used for classification. The model is evaluated using the f1-score.

The oversampling is only applied to the train data. Therefore the vectorizer and the model is saved as a pickle file. Both are used to apply the model to new text 
messages in the web application. For future work a pipeline including vectorizer and MLSMOTE can be used as imblearn pipeline does not support multilabel yet.

Due to lack of computational power only about half of the data is used to train the model. One can change this in 'train_classifier.py' line 31 and 32 by loading all rows.

## Web Application <a name="webpage"/>
Screenshots from the webpage:
1. Navbar with link to Udacity and Github Repo. In the top there is the input for a text message to process it in the model.
![Top section of the webpage](https://github.com/LollaPie/Data_Science_Disaster_Response/blob/main/images/Screenshot_1.png?raw=true)

2. Two plots of the distributions of categories and genres in the dataset.
![Bar chart of the categories](https://github.com/LollaPie/Data_Science_Disaster_Response/blob/main/images/Screenshot_2.png?raw=true)
![Pie chart of the genres](https://github.com/LollaPie/Data_Science_Disaster_Response/blob/main/images/Screenshot_3.png?raw=true)

## Acknowledgements & Licensing <a name="acknowledgements--licensing"/>
Credits to Figure Eight Inc. to provide the data and Udacity to provide the course and the support.
