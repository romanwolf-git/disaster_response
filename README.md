# Project: Disaster Response Pipeline
Classification of disaster messages
## Table of Content
- [Overview](#overview)
- [Installation](#installation)
- [About the Data](#about-the-data)
- [Extract from the Classification Report](#classfication-report)
- [Methods](#methods)
- [Web Application](#webpage)
- [Acknowledgements & Licensing](#acknowledgements--licensing)

## Overview <a name="overview"/>

Climate change has led to an increase in environmental disasters, making the role of emergency services more crucial than ever before. One of the challenges emergency services face during these disasters is the influx of text messages that need to be classified into specific categories to streamline the response process. This is particularly challenging during disasters when resources are stretched to their limits.

As part of my capstone project for the Udacity Data Science course, I have developed a machine learning model using Python to address this problem. The goal of this project is to create a tool that can automatically classify text messages sent during disasters into predefined categories. This tool can assist emergency services in efficiently allocating resources and responding to critical situations during disasters.

## Installation <a name="installation"/>
1. Clone repository and change to its directory.
```
git clone https://github.com/romanwolf-git/disaster_response
```

2. Install requirements.txt. Mac users may also need 'brew install cmake libomp'.
```
pip install -r requirements.txt
```

3. Change to 'data' directory and run 'process_data.py' to clean the data and store it in a database.
```
python process_data.py messages.csv categories.csv disaster_response.db
```
4. Change to 'models' directory and run 'data_pipeline.py' to train, tune and save the model.
```
python data_pipeline.py disaster_response.db
```
5. Run 'run.py' which outputs: "Serving Flask app 'run'" and the URL (http://127.0.0.1:3000) where its run. Open the URL in your browser.
```
python run.py
```

## About the Data <a name="about-the-data"/>
In this project, I utilized two primary datasets to develop and evaluate a machine learning model:
- categories.csv: This dataset provides multilabel categories for each message in our corpus. Each row in this dataset corresponds to a specific message and includes labels that categorize the message according to the types of assistance or information it may require during a disaster.
- messages.csv: This dataset contains the actual text messages. It includes the messages in their original language as well as their English translations. These text messages serve as the core data upon which our model is trained and tested.

One critical aspect of this dataset is its significant class imbalance. Imbalanced data means that some categories have far fewer instances than others. In fact, the category 'child_alone' doesn't have any representation. This imbalance can pose challenges during model training and evaluation, as it can lead to skewed model performance metrics and difficulties in effectively capturing minority classes.
To handle this imbalance, the data is resampled using [multi-label SMOTE](https://www.kaggle.com/code/tolgadincer/upsampling-multilabel-data-with-mlsmote).

## Methods <a name="methods"/>
The Python script 'process_data.py' is used to clean the data and save it to a SQLite database.

The Python script 'data_pipeline.py' loads the data from the database, builds a model, trains and tunes the model with resampled data, outputs a classification report, and saves the model in a serialized pickle binary. 
generates the machine learning model.

The model consists of scikit-learn's CountVectorizer, TFDIFTransformer and LightGBM's classifier. Resampling is done using the Multilabel Synthetic Minority Over-sampling Technique ([multi-label SMOTE](https://www.kaggle.com/code/tolgadincer/upsampling-multilabel-data-with-mlsmote)) with 500 additional training samples.

One of the project requirements is the use of a scikit-learn pipeline. However, resampling within a pipeline is only possible using imblearn, which currently does not support multi-labeling.

## Extract from the Classification Report <a name="classification-report"/>
| Average      | Precision | Recall | F1-Score   |
|--------------|-----------|--------|------------|
| Micro Avg    | 0.7929    | 0.6267 | **0.7001** |
| Macro Avg    | 0.5918    | 0.354  | **0.4208** |
| Weighted Avg | 0.7554    | 0.6267 | 0.6632     |
| Samples Avg  | 0.6339    | 0.5259 | 0.5312     |

## App <a name="webpage"/>
Screenshots from the application:
1. Front screen/dashboard of the application with:
   - Navigation bar,
   - Message input field
   - button for message classification,
   - 2 Plotly plots for training data overview.
![Fron screen/dashboard](https://github.com/romanwolf-git/disaster_response/blob/main/images/screenshot_app.png)

2. Message classification target page with selected categories
![Message classification target page](https://github.com/romanwolf-git/disaster_response/blob/main/images/screenshot_classification.png)

## Acknowledgements & Licensing <a name="acknowledgements--licensing"/>
Thanks to Figure Eight Inc. for providing the data and to Udacity for providing the course and support.
