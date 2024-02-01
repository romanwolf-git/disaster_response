"""
WEB APPLICATION

Script Execution via:
--> python run.py
"""

import json
import os
import pickle
import sys

from flask import Flask
from flask import render_template, request
import pandas as pd
import plotly

sys.path.append('../')
from config.definitions import root_dir
from models.data_pipeline import tokenize  # for the model to be loaded

app = Flask(__name__)

# load data
database_path = os.path.join(root_dir, 'data', 'disaster_response.db')
df = pd.read_sql_table(table_name='disaster_response.db', con=f'sqlite:///{database_path}')

# load model
model_path = os.path.join(root_dir, 'models', 'model.pkl')
with open(model_path, 'rb') as file:
    model, vec_tfidf = pickle.load(file)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    category_counts = df.iloc[:, 4:].sum()
    category_counts = category_counts.sort_index()
    category_names = category_counts.index

    category_counts_resampled = pd.read_csv('../data/resampling_counts.csv')
    category_counts_resampled.sort_values(by='Unnamed: 0', inplace=True)

    # create graphs for plotly
    graphs = [
        # plot #1: stacked bar plot with drop down menu
        {'data': [{
            'x': [category.replace('_', ' ').title() for category in category_names],
            'y': category_counts_resampled.loc[:, 'diff_0'].to_list(),
            'name': 'without resampling',
            'marker': {
                'color': '#23527C'
            },
            'type': 'bar'
        },
            {
            'x': [category.replace('_', ' ').title() for category in category_names],
            'y': category_counts_resampled.loc[:, 'diff_500'].to_list(),
            'name': 'n_sample = 500',
            'marker': {
                'color': '#02B3E4'
            },
            'type': 'bar'
        },
            {
                'x': [category.replace('_', ' ').title() for category in category_names],
                'y': category_counts_resampled.loc[:, 'diff_1000'].to_list(),
                'name': 'n_sample = 1000',
                'visible': False,
                'marker': {
                    'color': '#02B3E4'
                },
                'type': 'bar'
            },
            {
                'x': [category.replace('_', ' ').title() for category in category_names],
                'y': category_counts_resampled.loc[:, 'diff_2500'].to_list(),
                'name': 'n_sample = 2500',
                'visible': False,
                'marker': {
                    'color': '#02B3E4'
                },
                'type': 'bar'
            },
            {
                'x': [category.replace('_', ' ').title() for category in category_names],
                'y': category_counts_resampled.loc[:, 'diff_5000'].to_list(),
                'name': 'n_sample = 5000',
                'visible': False,
                'marker': {
                    'color': '#02B3E4'
                },
                'type': 'bar'
            }
        ],
            'layout': {
                'title': {
                    'text': 'Number of Data Points per Category',
                    'x': 0.07,
                },
                'yaxis': {
                    'title': 'Count',
                    'title_font': {
                        'size': 16
                    }},
                'xaxis': {
                    'title_font': {
                        'size': 16
                    },
                    'tickfont': {
                        'size': 16
                    },
                    'automargin': True,
                },
                'legend': dict(
                    yanchor='top',
                    y=1,
                    xanchor='right',
                    x=1),
                'barmode': 'stack',
                'updatemenus': list([
                    dict(
                        buttons=list([
                            dict(label='500',
                                 method='update',
                                 args=[{'visible': [True, True, False, False, False]}]),
                            dict(label='1000',
                                 method='update',
                                 args=[{'visible': [True, False, True, False, False]}]),
                            dict(label='2500',
                                 method='update',
                                 args=[{'visible': [True, False, False, True, False]}]),
                            dict(label='5000',
                                 method='update',
                                 args=[{'visible': [True, False, False, False, True]}]),
                        ]),
                        direction='down',
                        x=1,
                        xanchor='right',
                        y=1.25,
                        yanchor='top',
                    )
                ]),
                'annotations': [dict(text='n_sample =',
                                     x=31,
                                     y=1.2,
                                     yref='paper',
                                     align='right',
                                     showarrow=False
                                     ),
                                dict(text='Synthetically created samples<br>using <a '
                                          'href=\"https://www.kaggle.com/code/tolgadincer/upsampling-multilabel-data'
                                          '-with-mlsmote\">multi-label SMOTE</a>',
                                     x=7,
                                     y=6000,
                                     ay=-50,
                                     yref='y',
                                     align='left',
                                     showarrow=True
                                     ),
                                ]
            }},

        # plot #2: bar plot
        {'data': [{
            'x': [genre.title() for genre in genre_names],
            'y': genre_counts,
            'type': 'bar',
            'marker': {'color': '#23527C'}
        }],
            'layout': {
                'title': {
                    'text': 'Distribution of Message Genres',
                    'x': 0.07
                },
                'yaxis': {
                    'title': 'Count',
                    'title_font': {
                        'size': 16
                    }},
                'xaxis': {
                    'title': 'Genre',
                    'title_font': {
                        'size': 16
                    },
                    'tickfont': {
                        'size': 16
                    }}}},
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')
    # perform vectorization and tfidf on query
    query_transform = vec_tfidf.transform([query])

    # use model to predict classification for query
    classification_labels = model.predict(query_transform)[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()
