import json
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Heatmap
import plotly.express as px
import plotly.figure_factory as ff
#from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('data', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
     
   
    #show count per category
    categories_to_sum = df.columns[4:].to_list()
    category_counts = df[categories_to_sum].sum().sort_values(ascending=False)
    category_names = category_counts.index.to_list()
    corr = df[categories_to_sum].corr()
    
    # proportion requiring medical help
    medical_needed_labels = ['Medical Help Needed','Medical Help Not Needed']
    medical_needed_values = [df.query('medical_help != 0').count()['message']/df.shape[0],df.query('medical_help == 0').count()['message']/df.shape[0]]
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': {
                    'text' : 'Distribution of Message Genres',
                    'font': { 'size' : 30 , 'color': "Green"}
                },                
                'yaxis': {
                   'title': {
                        'text' : 'Count',
                        'font': { 'size' : 20 }
                    }

                },
                'xaxis': {
                    'title': {
                        'text' : 'Genre',
                        'font': { 'size' : 20 }
                    }
                },
                'height' : 400,
                'width' : 600
            }
        },
        {
            'data': [
                Bar(
                    x=category_counts,
                    y=category_names,
                    orientation='h'
                )
            ],

            'layout': {
                'title': {
                    'text' : 'Number of Message Category',
                    'font': { 'size' : 30 , 'color': "Green"}
                },
                'yaxis': {
                   'title': {
                        'text' : 'Category',
                        'font': { 'size' : 20 }
                    }
                },
                'xaxis': {
                    'title': {
                        'text' : 'Number',
                        'font': { 'size' : 20 }
                    }
                },
                'margin' : dict(l=200,r=20,t=50,b=50),
                'height' : 800
            }
        },
        {
            'data': [
                Bar(
                    x=medical_needed_labels,
                    y=medical_needed_values
                )
            ],

            'layout': {
                'title': {
                    'text' : 'Proportion of Medical Aid Messages',
                    'font': { 'size' : 30 , 'color': "Green"}
                },
                'yaxis': {
                    'title': {
                        'text' : 'Ratio of Messages',
                        'font': { 'size' : 20 }
                    }
                },
                'xaxis': {
                    'title': {
                        'text' : 'Category',
                        'font': { 'size' : 20 }
                    }
                },
                'height' : 400,
                'width' : 600
            }
        }
                   
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

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
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