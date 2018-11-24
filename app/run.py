import json
import plotly
import pandas as pd

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
from ML_Pipeline.train_classifier import tokenize
import message_language
import pickle
from collections import Counter

app = Flask(__name__)

# load data
engine = create_engine('sqlite:///ETL/DisasterResponse.db')
df = pd.read_sql_table('messages_categorized', engine)

# load model
model = pickle.load(open('ML_Pipeline/classifier.pkl', 'rb'))

def return_figures():
    """Creates two plotly visualizations on our training data

          Args:
              none

          Returns:
              list (dict): list containing the two plotly visualizations

        """

    # We first visualize the most common categories over all of our messages
    graph_one = []

    # extract data needed for visuals
    category_sum = df.iloc[:, 1:].select_dtypes(include=['number']).apply(sum, axis=0)
    top_categories = category_sum.sort_values(ascending=False)[:5].values.tolist()
    category_names = category_sum.sort_values(ascending=False)[:5].index.tolist()

    # create visuals
    graph_one.append(
        Bar(
            x=category_names,
            y=top_categories
        )
    )

    layout_one = dict(title='Top 5 Most Common Categories',
                      xaxis=dict(title='Category'),
                      yaxis=dict(title='Count'),
                      )

    # Our second visualization will display the top 10 tokens in the messages
    graph_two = []

    # We extract the data necessary
    tokens = []
    df.message.apply(lambda x: [tokens.append(y) for y in tokenize(x)])
    top_tuples = Counter(tokens).most_common(10)

    words = []
    counts = []

    for word, count in top_tuples:
        words.append(word)
        counts.append(count)

    # create visuals
    graph_two.append(
        Bar(
            x=words,
            y=counts
        )
    )

    layout_two = dict(title='Top 10 Most Common Words in Messages',
                      xaxis=dict(title='Word'),
                      yaxis=dict(title='Count'),
                      )

    figures = list()
    figures.append(dict(data=graph_one, layout=layout_one))
    figures.append(dict(data=graph_two, layout=layout_two))

    return figures

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    figures = return_figures()
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(figures)]
    graphJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)
    
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
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()