# Disaster Response Pipeline
Creation of an ETL and Machine Learning pipeline to build a model
 to classify messages during a disaster response. This model is then
 serialized and used as back-end for a web app where a user can 
 input any message and see the categories it relates to. The web
 app also displays useful visualizations about the training data
 available.
 
 ## Installation
 
 To run the web app, simply clone this directory. 
 
 The following libraries are necessary:
 * json
 * plotly
 * pandas
 * flask
 * sklearn
 * sqlalchemy
 * pickle
 * collections
 * langdetect
 * nltk
 * sys
 * re
 * numpy
 
 ## Structure
 
 This repository has the following structure.
 
 ### App
 
 This folder is where both pipeline as well as the webpages templates
 are saved. 
 
 **message_language.py:** Python module creating a custom 
 transformer to detect the language of text
 
 **run.py**: Python script for the Flask web app.
 
 #### ETL
 
 The folder containing the messages to classify, their categories 
 and the Python script for the ETL pipeline.
 
 **disaster_categories.csv**: a CSV file containing the categories
 associated with each individual message
 
 **disaster_messages.csv**: a CSV file containing the training set
 of messages to classify
 
 **process_data.py**: a Python script for an ETL pipeline for these
 messages and their categories
 
 **DisasterResponse.db**: the SQLite database where the cleaned
 training set is loaded at the end of the ETL pipeline
 
 #### ML_Pipeline
 
 A folder containing the Machine Learning pipeline to train a model
 and save it for future use.
 
 **classifier.pkl**: a pickle file containing a serialized version
 of our model, to be leveraged in our web app
 
 **message_language.py**: Python module creating a custom 
 transformer to detect the language of text
 
 **train_classifier.py**: a Python script leveraging nltk and
 scikit-learn to build a machine learning pipeline to create a 
 classification model for these messages
 
 ## Usage
 
 The main goal of this repository is to create a web app which
 displays visualizations on the training data available as well
 as allows user to leverage a trained model to classify any
 message inputted in the web app.
 
 There are two main uses:
 1. **Leveraging existing data**: 
 Two example datasets, disaster_categories.csv and
 disaster_messages.csv are already available in this repository.
 They have been processed, and the model trained. As a result,
 running run.py and clicking on the link for the local web app
 will output the web page with the visualization and the bar 
 to classify new messages
 
 2. **Leveraging new data**:
 The user can put any file in the ETL folder, replacing the 
 existing ones. In order ot leverage the web app, three steps are
 necessary.
 * process the files with `process_data.py`. One example call from
 the command line: `python process_data.py
              messages.csv categories.csv
              StoreDatabase.db`
 * train the model with `train_classifier.py`. One example call
 from the command line: `python train_classifier.py
 ../ETL/StoreDatabase.db pickle_file.pkl`
 
 * replace lines 17-18 and 21 with the appropriate paths in `run.py`
 and run this script to launch the web app. Visualizations and
 the model will be adapted to these new datasets
 
 ### Caveats
 
 1. The web app has not been deployed. To see the process to do so, 
 please refer to the web_app_movie_revenues repository
 2. The model takes a while (~2 hours) to train. This is due to the
 fact that GridSearch is leveraged to optimize the hyperparameters
 over the entire pipeline. This allows reaching upwards of 85% 
 accuracy on a testing set. This is also a step that happens only
 once, and then the serialized pickle file is used to make
 inferences.
 3. The web app takes a while (< 5 min) to load. This is due to 
 the visualization which extracts the most popular tokens in the
 training set, as this requires a lot of text processing. Should
 this visualization be removed, the loading time will be greatly
 reduced.
  

## Credits

This web app was built as part of the Data Scientist nanodegree
from Udacity. Templates with starter code were provided for each
step (ETL Pipeline, Machine Learning Pipeline, Web App). The rest
of the code is my own.

## Licensing

Copyright 2018 Celestin Hermez

Permission is hereby granted, free of charge, 
to any person obtaining a copy of this software 
and associated documentation files (the "Software"), 
to deal in the Software without restriction, 
including without limitation the rights to use, 
copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons 
to whom the Software is furnished to do so, subject to the 
following conditions:

The above copyright notice and this permission notice shall 
be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, 
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE 
USE OR OTHER DEALINGS IN THE SOFTWARE.