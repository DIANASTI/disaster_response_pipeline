# Disaster Response Pipeline Project

## Introduction
----------------------
This project is part of Udacity, Data Scientist Nanodegree Program, Disaster Response Pipeline Project.
The goal of this project is to use the real messages that were sent during disaster events and to create a machine learning pipeline that will categorize these events 
so that  the messages can be sent to an appropriate disaster relief agency. 
Time is critical during or even after disaster events, so such an application can be very effective in providing needed aid faster.


## Project Components
----------------------
There are three components  for this project.

1. ETL Pipeline
Loads the messages and categories datasets
Merges the two datasets
Cleans the data
Stores it in a SQLite database

2. ML Pipeline
Loads data from the SQLite database
Splits the dataset into training and test sets
Builds a text processing and machine learning pipeline
Trains and tunes a model using GridSearchCV
Outputs results on the test set
Exports the final model as a pickle file

3. Flask Web App
Web app that displays under which queues the messages are classified

## Dataset
----------------------
categories.csv - contains categories of the messages
messages.csv - contains disaster response messages, the messages are reflected in both original and english languages


## Project Structure
----------------------
 - app
	- template
		- master.html  -  main page of web app
| |- go.html  -  classification result page of web app
|- run.py  -  Flask file that runs app
 - data
|- categories.csv  -  data to process
|- messages.csv  -  data to process
|- process_data.py
|- InsertDatabaseName.db  -  database to save clean data to
 - models
|- train_classifier.py
|- classifier.pkl  -  saved model
 - README.md


## Instructions
----------------------
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`


