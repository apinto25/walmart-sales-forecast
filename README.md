# Walmart Sales Forecast
## Walmart sales forecasting project using ML algorithms and times Series models.

This project was developed in order to learn how to use different ML algorithms to predict the sales of a company and compare them with Time Series models.
For this project We used the Walmart sales dataset from the kaggle competition.

The codes are separated in four jupyter notebooks where we first explore the data and decided which characteristics of the dataset will be used fo the models. Then, we cleaned the dataset removing the columns with more than 40% of missing values and applied one hot encoding to categorical values. Finally, we built the machine learning algorithms and the time series models. The jupyter notebooks are listed below.

1. [Data Exploration](/exploring_walmart_datasets.ipynb)
2. [Data Cleaning](/data_cleaning_feature_engineering.ipynb) 
3. [Machine Learning Algorithms](/machine_learning_models.ipynb) 
4. [Times Series Models](/time_series_models.ipynb)

We also developed a stremlit app to visualize the results of the best models found for the ML algorithms.

## Requirements
Python 3.6 or higher

## Installation
To use in a python3 virtual environment, create the environment, activate it and run:

    $ pip3 install -r requirements.txt

## Usage
To run the jupyter-notebooks open the terminal and run:

    $ jupyter-notebooks

Then open the files in order and run each cell of the notebooks.


To open the stremalit app open the terminal and run:

    $ streamlit run app.py


### Authors
[Jose Miguel Correa](https://github.com/sandoval19) and [Ana Maria Pinto](https://github.com/apinto25/)
