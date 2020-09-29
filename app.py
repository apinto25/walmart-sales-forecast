import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pydeck as pdk
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


hide_streamlit_style = """
            <style> 
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


@st.cache(persist=True)
def load_data():
    data = pd.read_csv(DATA_SOURCE)
    return data


def split_dataset(data):
    y = data['Weekly_Sales']
    X = data.drop(['Date', 'Weekly_Sales'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    return X_train, X_test, y_train, y_test


def linear_regression(X_train, X_test, y_train, y_test):
    lr = LinearRegression(normalize=True, fit_intercept=True)
    lr.fit(X_train,y_train)
    y_pred = lr.predict(X_test)
    train_score = lr.score(X_train, y_train)
    test_score = lr.score(X_test, y_test)
    return y_pred, train_score, test_score


def knn_regression(X_train, X_test, y_train, y_test):
    knn = KNeighborsRegressor(n_neighbors=10,n_jobs=4)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    train_score = knn.score(X_train, y_train)
    test_score = knn.score(X_test, y_test)
    return y_pred, train_score, test_score


def tree_regression(X_train, X_test, y_train, y_test):
    tree = DecisionTreeRegressor(random_state=0)
    tree.fit(X_train,y_train)
    y_pred = tree.predict(X_test)
    train_score = tree.score(X_train, y_train)
    test_score = tree.score(X_test, y_test)
    return y_pred, train_score, test_score


def forest_regression(X_train, X_test, y_train, y_test):
    forest = RandomForestRegressor(n_estimators = 400,max_depth=15,n_jobs=5)
    forest.fit(X_train,y_train)
    y_pred = forest.predict(X_test)
    train_score = forest.score(X_train, y_train)
    test_score = forest.score(X_test, y_test)
    return y_pred, train_score, test_score


def update_json(key, value):
    with open("static/sample.json", 'r') as f:
        data = json.loads(f.read())

    data[key] = value

    with open("static/sample.json", 'w') as f:
        f.write(json.dumps(data, sort_keys=True, indent=4, separators=(',', ': ')))


def create_df(y_data, dates, model_type):
    df = pd.DataFrame()
    df["Value"] = y_data
    df["Date"] = dates.reset_index(drop=True) if model_type != "real" else dates 
    df["Data"] = model_type
    return df


def append_dfs(real_df, optons):
    with open("static/sample.json", 'r') as f:
        data = json.loads(f.read())
    
    for df, value in data.items():
        df = pd.read_csv(value, parse_dates=['Date'])
        real_df = real_df.append(df, ignore_index = True)
    
    if "real" not in options and len(options) == 0:
        options.append("real")

    final_df = real_df[real_df['Data'].isin(options)]
    return final_df


DATA_SOURCE = "data/walmart_clean.csv"

st.title("Walmart Sales Forecasting")
st.markdown(
    """This application is a Streamlit dashboard that can be used to visualize the
    results obtained from different Machine Learning algorithms to forecast the 
    sales of the Walmart dataset from Kaggle."""
)

st.markdown("## Training the models")

original_data = load_data()
data = original_data.copy()
X_train, X_test, y_train, y_test = split_dataset(data)

dates = pd.to_datetime(X_test[['Year', 'Month', 'Day']].apply(lambda row: '-'.join(row.values.astype(str)), axis=1)).dt.date

real_df = create_df(y_test, dates, "real")
real_df.to_csv('static/df_real.csv', index=None)
update_json("real", 'static/df_real.csv')



select = st.selectbox("ML algorithm", [
    "Linear Regression", 
    "KNN Regressor",
    "Decission Tree Regressor",
    "Random Forest Regressor",
])

if select == "Linear Regression":
    # selected = st.text_input("Hyperparameter", "5")
    # try:
    #     selected = int(selected)
    # except:
    #     st.error("Hyperparameter should be integer")

    if st.button("Try algorithm"):
        y_pred, train_score, test_score = linear_regression(X_train, X_test, y_train, y_test)
        st.write("Training set score: {:.2f}".format(train_score))
        st.write("Test set score: {:.2f}".format(test_score))
        linear_df = create_df(y_pred, dates, "linear regression")
        
        linear_df.to_csv('static/df_linear.csv', index=None)
        update_json("linear regression", 'static/df_linear.csv')

elif select == "KNN Regressor":
    if st.button("Try algorithm"):
        y_pred, train_score, test_score = knn_regression(X_train, X_test, y_train, y_test)
        st.write("Training set score: {:.2f}".format(train_score))
        st.write("Test set score: {:.2f}".format(test_score))
        knn_df = create_df(y_pred, dates, "knn regression")
        
        knn_df.to_csv('static/df_knn.csv', index=None)
        update_json("knn regression", 'static/df_knn.csv')

elif select == "Decission Tree Regressor":
    if st.button("Try algorithm"):
        y_pred, train_score, test_score = tree_regression(X_train, X_test, y_train, y_test)
        st.write("Training set score: {:.2f}".format(train_score))
        st.write("Test set score: {:.2f}".format(test_score))
        tree_df = create_df(y_pred, dates, "tree regression")
        
        tree_df.to_csv('static/df_tree.csv', index=None)
        update_json("tree regression", 'static/df_tree.csv')

elif select == "Random Forest Regressor":
    if st.button("Try algorithm"):
        y_pred, train_score, test_score = forest_regression(X_train, X_test, y_train, y_test)
        st.write("Training set score: {:.2f}".format(train_score))
        st.write("Test set score: {:.2f}".format(test_score))
        forest_df = create_df(y_pred, dates, "forest regression")

        forest_df.to_csv('static/df_forest.csv', index=None)
        update_json("forest regression", 'static/df_forest.csv')


with open("static/sample.json", 'r') as f:
    paths = json.loads(f.read())


st.markdown("## Plotting the models")
options = st.multiselect('Models to be ploted', list(paths.keys()))

if st.button("Plot models result"):
    final_df = append_dfs(real_df, options)
    
    fig, ax = plt.subplots()
    sales = sns.lineplot(data=final_df, x="Date", y="Value", hue="Data", ax=ax)
    plt.xticks(rotation=30)
    sales.set_title("Average of sales per week and store type", fontsize=14)
    sales.set_ylabel("Average of weekly sales")
    sales.set_xlabel("Week date")
    st.pyplot(fig)
