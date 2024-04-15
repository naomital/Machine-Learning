import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay

global_word_to_number = {}
word_number_counter = 1

def get_day_of_week(date_str: str):
    try:
        date_str = date_str.replace('‑', '-')
        # Convert the input string to a datetime object
        date_obj = pd.to_datetime(date_str, format="%Y-%m-%d")
        # Get the day of the week (0=Monday, 1=Tuesday, ..., 6=Sunday)
        day_of_week = date_obj.dayofweek
        # List of days of the week
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        # Return the day of the week
        return days[day_of_week]
    except ValueError as e:
        print(e)


def pre_process_binary(data, type):
    # transform the added column to be datetime objects.
    data["added"] = data["added"].apply(get_day_of_week)

    # create dummy columns
    if type == 'genre':
        str_columns = ["title", "artist", "top genre", "added", "artist type", "grouped_genre"]
    else:
        str_columns = ["title", "artist", "top genre", "added", "artist type", "top year"]
    dummies = ['added', 'artist type', 'year released']
    new_data = pd.get_dummies(data.drop(list(filter(lambda x: x not in dummies, str_columns)), axis=1))
    if type == 'genre':
        new_data["grouped_genre"] = data["grouped_genre"]
    return new_data



def genre(file_path='filtered_spotify_data.xlsx'):
    data = pd.read_excel(file_path)
    new_data = pre_process_binary(data, 'genre')

    X = new_data[['bpm', 'nrgy', 'dnce', 'dB', 'live', 'val', 'dur', 'acous', 'spch', 'pop',  'added_Friday', 'added_Monday',
       'added_Thursday', 'added_Tuesday', 'added_Wednesday',
       'artist type_Band/Group', 'artist type_Duo', 'artist type_Solo',
       'artist type_Trio', 'year released']]
    y = new_data['grouped_genre']
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.3, random_state=80)
    return X_train, X_test, y_train, y_test, new_data['grouped_genre'].unique()

def top_year(file_path='Spotify 2010 - 2019 Top 100 Songs.xlsx'):
    data = pd.read_excel(file_path)
    new_data = pre_process_binary(data, 'top year')
    X = new_data[['added_Friday', 'added_Monday', 'added_Thursday', 'added_Tuesday', 'added_Wednesday', 'year released']]
    y = data['top year']
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.3, random_state=80)
    return X_train, X_test, y_train, y_test, data['top year'].unique()


def print_results(model_name, data_type, is_train, accuracy, precision, recall):
    type_of_result = "validation"
    if(is_train):
        type_of_result = "training"
    print(f"question {data_type} {model_name} {type_of_result} Accuracy={accuracy:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")
def random_forest(data_type):
    X_train, X_test, y_train, y_test, classes = data_type()
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")

    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=classes).plot()
    plt.title(f'Random Forest  predict {data_type.__name__} Accuracy: {accuracy:.3f}')
    plt.show()
    print_results("Random Forest", data_type.__name__, False, accuracy, precision, recall)


def k_nn(data_type):
    k_values = [1, 3, 5, 7, 9]
    X_train, X_test, y_train, y_test, classes = data_type()
    for ki, k in enumerate(k_values):

        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")

        ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=classes).plot()
        plt.title(f'K-NN{k} predict {data_type.__name__} Accuracy: {accuracy:.3f}')
        plt.show()
        print(f"K-NN{k} predict {data_type.__name__} Accuracy: {accuracy:.3f}")
        print_results("K-NN", data_type.__name__, False, accuracy, precision, recall)



def  DecisionTree(data_type):
    X_train, X_test, y_train, y_test, classes = data_type()
    rf = DecisionTreeClassifier()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=classes).plot()
    #add the model name to the plot
    plt.title(f'Decision Tree predict {data_type.__name__} Accuracy: {accuracy:.3f}')
    plt.show()
    print_results("Decision Tree", data_type.__name__, False, accuracy, precision, recall)



if __name__ == '__main__':
    random_forest(genre)
    random_forest(top_year)
    k_nn(genre)
    k_nn(top_year)
    DecisionTree(genre)
    DecisionTree(top_year)
