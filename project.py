import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore') 

df = pd.read_excel("filtered_spotify_data.xlsx")

# data preprocessing
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

# drop null valued rows
df.dropna(inplace=True)

# transform the added column to be datetime objects.
df["added"] = df["added"].apply(get_day_of_week)

# create dummy columns
str_columns = ["title", "artist", "top genre", "added", "artist type", "grouped_genre"]
dummies = ["artist type", "added"]
new_df = pd.get_dummies(df.drop(list(filter(lambda x: x not in dummies, str_columns)), axis=1))
new_df["grouped_genre"] = df["grouped_genre"]
df = new_df

# utils

def training_loop(model, features, labels, is_train, question_num, model_name):

    predicted = model.predict(features)
    accuracy = accuracy_score(predicted,labels)
    precision = precision_score(labels, predicted, average="weighted")
    recall = recall_score(labels, predicted, average="weighted")
    
    print_results(model_name, question_num, is_train, accuracy, precision, recall)

    if(not is_train):
        title = f"{model_name} on question {question_num} accuracy is {accuracy:.3f}"
        confusion_matrix_plot(labels, predicted, model, title)

def print_results(model_name, question_num, is_train, accuracy, precision, recall):
    type_of_result = "validation"
    if(is_train):
        type_of_result = "training"
    print(f"question {question_num} {model_name} {type_of_result} Accuracy={accuracy:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")

def confusion_matrix_plot(labels, predicted, model, title):
    cm = confusion_matrix(labels, predicted, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    disp.ax_.set_title(title)

# question 1 - predicting what will be the top year for a song
features = df.drop(["top year", "grouped_genre"], axis=1)
labels = pd.DataFrame(df["top year"])

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, random_state = 42, test_size=0.25)
svm_model = SVC(kernel='linear')
svm_model.fit(train_features, train_labels)
training_loop(svm_model, train_features, train_labels, True, 1, "SVM")
training_loop(svm_model, test_features, test_labels, False, 1, "SVM")


logistic_regression_model = LogisticRegression(random_state=42)
logistic_regression_model.fit(train_features, train_labels)
training_loop(logistic_regression_model, train_features, train_labels, True, 1, "LogisticRegression")
training_loop(logistic_regression_model, test_features, test_labels, False, 1, "LogisticRegression")

print("\n")
# question 2 - classifying the songs genre
features = df.drop(["top year", "grouped_genre"], axis=1)
labels = pd.DataFrame(df["grouped_genre"])

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, random_state = 42, test_size=0.25)
svm_model = SVC(kernel='linear')
svm_model.fit(train_features, train_labels)
training_loop(svm_model, train_features, train_labels, True, 2, "SVM")
training_loop(svm_model, test_features, test_labels, False, 2, "SVM")


logistic_regression_model = LogisticRegression(random_state=42)
logistic_regression_model.fit(train_features, train_labels)
training_loop(logistic_regression_model, train_features, train_labels, True, 2, "LogisticRegression")
training_loop(logistic_regression_model, test_features, test_labels, False, 2, "LogisticRegression")

plt.show()