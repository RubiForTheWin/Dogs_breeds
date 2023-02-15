import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data():
    data_frame = pd.read_csv("titanic_data.csv")  # קריאת הנתונים
    print(data_frame.describe())  # מדפיס את תכונות הנתונים
    print("-------------------")

    print(data_frame.head())  # מדפיס את הנתונים
    print("-------------------")

    print(data_frame.isna().sum())

    avg_age = data_frame["Age"].mean()
    data_frame["Age"].replace(to_replace=np.nan, value=avg_age, inplace=True)
    # drop null data
    data_frame.drop("Cabin", axis=1, inplace=True)
    data_frame.dropna(inplace=True)
    # create dummy variables for sex and embarked columns
    sex_data = pd.get_dummies(data_frame["Sex"], drop_first=True)
    embarked_data = pd.get_dummies(data_frame["Embarked"])
    data_frame = pd.concat([data_frame, sex_data, embarked_data], axis=1)
    data_frame.drop(["Name", "PassengerId", "Ticket", "Sex", "Embarked"], axis=1, inplace=True)

    y = data_frame["Survived"].to_numpy().reshape(-1, 1)
    x = data_frame.drop("Survived", axis=1)

    return x, y

    # def input_missing_age(columns):
    # age = columns[0]
    # if pd.isnull(age):
    # rv = date_frame["Age"].mean()
    # else:
    # rv = age
    # return rv


def gradient_decent(x, y):
    ones = np.ones(len(x), dtype=int)
    ones = ones.reshape(-1, 1)  # קובע מימדים חדשים לוקטור, ה- -1 אומר לתוכנה שתקבע לבד את מספר השורות
    x = np.concatenate((ones, x), axis=1)  # מחבר (משרשר) בין 2 הוקטורים במקביל
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=8)
    w = train(x_train, y_train)
    print(f"w= {w}")
    test(w, x_test, y_test)


def test(w, x_test, y_test):
    sig = sigmoid(x_test @ w)
    y_hat = np.ones((y_test.shape[0], 1))
    y_hat[np.where(sig < 0.5)] = 0

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for y_i, y_hat, in zip(y_test, y_hat):
        if y_i == 1 and y_hat == 1:
            tp += 1
        if y_i == 0 and y_hat == 0:
            tn += 1
        if y_i == 0 and y_hat == 1:
            fp += 1
        if y_i == 1 and y_hat == 0:
            fn += 1
    p = tp / (tp+fp)
    r = tp / (tp+fn)
    f1 = 2 * (r * p) / (p+r)
    accuracy = (tp+tn) / (tp+tn+fp+fn)
    print(f"p={p}, r={r}, f1={f1}, accuracy={accuracy}, tp={tp}, tn={tn}, fp={fp}, fn={fn} ")


def train(x, y):
    print("train:")
    alpha = 0.001  # גודל הקפיצה
    count = len(y)  # מספר המשתנים
    rounds = 500000  # מספר הסיבובים
    prev_cost = None
    w = np.zeros(x.shape[1]).reshape(-1, 1)  # קובע מימדים חדשים לוקטור, ה- -1 אומר לתוכנה שתקבע לבד את מספר השורות

    for i in range(rounds):
        z = x @ w  # חישוב התוצאות עם w נוכחי
        y_hat = sigmoid(z)  # פעולה המחשבת את y_hat
        dw = (1 / count) * x.T @ (y_hat - y)  # חישוב הנגזרת של ה- cost לפי ה w
        w = w - alpha * dw  # עדכון ה- w לפי הנגזרת
        cost = cross_entropy(y_hat, y)  # בדיקת התכנסות - תנאי יציאה מוקדם יותר

        if i < 10 or i % 100 == 0:
            print(f"gradiant decent: cost after {i} rounds is {cost}")  # מדפיס את ה cost בכל סיבוב
        if prev_cost is not None and np.abs(prev_cost - cost) < 1e-5:
            print(f"Converged at iteration {i}, cost ={cost}")
            break
        prev_cost = cost
    print(f"w= {w}")
    return w


def cross_entropy(y_hat, y):
    rv = -np.sum(y*np.log(y_hat) + (1-y) * np.log(1-y_hat))  # פונקציה המחשבת את ה cost
    return rv


def sigmoid(x):
    rv = 1/(1+np.exp(-x))  # פעולה המחשבת את y_hat
    return rv


def main():
    x, y = load_data()  # קריאה לפונקציה
    gradient_decent(x, y)  # קריאה לפונקציה

    print("done")


main()
