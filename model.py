import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data.csv")

X = df[["Height", "Weight", "Eye"]]
X = X.replace(["Brown", "Blue"], [1, 0])

y = df["Species"]

clf = LogisticRegression()
clf.fit(X, y)

# Going to pickle the model which would separate the training process and the 
# user experience (by removing model training time)
import joblib

joblib.dump(clf, "clf.pkl")