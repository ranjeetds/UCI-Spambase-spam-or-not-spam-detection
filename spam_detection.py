# Importing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, cross_val_score

# Setting random seed to have results repeatability
np.random.seed(23)

# Loading dataset into pandas dataframe
data = pd.read_csv('../spambase.data', names=[x for x in range(58)])

# First lest shuffle the dataset
data = data.sample(frac=1).reset_index(drop=True)

# Now lets split dataset into features and target
Y = data[57]
del data[57]
X = data

# Now lets lets split dataset into train and test set using scikit learn
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

# Now using scikit learn lets sacle dataset using standardscalar
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Lets do Cross validation in order to check whether model is performing consistantly and not over fitting
clf_cv = LogisticRegression()
kf = KFold(n_splits=10, shuffle=True)
cv_results = cross_val_score(clf_cv, X_train, y_train, cv=kf, scoring="accuracy", n_jobs=-1)

print("\nCross validation score is: \n")
print(cv_results)
print
# CV results indecate model is performing well and can be used on test set

# Hence, Now lets use logistic regression to see how it performs
clf = LogisticRegression().fit(X_train, y_train)

predictions = clf.predict(X_test)

# Now lets evaluate the results
print
print("Classification results without any parameter tuning or CV are\n")
print(classification_report(predictions, y_test, labels=[0,1]))
print

# As results are considerable lets not do hyperparamter tuning