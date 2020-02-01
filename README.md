# UCI-Spambase-spam-or-not-spam-detection
A simple Logistic regression classification to identify whether email is spam or not spam built using python and scikit learn

## Dataset used -
- UCI Spambase dataset [https://archive.ics.uci.edu/ml/datasets/spambase]

## Requirements 
- Install ```Pandas``` ```Numpy``` and ```Scikit learn```
- Python3.X

## How to run?
- ```Python3.x spam_detection.py```

## Results

```

Cross validation score is: 

[0.93167702 0.90993789 0.9068323  0.9378882  0.94720497 0.92236025
 0.92236025 0.92857143 0.91304348 0.91925466]

```
```
Classification results without any parameter tuning or CV are

              precision    recall  f1-score   support

           0       0.97      0.92      0.94       883
           1       0.87      0.94      0.91       498

    accuracy                           0.93      1381
   macro avg       0.92      0.93      0.92      1381
weighted avg       0.93      0.93      0.93      1381
```
