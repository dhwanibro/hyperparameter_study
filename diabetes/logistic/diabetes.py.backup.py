import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler #forces data to 0 and 1 so better for extreme outliers not distorting scaling
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ### The target variable is "Diabetes_binary"
# 0 = no diabetes
# 1 = diabetes

df = pd.read_csv("diabetes_binary.csv")


X = df.drop("Diabetes_binary", axis = 1) #drop the column which has all ans to prevent peeking and all
Y = df["Diabetes_binary"]
# if OG data has imbalance then test and train data has same imbalance ratio using stratisfy. test_size = 0.2 means .2 test .8 train
X_train, X_test, Y_train, Y_test = train_test_split(X, Y , test_size=0.2, random_state=42, stratify=Y) 

#creating an instance of the standarscalar class
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
#the scaler object stores the mean and std of every column inside itself. transform j applies it 

#applying the train data's mean and sd to this, will not fit it cause that causes leakage 
X_test = scaler.transform(X_test)

#  Hyperparam -> C
# C = 1/lambda (inverse regularization strength)
# low C → strong penalty → simpler model
# high C → weak penalty → model fits more → maybe overfit

C_values = [0.001, 0.01, 0.1, 1, 10, 100]

#for l2
test_acc_list = []
#train_acc_list = []
for C in C_values:
    model = LogisticRegression( C=C, max_iter=2000)
    model.fit(X_train,Y_train)
    # Predict on TRAIN data
    # (checks if model learned the training set well)
    Y_train_pred = model.predict(X_train)
# Predict on TEST data
    # (checks generalisation on unseen data)
    Y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(Y_train,Y_train_pred)
    test_acc = accuracy_score(Y_test, Y_test_pred)

  #  train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)


min_acc = min(test_acc_list)
max_acc = max(test_acc_list)

relative_change = (max_acc - min_acc) / min_acc
percent_change = relative_change * 100
with open("logreg_sensitivity.txt", "w") as f:
    f.write(f"Relative sensitivity to C: {percent_change:.4f}%\n")

plt.figure()
plt.plot(C_values, test_acc_list, marker = "o")

#Because C jumps like: 0.001 → 0.01 → 0.1 → 1 → 10 → 100
#A log scale spreads them evenly.
plt.xscale("log")

plt.xlabel("c value")
plt.ylabel("test accuracy")
plt.title("Sensitivity of Logistic Regression to C")
plt.grid(True)
plt.show()