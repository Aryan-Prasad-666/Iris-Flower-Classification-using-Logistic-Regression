import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

dataset = load_iris()

print(dataset.keys())

X_train, X_test, Y_train, Y_test = train_test_split(dataset.data, dataset.target, test_size=0.2, random_state=42)

print(len(X_train))
print(len(X_test))

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=10000)

model.fit(X_train, Y_train)

print(model.score(X_test, Y_test))

from sklearn.metrics import confusion_matrix
predicted = model.predict(X_test)
c_mat = confusion_matrix(Y_test, predicted)

print(c_mat)

sns.heatmap(c_mat, cmap="Blues", annot=True, fmt="d", xticklabels=dataset.target_names, yticklabels=dataset.target_names)
plt.xlabel("Predicted", fontsize=12) 
plt.ylabel("Actual", fontsize = 12)
plt.title("Confusion Matrix")
plt.show()