import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import the dataset, replace dataset filename as needed
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting data into training and test
# import scikit-learn package
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)