import tensorflow as tf
import keras
import pandas as pd
import matplotlib.pyplot as plt




mnist=keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape)