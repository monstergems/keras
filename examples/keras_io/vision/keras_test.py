import tensorflow as tf
import keras
import pandas as pd
import matplotlib.pyplot as plt



fashion_mnist=keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
#print(X_train_full.shape)

print(len(X_train_full))

X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat","Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

#print(class_names[y_train[0]])


model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))
optimizer_sgd=keras.optimizers.SGD(learning_rate=0.01)
model.compile(loss="sparse_categorical_crossentropy",optimizer=optimizer_sgd, metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=5,validation_data=(X_valid, y_valid)) #maar 5 epchos om voor de snelheid aangezien ik maar aan het testen ben

show_graph=False
if show_graph:
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
    plt.show()

print("evaluatie is")
print(model.evaluate(X_test, y_test))

X_new = X_test[:3]
y_proba = model.predict(X_new)
print(y_proba.round(2))