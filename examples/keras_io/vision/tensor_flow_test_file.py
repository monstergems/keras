import tensorflow as tf
import keras
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))



def build_MLP(n_nodes=200,activation_func="relu",learning_rate=0.01,input_Shape=[28,28],output_shape=10,use_dropout=False,dropout_rate=0.2,use_l1=False,use_l2=False,l1=0.005,l2=0.001):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=input_Shape))
    
    if use_l1 and use_l2==False:
        model.add(keras.layers.Dense(n_nodes, activation=activation_func))
    elif use_l1 and use_l2==True:
        model.add(keras.layers.Dense(n_nodes, activation=activation_func,kernel_regularizer=keras.regularizers.L1L2(l1=l1,l2=l2)))
    elif use_l1==True:
        model.add(keras.layers.Dense(n_nodes, activation=activation_func,kernel_regularizer=keras.regularizers.L1(l1=l1)))
    else:
        model.add(keras.layers.Dense(n_nodes, activation=activation_func,kernel_regularizer=keras.regularizers.L2(l2=l2)))

    if use_dropout==True:
        model.add(keras.layers.Dropout(rate=dropout_rate))
        
    model.add(keras.layers.Dense(output_shape, activation="softmax"))
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(loss="sparse_categorical_crossentropy",optimizer=optimizer, metrics=["accuracy"])
    return model

def build_cnn(n_nodes=128,activation_func="relu",lr=0.01,input_shape=[28,28,1],output_shape=10,use_dropout=True,dropout_rate=0.5,use_l1=False,use_l2=False,l1=0.005,l2=0.005):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(64,7,activation=activation_func,padding="same",input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D(2))
    model.add(keras.layers.Conv2D(128,3,activation=activation_func,padding="same"))
    model.add(keras.layers.Conv2D(128,3,activation=activation_func,padding="same"))
    model.add(keras.layers.MaxPooling2D(2))
    model.add(keras.layers.Flatten())

    if use_l1 and use_l2==False:
        model.add(keras.layers.Dense(n_nodes,activation=activation_func))
    elif use_l1 and use_l2==True:
        model.add(keras.layers.Dense(n_nodes,activation=activation_func,kernel_regularizer=keras.regularizers.L1L2(l1=l1,l2=l2)))
    elif use_l1==True:
        model.add(keras.layers.Dense(n_nodes,activation=activation_func,kernel_regularizer=keras.regularizers.L1(l1=l1)))
    else:
        model.add(keras.layers.Dense(n_nodes,activation=activation_func,kernel_regularizer=keras.regularizers.L2(l2=l2)))

    if use_dropout==True:
        model.add(keras.layers.Dropout(dropout_rate))

    if use_l1 and use_l2==False:
        model.add(keras.layers.Dense(n_nodes/2,activation=activation_func))
    elif use_l1 and use_l2==True:
        model.add(keras.layers.Dense(n_nodes/2,activation=activation_func,kernel_regularizer=keras.regularizers.L1L2(l1=l1,l2=l2)))
    elif use_l1==True:
        model.add(keras.layers.Dense(n_nodes/2,activation=activation_func,kernel_regularizer=keras.regularizers.L1(l1=l1)))
    else:
        model.add(keras.layers.Dense(n_nodes/2,activation=activation_func,kernel_regularizer=keras.regularizers.L2(l2=l2)))

    if use_dropout==True:
        model.add(keras.layers.Dropout(dropout_rate))
        
    model.add(keras.layers.Dense(output_shape,activation="softmax"))

    optimizer = keras.optimizers.SGD(learning_rate=lr)
    model.compile(loss="sparse_categorical_crossentropy",optimizer=optimizer, metrics=["accuracy"])
    return model

test_model=build_cnn()
print(test_model.summary())
