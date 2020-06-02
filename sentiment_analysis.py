import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

[(x_train,y_train), (x_test,y_test)] = tf.keras.datasets.imdb.load_data(
    path="imdb.npz")

x_train = pad_sequences(x_train)
x_train = np.array(x_train)
y_train = np.array(y_train) 
vocab_size = np.max(x_train)
print(y_train[0])
print(np.shape(x_train))
print(np.shape(y_train))
## Training data prepared
word_embeddings_dim = 50

input_layer  = keras.Input(shape=(None,), name="InputLayer")
embedding_layer_init = layers.Embedding(vocab_size+1,word_embeddings_dim)
embedding_layer = embedding_layer_init(input_layer)

BiRNN = layers.Bidirectional(layers.LSTM(word_embeddings_dim, return_sequences=True))(embedding_layer)
LSTM_RNN = layers.LSTM(2*word_embeddings_dim)(BiRNN)
LSTM_RNN = layers.Dropout(0.1)(LSTM_RNN)

first_dense = layers.Dense(2*word_embeddings_dim, name="firstDense")(LSTM_RNN)
first_dense = layers.Dropout(0.2)(first_dense)
second_dense = layers.Dense(0.5*word_embeddings_dim, name="secondDense")(first_dense)
second_dense = layers.Dropout(0.2)(second_dense)

output_layer = layers.Dense(2,name="OutputLayer",activation = "softmax")(second_dense)

model = keras.Model(
    inputs=[input_layer],
    outputs=[output_layer]
)


keras.utils.plot_model(model, "sentiment_classification.png", show_shapes=True)
opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-5)

model.compile(
    optimizer=opt,               
    loss= 'sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    {"InputLayer": x_train},
    {"OutputLayer": y_train},
    epochs=4,
    batch_size= 20
)

model.save('sentiment_classify')





