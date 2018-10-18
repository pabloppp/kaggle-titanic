from keras.layers.normalization import BatchNormalization
from tools import prepare_dataset, load_test
from keras.layers import Dense, Dropout, Activation, LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam, Adagrad, Adadelta, Nadam
import numpy as np

y, x = prepare_dataset()

# x_train = x[:614]
# y_train = y[:614]

# x_valid = x[614:]
# y_valid = y[614:]

print(x.shape)

chunks = 10
chunk_size = np.ceil(1.0 * x.shape[0] / chunks)

model = Sequential()
model.add(Dense(300, input_shape=(10,)))
model.add(LeakyReLU(0.1))

model.add(Dense(100))
model.add(LeakyReLU(0.1))

model.add(Dense(20))
model.add(LeakyReLU(0.1))

model.add(Dense(20))
model.add(LeakyReLU(0.1))

model.add(Dense(20))
model.add(LeakyReLU(0.1))

model.add(Dense(20))
model.add(LeakyReLU(0.1))

model.add(Dense(1, activation="sigmoid"))

optimizer = Nadam(lr=0.001)
model.compile(optimizer, 'binary_crossentropy', ['accuracy'])

model.fit(x, y, validation_split=0.05, batch_size=512, epochs=300)

# print(model.evaluate(x_valid, y_valid))

ids, x_test = load_test()
predictions = np.round(model.predict(x_test))

f = open("submission_perceptron.csv", "w")
f.write("PassengerId,Survived\n")
for index, id_number in enumerate(ids):
    f.write(str(int(id_number[0])) + "," + str(int(predictions[index])) + "\n")

f.close()
