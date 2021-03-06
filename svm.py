from sklearn.svm import LinearSVC
from sklearn.preprocessing import Normalizer
from tools import prepare_dataset, load_test

y, x = prepare_dataset()

# x_train = x[:614]
# y_train = y[:614].reshape(-1, )

# x_valid = x[614:]
# y_valid = y[614:].reshape(-1, )

clf = LinearSVC(max_iter=10000, C=0.008)
clf.fit(x, y.reshape(-1, ))

print(clf.score(x, y.reshape(-1, )))

ids, x = load_test()
predictions = clf.predict(x)

for index, id_number in enumerate(ids):
    print(str(int(id_number[0])) + "," + str(int(predictions[index])))
