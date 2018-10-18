from sklearn.svm import SVC
from sklearn.preprocessing import Normalizer
from tools import prepare_dataset, load_test

y, x = prepare_dataset()

transformer = Normalizer()
transformer.fit(x)

x_train = x[:614]
y_train = y[:614].reshape(-1, )

x_valid = x[614:]
y_valid = y[614:].reshape(-1, )

clf = SVC(C=1)
clf.fit(x_train, y_train)

print(clf.score(x_valid, y_valid))

ids, x = load_test()
predictions = clf.predict(x)

for index, id_number in enumerate(ids):
    print(str(int(id_number[0])) + "," + str(int(predictions[index])))
