from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator
from tools import prepare_dataset, load_test

y, x = prepare_dataset()

x_train = x[:614]
y_train = y[:614].reshape(-1, )

x_valid = x[614:]
y_valid = y[614:].reshape(-1, )

# Average CV score on the training set was:0.8469063987308303
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=DecisionTreeClassifier(criterion="gini", max_depth=4, min_samples_leaf=15, min_samples_split=11)),
    StackingEstimator(estimator=BernoulliNB(alpha=0.01, fit_prior=False)),
    StackingEstimator(estimator=RandomForestClassifier(bootstrap=False, criterion="entropy", max_features=0.45, min_samples_leaf=3, min_samples_split=9, n_estimators=100)),
    StackingEstimator(estimator=ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.9000000000000001, min_samples_leaf=12, min_samples_split=16, n_estimators=100)),
    GaussianNB()
)

exported_pipeline.fit(x_train, y_train)

ids, x = load_test()
predictions = exported_pipeline.predict(x)

for index, id_number in enumerate(ids):
    print(str(int(id_number[0])) + "," + str(int(predictions[index])))