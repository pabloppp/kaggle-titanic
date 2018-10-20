from tpot import TPOTClassifier
from tools import prepare_dataset

y, x = prepare_dataset()

x_train = x[:614]
y_train = y[:614].reshape(-1, )

x_valid = x[614:]
y_valid = y[614:].reshape(-1, )

pipeline_optimizer = TPOTClassifier(generations=50, population_size=20, cv=5, random_state=42, verbosity=2)

pipeline_optimizer.fit(x_train, y_train)
print(pipeline_optimizer.score(x_train, y_train))

pipeline_optimizer.export('tpot_exported_pipeline.py')