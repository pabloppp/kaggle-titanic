import pandas as pd


def prepare_dataset():
    train = pd.read_csv("train.csv")
    train = train[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked"]]

    train["Sex"] = train["Sex"].fillna(1)
    train["Pclass"] = train["Pclass"].fillna(3)
    train["Embarked"] = train["Embarked"].fillna("C")
    train["Age"] = train["Age"].fillna(30) / 80.0

    train['Sex_1'] = train["Sex"].map({"male": 1, "female": 0})
    train['Sex_2'] = train["Sex"].map({"male": 0, "female": 1})
    train['Class_1'] = train["Pclass"].map({1: 1, 2: 0, 3: 0})
    train['Class_2'] = train["Pclass"].map({1: 0, 2: 1, 3: 0})
    train['Class_3'] = train["Pclass"].map({1: 0, 2: 0, 3: 1})

    train['Embarked_1'] = train["Embarked"].map({"C": 1, "Q": 0, "S": 0})
    train['Embarked_2'] = train["Embarked"].map({"C": 0, "Q": 1, "S": 0})
    train['Embarked_3'] = train["Embarked"].map({"C": 0, "Q": 0, "S": 1})

    train['FamilySize'] = train["SibSp"] + train["Parch"] + 1

    train = train.drop("Sex", axis=1).drop("Pclass", axis=1).drop("SibSp", axis=1).drop("Parch", axis=1).drop("Embarked", axis=1).as_matrix()

    return train[:, :1], train[:, 1:]


def load_test():
    train = pd.read_csv("test.csv")
    train = train[["PassengerId", "Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked"]]

    train["Sex"] = train["Sex"].fillna(1)
    train["Pclass"] = train["Pclass"].fillna(3)
    train["Age"] = train["Age"].fillna(30) / 80.0
    train["Embarked"] = train["Embarked"].fillna("C")

    train['Sex_1'] = train["Sex"].map({"male": 1, "female": 0})
    train['Sex_2'] = train["Sex"].map({"male": 0, "female": 1})
    train['Class_1'] = train["Pclass"].map({1: 1, 2: 0, 3: 0})
    train['Class_2'] = train["Pclass"].map({1: 0, 2: 1, 3: 0})
    train['Class_3'] = train["Pclass"].map({1: 0, 2: 0, 3: 1})

    train['Embarked_1'] = train["Embarked"].map({"C": 1, "Q": 0, "S": 0})
    train['Embarked_2'] = train["Embarked"].map({"C": 0, "Q": 1, "S": 0})
    train['Embarked_3'] = train["Embarked"].map({"C": 0, "Q": 0, "S": 1})

    train['FamilySize'] = train["SibSp"] + train["Parch"] + 1

    train = train.drop("Sex", axis=1).drop("Pclass", axis=1).drop("SibSp", axis=1).drop("Parch", axis=1).drop("Embarked", axis=1).as_matrix()

    return train[:, :1], train[:, 1:]


def prepare_dataset_featured():
    train = pd.read_csv("train.csv")
    train = train[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked"]]

    train['Sex'] = train["Sex"].map({"male": 10, "female": 0})

    def combine_sex_class(row):
        return row.Pclass + row.Sex

    train['Sex_Pclass'] = train.apply(combine_sex_class, axis=1)

    train["Sex"] = train["Sex"].fillna(1)
    train["Pclass"] = train["Pclass"].fillna(3)
    train["Embarked"] = train["Embarked"].fillna("C")
    train["Age"] = train["Age"].fillna(30) / 80.0

    train['Sex_1'] = train["Sex"].map({"male": 1, "female": 0})
    train['Sex_2'] = train["Sex"].map({"male": 0, "female": 1})
    train['Class_1'] = train["Pclass"].map({1: 1, 2: 0, 3: 0})
    train['Class_2'] = train["Pclass"].map({1: 0, 2: 1, 3: 0})
    train['Class_3'] = train["Pclass"].map({1: 0, 2: 0, 3: 1})

    train['Embarked_1'] = train["Embarked"].map({"C": 1, "Q": 0, "S": 0})
    train['Embarked_2'] = train["Embarked"].map({"C": 0, "Q": 1, "S": 0})
    train['Embarked_3'] = train["Embarked"].map({"C": 0, "Q": 0, "S": 1})

    train['FamilySize'] = train["SibSp"] + train["Parch"] + 1

    train = train.drop("Sex", axis=1).drop("Pclass", axis=1).drop("SibSp", axis=1).drop("Parch", axis=1).drop("Embarked", axis=1).as_matrix()

    return train[:, :1], train[:, 1:]