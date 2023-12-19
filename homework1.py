import pickle
import gzip
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

class MNISTDataset:
    def __init__(self, train=True):
        with gzip.open("mnist.pkl.gz", "rb") as fd:
            (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = pickle.load(fd, encoding="latin")

        if train:
            self.inputs = np.concatenate((train_x, test_x))
            self.labels = np.concatenate((train_y, test_y))
        else:
            self.inputs = valid_x
            self.labels = valid_y


# o folosim ca sa putem testa mai multi clasificatori slabi/de baza
def run(base_classifier):
    dataset_train = MNISTDataset(train=True)
    dataset_test = MNISTDataset(train=False)

    adaboost_model = AdaBoostClassifier(base_classifier, n_estimators=40, random_state=1234)
    adaboost_model.fit(dataset_train.inputs, dataset_train.labels)

    # Evaluare pe setul de training
    predicted_lables_train = adaboost_model.predict(dataset_train.inputs)
    train_accuracy = accuracy_score(dataset_train.labels, predicted_lables_train)
    print("Train accuracy: %.4f" % train_accuracy)


    # Evaluare pe setul de test
    predicted_lables_test = adaboost_model.predict(dataset_test.inputs)
    test_accuracy = accuracy_score(dataset_test.labels, predicted_lables_test)
    print("Test accuracy: %.4f" % test_accuracy)


print("Decision Tree")
run(DecisionTreeClassifier(max_depth=1))

print("Naive Bayes")
run(GaussianNB())

# Rezultate obtinute (cu 40 de clasificatori slabi):
    # Decision Tree (adaugand max_depth=1)
        # Train accuracy: 0.7051
        # Test accuracy: 0.7187
    # Naive Bayes
        # Train accuracy: 0.6567
        # Test accuracy: 0.6608