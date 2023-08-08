from himpurity.classes import ClassHierarchy
from himpurity.criterion import MHICriterion

from sklearn.tree import DecisionTreeClassifier

from himpurity.criterion import ClassifierCriterionBinder

import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

from himpurity.tests.utils import make_digits_dataset


X, y = make_digits_dataset(
    targets=[1, 7, 3, 8, 9],
    as_str=True, )

random_state = 42


class EquidistantDataset(object):
    def __init__(self, n_classes=3, random_state=42):
        self.n_classes = n_classes
        self.random_state = random_state

    def _initialise(self):
        if not hasattr(self, 'has_been_initialised'):

            # TODO come up with general formula
            if self.n_classes == 2:
                self.centers = np.array([[-0.5], [0.5]])
                self.n_features = self.n_classes - 1
            elif self.n_classes == 3:
                self.centers = np.array([
                    [0, 0.5/np.cos(np.radians(30))],
                    [-0.5, -0.5*np.tan(np.radians(30))],
                    [+0.5, -0.5*np.tan(np.radians(30))]])
                np.random.seed(self.random_state)
                self.rotation_angle = np.random.rand(1)
                self.rotation = np.array([[np.cos(self.rotation_angle), -np.sin(self.rotation_angle)],
                                          [np.sin(self.rotation_angle),  np.cos(self.rotation_angle)]
                                         ])
                self.centers = np.inner(self.rotation.reshape(2, 2), self.centers).T
                self.n_features = self.n_classes - 1
            else:
                self.centers = np.eye(self.n_classes)[np.arange(self.n_classes)]
                self.n_features = self.n_classes
            self.cluster_std = 0.5
            self.sample_iteration = 0
            self.has_been_initialised = True

    def sample(self, n_samples):
        self._initialise()
        self.sample_iteration += 1
        return make_blobs(n_samples=n_samples, n_features=self.n_features,
                          centers=self.centers, cluster_std=self.cluster_std,
                          shuffle=True,
                          random_state=self.random_state + self.sample_iteration,
                          return_centers=False)

n_classes = 7

generator = EquidistantDataset(n_classes=n_classes)
X_train, y_train = generator.sample(1000)


ROOT = "R"
hierarchy_dict = {
    "R": ["A", "B"],
    "A": [1, 2],
    "B": ["C", "D"],
    "C": [3, 4],
    "D": [5, "E"],
    "E": [6, 7]
}
ch_hierarchy = ClassHierarchy(hierarchy_dict, root=ROOT)
global_params = {"max_depth": 3, "random_state": random_state}

clf = ClassifierCriterionBinder(DecisionTreeClassifier, MHICriterion,
                                ch_hierarchy, clf_params=global_params)

y_train = y_train.astype(str)
clf.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

X_test, y_test = generator.sample(1000)
y_test = y_test.astype(str)
predictions = clf.predict(X_test)

cm = confusion_matrix(y_test, predictions)

disp = ConfusionMatrixDisplay(cm)
disp.plot()
