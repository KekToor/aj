import glob
import random

import cv2 as cv
from sklearn.neural_network import MLPClassifier


def load_ds(path: str) -> list[tuple[str, int]]:
    bikes = [(f, 0) for f in glob.glob(f'{path}/Bike/*')]
    cars = [(f, 1) for f in glob.glob(f'{path}/Car/*')]
    result = bikes + cars
    random.shuffle(result)
    return result


def load_img(path: str):
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return cv.resize(img, (96,96))


def create_hog_descriptor() -> cv.HOGDescriptor:
    win_size = 96, 96
    block_size = 32, 32
    block_stride = 16, 16
    cell_size = 8, 8
    nbins = 9
    deriv_aperture = 1
    win_sigma = -1
    histogram_norm_type = 0
    l2_hys_threshold = 0.2
    gamma_correction = 1
    nlevels = 64
    signed_gradients = True

    return cv.HOGDescriptor(
        win_size,
        block_size,
        block_stride,
        cell_size,
        nbins,
        deriv_aperture,
        win_sigma,
        histogram_norm_type,
        l2_hys_threshold,
        gamma_correction,
        nlevels,
        signed_gradients
    )


def compute_signals(ds) -> tuple[list, list]:
    hog = create_hog_descriptor()
    signals = []
    labels = []

    for img, label in ds:
        img = load_img(img)
        signals.append(hog.compute(img))
        labels.append(label)

    return signals, labels


def split_ds(ds: list[tuple[str, int]]):
    train, test = [], []

    for item in ds:
        if random.random() < 0.2:
            test.append(item)
        else:
            train.append(item)

    return train, test


def train_classifier(ds, classifier):
    signals, labels = ds
    classifier.fit(signals, labels)


def test_classifier(ds, classifier):
    signals, labels = ds
    preds = classifier.predict(signals)

    correct = 0

    for pred, exp in zip(preds, labels):
        correct += (pred == exp)

    return correct / len(preds)


def test_conf(train, test, classifier_class, params):
    classifier = classifier_class(**params)
    train_classifier(train, classifier)
    accuracy = test_classifier(test, classifier)

    print(f'{classifier_class.__name__}({params}): {accuracy:.3f}')


def test_all_configurations(train, test, classifier_class, params):
    for p in params:
        test_conf(train, test, classifier_class, p)


def test_mlp(train, test):
    params = [
        {'hidden_layer_sizes': (18,), 'solver': 'lbfgs', 'activation': 'relu'},
        {'hidden_layer_sizes': (18,), 'solver': 'lbfgs', 'activation': 'logistic'},
        {'hidden_layer_sizes': (18, 18), 'solver': 'lbfgs', 'activation': 'relu'},
        {'hidden_layer_sizes': (18, 18), 'solver': 'lbfgs', 'activation': 'logistic'},
    ]
    test_all_configurations(train, test, MLPClassifier, params)


def test_all(dataset: str) -> None:
    ds = load_ds(dataset)
    train, test = split_ds(ds)
    train, test = compute_signals(train), compute_signals(test)
    test_mlp(train, test)

