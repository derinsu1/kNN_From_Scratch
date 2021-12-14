import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors


def readData(path):
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        data = []
        for row in csv_reader:
            data.append([float(row[1]), float(row[2]), int(row[0])])
        return data


def kNN(k, trainData, testData, bound):
    # bound = True -> Operates for plotting the decision boundaries
    correctEstimates = 0
    estimates = []
    for testRow in testData:
        distances = []
        for trainRow in trainData:
            dist = np.math.sqrt(((trainRow[0] - testRow[0]) ** 2) + ((trainRow[1] - testRow[1]) ** 2))
            distances.append([dist, trainRow[2]])
        distances.sort()
        nearest = distances[:k]  # Get the k nearest points
        votes = [0, 0, 0]
        for c in nearest:
            if c[0] < 0.01:  # Handle points that are at the same spot or too near
                c[0] = 0.01
            votes[c[1] - 1] += 1 / c[0]  # Calculate the votes for every class
        estimatedClass = 1 + votes.index(max(votes))  # Get the class with the highest vote
        if bound:
            estimates.append(estimatedClass)
        elif testRow[2] == estimatedClass:
            correctEstimates += 1
    if not bound:
        print("k =", k, "     ", "{:.2f}".format(correctEstimates / len(testData) * 100), "      ",
              len(testData) - correctEstimates, "/", len(testData))
    else:
        return estimates


def scikitAccuracy(k, trainData, testData):
    X = []
    y = []
    test = []
    testResult = []
    for trainRow in trainData:
        X.append([trainRow[0], trainRow[1]])
        y.append(trainRow[2])
    for testRow in testData:
        test.append([testRow[0], testRow[1]])
        testResult.append(testRow[2])
    clf = neighbors.KNeighborsClassifier(k, weights='distance')
    clf.fit(X, y)
    predictions = clf.predict(test)
    correctEstimates = 0
    for index, t in enumerate(testResult):
        if t == predictions[index]:
            correctEstimates += 1
    print("k =", k, "     ", "{:.2f}".format(correctEstimates / len(testData) * 100), "      ",
          len(testData) - correctEstimates, "/", len(testData))


def decisionBoundaries(k, trainData):
    # Draws the decision boundaries using sci-kit learn's k-NN algorithm
    X = []
    y = []
    x_min = 1000
    x_max = 0
    y_min = 1000
    y_max = 0
    for trainRow in trainData:
        if trainRow[0] < x_min:
            x_min = trainRow[0]
        elif trainRow[0] > x_max:
            x_max = trainRow[0]
        if trainRow[1] < y_min:
            y_min = trainRow[1]
        elif trainRow[1] > y_max:
            y_max = trainRow[1]
        X.append([trainRow[0], trainRow[1]])
        y.append(trainRow[2])

    h = .01  # step size in the mesh
    cmap_light = ListedColormap(['#61ff5e', '#fc6d91', '#2beaff'])
    cmap_bold = ['#3da13b', '#a3465f', '#1c8f9c']
    clf = neighbors.KNeighborsClassifier(k, weights='distance')
    clf.fit(X, y)
    xx, yy = np.meshgrid(np.arange(x_min - 0.5, x_max + 0.5, h), np.arange(y_min - 0.5, y_max + 0.5, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light)
    sns.scatterplot(x=list(zip(*X))[0], y=list(zip(*X))[1], hue=y, palette=cmap_bold, alpha=1.0, edgecolor="black")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Decision Boundaries        k=%i" % k)
    plt.xlabel("x coordinates")
    plt.ylabel("y coordinates")
    plt.show()


def customDecisionBoundaries(k, trainData):
    # Draws the decision boundaries using my own k-NN algorithm
    X = []
    y = []
    x_min = 1000
    x_max = 0
    y_min = 1000
    y_max = 0
    for trainRow in trainData:
        if trainRow[0] < x_min:
            x_min = trainRow[0]
        elif trainRow[0] > x_max:
            x_max = trainRow[0]
        if trainRow[1] < y_min:
            y_min = trainRow[1]
        elif trainRow[1] > y_max:
            y_max = trainRow[1]
        X.append([trainRow[0], trainRow[1]])
        y.append(trainRow[2])

    h = .01  # step size in the mesh
    cmap_light = ListedColormap(['#61ff5e', '#fc6d91', '#2beaff'])
    cmap_bold = ['#3da13b', '#a3465f', '#1c8f9c']
    xx, yy = np.meshgrid(np.arange(x_min - 0.5, x_max + 0.5, h), np.arange(y_min - 0.5, y_max + 0.5, h))
    Z = np.c_[xx.ravel(), yy.ravel()]
    Z = kNN(k, trainData, Z, True)    # Apply the k-NN algorithm to get predictions for all points in the mesh
    Z = np.array(Z).reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light)
    sns.scatterplot(x=list(zip(*X))[0], y=list(zip(*X))[1], hue=y, palette=cmap_bold, alpha=1.0, edgecolor="black")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Decision Boundaries        k=%i" % k)
    plt.xlabel("x coordinates")
    plt.ylabel("y coordinates")
    plt.show()


train = readData("data_training.csv")
test = readData("data_test.csv")
print("        Accuracy (%)   Error Count")
i = 1
while i < 15:
    if i == 13:
        i = 15
    kNN(i, train, test, False)
    i += 2
print()
print("Sci-kit Accuracy (%)   Error Count")
i = 1
while i < 15:
    if i == 13:
        i = 15
    scikitAccuracy(i, train, test)
    i += 2

# Change the k values to test with different values
#decisionBoundaries(25, train)
customDecisionBoundaries(25, train)
