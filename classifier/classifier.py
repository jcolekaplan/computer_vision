"""
    Jacob Kaplan
    classifier.py

    Takes a background name and a command (train or test)
    If train, takes the descriptor vectors of all the images and trains a
        classifier for that background
    If test, takes the classifier for that background and goes through the test
        directory for that background and output stats about its accuracy


    o   o
                  /^^^^^7
    '  '     ,oO))))))))Oo,
           ,'))))))))))))))), /{
      '  ,'o  ))))))))))))))))={
         >    ))))))))))))))))={
         `,   ))))))\ \)))))))={
           ',))))))))\/)))))' \{
             '*O))))))))O*'

    You're trying to classify pictures of the ocean? I could have done that!
"""

import sys
import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from p1_descriptor import find_images, get_sub_imgs, get_desc_vec

def train_classifer(desc_vecs, back_name):
    """
    Takes a set of description vectors and a background name
    Creates a LinearSVC classifier for that background with its unique C value
    Fits the data of all description vectors to the classifier
    """
    back_names = ["grass", "ocean", "redcarpet", "road", "wheatfield"]
    num_vecs = len(desc_vecs)
    C_vals = [7, 4, 1, 1, 2]
    nb = num_vecs // 5
    labels = np.hstack((np.full((nb), 1), np.full((nb), 2), np.full((nb), 3), \
                        np.full((nb), 4), np.full((nb), 5)))
    ind = back_names.index(back_name)
    c = C_vals[ind]
    classifier = LinearSVC(C = c, tol = 1e-5)
    classifier.fit(desc_vecs, labels)
    return classifier

def test_classifer(test_vecs, clf, back_name):
    """
    Takes in a set of decription vectors, a LinearSVC classifier, and a
        background name
    Go through each description vector:
        Output stats every time its comparing a new class of images
        Add the stats to this classifier's row in the confusion matrix
        Get the classifier's prediction of what the vector is describing
        If prediction is true then increment number of correct predictions
    Return the classifier's row in the confusion matrix

    E.g. it might return something like this, where each number is the
        percentage of the images the ocean classifier thought were ocean:
            Road Ocean Redcarpet Wheatfield Grass
    Ocean      3    77         1         15    21
    """
    num_vecs = len(test_vecs)
    nb = num_vecs // 5
    back_names = ["road", "ocean", "redcarpet", "wheatfield", "grass"]
    target = back_names.index(back_name) + 1

    predict_count = 0
    ind = 0
    cnf_row = []
    for i in range(num_vecs):
        if (i != 0 and ((i+1) // nb == (i+1) / nb)):
            print("Testing: {}".format(back_names[ind]))
            print("{}% match as {}".format((100*predict_count/nb), back_name))
            cnf_row.append(100*predict_count/nb)
            predict_count = 0
            ind += 1

        test_vec = test_vecs[i]
        prediction = clf.predict(np.asarray(test_vec).reshape(1,-1))
        if prediction == target:
            predict_count += 1
    return cnf_row

def plot_confusion_matrix(cm, classes, title="Confusion Matrix",
                        normalize=False,cmap=plt.cm.Blues):
    """
    Scikit learn's fancy-pants confusion matrix plotter:
    scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

if __name__ == "__main__":
    """
    Handle command line arguments
    """
    if len(sys.argv) != 4:
        print("Usage: {0} dir train background\nUsage: {0} dir test test_dir" \
            .format(sys.argv[0]))
        sys.exit()
    else:
        dir_name = sys.argv[1].lower()
        command = sys.argv[2].lower()

    """
    Load descriptors
    """
    descriptor_vecs = list()
    try:
        infile = open("{}/desc.txt".format(dir_name), "rb")
        descriptor_vecs = pickle.load(infile)
        infile.close()
    except FileNotFoundError:
        print("{} not found!".format(dir_name))
        sys.exit()

    """
    Check for valid background and command
    """
    if command != "train" and command != "test":
        print("Command must be train or test!")
        sys.exit()

    backgrounds = ["road", "ocean", "redcarpet", "wheatfield", "grass"]
    if command == "train":
        background_name = sys.argv[3].lower()
        if background_name not in backgrounds:
            print("Background must be grass, ocean, redcarpet, road, or wheatfield")
            sys.exit()

    """
    Train classifier, pickle it
    """
    if command == "train":
        clf = train_classifer(descriptor_vecs, background_name)
        outfile = open("{}/{}_clf.pkl".format(dir_name, background_name), "wb")
        pickle.dump(clf, outfile)
        outfile.close()

    """
    Test all five classifiers
    Read in each classifier
    For each classifier:
        Unpickle it
        Run test_classifier and get its unique row in confusion matrix
    Build the confusion matrix and plot it
    """
    if command == "test":

        test_dir_name = sys.argv[3].lower()
        try:
            infile = open("{}/desc.txt".format(test_dir_name), "rb")
            testing_vecs = pickle.load(infile)
            infile.close()
        except FileNotFoundError:
            print("{} not found!".format(test_dir_name))
            sys.exit()

        cnf = []
        for i in range(5):
            print("Running {} classifer ===================================="\
                    .format(backgrounds[i]))
            clf_infile = open("{}/{}_clf.pkl".format(dir_name, backgrounds[i]), "rb")
            clf = pickle.load(clf_infile)
            clf_infile.close()
            row = test_classifer(testing_vecs, clf, backgrounds[i])
            cnf.append(row)


        cnf_matrix = np.asarray(cnf)
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes= backgrounds,
            title='Confusion matrix')
        plt.show()
