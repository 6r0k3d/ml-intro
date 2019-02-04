import os
import ember
import argparse
import graphviz
import pickle
import pandas
import matplotlib.pyplot as plt
from sklearn import metrics


# Paradigm
# import
# instantiate
# fit
# predict

def evaluate_threshold(tpr, fpr, thresholds, threshold):
    print('Sensitivity:', tpr[thresholds > threshold][-1])
    print('Specificity:', 1-fpr[thresholds > threshold][-1])
    print()

def main():
    debug = True
    prog = "train_ember"
    descr = "Train an ember model from a directory with raw feature files"
    parser = argparse.ArgumentParser(prog=prog, description=descr)
    parser.add_argument("datadir", metavar="DATADIR", type=str, help="Directory with raw features")
    args = parser.parse_args()

    # If model data doesn't exist yet, create it from raw features
    if not os.path.exists(args.datadir) or not os.path.isdir(args.datadir):
        parser.error("{} is not a directory with raw feature files".format(args.datadir))

    X_train_path = os.path.join(args.datadir, "X_train.dat")
    y_train_path = os.path.join(args.datadir, "y_train.dat")
    if not (os.path.exists(X_train_path) and os.path.exists(y_train_path)):
        print("Creating vectorized features")
        ember.create_vectorized_features(args.datadir)

    # Get training and testing data
    X_train, y_train, X_test, y_test = ember.read_vectorized_features(args.datadir)

    if debug:
        print("X_train shape: ", X_train.shape)
        print("y_train shape: ", y_train.shape)
        print("X_test shape: ", X_test.shape)
        print("y_test shape: ", y_test.shape)

    # Convert memmap to pandas series for metrics
    y_test = pandas.Series(data=y_test)

    # Decision Tree Learner
    from sklearn import tree
    tree_clf = tree.DecisionTreeClassifier(max_depth=5)

    """
    tree_model_path = os.path.join(args.datadir, "tree_model.p")
    # Train model if it doesn't exist
    if not (os.path.exists(tree_model_path)):
        print("Training model")
        tree_clf.fit(X_train, y_train)
        pickle.dump(tree_clf, open("tree_model.p","wb"))

    saved_tree_clf = pickle.load(open("tree_model.p", "rb"))
    """ 

    tree_clf.fit(X_train, y_train)
    tree_dot = tree.export_graphviz(tree_clf, out_file=None)
    graph = graphviz.Source(tree_dot)
    graph.render("tree")

    y_pred = tree_clf.predict(X_test)

    print("\n##### Metrics #####\n")
    print("Accuracy Score")
    print(metrics.accuracy_score(y_test, y_pred), "\n")

    print("Class distribution\n", y_test.value_counts(), "\n")
    print("Average Malware: ", y_test.mean())
    print("Average Benign: ", 1 - y_test.mean())
    print("Null Accuracy: ", max(y_test.mean(), 1 - y_test.mean()), "\n")

    print("Confusion Matrix")
    print("[[TN   FP]\n [FN   TP]]\n")
    confusion = metrics.confusion_matrix(y_test, y_pred)
    print(confusion, "\n")

    TP = confusion[1,1]
    TN = confusion[0,0]
    FP = confusion[0,1]
    FN = confusion[1,0]

    print("Accuracy: how often is the classifier right?")
    print("Accuracy from Confusion Matrix")
    print("(TP + TN) / float(TP + TN + FP + FN)")
    print((TP + TN) / float(TP + TN + FP + FN),"\n")

    print("Classification Error: How often is the classifier wrong?")
    print("(1 - metrics.accuracy_score(y_test, y_pred)")
    print((1 - metrics.accuracy_score(y_test, y_pred)),"\n")

    print("Sensitivity: When the actual value is positive,\n how often is the prediction right?")
    print("Also call 'recall'")
    print("TP / float(TP + FN)")
    print((TP / float(TP + FN)),"\n")
    print(metrics.recall_score(y_test, y_pred))

    print("Specificity: When value is negative, how often is the prediction right?")
    print("TN / float(TN + FP)")
    print(TN / float(TN + FP),"\n")

    print("False Pos. Rate: When the actual value is negative,\n how often is the prediction wrong?")
    print("FP / float(TN + FP)")
    print(FP / float(TN + FP),"\n")

    print("Precision: When a positive value is predicted,\n how often is the prediction right?")
    print("TP / float(TP + FP)")
    print(TP / float(TP + FP),"\n")

    # Classification threshold
    # MUST USE y_pred_prod with the positive class!!!
    y_pred_prob = tree_clf.predict_proba(X_test)[:, 1]
    
    # ROC: Choose a threshold that balances sensitivity and specificity
    # Ideal plot hugs top left of graph: high sensitivity and high specificity
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
    plt.plot(fpr, tpr)
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.title('ROC curve for malware classifier')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.grid(True)
    plt.show()

    evaluate_threshold(tpr, fpr, thresholds, 0.5)

    # AUC: percentage of ROC plot that is under the curve
    # Higher AUC indicates top left graph ROC
    # Useful for imbalanced classes
    print("AUC")
    print(metrics.roc_auc_score(y_test, y_pred_prob),"\n")

if __name__ == "__main__":
    main()
