import pandas as pd
import graphviz
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.externals.six import StringIO
import pydotplus

train = pd.read_csv('train.csv', sep=',')
test = pd.read_csv('test.csv', sep=',')

train_X = pd.DataFrame(train, columns=train.columns[:-1])
train_Y = train['Class']

test_X = pd.DataFrame(test, columns=test.columns[:-1])
test_Y = test['Class']

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_X, train_Y)


def show_plot():
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=train.columns[:-1],
                                    class_names=[str(x)
                                                 for x in [1, 2, 3, 4, 5, 6, 7]],
                                    filled=True, rounded=True)
    graph = graphviz.Source(dot_data)
    graph.view()

def wlasne_drzewo():
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("my-tree.pdf")


#clf = tree.DecisionTreeClassifier()
#clf = clf.fit(train_X, train_Y)
#show_plot()

#clf = tree.DecisionTreeClassifier(max_depth=4)
#clf = clf.fit(train_X, train_Y)
#show_plot()

#clf = tree.DecisionTreeClassifier(max_leaf_nodes=3)
#clf = clf.fit(train_X, train_Y)
#show_plot()

#clf = tree.DecisionTreeClassifier(min_impurity_split=0.21)
#clf = clf.fit(train_X, train_Y)
#show_plot()

#clf = tree.DecisionTreeClassifier(presort=True)
#clf = clf.fit(train_X, train_Y)
#show_plot()

clf = tree.DecisionTreeClassifier(presort=True, max_depth=8)
clf = clf.fit(train_X, train_Y)
y=clf.predict(train_X)
wlasne_drzewo()


print('TRAIN SET')
print(confusion_matrix(clf.predict(train_X), train_Y))
print(sum(clf.predict(train_X) == train_Y) / len(train_X))
print('TEST SET')
print(confusion_matrix(clf.predict(test_X), test_Y))
print(sum(clf.predict(test_X) == test_Y) / len(test_X))
