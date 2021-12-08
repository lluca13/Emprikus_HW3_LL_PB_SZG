# 1.excercise
# Create an estimation of gender on the test set!
# Count the number of male and female friends of each node, and predict accordingly.


# We have loaded the packages and function which we will need later.
import pandas as pd
from hw_demo_estimation import etl, graph_manipulation as gm, data_viz as dv
from tqdm import tqdm
import networkx as nx
from sklearn.metrics import accuracy_score, confusion_matrix


# We have loaded the data, with the function which also cleans it, and selects relevant profiles only.
# We have created a graph as well from the data.
nodes, edges = etl.load_and_select_profiles_and_edges()
G = gm.create_graph_from_nodes_and_edges(nodes, edges)


# We have splitted the database to a train and test database.
is_train = nodes['TRAIN_TEST'] == 'TRAIN'
nodes_train = nodes[is_train]
is_test = nodes['TRAIN_TEST'] == 'TEST'
nodes_test = nodes[is_test]


# we created graphs from the test and train data base.
G_test = gm.create_graph_from_nodes_and_edges(nodes_test, edges)
G_train = gm.create_graph_from_nodes_and_edges(nodes_train, edges)


# Now we should count the number of friends by gender for each node, but it would be a really computing-intensive task,
# to go through all the 394590 rows in the nodes_train dataset.
# That is why we connected the counting-process directly with the prediction.
# We counted through iteration the neighbors of each node grouped by gender.
# If a node had more female neighbors than male,
# we predicted it's gender female, otherwise male.

pred_gend = []
for userid in tqdm(nodes_test.user_id):
    neighbors_it = set(G_train.neighbors(userid))
    genders = nodes_train[nodes_train.user_id.isin(neighbors_it)].dropna().groupby('gender').count()
    try:
        males = genders.loc[1, 'user_id']
    except:
        males = 0
    try:
        females = genders.loc[0, 'user_id']
    except:
        females = 0

    if males >= females:
        pred_gend.append(1)
    else:
        pred_gend.append(0)

prediction = pd.DataFrame(dict(user_id = nodes_test.user_id, gender = pred_gend))


# We created a function to measure the accuracy with an accuracy score and
# a confusion matrix of any prediction which we can use any time, to see the results at one time.

def accuracy_check(test, pred):
    accuracyscore = accuracy_score(test, pred)
    confusionmatrix = confusion_matrix(test, pred)
    print(f"The confusion matrix is:\n{confusionmatrix}\n\nThe accuracy score is: {accuracyscore}")


# This code below doesn't work, while in the given data the test nodes had no gender values,
# that is why they are not comparable with the results of the prediction:(

y_test = nodes_test.gender.dropna()
y_prediction = prediction.gender
accuracy_check(y_test, y_prediction)
