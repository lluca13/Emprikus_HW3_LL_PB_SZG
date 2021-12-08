# 2.exercise
# Plotting

# Packages we might need
import pandas as pd
from hw_demo_estimation import etl, graph_manipulation as gm, data_viz as dv
from tqdm import tqdm
import networkx as nx
from sklearn.metrics import accuracy_score, confusion_matrix

# Packages for plotting
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

# We have loaded the data, with the function which also cleans it, and selects relevant profiles only.
nodes, edges = etl.load_and_select_profiles_and_edges()
# we've created a graph as well from the data.
G = gm.create_graph_from_nodes_and_edges(nodes, edges)

# Degree distribution plot
dv.plot_degree_distribution(G)

# This log-log plot has been created to show the mutual distribution of the graph, itself.
# According to our studies, this degree distribution is not scale-free, because after a while it breaks down.
# A log-log plot has two dimensions, and both of the axes show the logarithmic value of the two variables.


# Age distribution plot
# descriptive analytics of the nodes
# age distribution by gender
dv.plot_age_distribution_by_gender(nodes)

# This frequency histogram shows us the age distribution of the graph.
# According to the histogram, a huge number of young people (aged between 20 and 25) are in the dataset. The main reason is that this dataset is from an online platform, which is nowadays used mainly by the younger generation.
# After the age of 25, the histogram shows us a huge decline. A people are getting older, the less the chances are that they use online networks to keep contact with each other.
# The distribution by sex indicates that there are more male users in the network than female ones. However, after the age of 40 there are more and more female users.
# The maximum age of the dataset is 55. This is just only a tiny proportion of this huge dataset.


# 3. plot
dv.plot_node_degree_by_gender(nodes, G)

# This line chart above shows interesting pieces of information about the degree of both female and male users.
# The chart uses the number of the genders, such as 0 is for women and 1 for men.
# The line chart achieve its peak values at the age of nearly 18-20 among women and about 20-22 among male users.
# After these peaks, a huge shrinkage brings into being. After the age of 35, the are still more women than man in the graph.
# The small number of low values is due to the fact that there are fewer people, who are still part of the online world.

# We need the same for neighbor connectivity, triadic closure (local clustering coefficient)
# we need to add these attributes to the nodes dataframe so we can plot them

# adding two more attributes to the nodes dataframe
nodes = nodes.assign(connectivity=nodes.user_id.map(nx.average_neighbor_degree(G)))
nodes = nodes.assign(tri_closure=nodes.user_id.map(nx.clustering(G)))


# We have added to the function of plotting these attributes as a new parameter
def plot_node_stats_by_gender(nodes, G, stats):
    """Plot the average of node degree across age and gender"""
    # TODO: this could be generalized for different node level statistics as well!

    nodes_w_degree = nodes.set_index("user_id").merge(
        pd.Series(dict(G.degree)).to_frame(),
        how="left",
        left_index=True,
        right_index=True,
    )
    nodes_w_degree = nodes_w_degree.rename({0: "degree"}, axis=1)
    plot_df = (
        nodes_w_degree.groupby(["AGE", "gender"]).agg({"degree": "mean"}).reset_index()
    )

    if stats == "degree":
        sns.lineplot(data=plot_df, x="AGE", y="degree", hue="gender")
    elif stats == "connectivity":
        sns.lineplot(data=nodes, x="AGE", y="connectivity", hue="gender")
    else:
        sns.lineplot(data=nodes, x="AGE", y="tri_closure", hue="gender")


# 3.2 plot
plot_node_stats_by_gender(nodes, G, 'connectivity')

# For the connectivity chart a peak around between 20 and 25 can also been seen, likewise the graph before. After this age, a
# decline is also visible, meaning people tend to speak less as they age. However, a second bump is also noticable between 43-46.
# According to the chart male users have higher connectivity, than women.


# 3.3 plot
plot_node_stats_by_gender(nodes, G, 'tri_closure')

# This chart provides us information about the triadic closure, the local clustering coefficient.
# The clustering coefficient both for female and male users declines until the age of 40.
# After this decline, huge clustering coefficients can be seen for both sexes.
# Young people using social media platforms usually extend their acquaintanceship.
# Elder people have less but stable connection. As a result of this fact, higher clustering coefficients can be obtained at this age by the data.


# 4. plot
edges_w_features = gm.add_node_features_to_edges(nodes, edges)
dv.plot_age_relations_heatmap(edges_w_features)


# We have modified the underlying function to separate genders and to try out without logging, and normalizing as well
def plot_age_relations_heatmap_methods(edges_w_features, method):
    """Plot a heatmap that represents the distribution of edges"""
    # TODO: check what happpens without logging
    # TODO: instead of logging check what happens if you normalize with the row sum
    #  make sure you figure out an interpretation of that as well!
    # TODO: separate these charts by gender as well
    # TODO: column names could be nicer
    plot_df = edges_w_features.groupby(["gender_x", "gender_y", "AGE_x", "AGE_y"]).agg(
        {"smaller_id": "count"}
    )
    plot_df_w_w = plot_df.loc[(0, 0)].reset_index()
    plot_df_heatmap = plot_df_w_w.pivot_table(
        index="AGE_x", columns="AGE_y", values="smaller_id"
    ).fillna(0)
    plot_df_heatmap_logged = np.log(plot_df_heatmap + 1)
    plot_df_heatmap_normed = np.linalg.norm(plot_df_heatmap + 1, ord='fro')

    if method == "log":
        sns.heatmap(plot_df_heatmap_logged)
    elif method == "norm":
        sns.heatmap(plot_df_heatmap_normed)
    else:
        sns.heatmap(plot_df_heatmap)


# 4.1. plot
# We can show the logged, the normed, and the basic (no-logging) heatmap. Unfortunately the normed plot is not working.
# Basic version:
plot_age_relations_heatmap_methods(edges_w_features, "basic")

# 4.2. plot
plot_age_relations_heatmap_methods(edges_w_features, "log")


# This logged version of the heat map provides a lot of valuable information about people's life.
# The x axis shows the age of men, and the y axis shows the age of women. This data corresponds to gender and age homophily.
# At the age of 25, the most striking feature is that contact with younger people is more common than with older people.
# Here, in our opinion, people are in the phase of getting to know each other.
# In this section of life, men are older than women. Then, after the period, this trend becomes less and less apparent.
# Presumably, after the dating period, people will be immersed in everyday life. Friendship among men and women after a while ceases to exist.
# At the age of 49, as it can be seen from the heat map, we get an exceptionally high value.


#function for 5.1 plot
#MM version
def plot_age_relations_heatmap_methods_MM(edges_w_features, method):
    """Plot a heatmap that represents the distribution of edges"""
    # TODO: check what happpens without logging
    # TODO: instead of logging check what happens if you normalize with the row sum
    #  make sure you figure out an interpretation of that as well!
    # TODO: separate these charts by gender as well
    # TODO: column names could be nicer
    plot_df = edges_w_features.groupby(["gender_y", "gender_y", "AGE_x", "AGE_y"]).agg(
        {"smaller_id": "count"}
    )
    plot_df_w_w = plot_df.loc[(0, 0)].reset_index()
    plot_df_heatmap = plot_df_w_w.pivot_table(
        index="AGE_x", columns="AGE_y", values="smaller_id"
    ).fillna(0)
    plot_df_heatmap_logged = np.log(plot_df_heatmap + 1)
    plot_df_heatmap_normed = np.linalg.norm(plot_df_heatmap + 1, ord='fro')
    
    if method == "log":
        sns.heatmap(plot_df_heatmap_logged)
    elif method == "norm":
        sns.heatmap(plot_df_heatmap_normed)
    else:
        sns.heatmap(plot_df_heatmap)


#5.1 plot
plot_age_relations_heatmap_methods_MM(edges_w_features, "basic")

#Log version
plot_age_relations_heatmap_methods_MM(edges_w_features, "log")


# On this logged version of the heat map both the axes belong to male users. In this section we wanted to get an answer how men keep contact with each other, which age groups "belong together".
# A diagonal can be seen in the middle of the heat map. It means, people in the same age group communicate with each other the most. Looking at the area next to the diagonal, it is clear that the young age (up to 30 years) group communicates nearly in the same way with friends who are a few years older. This means, that men favour more bigger social connections. Also in the previous graph, at the age of 49, we get an exceptionally high value.


#function for 5.2 plot
#FF version
def plot_age_relations_heatmap_methods_FF(edges_w_features, method):
    """Plot a heatmap that represents the distribution of edges"""
    # TODO: check what happpens without logging
    # TODO: instead of logging check what happens if you normalize with the row sum
    #  make sure you figure out an interpretation of that as well!
    # TODO: separate these charts by gender as well
    # TODO: column names could be nicer
    plot_df = edges_w_features.groupby(["gender_x", "gender_x", "AGE_x", "AGE_y"]).agg(
        {"smaller_id": "count"}
    )
    plot_df_w_w = plot_df.loc[(0, 0)].reset_index()
    plot_df_heatmap = plot_df_w_w.pivot_table(
        index="AGE_x", columns="AGE_y", values="smaller_id"
    ).fillna(0)
    plot_df_heatmap_logged = np.log(plot_df_heatmap + 1)
    plot_df_heatmap_normed = np.linalg.norm(plot_df_heatmap + 1, ord='fro')
    
    if method == "log":
        sns.heatmap(plot_df_heatmap_logged)
    elif method == "norm":
        sns.heatmap(plot_df_heatmap_normed)
    else:
        sns.heatmap(plot_df_heatmap)

#5.2 plot
plot_age_relations_heatmap_methods_FF(edges_w_features, "basic")

#Log version
plot_age_relations_heatmap_methods_XX(edges_w_features, "log")


# On this logged version of the heat map both the axes belong to female users. 
# In general, most friendships are made at a younger age, which is true for both genders. People in the same age group communicate with each other the most. Looking at the area next to the diagonal, it is clear that the young age (up to 30 years) group communicates nearly in the same way with friends who are a few years older.
