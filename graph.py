import json
import networkx as nx
import matplotlib.pyplot as plt

#reads json of graph json file
f = open('graph.json')

data = json.load(f)
print(data)
print(len(data))

from matplotlib.pyplot import figure, text

g = nx.DiGraph(data)
pos = nx.spring_layout(g)
figure(figsize=(40,7))
d = dict(g.degree)
nx.draw(g, pos=pos, node_color='orange',
        with_labels=False,
        node_size=100,
        arrowsize=5)
for node, (x, y) in pos.items():
    text(x, y, node, fontsize=10, ha='center', va='center')
plt.show()