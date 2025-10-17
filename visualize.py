

import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import json

# --- Define nodes from your docker-compose excerpt ---
nodes = [
    # neighborhood: docker
    {"name":"nodeX","id":"X","ip":"192.168.1.199","neigh":"docker"},
    {"name":"nodeY","id":"Y","ip":"192.168.1.200","neigh":"docker"},
    {"name":"nodeZ","id":"Z","ip":"192.168.1.201","neigh":"docker"},
    {"name":"nodeW","id":"W","ip":"192.168.1.202","neigh":"docker"},
    # neighborhood: docker2
    {"name":"nodeP","id":"P","ip":"192.168.1.204","neigh":"docker2"},
    {"name":"nodeQ","id":"Q","ip":"192.168.1.205","neigh":"docker2"},
    {"name":"nodeR","id":"R","ip":"192.168.1.206","neigh":"docker2"},
    {"name":"nodeL","id":"L","ip":"192.168.1.207","neigh":"docker2"},
    # neighborhood: docker3
    {"name":"nodeA","id":"A","ip":"192.168.1.208","neigh":"docker3"},
    {"name":"nodeB","id":"B","ip":"192.168.1.209","neigh":"docker3"},
    {"name":"nodeC","id":"C","ip":"192.168.1.210","neigh":"docker3"},
    {"name":"nodeD","id":"D","ip":"192.168.1.211","neigh":"docker3"},
    # neighborhood: docker4
    {"name":"nodeE","id":"E","ip":"192.168.1.212","neigh":"docker4"},
    {"name":"nodeF","id":"F","ip":"192.168.1.213","neigh":"docker4"},
    {"name":"nodeG","id":"G","ip":"192.168.1.214","neigh":"docker4"},
    {"name":"nodeH","id":"H","ip":"192.168.1.215","neigh":"docker4"},
    # coordinator
    {"name":"coordinator","id":"coordinator","ip":"192.168.1.203","neigh":"coordinator"},
]

# Group by neighborhood
from collections import defaultdict
groups = defaultdict(list)
for n in nodes:
    groups[n["neigh"]].append(n)

# Choose leaders (first listed per neighborhood except coordinator)
leaders = {}
for neigh, lst in groups.items():
    if neigh == "coordinator":
        continue
    leaders[neigh] = lst[0]["name"]  # deterministic choice

# Build graph
G = nx.Graph()
for n in nodes:
    label = f'{n["id"]}\n{n["ip"]}'
    G.add_node(n["name"], label=label, neigh=n["neigh"])

# Intra-neighborhood ring edges
for neigh, lst in groups.items():
    if neigh == "coordinator":
        continue
    order = [n["name"] for n in lst]
    # ring
    for i in range(len(order)):
        a = order[i]
        b = order[(i+1) % len(order)]
        G.add_edge(a, b, kind="ring")

# Coordinator star edges (to leaders only)
for neigh, leader in leaders.items():
    G.add_edge("coordinator", leader, kind="star")

# Layout: place neighborhoods in quadrants; coordinator at center
pos = {}
# set manual cluster anchors
cluster_centers = {
    "docker": (-1.5,  1.0),
    "docker2": ( 1.5,  1.0),
    "docker3": (-1.5, -1.0),
    "docker4": ( 1.5, -1.0),
    "coordinator": (0.0, 0.0),
}

import math

for neigh, lst in groups.items():
    if neigh == "coordinator":
        pos["coordinator"] = cluster_centers["coordinator"]
        continue
    cx, cy = cluster_centers[neigh]
    r = 0.6
    k = len(lst)
    for i, n in enumerate(lst):
        angle = 2*math.pi*i/k
        pos[n["name"]] = (cx + r*math.cos(angle), cy + r*math.sin(angle))

# Draw
plt.figure(figsize=(10, 8))

# Draw intra-neighborhood rings
ring_edges = [(u,v) for u,v,d in G.edges(data=True) if d.get("kind")=="ring"]
nx.draw_networkx_edges(G, pos, edgelist=ring_edges, width=1.5)

# Draw coordinator star edges
star_edges = [(u,v) for u,v,d in G.edges(data=True) if d.get("kind")=="star"]
nx.draw_networkx_edges(G, pos, edgelist=star_edges, style="dashed", width=1.5)

# Node sets
coord_nodes = ["coordinator"]
neigh_nodes = [n["name"] for n in nodes if n["neigh"]!="coordinator"]

nx.draw_networkx_nodes(G, pos, nodelist=coord_nodes, node_size=1200)
nx.draw_networkx_nodes(G, pos, nodelist=neigh_nodes, node_size=1000)

labels = {n["name"]: G.nodes[n["name"]]["label"] for n in nodes}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

for neigh, (cx, cy) in cluster_centers.items():
    if neigh == "coordinator":
        continue
    plt.text(cx, cy+1.0, neigh, ha="center", va="center")

plt.axis("off")
png_path = "/mnt/data/docker_topology.png"
plt.tight_layout()
plt.savefig(png_path, dpi=200, bbox_inches="tight")

dot_lines = []
dot_lines.append("digraph FederatedDocker {")
dot_lines.append('  rankdir=LR;')
dot_lines.append('  node [shape=box];')

# subgraphs
cluster_id = 0
for neigh, lst in groups.items():
    if neigh == "coordinator":
        continue
    dot_lines.append(f'  subgraph cluster_{cluster_id} {{')
    dot_lines.append(f'    label="{neigh}";')
    for n in lst:
        dot_lines.append(f'    {n["name"]} [label="{n["id"]}\\n{n["ip"]}"];')
    # ring edges
    order = [n["name"] for n in lst]
    for i in range(len(order)):
        a = order[i]; b = order[(i+1)%len(order)]
        dot_lines.append(f'    {a} -> {b} [dir=both];')
    dot_lines.append('  }')
    cluster_id += 1

# coordinator and star edges
dot_lines.append('  coordinator [shape=ellipse, label="coordinator\\n192.168.1.203"];')
for neigh, leader in leaders.items():
    dot_lines.append(f'  coordinator -> {leader} [style=dashed];')

dot_lines.append("}")
dot_path = ""
Path(dot_path).write_text("\n".join(dot_lines), encoding="utf-8")
