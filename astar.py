import numpy as np
from PIL import Image
import random
import math


R = 480
C = 844
text = open("Colorado_844x480.dat").read().split("   ")[1:]
data = np.zeros((R, C), dtype=float)
for r in range(R):
    for c in range(C):
        data[r, c] = float(text[r * C + c].strip())
rgb = np.stack([data for _ in range(3)], axis=2)
rgb = np.uint8((rgb - rgb.min()) / rgb.max() * 255)


def heuristic(a, b, modifier=None):
    if modifier == "bfs":
        return 0
    return 5 * abs(data[b] - data[a])


def distance(a, b, modifier=None):
    if modifier == "greedy":
        return 0
    return abs(data[b] - data[a])


def neighbors(point):
    r = point[0]
    c = point[1]
    return [(r - 1, c - 1), (r - 1, c), (r - 1, c + 1), (r, c - 1), (r, c + 1), (r + 1, c - 1), (r + 1, c), (r + 1, c + 1)]


def valid(point):
    return 0 <= point[0] < R and 0 <= point[1] < C


def reconstruct_path(path_dict, best_node):
    path = []
    path.insert(0, best_node)
    while best_node in path_dict:
        best_node = path_dict[best_node]
        path.insert(0, best_node)
    return path


# https://en.wikipedia.org/wiki/A*_search_algorithm#Pseudocode
def astar(start, end, modifier=None):
    open_nodes = []
    open_nodes.append(start)
    path_dict = {}
    g = {}
    g[start] = 0
    f = {}
    f[start] = heuristic(start, end, modifier)
    explored = 0
    while len(open_nodes) > 0:
        best_node = None
        best_val = math.inf
        for node in open_nodes:
            if f.get(node, math.inf) < best_val:
                best_node = node
                best_val = f.get(node, math.inf)
        open_nodes.remove(best_node)
        explored += 1
        if best_node == end:
            return reconstruct_path(path_dict, best_node), explored
        for neighbor in neighbors(best_node):
            if valid(neighbor):
                temp_g = g[best_node] + distance(neighbor, end, modifier)
                if temp_g < g.get(neighbor, math.inf):
                    path_dict[neighbor] = best_node
                    g[neighbor] = temp_g
                    f[neighbor] = g[neighbor] + heuristic(neighbor, end, modifier)
                    if neighbor not in open_nodes:
                        open_nodes.append(neighbor)


def add_path(rgb, path, color="random"):
    for point in path:
        if color == "random":
            rgb[point[0], point[1]] = [random.randint(0, 255) for _ in range(3)]
        if color == "green":
            rgb[point[0], point[1], 1] = 255
        if color == "red":
            rgb[point[0], point[1], 0] = 255
        if color == "blue":
            rgb[point[0], point[1], 2] = 255


def total_dist(path):
    total = 0
    for i in range(len(path) - 1):
        total += distance(path[i], path[i + 1])
    return total


for i in range(10):
    rgb_copy = rgb.copy()
    point = (random.randint(50, 150), random.randint(50, 150))
    astar_path, astar_explored = astar((0, 0), point)
    add_path(rgb_copy, astar_path, "red")
    bfs_path, bfs_explored = astar((0, 0), point, "bfs")
    add_path(rgb_copy, bfs_path, "blue")
    greedy_path, greedy_explored = astar((0, 0), point, "greedy")
    add_path(rgb_copy, greedy_path, "green")
    print("POINT %d - (%d, %d)" % (i + 1, point[0], point[1]))
    print("TILES EXPLORED: A* - %d, bfs - %d, greedy search - %d" % (astar_explored, bfs_explored, greedy_explored))
    print("TOTAL ELEVATION CHANGE: A* - %d, bfs - %d, greedy search - %d" % (total_dist(astar_path), total_dist(bfs_path), total_dist(greedy_path)))
    print("PATH LENGTH: A* - %d, bfs - %d, greedy search - %d" % (len(astar_path), len(bfs_path), len(greedy_path)))

    Image.fromarray(rgb_copy).show()