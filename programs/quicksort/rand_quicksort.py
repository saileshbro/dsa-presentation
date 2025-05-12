import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
np.random.seed(42)
class RecNode:
    def __init__(self, l, r, pivot_idx, pivot_val, depth, subarray, orig_pivot_idx):
        self.l = l
        self.r = r
        self.pivot_idx = pivot_idx
        self.pivot_val = pivot_val
        self.depth = depth
        self.subarray = subarray  # snapshot at this call
        self.orig_pivot_idx = orig_pivot_idx  # index in the original array
        self.children = []

class RandomizedQuicksortVisualizer:
    def __init__(self, arr):
        self.original = copy.deepcopy(arr)
        self.arr = copy.deepcopy(arr)
        self.recursion_tree = None
        self.step = 0  # For saving images

    def choose_pivot(self, l, r):
        return random.randint(l, r)

    def partition(self, l, r, p_idx):
        A = self.arr
        A[l], A[p_idx] = A[p_idx], A[l]
        pivot = A[l]
        i = l + 1
        for j in range(l + 1, r + 1):
            if A[j] <= pivot:
                A[i], A[j] = A[j], A[i]
                i += 1
        A[l], A[i - 1] = A[i - 1], A[l]
        return i - 1

    def find_original_index(self, l, r, pivot_val, subarray):
        # Find the index of the pivot value in the original array, within the subarray bounds
        for i in range(l, r + 1):
            if self.original[i] == pivot_val:
                return i
        # If not found (shouldn't happen), fallback to -1
        return -1

    def print_tree(self, node=None, indent="", is_left=True):
        if node is None:
            node = self.recursion_tree
        if node is None:
            return
        prefix = indent + ("├── " if is_left else "└── ")
        if node.pivot_val is not None:
            print(f"{prefix}[{node.l},{node.r}] pivot={node.pivot_val} arr={node.subarray}")
        else:
            print(f"{prefix}[{node.l},{node.r}] arr={node.subarray}")
        for i, child in enumerate(node.children):
            self.print_tree(child, indent + ("│   " if is_left else "    "), i < len(node.children) - 1)

    def visualize_tree(self):
        if self.recursion_tree is None:
            return
        G = nx.DiGraph()
        labels = {}
        def add_edges(node, parent_id=None):
            node_id = id(node)
            G.add_node(node_id)  # Ensure every node is added!
            label = f"[{node.l},{node.r}]\narr={node.subarray}"
            if node.pivot_val is not None:
                label += f"\npivot={node.pivot_val}"
            labels[node_id] = label
            if parent_id is not None:
                G.add_edge(parent_id, node_id)
            for child in node.children:
                add_edges(child, node_id)
        add_edges(self.recursion_tree)
        pos = graphviz_layout(G, prog="dot")
        plt.figure(figsize=(10, 6))
        nx.draw(G, pos, with_labels=False, arrows=False, node_size=2000, node_color="#e0e0e0")
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        plt.title(f"Quicksort Recursion Tree (Step {self.step})")
        plt.savefig(f"step_{self.step}.png")
        plt.close()
        self.step += 1

    def quicksort(self, l, r, depth=0, parent=None):
        subarray_snapshot = self.arr[l:r+1] if l <= r else []
        pivot_idx = None
        pivot_val = None
        orig_pivot_idx = None
        if r - l + 1 <= 1:
            node = RecNode(l, r, pivot_idx, pivot_val, depth, subarray_snapshot, orig_pivot_idx)
            if parent is not None:
                parent.children.append(node)
            else:
                self.recursion_tree = node
            self.print_tree()
            self.visualize_tree()
            return
        p_idx = self.choose_pivot(l, r)
        pivot_val = self.arr[p_idx]
        orig_pivot_idx = self.find_original_index(l, r, pivot_val, subarray_snapshot)
        new_pivot_idx = self.partition(l, r, p_idx)
        node = RecNode(l, r, new_pivot_idx, pivot_val, depth, subarray_snapshot, orig_pivot_idx)
        if parent is not None:
            parent.children.append(node)
        else:
            self.recursion_tree = node
        self.print_tree()
        self.visualize_tree()
        self.quicksort(l, new_pivot_idx - 1, depth + 1, node)
        self.quicksort(new_pivot_idx + 1, r, depth + 1, node)



if __name__ == "__main__":
    arr = [15, 3, 1, 10, 9, 0, 6, 4]
    # arr = np.random.randint(-10, 10, 16)
    print("Original array:", arr)
    visualizer = RandomizedQuicksortVisualizer(arr)
    visualizer.quicksort(0, len(arr) - 1)
    print("Sorted array:", visualizer.arr)