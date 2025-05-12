import os
import random

# Parameters
ARRAY_LEN = 7
ARRAY_MIN = 1
ARRAY_MAX = 20
OUTPUT_TEX = os.path.join(os.path.dirname(__file__), "quicksort_steps.tex")

# Generate a random array
array = random.sample(range(ARRAY_MIN, ARRAY_MAX), ARRAY_LEN)

frames = []
tree_nodes = []  # For recursion tree

# Helper: TikZ for array with borders and color highlights
def tikz_array(arr, pivot_idx=None, left_idxs=None, right_idxs=None, name=None):
    tikz = [r"\begin{tikzpicture}[baseline=(current bounding box.center),every node/.style={font=\small}]",]
    for i, v in enumerate(arr):
        color = "white"
        text = str(v)
        draw_opts = "thick"
        if pivot_idx is not None and i == pivot_idx:
            color = "red!20"
            text = r"\textcolor{red}{%s}" % v
            draw_opts += ",fill=red!20"
        elif left_idxs and i in left_idxs:
            color = "green!20"
            text = r"\textcolor{green!60!black}{%s}" % v
            draw_opts += ",fill=green!20"
        elif right_idxs and i in right_idxs:
            color = "blue!20"
            text = r"\textcolor{blue}{%s}" % v
            draw_opts += ",fill=blue!20"
        tikz.append(rf"  \draw[{draw_opts}] ({i},0) rectangle ++(1,1);")
        tikz.append(rf"  \node at ({i}+0.5,0.5) {{{text}}};")
    if name:
        tikz.append(rf"  \node[below=6pt] at ({len(arr)/2},0) {{{name}}};")
    tikz.append(r"\end{tikzpicture}")
    return "\n".join(tikz)

# Recursion tree node structure
def add_tree_node(tree, arr, parent_idx=None):
    node_id = len(tree)
    tree.append({'id': node_id, 'arr': arr, 'parent': parent_idx})
    return node_id

def tikz_tree(tree, highlight_idx):
    if not tree:
        return ""
    # Build children map
    children = {n['id']: [] for n in tree}
    for n in tree:
        if n['parent'] is not None:
            children[n['parent']].append(n['id'])
    def node_label(n):
        arr_str = ','.join(map(str, n['arr']))
        if n['id'] == highlight_idx:
            return r"[fill=yellow!30,draw=red,thick]{\textbf{%s}}" % arr_str
        else:
            return r"[draw,thick]{%s}" % arr_str
    def build(idx):
        n = tree[idx]
        label = node_label(n)
        if children[idx]:
            return label + " " + " ".join([f"child {{ node{build(c)} }}" for c in children[idx]])
        else:
            return label
    # Root node: must start with \node
    return (
        "\\begin{center}\\begin{tikzpicture}[level distance=1.2cm,sibling distance=2.2cm,every node/.style={font=\\small},grow=down]"
        "\n"
        "\\node" + build(0) + ";"
        "\\end{tikzpicture}\\end{center}"
    )

# Recursive function to generate frames
def quicksort_tex(arr, depth=0, max_depth=3, parent_idx=None, tree=None):
    if tree is None:
        tree = []
    node_idx = add_tree_node(tree, arr, parent_idx)
    if len(arr) == 0:
        return
    if len(arr) == 1:
        frames.append(f"""
% --- Quicksort Base Case ---
\\begin{{frame}}{{Quicksort Base Case}}
{tikz_array(arr)}
\\medskip
Array: {arr[0]} (sorted)
\\end{{frame}}
""")
        return
    if depth > max_depth:
        frames.append(f"""
% --- Quicksort Max Depth ---
\\begin{{frame}}{{Quicksort (Depth Limit)}}
{tikz_array(arr)}
\\medskip
Array: {', '.join(map(str, arr))} (recursion limit)
\\end{{frame}}
""")
        return
    pivot_idx = random.randint(0, len(arr)-1)
    pivot_val = arr[pivot_idx]
    left = [x for x in arr if x < pivot_val]
    right = [x for x in arr if x > pivot_val]
    left_idxs = [i for i, x in enumerate(arr) if x < pivot_val]
    right_idxs = [i for i, x in enumerate(arr) if x > pivot_val]
    # --- Pivot selection slide ---
    frames.append(f"""
% --- Quicksort Pivot Selection ---
\\begin{{frame}}{{Quicksort: Select Pivot}}
{tikz_array(arr, pivot_idx, left_idxs, right_idxs, name="Main Array")}
\\medskip
\\textbf{{Pivot index:}} {pivot_idx}\\
\\textbf{{Pivot value:}} {pivot_val}
\\end{{frame}}
""")
    # --- Partitioning slide: left, pivot, right, one at a time ---
    left_str = ', '.join(map(str, left)) if left else '---'
    right_str = ', '.join(map(str, right)) if right else '---'
    frames.append(f"""
% --- Quicksort Partitioning Step ---
\\begin{{frame}}{{Quicksort: Partitioning}}
{tikz_array(arr, pivot_idx, left_idxs, right_idxs, name="Main Array")}
\\medskip
\\textbf{{Partition:}}\\
\\textcolor{{green!60!black}}{{Left:}} {left_str}
\\pause
\\textcolor{{red}}{{Pivot:}} {pivot_val}
\\pause
\\textcolor{{blue}}{{Right:}} {right_str}
\\end{{frame}}
""")
    # --- Recursion tree slide ---
    frames.append(f"""
% --- Quicksort Recursion Tree ---
\\begin{{frame}}{{Quicksort: Recursion Tree}}
{tikz_tree(tree, node_idx)}
\\end{{frame}}
""")
    if left:
        quicksort_tex(left, depth+1, max_depth, node_idx, tree)
    if right:
        quicksort_tex(right, depth+1, max_depth, node_idx, tree)

# Add a comment at the top for required packages
header = "% This file is auto-generated. Requires: \\usepackage{{tikz}}, \\usetikzlibrary{{trees}}, \\usepackage{{xcolor}} in your main .tex.\n"

# Generate all frames
quicksort_tex(array)

with open(OUTPUT_TEX, "w") as f:
    f.write(header)
    for s in frames:
        f.write(s)

print(f"Generated {OUTPUT_TEX} with deep recursive quicksort visualization and recursion tree.")