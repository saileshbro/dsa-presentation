import matplotlib.pyplot as plt
import numpy as np
import os

# Example array and random pivot
arr = [7, 5, 9, 1, 3, 4, 8, 6]
pivot = 5

# Partition the array
left = [x for x in arr if x < pivot]
middle = [x for x in arr if x == pivot]
right = [x for x in arr if x > pivot]

# Plotting
fig, ax = plt.subplots(figsize=(8, 2))

# Draw the array
for i, val in enumerate(arr):
    color = 'tab:blue'
    if val < pivot:
        color = 'tab:green'
    elif val == pivot:
        color = 'tab:red'
    elif val > pivot:
        color = 'tab:orange'
    ax.bar(i, 1, color=color, edgecolor='k')
    ax.text(i, 0.5, str(val), ha='center', va='center', fontsize=14, color='white')

# Annotate pivot
pivot_idx = arr.index(pivot)
ax.annotate('pivot', xy=(pivot_idx, 1), xytext=(pivot_idx, 1.3),
            arrowprops=dict(facecolor='black', shrink=0.05),
            ha='center', fontsize=12)

# Hide axes
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(-0.5, len(arr)-0.5)
ax.set_ylim(0, 1.5)
ax.set_title('Quicksort: Partitioning Step with Random Pivot', fontsize=14)

# Save to diagrams folder
output_dir = os.path.join(os.path.dirname(__file__), '../../presentation/diagrams')
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'quicksort_step.png'), bbox_inches='tight')
plt.close()