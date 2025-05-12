import matplotlib.pyplot as plt
import numpy as np
import os

# Example array and indices to compare
arr = [1, 3, 4, 5, 7, 8, 9]
i, j = 1, 5  # Compare elements at index 1 and 5 (3 and 8)

fig, ax = plt.subplots(figsize=(8, 2))

# Draw the array
for idx, val in enumerate(arr):
    color = 'tab:blue'
    if idx == i or idx == j:
        color = 'tab:red'
    ax.bar(idx, 1, color=color, edgecolor='k')
    ax.text(idx, 0.5, str(val), ha='center', va='center', fontsize=14, color='white')

# Annotate compared elements
ax.annotate('i', xy=(i, 1), xytext=(i, 1.3), arrowprops=dict(facecolor='black', shrink=0.05), ha='center', fontsize=12)
ax.annotate('j', xy=(j, 1), xytext=(j, 1.3), arrowprops=dict(facecolor='black', shrink=0.05), ha='center', fontsize=12)

# Annotate probability
mid = (i + j) / 2
ax.text(mid, 1.1, r'$Pr[R_{ij}] = \frac{2}{j-i+1}$', ha='center', va='bottom', fontsize=14, color='black')

# Hide axes
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(-0.5, len(arr)-0.5)
ax.set_ylim(0, 1.5)
ax.set_title('Probability of Comparison in Quicksort', fontsize=14)

# Save to diagrams folder
output_dir = os.path.join(os.path.dirname(__file__), '../../presentation/diagrams')
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'comparison_prob.png'), bbox_inches='tight')
plt.close()