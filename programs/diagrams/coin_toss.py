import matplotlib.pyplot as plt
import os

def draw_coin_toss_tree():
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis('off')

    # Nodes positions
    nodes = {
        'start': (0, 0),
        'heads': (-2, -2),
        'tails': (2, -2),
        'red': (-3, -4),
        'blue': (-1, -4)
    }

    # Draw edges
    ax.plot([0, -2], [0, -2], 'k-')  # start to heads
    ax.plot([0, 2], [0, -2], 'k-')   # start to tails
    ax.plot([-2, -3], [-2, -4], 'k-')  # heads to red
    ax.plot([-2, -1], [-2, -4], 'k-')  # heads to blue

    # Draw nodes
    for name, (x, y) in nodes.items():
        ax.plot(x, y, 'o', color='tab:blue' if name == 'start' else 'tab:gray', markersize=12)
        ax.text(x, y+0.3, name.capitalize(), ha='center', fontsize=12)

    # Probabilities
    ax.text(-1, -1, 'p', fontsize=12, color='tab:green')
    ax.text(1, -1, '1-p', fontsize=12, color='tab:green')
    ax.text(-2.7, -3, '1/2', fontsize=12, color='tab:red')
    ax.text(-1.3, -3, '1/2', fontsize=12, color='tab:red')

    # Tie colors
    ax.text(-3, -4.4, 'Red Tie', ha='center', fontsize=12, color='tab:red')
    ax.text(-1, -4.4, 'Blue Tie', ha='center', fontsize=12, color='tab:blue')
    ax.text(2, -2.4, 'No Tie', ha='center', fontsize=12, color='tab:gray')

    ax.set_xlim(-4, 3)
    ax.set_ylim(-5, 1)
    ax.set_title('Coin Toss: Probability Tree', fontsize=14)

    # Save to diagrams folder
    output_dir = os.path.join(os.path.dirname(__file__), '../../presentation/diagrams')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'coin_toss.png'), bbox_inches='tight')
    plt.close()

draw_coin_toss_tree()