import matplotlib.pyplot as plt
import seaborn as sns



def regrets_plot(regrets):
    # Define custom colors for groups of three in different shades
    legend_colors = {
        "ecdf": "#FF69B4",  # pink
        "kde": "#800080",  # purple
        "rde(50x10)": "#ADD8E6",  # light blue
        "rde(50x100)": "#4682B4",  # blue
        "rde(50x1000)": "#00008B",  # dark blue
        "rde(100x10)": "#98FB98",  # light green
        "rde(100x100)": "#32CD32",  # green
        "rde(100x1000)": "#006400",  # dark green
        "rde(1000x10)": "#FFD700",  # light yellow
        "rde(1000x100)": "#FFA500",  # orange
        "rde(1000x1000)": "#FF4500",  # dark orange
    }
    titles = [10, 20, 50, 100]

    # Set up 2x2 subplots
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)

    # Plot overlayed KDEs in each subplot with custom colors
    t = 0
    for i in range(2):
        for j in range(2):
            for k, color in enumerate(legend_colors.values()):
                sns.kdeplot(
                    regrets[11 * t + k],
                    ax=axes[i, j],
                    color=color,
                )
            axes[i, j].set_title(f"n={titles[t]}")
            t += 1

    # Create a shared legend for the entire figure
    fig.legend(
        [key for key in legend_colors.keys()],
        loc="center right",
        bbox_to_anchor=(1.25, 0.5),
        title="Legend",
    )

    # Zoom in
    plt.xlim(0, 0.1)

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()
