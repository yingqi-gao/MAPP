import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# Define custom colors for groups of three in different shades
legend_dicts = {
    "ecdf": ("black", ":"),
    "kde": ("red", "-"),
    "rde(50x10)": ("purple", "--"),
    "rde(50x200)": ("blue", "-."),
    "rde(200x200)": ("green", (0, (5, 1, 1, 1, 1, 1, 1, 1))),
}

def regrets_plot(regrets, ns, zoomin=(0, 0.1), wlegend=False):
    # Set up 2x2 subplots
    fig, axes = plt.subplots(2, 2, dpi=600)

    # Plot overlayed KDEs in each subplot with custom colors
    t = 0
    for i in range(2):
        for j in range(2):
            for k, legends in enumerate(legend_dicts.values()):
                if k == 2 or k == 4:
                    continue
                sns.kdeplot(
                    regrets[5 * t + k],
                    ax=axes[i, j],
                    color=legends[0],
                    linestyle=legends[1],
                )
            inset_ax = inset_axes(
                axes[i, j], width="50%", height="50%", loc="upper right"
            )
            for k, legends in enumerate(legend_dicts.values()):
                if k == 2 or k == 4:
                    continue
                sns.kdeplot(
                    regrets[5 * t + k],
                    ax=inset_ax,
                    color=legends[0],
                    linestyle=legends[1],
                )
            inset_ax.set_xlim(zoomin[0], zoomin[1])
            axes[i, j].set_title(f"n={ns[t]}")
            inset_ax.set_ylabel("")
            axes[i, j].set_ylabel("")
            t += 1

    # Add shared x and y labels for the whole figure
    fig.text(0.5, 0, "Regret", ha="center", va="center", fontsize=14)
    fig.text(0, 0.5, "Frequency", ha="center", va="center", rotation="vertical", fontsize=14)

    if wlegend is True:
        # Create a shared legend for the entire figure
        fig.legend(
            ["ecdf", "kde", "rde"],
            loc="center right",
            bbox_to_anchor=(1.25, 0.5),
            title="Legend",
        )

    # Display the plot
    plt.show()
