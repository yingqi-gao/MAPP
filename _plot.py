import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def regrets_plot(regrets, ns, zoomin=(0, 0.1)):
    # Define custom colors for groups of three in different shades
    legend_dicts = {
        "ecdf": ("black", ":"),
        "kde": ("red", "-"),
        "rde(50x10)": ("purple", "--"),
        "rde(50x200)": ("blue", "-."),
        "rde(200x200)": ("green", (0, (5, 1, 1, 1, 1, 1, 1, 1))),
    }

    # Set up 2x2 subplots
    fig, axes = plt.subplots(2, 2)

    # Plot overlayed KDEs in each subplot with custom colors
    t = 0
    for i in range(2):
        for j in range(2):
            for k, legends in enumerate(legend_dicts.values()):
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
                sns.kdeplot(
                    regrets[5 * t + k],
                    ax=inset_ax,
                    color=legends[0],
                    linestyle=legends[1],
                )
            inset_ax.set_xlim(zoomin[0], zoomin[1])
            axes[i, j].set_title(f"n={ns[t]}")
            t += 1

    # # Create a shared legend for the entire figure
    # fig.legend(
    #     [key for key in legend_dicts.keys()],
    #     loc="center right",
    #     bbox_to_anchor=(1.25, 0.5),
    #     title="Legend",
    # )

    # Adjust the layout and display the plot
    plt.tight_layout()
    plt.show()
