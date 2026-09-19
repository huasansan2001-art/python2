"""
visualizer.py

Plotting helpers for Section 2 of the FMA analysis (genre-pair PCA centroid
distances over time): plot_pair_distance_trends (2.1) and
plot_pair_time_strip (2.2).

All methods are called as `Visualizer.method(...)` (static, no instance
state) to match the calling convention used throughout report.ipynb.
"""

from typing import List, Tuple

import matplotlib.pyplot as plt
import seaborn as sns

from genre_pair_analysis import GenrePairResult, PERIOD_WIDTH

# Dark theme shared with visualization.py so all plots in the report match.
BG = "#0d1117"
GRID_COLOR = "#444444"
SPINE_COLOR = "#333333"
TEXT_COLOR = "#aaaaaa"


class Visualizer:

    # ================================================================
    # 2.1 — Distance trends, small multiples by anchor genre
    # ================================================================
    @staticmethod
    def plot_pair_distance_trends(pair_result: GenrePairResult):
        genres = pair_result.genres
        periods = pair_result.periods
        midpoints = [p + PERIOD_WIDTH / 2 for p in periods]
        dist = pair_result.pair_distances

        n = len(genres)
        ncols = 4
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5 * ncols, 3.2 * nrows), facecolor=BG
        )
        axes = axes.flatten()

        palette = sns.color_palette("tab10", n_colors=n)
        color_map = dict(zip(genres, palette))

        for i, anchor in enumerate(genres):
            ax = axes[i]
            ax.set_facecolor(BG)

            for other in genres:
                if other == anchor:
                    continue
                col = f"{anchor}|{other}" if f"{anchor}|{other}" in dist.columns else f"{other}|{anchor}"
                if col not in dist.columns:
                    continue
                y = dist[col].reindex(periods)
                ax.plot(
                    midpoints, y.values,
                    marker="o", linewidth=1.5, markersize=4,
                    color=color_map[other], label=other,
                )

            ax.set_title(f'distances from "{anchor}"', color="white", fontsize=10)
            ax.set_xlabel("period midpoint", color=TEXT_COLOR, fontsize=8)
            ax.set_ylabel("centroid distance (PCA-2D)", color=TEXT_COLOR, fontsize=8)
            ax.tick_params(colors=TEXT_COLOR, labelsize=7)
            ax.grid(linestyle="--", alpha=0.3, color=GRID_COLOR)
            for spine in ax.spines.values():
                spine.set_color(SPINE_COLOR)

        # hide any unused axes (e.g. 8 genres in a 4x2 grid fills exactly, but
        # this keeps the function generic for other genre counts)
        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        handles = [
            plt.Line2D([0], [0], color=color_map[g], marker="o", linewidth=1.5, label=g)
            for g in genres
        ]
        fig.legend(
            handles=handles, loc="lower center", ncol=min(n, 8), fontsize=8,
            facecolor="#1a1a2e", labelcolor="white", framealpha=0.3,
        )

        pc1, pc2 = pair_result.explained_variance
        fig.suptitle(
            f"Genre-pair centroid distances over time (PC1 {pc1}%, PC2 {pc2}%)",
            color="white", fontsize=13, fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0.06, 1, 0.94])

    # ================================================================
    # 2.2 — Per-pair time strip (interesting pairs)
    # ================================================================
    @staticmethod
    def plot_pair_time_strip(pair_result: GenrePairResult, pairs: List[Tuple[str, str]]):
        periods = pair_result.periods
        X = pair_result.X_pca

        n_rows = len(pairs)
        n_cols = len(periods)

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(1.8 * n_cols, 1.8 * n_rows),
            facecolor=BG,
        )
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        palette = sns.color_palette("tab10", n_colors=len(pair_result.genres))
        color_map = dict(zip(pair_result.genres, palette))

        for r, (genre_a, genre_b) in enumerate(pairs):
            for c, period in enumerate(periods):
                ax = axes[r, c]
                ax.set_facecolor(BG)

                for genre in (genre_a, genre_b):
                    color = color_map[genre]
                    cell = X[(X["period"] == period) & (X["genre"] == genre)]
                    if len(cell):
                        ax.scatter(
                            cell["PC1"], cell["PC2"],
                            s=4, alpha=0.5, color=color, linewidths=0,
                        )
                    centroid = pair_result.centroids.get((period, genre))
                    if centroid is not None:
                        ax.scatter(
                            centroid[0], centroid[1],
                            marker="x", s=60, color=color, linewidths=2,
                        )

                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_color(SPINE_COLOR)

                if r == 0:
                    ax.set_title(f"{period}-{period + PERIOD_WIDTH}", color="white", fontsize=8)
                if c == 0:
                    ax.set_ylabel(
                        f"{genre_a}\nvs\n{genre_b}",
                        color="white", fontsize=7, rotation=0,
                        ha="right", va="center", labelpad=30,
                    )

        fig.suptitle(
            "Per-pair time strip — points + centroids by 5y period",
            color="white", fontsize=13, fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])
