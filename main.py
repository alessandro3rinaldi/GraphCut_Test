import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.neighbors import KDTree
import graphcuts
import pointclouds


# ── Visualization ─────────────────────────────────────────────────────────────

def show_results(points, gt_labels, labels_dict, circles=None, show_circles=True):
    n_plots = 1 + len(labels_dict)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))

    # Ground truth
    colors = np.where(gt_labels == 0, 'green', 'red')
    axes[0].scatter(points[:, 0], points[:, 1], c=colors, s=5)
    axes[0].set_title("Ground Truth")
    axes[0].set_aspect('equal')

    # Each method
    for ax, (name, labels) in zip(axes[1:], labels_dict.items()):
        acc    = (labels == gt_labels).mean()
        colors = np.where(labels == 0, 'green', 'red')
        ax.scatter(points[:, 0], points[:, 1], c=colors, s=5)
        ax.set_title(f"{name}\nAcc: {acc:.2%}")
        ax.set_aspect('equal')
        print(f"Accuracy {name}: {acc:.2%}")

        # Draw circle only if show_circles=True and circle is provided for this method
        if show_circles and circles is not None and name in circles:
            center, radius = circles[name]
            if center is not None and radius is not None:
                patch = patches.Circle(center, radius,
                                       fill=False, edgecolor='blue',
                                       linewidth=1.5, linestyle='--',
                                       label='fitted circle')
                ax.add_patch(patch)
                ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    #points, gt_labels = pointclouds.noisy_circle(N_inliers=150, N_outliers=400, noise=0.02)
    #points, gt_labels = pointclouds.noisy_square(N_inliers=150, N_outliers=400, noise=0.02)
    points, gt_labels = pointclouds.two_tangent_circles(N_inliers=150, N_outliers=400, noise=0.02)

    # RANSAC functions also return center and radius of the best circle
    labels_ransac,    c1, r1 = graphcuts.ransac_circle(points, epsilon=0.3, n_iter=500)
    labels_ransac_gc, c2, r2 = graphcuts.ransac_with_graphcut(points, epsilon=0.3, n_iter=500, k=6)

    labels_dict = {
        "PyMaxflow (BK)":    graphcuts.graphcut_pymaxflow(points, k=6, inlier_bias=1),
        "NetworkX (min-cut)": graphcuts.graphcut_networkx(points, k=4, inlier_bias=0.9),
        "RANSAC":             labels_ransac,
        "RANSAC + GraphCut":  labels_ransac_gc,
    }

    # Only RANSAC methods have a fitted circle to draw
    circles = {
        "RANSAC":            (c1, r1),
        "RANSAC + GraphCut": (c2, r2),
    }

    show_results(points, gt_labels, labels_dict, circles=circles, show_circles=False)


if __name__ == "__main__":
    main()