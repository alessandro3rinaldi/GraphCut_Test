import numpy as np


def noisy_circle(N_inliers=200, N_outliers=600, radius=5.0, noise=0.05):
    """Circle with gaussian noise and random outliers."""
    angles  = np.linspace(0, 2 * np.pi, N_inliers)
    inliers = np.stack([
        radius * np.cos(angles),
        radius * np.sin(angles),
        np.zeros(N_inliers)
    ], axis=1)
    inliers += np.random.normal(0, noise, inliers.shape)

    outliers       = np.random.uniform(-radius * 1.5, radius * 1.5, (N_outliers, 3))
    outliers[:, 2] = 0

    points = np.concatenate([inliers, outliers])
    labels = np.array([0] * N_inliers + [1] * N_outliers)
    return points, labels


def noisy_ellipse(N_inliers=200, N_outliers=600, a=6.0, b=3.0, noise=0.05):
    """Ellipse with semi-axes a (x) and b (y), gaussian noise and random outliers."""
    angles  = np.linspace(0, 2 * np.pi, N_inliers)
    inliers = np.stack([
        a * np.cos(angles),
        b * np.sin(angles),
        np.zeros(N_inliers)
    ], axis=1)
    inliers += np.random.normal(0, noise, inliers.shape)

    outliers       = np.random.uniform(-a * 1.5, a * 1.5, (N_outliers, 3))
    outliers[:, 2] = 0

    points = np.concatenate([inliers, outliers])
    labels = np.array([0] * N_inliers + [1] * N_outliers)
    return points, labels


def noisy_rectangle(N_inliers=200, N_outliers=600, width=8.0, height=4.0, noise=0.05):
    """Axis-aligned rectangle perimeter with gaussian noise and random outliers."""
    n_per_side = N_inliers // 4

    # Points along each side
    top    = np.stack([np.linspace(-width/2, width/2, n_per_side),  np.full(n_per_side,  height/2), np.zeros(n_per_side)], axis=1)
    bottom = np.stack([np.linspace(-width/2, width/2, n_per_side),  np.full(n_per_side, -height/2), np.zeros(n_per_side)], axis=1)
    left   = np.stack([np.full(n_per_side, -width/2),  np.linspace(-height/2, height/2, n_per_side), np.zeros(n_per_side)], axis=1)
    right  = np.stack([np.full(n_per_side,  width/2),  np.linspace(-height/2, height/2, n_per_side), np.zeros(n_per_side)], axis=1)

    inliers  = np.concatenate([top, bottom, left, right])
    inliers += np.random.normal(0, noise, inliers.shape)

    outliers       = np.random.uniform(-width, width, (N_outliers, 3))
    outliers[:, 2] = 0

    points = np.concatenate([inliers, outliers])
    labels = np.array([0] * len(inliers) + [1] * N_outliers)
    return points, labels


def noisy_square(N_inliers=200, N_outliers=600, side=6.0, noise=0.05):
    """Square (special case of rectangle with equal sides)."""
    return noisy_rectangle(N_inliers, N_outliers, width=side, height=side, noise=noise)


def two_tangent_circles(N_inliers=200, N_outliers=600, radius1=5.0, radius2=3.0, noise=0.05):
    """
    Two tangent circles — useful to test if GC bleeds across boundaries.
    Labels: 0 = circle 1, 1 = circle 2, 1 = outlier. (for now just two labels)
    """
    def make_circle(cx, r, n):
        angles = np.linspace(0, 2 * np.pi, n)
        pts    = np.stack([cx + r * np.cos(angles),
                           r * np.sin(angles),
                           np.zeros(n)], axis=1)
        pts   += np.random.normal(0, noise, pts.shape)
        return pts

    center2  = radius1 + radius2
    inliers1 = make_circle(0,       radius1, N_inliers)
    inliers2 = make_circle(center2, radius2, N_inliers)

    span           = center2 + radius2 + 2
    outliers       = np.random.uniform(-span, span, (N_outliers, 3))
    outliers[:, 2] = 0

    points = np.concatenate([inliers1, inliers2, outliers])
    labels = np.array([0] * N_inliers + [1] * N_inliers + [1] * N_outliers)
    return points, labels