#pragma once
/**
 * cv_segmentation.h — From-scratch segmentation algorithms for CV_04
 * Implements: K-means, Region Growing, Agglomerative, Mean Shift
 * OpenCV used ONLY for cv::Mat storage.
 */
#include <opencv2/core.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <queue>
#include <random>
#include <array>
#include <map>

namespace cv4 {

/* ─── Color table for labeling segmented regions ─── */
inline cv::Vec3b label_color(int label) {
    // Deterministic distinct colors via golden-ratio hue spacing
    double hue = std::fmod(label * 137.508, 360.0);
    double h = hue / 60.0;
    int hi = (int)h % 6;
    double f = h - (int)h;
    double q = 1.0 - f;
    double r = 0, g = 0, b = 0;
    switch (hi) {
        case 0: r=1; g=f; b=0; break;
        case 1: r=q; g=1; b=0; break;
        case 2: r=0; g=1; b=f; break;
        case 3: r=0; g=q; b=1; break;
        case 4: r=f; g=0; b=1; break;
        case 5: r=1; g=0; b=q; break;
    }
    return cv::Vec3b((uchar)(b*220+35), (uchar)(g*220+35), (uchar)(r*220+35));
}

/* ═══════════════════════════════════════════════════════════════
 *  1. K-MEANS CLUSTERING (from scratch)
 *
 *  For grayscale: 1D feature (intensity).
 *  For color: 3D feature (B, G, R channels).
 *
 *  Steps:
 *    1. Initialize k centroids using k-means++ for stability.
 *    2. Assignment: assign each pixel to nearest centroid (L2).
 *    3. Update: recompute centroids as mean of assigned pixels.
 *    4. Repeat until labels stabilize or max_iter reached.
 * ═══════════════════════════════════════════════════════════════ */

struct SegmentResult {
    cv::Mat output;       // Segmented/colored image
    int num_clusters;
    int iterations;
    double time_ms;
};

inline SegmentResult kmeans_segment(const cv::Mat& img, int k, int max_iter) {
    int rows = img.rows, cols = img.cols;
    int N = rows * cols;
    int ch = img.channels();
    k = std::clamp(k, 2, 16);
    max_iter = std::clamp(max_iter, 1, 100);

    // Flatten pixels into feature vectors
    std::vector<std::vector<double>> pixels(N, std::vector<double>(ch));
    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x) {
            int idx = y * cols + x;
            if (ch == 1) {
                pixels[idx][0] = img.at<uchar>(y, x);
            } else {
                cv::Vec3b c = img.at<cv::Vec3b>(y, x);
                pixels[idx][0] = c[0];
                pixels[idx][1] = c[1];
                pixels[idx][2] = c[2];
            }
        }

    // K-means++ initialization
    std::mt19937 rng(42);
    std::vector<std::vector<double>> centroids(k, std::vector<double>(ch, 0));
    std::vector<double> min_dist(N, 1e30);

    // First centroid: random pixel
    centroids[0] = pixels[rng() % N];

    for (int c = 1; c < k; ++c) {
        // Compute distance from each pixel to nearest existing centroid
        double total = 0;
        for (int i = 0; i < N; ++i) {
            double d = 0;
            for (int j = 0; j < ch; ++j) {
                double diff = pixels[i][j] - centroids[c - 1][j];
                d += diff * diff;
            }
            min_dist[i] = std::min(min_dist[i], d);
            total += min_dist[i];
        }
        // Weighted random selection proportional to D²
        double r = std::uniform_real_distribution<>(0, total)(rng);
        double cum = 0;
        int chosen = 0;
        for (int i = 0; i < N; ++i) {
            cum += min_dist[i];
            if (cum >= r) { chosen = i; break; }
        }
        centroids[c] = pixels[chosen];
    }

    // Iterative assignment + update
    std::vector<int> labels(N, 0);
    int iters = 0;

    for (int iter = 0; iter < max_iter; ++iter) {
        iters++;
        bool changed = false;

        // Assignment step
        for (int i = 0; i < N; ++i) {
            double best = 1e30;
            int best_c = 0;
            for (int c = 0; c < k; ++c) {
                double d = 0;
                for (int j = 0; j < ch; ++j) {
                    double diff = pixels[i][j] - centroids[c][j];
                    d += diff * diff;
                }
                if (d < best) { best = d; best_c = c; }
            }
            if (labels[i] != best_c) { labels[i] = best_c; changed = true; }
        }

        if (!changed) break;

        // Update step: recompute centroids
        std::vector<std::vector<double>> sums(k, std::vector<double>(ch, 0));
        std::vector<int> counts(k, 0);
        for (int i = 0; i < N; ++i) {
            counts[labels[i]]++;
            for (int j = 0; j < ch; ++j)
                sums[labels[i]][j] += pixels[i][j];
        }
        for (int c = 0; c < k; ++c) {
            if (counts[c] > 0)
                for (int j = 0; j < ch; ++j)
                    centroids[c][j] = sums[c][j] / counts[c];
        }
    }

    // Build output: map each pixel to its centroid color
    cv::Mat output(rows, cols, CV_8UC3);
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            int lbl = labels[y * cols + x];
            output.at<cv::Vec3b>(y, x) = label_color(lbl);
        }
    }

    return {output, k, iters, 0};
}

/* ═══════════════════════════════════════════════════════════════
 *  2. REGION GROWING
 *
 *  BFS-based expansion from seed points.
 *  Similarity: |pixel_intensity - region_mean| < threshold
 *  Uses 4-connectivity. Updates region mean incrementally.
 * ═══════════════════════════════════════════════════════════════ */

inline SegmentResult region_growing(const cv::Mat& img,
                                     const std::vector<std::pair<int,int>>& seeds,
                                     double threshold) {
    int rows = img.rows, cols = img.cols;
    int ch = img.channels();

    // Label map: -1 = unlabeled
    std::vector<int> labels(rows * cols, -1);
    int num_regions = 0;

    // Store region mean for each label
    std::vector<std::vector<double>> region_sums;
    std::vector<int> region_counts;

    auto get_pixel = [&](int y, int x) -> std::vector<double> {
        std::vector<double> p(ch);
        if (ch == 1) {
            p[0] = img.at<uchar>(y, x);
        } else {
            auto c = img.at<cv::Vec3b>(y, x);
            p[0] = c[0]; p[1] = c[1]; p[2] = c[2];
        }
        return p;
    };

    auto pixel_dist = [&](const std::vector<double>& a,
                          const std::vector<double>& b) -> double {
        double d = 0;
        for (int i = 0; i < ch; ++i) {
            double diff = a[i] - b[i];
            d += diff * diff;
        }
        return std::sqrt(d);
    };

    // 4-connectivity neighbors
    const int dx[] = {0, 0, 1, -1};
    const int dy[] = {1, -1, 0, 0};

    // Grow from each seed
    for (auto& [sy, sx] : seeds) {
        if (sy < 0 || sy >= rows || sx < 0 || sx >= cols) continue;
        if (labels[sy * cols + sx] >= 0) continue; // already labeled

        int lbl = num_regions++;
        region_sums.push_back(get_pixel(sy, sx));
        region_counts.push_back(1);
        labels[sy * cols + sx] = lbl;

        std::queue<std::pair<int,int>> q;
        q.push({sy, sx});

        while (!q.empty()) {
            auto [cy, cx] = q.front(); q.pop();

            for (int d = 0; d < 4; ++d) {
                int ny = cy + dy[d], nx = cx + dx[d];
                if (ny < 0 || ny >= rows || nx < 0 || nx >= cols) continue;
                if (labels[ny * cols + nx] >= 0) continue;

                // Compute distance to region mean
                std::vector<double> pix = get_pixel(ny, nx);
                std::vector<double> mean(ch);
                for (int i = 0; i < ch; ++i)
                    mean[i] = region_sums[lbl][i] / region_counts[lbl];

                if (pixel_dist(pix, mean) < threshold) {
                    labels[ny * cols + nx] = lbl;
                    for (int i = 0; i < ch; ++i)
                        region_sums[lbl][i] += pix[i];
                    region_counts[lbl]++;
                    q.push({ny, nx});
                }
            }
        }
    }

    // Build output
    cv::Mat output(rows, cols, CV_8UC3, cv::Scalar(30, 30, 30));
    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x) {
            int lbl = labels[y * cols + x];
            if (lbl >= 0)
                output.at<cv::Vec3b>(y, x) = label_color(lbl);
        }

    return {output, num_regions, 1, 0};
}

/* ═══════════════════════════════════════════════════════════════
 *  3. AGGLOMERATIVE CLUSTERING
 *
 *  Bottom-up merging with average linkage on downsampled blocks.
 *  1. Divide image into small blocks → initial clusters.
 *  2. Compute mean color per block.
 *  3. Repeatedly merge the two closest clusters (by color dist).
 *  4. Stop when desired number of clusters reached.
 *  5. Map block labels back to full resolution.
 * ═══════════════════════════════════════════════════════════════ */

inline SegmentResult agglomerative_segment(const cv::Mat& img, int num_clusters) {
    int rows = img.rows, cols = img.cols;
    int ch = img.channels();
    num_clusters = std::clamp(num_clusters, 2, 16);

    // Adaptive block size: keep total blocks ≤ ~500 for tractability
    // O(n³) merge loop: 500³ = 125M which is fine, 4000³ = 64B which is not
    int bsize = 50;
    while ((rows / bsize) * (cols / bsize) > 500 && bsize < 64)
        bsize *= 2;

    int brows = (rows + bsize - 1) / bsize;
    int bcols = (cols + bsize - 1) / bsize;
    int nb = brows * bcols;

    // Compute mean color per block
    struct Cluster {
        std::vector<double> color;
        int count; // number of blocks in this cluster
        std::vector<int> blocks; // block indices
    };

    std::vector<Cluster> clusters(nb);
    for (int by = 0; by < brows; ++by) {
        for (int bx = 0; bx < bcols; ++bx) {
            int bi = by * bcols + bx;
            clusters[bi].color.resize(ch, 0);
            clusters[bi].count = 1;
            clusters[bi].blocks = {bi};

            int npix = 0;
            for (int y = by*bsize; y < std::min((by+1)*bsize, rows); ++y)
                for (int x = bx*bsize; x < std::min((bx+1)*bsize, cols); ++x) {
                    if (ch == 1) {
                        clusters[bi].color[0] += img.at<uchar>(y, x);
                    } else {
                        auto c = img.at<cv::Vec3b>(y, x);
                        clusters[bi].color[0] += c[0];
                        clusters[bi].color[1] += c[1];
                        clusters[bi].color[2] += c[2];
                    }
                    npix++;
                }
            for (int j = 0; j < ch; ++j)
                clusters[bi].color[j] /= std::max(npix, 1);
        }
    }

    // Active cluster indices
    std::vector<bool> active(nb, true);
    int active_count = nb;

    // Merge until desired clusters reached
    int iters = 0;
    while (active_count > num_clusters) {
        // Find two closest active clusters
        double min_dist = 1e30;
        int mi = -1, mj = -1;

        std::vector<int> act_idx;
        act_idx.reserve(active_count);
        for (int i = 0; i < nb; ++i)
            if (active[i]) act_idx.push_back(i);

        for (int ii = 0; ii < (int)act_idx.size(); ++ii) {
            for (int jj = ii + 1; jj < (int)act_idx.size(); ++jj) {
                int a = act_idx[ii], b = act_idx[jj];
                double d = 0;
                for (int c = 0; c < ch; ++c) {
                    double diff = clusters[a].color[c] - clusters[b].color[c];
                    d += diff * diff;
                }
                if (d < min_dist) {
                    min_dist = d;
                    mi = a; mj = b;
                }
            }
        }

        if (mi < 0) break;

        // Merge mj into mi (weighted average color)
        int total = clusters[mi].count + clusters[mj].count;
        for (int c = 0; c < ch; ++c)
            clusters[mi].color[c] =
                (clusters[mi].color[c] * clusters[mi].count +
                 clusters[mj].color[c] * clusters[mj].count) / total;
        clusters[mi].count = total;
        clusters[mi].blocks.insert(clusters[mi].blocks.end(),
                                    clusters[mj].blocks.begin(),
                                    clusters[mj].blocks.end());
        active[mj] = false;
        active_count--;
        iters++;
    }

    // Extract the final cluster colors
    std::vector<std::vector<double>> final_colors;
    for (int i = 0; i < nb; ++i) {
        if (active[i]) {
            final_colors.push_back(clusters[i].color);
        }
    }

    // Map each pixel to the nearest final cluster color to avoid blockiness
    cv::Mat output(rows, cols, CV_8UC3);
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            std::vector<double> p(ch);
            if (ch == 1) {
                p[0] = img.at<uchar>(y, x);
            } else {
                auto c = img.at<cv::Vec3b>(y, x);
                p[0] = c[0]; p[1] = c[1]; p[2] = c[2];
            }

            double min_dist = 1e30;
            int best_lbl = 0;
            for (int i = 0; i < (int)final_colors.size(); ++i) {
                double d = 0;
                for (int c = 0; c < ch; ++c) {
                    double diff = p[c] - final_colors[i][c];
                    d += diff * diff;
                }
                if (d < min_dist) {
                    min_dist = d;
                    best_lbl = i;
                }
            }
            output.at<cv::Vec3b>(y, x) = label_color(best_lbl);
        }
    }

    return {output, num_clusters, iters, 0};
}

/* ═══════════════════════════════════════════════════════════════
 *  4. MEAN SHIFT CLUSTERING
 *
 *  Iterative mode-seeking in joint spatial-color space.
 *  For each pixel (x, y, c1, c2, c3):
 *    1. Define a window with spatial_radius and color_radius.
 *    2. Find all pixels within both radii.
 *    3. Shift toward the mean of the window.
 *    4. Repeat until convergence (shift < epsilon).
 *  After convergence, merge nearby modes and assign labels.
 *
 *  For tractability, we subsample then map back.
 * ═══════════════════════════════════════════════════════════════ */

inline SegmentResult mean_shift_segment(const cv::Mat& img,
                                         double spatial_radius,
                                         double color_radius) {
    int rows = img.rows, cols = img.cols;
    int ch = img.channels();

    // More aggressive subsampling for tractability
    int step = 1;
    if (rows * cols > 40000)  step = 2;
    if (rows * cols > 100000) step = 3;
    if (rows * cols > 200000) step = 4;
    if (rows * cols > 400000) step = 5;

    int srows = (rows + step - 1) / step;
    int scols = (cols + step - 1) / step;
    int sN = srows * scols;

    // Build feature vectors: [y, x, c0, c1, c2]
    int fdim = 2 + ch;
    std::vector<std::vector<double>> features(sN, std::vector<double>(fdim));
    std::vector<std::vector<double>> modes(sN, std::vector<double>(fdim));

    for (int sy = 0; sy < srows; ++sy) {
        for (int sx = 0; sx < scols; ++sx) {
            int y = std::min(sy * step, rows - 1);
            int x = std::min(sx * step, cols - 1);
            int si = sy * scols + sx;
            features[si][0] = y;
            features[si][1] = x;
            if (ch == 1) {
                features[si][2] = img.at<uchar>(y, x);
            } else {
                auto c = img.at<cv::Vec3b>(y, x);
                features[si][2] = c[0];
                features[si][3] = c[1];
                features[si][4] = c[2];
            }
            modes[si] = features[si];
        }
    }

    // ── Spatial grid for O(1) neighbor lookup ──
    // Grid cell size = spatial_radius so we only check 3×3 neighborhood
    double cell_size = std::max(spatial_radius, 1.0);
    int grid_rows = (int)std::ceil(rows / cell_size) + 1;
    int grid_cols = (int)std::ceil(cols / cell_size) + 1;

    // Build grid: cell → list of pixel indices
    auto grid_id = [&](double y, double x) -> int {
        int gy = std::clamp((int)(y / cell_size), 0, grid_rows - 1);
        int gx = std::clamp((int)(x / cell_size), 0, grid_cols - 1);
        return gy * grid_cols + gx;
    };

    double sr2 = spatial_radius * spatial_radius;
    double cr2 = color_radius * color_radius;
    const int MAX_MS_ITER = 15;
    const double EPSILON = 1.0;

    for (int iter = 0; iter < MAX_MS_ITER; ++iter) {
        // Rebuild grid each iteration (modes shift)
        std::vector<std::vector<int>> grid(grid_rows * grid_cols);
        for (int i = 0; i < sN; ++i) {
            int gid = grid_id(modes[i][0], modes[i][1]);
            grid[gid].push_back(i);
        }

        bool any_moved = false;
        for (int i = 0; i < sN; ++i) {
            std::vector<double> sum(fdim, 0);
            int count = 0;

            // Only check grid cells within spatial radius (3×3 neighborhood)
            int gy = std::clamp((int)(modes[i][0] / cell_size), 0, grid_rows - 1);
            int gx = std::clamp((int)(modes[i][1] / cell_size), 0, grid_cols - 1);

            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int ny = gy + dy, nx = gx + dx;
                    if (ny < 0 || ny >= grid_rows || nx < 0 || nx >= grid_cols) continue;
                    int gid = ny * grid_cols + nx;

                    for (int j : grid[gid]) {
                        // Spatial distance
                        double ds = 0;
                        for (int d = 0; d < 2; ++d) {
                            double diff = modes[i][d] - modes[j][d];
                            ds += diff * diff;
                        }
                        if (ds > sr2) continue;

                        // Color distance
                        double dc = 0;
                        for (int d = 2; d < fdim; ++d) {
                            double diff = modes[i][d] - modes[j][d];
                            dc += diff * diff;
                        }
                        if (dc > cr2) continue;

                        for (int d = 0; d < fdim; ++d)
                            sum[d] += modes[j][d];
                        count++;
                    }
                }
            }

            if (count > 0) {
                double shift = 0;
                for (int d = 0; d < fdim; ++d) {
                    double new_val = sum[d] / count;
                    double diff = new_val - modes[i][d];
                    shift += diff * diff;
                    modes[i][d] = new_val;
                }
                if (std::sqrt(shift) > EPSILON) any_moved = true;
            }
        }
        if (!any_moved) break;
    }

    // Merge modes: cluster converged modes that are close
    std::vector<int> labels(sN, -1);
    int num_labels = 0;
    std::vector<std::vector<double>> final_modes;

    for (int i = 0; i < sN; ++i) {
        if (labels[i] >= 0) continue;

        int found = -1;
        for (int m = 0; m < (int)final_modes.size(); ++m) {
            double ds = 0, dc = 0;
            for (int d = 0; d < 2; ++d) {
                double diff = modes[i][d] - final_modes[m][d];
                ds += diff * diff;
            }
            for (int d = 2; d < fdim; ++d) {
                double diff = modes[i][d] - final_modes[m][d];
                dc += diff * diff;
            }
            if (ds < sr2 && dc < cr2) { found = m; break; }
        }

        if (found >= 0) {
            labels[i] = found;
        } else {
            labels[i] = num_labels;
            final_modes.push_back(modes[i]);
            num_labels++;
        }
    }

    // Map subsampled labels back to full resolution (nearest neighbor)
    cv::Mat output(rows, cols, CV_8UC3);
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            int sy = std::min(y / step, srows - 1);
            int sx = std::min(x / step, scols - 1);
            int lbl = labels[sy * scols + sx];
            if (lbl >= 0 && lbl < (int)final_modes.size()) {
                output.at<cv::Vec3b>(y, x) = label_color(lbl);
            } else {
                output.at<cv::Vec3b>(y, x) = label_color(0);
            }
        }
    }

    return {output, num_labels, MAX_MS_ITER, 0};
}

} // namespace cv4
