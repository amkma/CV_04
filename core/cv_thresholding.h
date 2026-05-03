#pragma once
/**
 * cv_thresholding.h — From-scratch thresholding algorithms for CV_04
 *
 * Implements:
 *   1. Optimal (iterative) thresholding
 *   2. Otsu's thresholding (maximize between-class variance)
 *   3. Multi-level spectral thresholding (recursive Otsu, >2 classes)
 *   4. Local (adaptive) thresholding (block-based Otsu)
 *
 * OpenCV is used ONLY for cv::Mat storage. All logic is from scratch.
 */

#include <opencv2/core.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <climits>

namespace cv4 {

/* ═══════════════════════════════════════════════════════════════════
 *  HISTOGRAM — compute 256-bin histogram from a grayscale image
 *
 *  Iterates every pixel, increments the count for that intensity.
 *  Returns a vector of 256 integer counts.
 * ═══════════════════════════════════════════════════════════════════ */

inline std::vector<int> compute_histogram(const cv::Mat& gray) {
    std::vector<int> hist(256, 0);
    for (int y = 0; y < gray.rows; ++y)
        for (int x = 0; x < gray.cols; ++x)
            hist[gray.at<uchar>(y, x)]++;
    return hist;
}

/* ═══════════════════════════════════════════════════════════════════
 *  APPLY THRESHOLD — binarize a grayscale image at a given threshold
 *
 *  Pixels > T become 255 (foreground), else 0 (background).
 * ═══════════════════════════════════════════════════════════════════ */

inline cv::Mat apply_threshold(const cv::Mat& gray, int T) {
    cv::Mat result(gray.rows, gray.cols, CV_8UC1);
    for (int y = 0; y < gray.rows; ++y)
        for (int x = 0; x < gray.cols; ++x)
            result.at<uchar>(y, x) = (gray.at<uchar>(y, x) > T) ? 255 : 0;
    return result;
}

/* ═══════════════════════════════════════════════════════════════════
 *  1. OPTIMAL (ITERATIVE) THRESHOLDING
 *
 *  Algorithm (from Tutorial 2):
 *    1. Initialize T as the mean intensity of the entire image.
 *    2. Partition pixels into two groups:
 *         G1 = {pixels with intensity <= T}
 *         G2 = {pixels with intensity >  T}
 *    3. Compute mean intensities μ1 (of G1) and μ2 (of G2).
 *    4. Update T = (μ1 + μ2) / 2.
 *    5. Repeat steps 2–4 until |T_new - T_old| < 0.5 (convergence).
 *
 *  This method works well when the histogram is roughly bimodal
 *  and the two modes have similar spread.
 * ═══════════════════════════════════════════════════════════════════ */

struct ThresholdResult {
    cv::Mat output;       // Thresholded binary image
    int threshold;        // Final threshold value found
    int iterations;       // Number of iterations to converge
    double time_ms;       // Processing time
};

inline ThresholdResult optimal_threshold(const cv::Mat& gray) {
    // Step 1: compute histogram and initial T = global mean
    std::vector<int> hist = compute_histogram(gray);
    int total_pixels = gray.rows * gray.cols;

    // Global mean as initial threshold estimate
    double sum = 0;
    for (int i = 0; i < 256; ++i) sum += (double)i * hist[i];
    double T = sum / total_pixels;

    int iterations = 0;
    const int MAX_ITER = 200;

    // Step 2-5: iterate until convergence
    while (iterations < MAX_ITER) {
        // Partition into G1 (<=T) and G2 (>T), compute their means
        double sum1 = 0, sum2 = 0;
        int count1 = 0, count2 = 0;

        for (int i = 0; i < 256; ++i) {
            if (i <= (int)T) {
                sum1 += (double)i * hist[i];
                count1 += hist[i];
            } else {
                sum2 += (double)i * hist[i];
                count2 += hist[i];
            }
        }

        // Compute group means (guard against empty groups)
        double mu1 = (count1 > 0) ? sum1 / count1 : 0.0;
        double mu2 = (count2 > 0) ? sum2 / count2 : 0.0;

        // New threshold = midpoint of the two means
        double T_new = (mu1 + mu2) / 2.0;
        iterations++;

        // Check convergence: if T barely changed, stop
        if (std::abs(T_new - T) < 0.5) {
            T = T_new;
            break;
        }
        T = T_new;
    }

    int final_T = (int)std::round(T);
    final_T = std::clamp(final_T, 0, 255);

    ThresholdResult result;
    result.output = apply_threshold(gray, final_T);
    result.threshold = final_T;
    result.iterations = iterations;
    return result;
}

/* ═══════════════════════════════════════════════════════════════════
 *  2. OTSU'S THRESHOLDING
 *
 *  Algorithm (from Tutorial 2):
 *    Exhaustively search all possible thresholds t ∈ [0, 255]
 *    and select the one that maximizes the between-class variance:
 *
 *      σ²_B(t) = ω₀(t) · ω₁(t) · [μ₀(t) - μ₁(t)]²
 *
 *    where:
 *      ω₀(t) = Σ p(i) for i=0..t       (probability of class 0)
 *      ω₁(t) = Σ p(i) for i=t+1..255   (probability of class 1)
 *      μ₀(t) = Σ i·p(i)/ω₀ for i=0..t  (mean of class 0)
 *      μ₁(t) = Σ i·p(i)/ω₁ for i=t+1.. (mean of class 1)
 *      p(i)  = hist[i] / total_pixels   (normalized histogram)
 *
 *  This is optimal in the sense of minimum intra-class variance
 *  (equivalently, maximum inter-class variance).
 * ═══════════════════════════════════════════════════════════════════ */

inline int otsu_find_threshold(const std::vector<int>& hist, int total_pixels,
                               int range_start = 0, int range_end = 255) {
    // Compute normalized probabilities and cumulative sums
    // over the specified range [range_start, range_end]
    double best_sigma = -1.0;
    int best_t = range_start;

    // Precompute total weight and mean for the given range
    double w_total = 0, mu_total = 0;
    for (int i = range_start; i <= range_end; ++i) {
        w_total += hist[i];
        mu_total += (double)i * hist[i];
    }

    if (w_total == 0) return (range_start + range_end) / 2;

    // Sweep through all thresholds in the range
    double w0 = 0, sum0 = 0;

    for (int t = range_start; t < range_end; ++t) {
        w0 += hist[t];                    // cumulative weight of class 0
        double w1 = w_total - w0;         // weight of class 1
        if (w0 == 0 || w1 == 0) continue; // need both classes non-empty

        sum0 += (double)t * hist[t];      // cumulative sum for class 0
        double sum1 = mu_total - sum0;    // sum for class 1

        double mu0 = sum0 / w0;           // mean of class 0
        double mu1 = sum1 / w1;           // mean of class 1

        // Between-class variance: σ²_B = ω₀ · ω₁ · (μ₀ - μ₁)²
        double diff = mu0 - mu1;
        double sigma_b = w0 * w1 * diff * diff;

        if (sigma_b > best_sigma) {
            best_sigma = sigma_b;
            best_t = t;
        }
    }

    return best_t;
}

inline ThresholdResult otsu_threshold(const cv::Mat& gray) {
    std::vector<int> hist = compute_histogram(gray);
    int total = gray.rows * gray.cols;

    int T = otsu_find_threshold(hist, total, 0, 255);

    ThresholdResult result;
    result.output = apply_threshold(gray, T);
    result.threshold = T;
    result.iterations = 1; // Otsu is single-pass (exhaustive search)
    return result;
}

/* ═══════════════════════════════════════════════════════════════════
 *  3. MULTI-LEVEL (SPECTRAL) THRESHOLDING
 *
 *  Extension of Otsu for more than 2 classes (Tutorial 2):
 *    To segment into N classes, we need N-1 thresholds.
 *    Strategy: recursively apply Otsu to split ranges.
 *
 *    1. Apply Otsu on [0, 255] → get threshold T1
 *    2. Apply Otsu on [0, T1] → get T0 (if needed)
 *    3. Apply Otsu on [T1+1, 255] → get T2 (if needed)
 *    4. Continue recursively until we have N-1 thresholds.
 *
 *  Each pixel is assigned to a class based on which range it falls
 *  in, and colored with evenly spaced gray levels for visualization.
 * ═══════════════════════════════════════════════════════════════════ */

struct SpectralResult {
    cv::Mat output;                 // Multi-level segmented image
    std::vector<int> thresholds;   // The N-1 threshold values found
    int num_classes;                // Number of classes
    double time_ms;
};

// Helper: recursively find thresholds by splitting histogram ranges
inline void find_thresholds_recursive(const std::vector<int>& hist,
                                       int total_pixels,
                                       int range_start, int range_end,
                                       int depth, int max_depth,
                                       std::vector<int>& thresholds) {
    if (depth >= max_depth) return;
    if (range_end - range_start < 2) return; // range too small to split

    // Find Otsu threshold within this sub-range
    int T = otsu_find_threshold(hist, total_pixels, range_start, range_end);

    // Only add if T is meaningfully inside the range
    if (T > range_start && T < range_end) {
        thresholds.push_back(T);

        // Recursively split left sub-range [range_start, T]
        find_thresholds_recursive(hist, total_pixels,
                                  range_start, T,
                                  depth + 1, max_depth, thresholds);

        // Recursively split right sub-range [T+1, range_end]
        find_thresholds_recursive(hist, total_pixels,
                                  T + 1, range_end,
                                  depth + 1, max_depth, thresholds);
    }
}

inline SpectralResult spectral_threshold(const cv::Mat& gray, int num_classes) {
    num_classes = std::clamp(num_classes, 2, 8);
    int num_thresholds = num_classes - 1;

    std::vector<int> hist = compute_histogram(gray);
    int total = gray.rows * gray.cols;

    // Find thresholds recursively
    std::vector<int> thresholds;
    // max_depth controls how many recursive splits we do
    // For N thresholds, we need ceil(log2(N+1)) levels
    int max_depth = (int)std::ceil(std::log2(num_classes));
    find_thresholds_recursive(hist, total, 0, 255, 0, max_depth, thresholds);

    // Sort and keep only the needed number of thresholds
    std::sort(thresholds.begin(), thresholds.end());

    // Remove duplicates
    thresholds.erase(std::unique(thresholds.begin(), thresholds.end()),
                     thresholds.end());

    // If we got more thresholds than needed, keep the most evenly spaced ones
    while ((int)thresholds.size() > num_thresholds && !thresholds.empty()) {
        // Remove the threshold that creates the smallest gap
        int min_gap = INT_MAX;
        int min_idx = 0;
        for (int i = 0; i < (int)thresholds.size(); ++i) {
            int left = (i == 0) ? 0 : thresholds[i - 1];
            int right = (i == (int)thresholds.size() - 1) ? 255 : thresholds[i + 1];
            int gap = right - left;
            if (gap < min_gap) {
                min_gap = gap;
                min_idx = i;
            }
        }
        thresholds.erase(thresholds.begin() + min_idx);
    }

    // If we got fewer thresholds than needed, fill with evenly spaced ones
    while ((int)thresholds.size() < num_thresholds) {
        // Find the largest gap and split it
        int best_gap = -1, best_pos = -1;
        std::vector<int> boundaries = {0};
        for (int t : thresholds) boundaries.push_back(t);
        boundaries.push_back(255);

        for (int i = 0; i + 1 < (int)boundaries.size(); ++i) {
            int gap = boundaries[i + 1] - boundaries[i];
            if (gap > best_gap) {
                best_gap = gap;
                best_pos = i;
            }
        }
        int new_t = (boundaries[best_pos] + boundaries[best_pos + 1]) / 2;
        thresholds.push_back(new_t);
        std::sort(thresholds.begin(), thresholds.end());
    }

    // Assign each pixel to a class and map to evenly-spaced gray levels
    // Class i spans [ thresholds[i-1]+1, thresholds[i] ]
    // Class 0: [0, thresholds[0]]
    // Class N-1: [thresholds[N-2]+1, 255]
    cv::Mat output(gray.rows, gray.cols, CV_8UC1);

    for (int y = 0; y < gray.rows; ++y) {
        for (int x = 0; x < gray.cols; ++x) {
            int val = gray.at<uchar>(y, x);

            // Determine which class this pixel belongs to
            int cls = 0;
            for (int t = 0; t < (int)thresholds.size(); ++t) {
                if (val > thresholds[t]) cls = t + 1;
            }

            // Map class to evenly spaced gray level
            // Class 0 → 0, Class N-1 → 255
            int level = (num_classes > 1)
                        ? (int)(255.0 * cls / (num_classes - 1))
                        : 0;
            output.at<uchar>(y, x) = (uchar)std::clamp(level, 0, 255);
        }
    }

    SpectralResult result;
    result.output = output;
    result.thresholds = thresholds;
    result.num_classes = num_classes;
    return result;
}

/* ═══════════════════════════════════════════════════════════════════
 *  4. LOCAL (ADAPTIVE) THRESHOLDING
 *
 *  Algorithm (from Tutorial 2):
 *    Handles non-uniform illumination by applying thresholding
 *    independently to local regions of the image.
 *
 *    1. Divide the image into non-overlapping blocks of size B×B.
 *    2. For each block, compute its own Otsu threshold.
 *    3. Apply that local threshold to binarize the block.
 *    4. Edge blocks that are smaller than B×B are handled
 *       by computing Otsu on whatever pixels are available.
 *
 *  This is effective for documents, medical images, and scenes
 *  with uneven lighting where a single global threshold fails.
 * ═══════════════════════════════════════════════════════════════════ */

struct LocalThresholdResult {
    cv::Mat output;        // Locally thresholded binary image
    int block_size;        // Block size used
    int num_blocks;        // Total number of blocks processed
    double time_ms;
};

inline LocalThresholdResult local_threshold(const cv::Mat& gray, int block_size) {
    block_size = std::max(8, block_size); // minimum block size of 8

    cv::Mat output(gray.rows, gray.cols, CV_8UC1, cv::Scalar(0));
    int num_blocks = 0;

    // Iterate over blocks
    for (int by = 0; by < gray.rows; by += block_size) {
        for (int bx = 0; bx < gray.cols; bx += block_size) {
            // Determine actual block dimensions (handle edge blocks)
            int bh = std::min(block_size, gray.rows - by);
            int bw = std::min(block_size, gray.cols - bx);

            // Compute local histogram for this block
            std::vector<int> local_hist(256, 0);
            int local_total = 0;

            for (int y = by; y < by + bh; ++y) {
                for (int x = bx; x < bx + bw; ++x) {
                    local_hist[gray.at<uchar>(y, x)]++;
                    local_total++;
                }
            }

            // Find Otsu threshold for this block
            int local_T = otsu_find_threshold(local_hist, local_total, 0, 255);

            // Apply threshold to this block
            for (int y = by; y < by + bh; ++y) {
                for (int x = bx; x < bx + bw; ++x) {
                    output.at<uchar>(y, x) =
                        (gray.at<uchar>(y, x) > local_T) ? 255 : 0;
                }
            }

            num_blocks++;
        }
    }

    LocalThresholdResult result;
    result.output = output;
    result.block_size = block_size;
    result.num_blocks = num_blocks;
    return result;
}

} // namespace cv4
