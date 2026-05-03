#pragma once
/**
 * cv_custom_algorithms.h — From-scratch CV algorithms for CV_03
 *
 * Assignment 3: Feature Detection, Descriptors & Matching
 *   1. Harris Corner Detection + λ⁻
 *   2. SIFT Feature Descriptors
 *   3. Feature Matching (SSD + NCC)
 *
 * OpenCV is used ONLY for cv::Mat storage and basic types.
 * All algorithmic logic is implemented from scratch.
 */

#include <opencv2/core.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <queue>
#include <array>
#include <functional>

namespace custom {

constexpr double PI = 3.14159265358979323846;

/* ═══════════════════════════════════════════════════════════════════
 *  DATA STRUCTURES
 * ═══════════════════════════════════════════════════════════════════ */

struct KeyPoint {
    double x, y;
    double scale;
    double orientation;  // radians
    double response;     // corner strength / DoG value
    int octave;
};

struct MatchResult {
    int idx1, idx2;
    double distance;     // SSD distance or 1-NCC
};

/* ═══════════════════════════════════════════════════════════════════
 *  PIXEL ACCESS HELPERS
 * ═══════════════════════════════════════════════════════════════════ */

inline double get_pix(const cv::Mat& m, int y, int x) {
    if (y < 0) y = 0; if (y >= m.rows) y = m.rows - 1;
    if (x < 0) x = 0; if (x >= m.cols) x = m.cols - 1;
    if (m.type() == CV_8UC1)  return m.at<uchar>(y, x);
    if (m.type() == CV_32FC1) return m.at<float>(y, x);
    if (m.type() == CV_64FC1) return m.at<double>(y, x);
    return 0;
}

inline double get_pix_safe(const cv::Mat& m, int y, int x) {
    if (y < 0 || y >= m.rows || x < 0 || x >= m.cols) return 0;
    return get_pix(m, y, x);
}

inline void set_pixel_color(cv::Mat& img, int x, int y, const cv::Scalar& c) {
    if (x < 0 || x >= img.cols || y < 0 || y >= img.rows) return;
    if (img.channels() == 3)
        img.at<cv::Vec3b>(y, x) = cv::Vec3b((uchar)c[0], (uchar)c[1], (uchar)c[2]);
    else
        img.at<uchar>(y, x) = (uchar)c[0];
}

/* ═══════════════════════════════════════════════════════════════════
 *  GAUSSIAN BLUR (separable, from scratch)
 * ═══════════════════════════════════════════════════════════════════ */

inline std::vector<double> make_gaussian_kernel(int ksize, double sigma) {
    int half = ksize / 2;
    std::vector<double> k(ksize);
    double sum = 0;
    for (int i = 0; i < ksize; ++i) {
        double x = i - half;
        k[i] = std::exp(-(x * x) / (2.0 * sigma * sigma));
        sum += k[i];
    }
    for (auto& v : k) v /= sum;
    return k;
}

inline cv::Mat gaussian_blur(const cv::Mat& src, int ksize, double sigma) {
    auto k = make_gaussian_kernel(ksize, sigma);
    int half = ksize / 2;

    // horizontal pass
    cv::Mat tmp(src.rows, src.cols, CV_64FC1, cv::Scalar(0));
    for (int y = 0; y < src.rows; ++y)
        for (int x = 0; x < src.cols; ++x) {
            double v = 0;
            for (int i = -half; i <= half; ++i)
                v += get_pix(src, y, std::clamp(x + i, 0, src.cols - 1)) * k[i + half];
            tmp.at<double>(y, x) = v;
        }
    // vertical pass
    cv::Mat dst(src.rows, src.cols, CV_64FC1, cv::Scalar(0));
    for (int y = 0; y < src.rows; ++y)
        for (int x = 0; x < src.cols; ++x) {
            double v = 0;
            for (int i = -half; i <= half; ++i)
                v += tmp.at<double>(std::clamp(y + i, 0, src.rows - 1), x) * k[i + half];
            dst.at<double>(y, x) = v;
        }
    return dst;
}

// Version that returns CV_8UC1 (for image output)
inline cv::Mat gaussian_blur_u8(const cv::Mat& src, int ksize, double sigma) {
    cv::Mat f64 = gaussian_blur(src, ksize, sigma);
    cv::Mat dst(src.rows, src.cols, CV_8UC1);
    for (int y = 0; y < src.rows; ++y)
        for (int x = 0; x < src.cols; ++x)
            dst.at<uchar>(y, x) = (uchar)std::clamp(f64.at<double>(y, x), 0.0, 255.0);
    return dst;
}

/* ═══════════════════════════════════════════════════════════════════
 *  SOBEL OPERATOR (3×3, output CV_64F)
 * ═══════════════════════════════════════════════════════════════════ */

inline cv::Mat sobel(const cv::Mat& src, int dx, int dy) {
    double Kx[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
    double Ky[3][3] = {{-1,-2,-1},{0,0,0},{1,2,1}};

    cv::Mat dst(src.rows, src.cols, CV_64FC1, cv::Scalar(0));
    for (int y = 1; y < src.rows - 1; ++y)
        for (int x = 1; x < src.cols - 1; ++x) {
            double val = 0;
            for (int ky = -1; ky <= 1; ++ky)
                for (int kx = -1; kx <= 1; ++kx) {
                    double p = get_pix(src, y + ky, x + kx);
                    if (dx == 1) val += p * Kx[ky + 1][kx + 1];
                    else         val += p * Ky[ky + 1][kx + 1];
                }
            dst.at<double>(y, x) = val;
        }
    return dst;
}

/* ═══════════════════════════════════════════════════════════════════
 *  CONVERT TO GRAYSCALE (from scratch)
 * ═══════════════════════════════════════════════════════════════════ */

inline cv::Mat to_grayscale(const cv::Mat& src) {
    if (src.channels() == 1) return src.clone();
    cv::Mat gray(src.rows, src.cols, CV_8UC1);
    for (int y = 0; y < src.rows; ++y)
        for (int x = 0; x < src.cols; ++x) {
            cv::Vec3b bgr = src.at<cv::Vec3b>(y, x);
            // Standard luminosity formula
            gray.at<uchar>(y, x) = (uchar)(0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2]);
        }
    return gray;
}

/* ═══════════════════════════════════════════════════════════════════
 *  IMAGE RESIZE (bilinear interpolation, from scratch)
 * ═══════════════════════════════════════════════════════════════════ */

inline cv::Mat resize_image(const cv::Mat& src, int new_rows, int new_cols) {
    cv::Mat dst(new_rows, new_cols, src.type(), cv::Scalar(0));
    double ry = (double)src.rows / new_rows;
    double rx = (double)src.cols / new_cols;

    for (int y = 0; y < new_rows; ++y)
        for (int x = 0; x < new_cols; ++x) {
            double sy = y * ry, sx = x * rx;
            int y0 = (int)sy, x0 = (int)sx;
            int y1 = std::min(y0 + 1, src.rows - 1);
            int x1 = std::min(x0 + 1, src.cols - 1);
            double fy = sy - y0, fx = sx - x0;

            double v = get_pix(src, y0, x0) * (1-fx) * (1-fy)
                     + get_pix(src, y0, x1) * fx * (1-fy)
                     + get_pix(src, y1, x0) * (1-fx) * fy
                     + get_pix(src, y1, x1) * fx * fy;

            if (dst.type() == CV_8UC1)
                dst.at<uchar>(y, x) = (uchar)std::clamp(v, 0.0, 255.0);
            else if (dst.type() == CV_64FC1)
                dst.at<double>(y, x) = v;
            else if (dst.type() == CV_32FC1)
                dst.at<float>(y, x) = (float)v;
        }
    return dst;
}

/* ═══════════════════════════════════════════════════════════════════
 *  1. HARRIS CORNER DETECTION (from scratch)
 *
 *  Steps:
 *    1. Compute gradients Ix, Iy using Sobel
 *    2. Compute products Ix², Iy², IxIy
 *    3. Apply Gaussian window to each product
 *    4. Compute R = det(M) - k·trace(M)²
 *    5. Non-maximum suppression
 *    6. Threshold
 * ═══════════════════════════════════════════════════════════════════ */

inline std::vector<KeyPoint> harris_corners(
    const cv::Mat& gray,
    double k_param = 0.04,
    double threshold = 1e6,
    int nms_radius = 5,
    int gauss_ksize = 5,
    double gauss_sigma = 1.0)
{
    int rows = gray.rows, cols = gray.cols;

    // 1. Smooth the image
    cv::Mat blurred = gaussian_blur(gray, gauss_ksize, gauss_sigma);

    // 2. Compute gradients
    cv::Mat Ix = sobel(blurred, 1, 0);
    cv::Mat Iy = sobel(blurred, 0, 1);

    // 3. Compute products
    cv::Mat Ix2(rows, cols, CV_64FC1);
    cv::Mat Iy2(rows, cols, CV_64FC1);
    cv::Mat IxIy(rows, cols, CV_64FC1);

    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x) {
            double ix = Ix.at<double>(y, x);
            double iy = Iy.at<double>(y, x);
            Ix2.at<double>(y, x) = ix * ix;
            Iy2.at<double>(y, x) = iy * iy;
            IxIy.at<double>(y, x) = ix * iy;
        }

    // 4. Apply Gaussian window to each product
    cv::Mat Sx2 = gaussian_blur(Ix2, gauss_ksize, gauss_sigma);
    cv::Mat Sy2 = gaussian_blur(Iy2, gauss_ksize, gauss_sigma);
    cv::Mat Sxy = gaussian_blur(IxIy, gauss_ksize, gauss_sigma);

    // 5. Compute Harris response R = det(M) - k·trace(M)²
    cv::Mat R(rows, cols, CV_64FC1, cv::Scalar(0));
    double maxR = 0;

    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x) {
            double a = Sx2.at<double>(y, x);
            double b = Sxy.at<double>(y, x);
            double c = Sy2.at<double>(y, x);
            double det = a * c - b * b;
            double trace = a + c;
            double r = det - k_param * trace * trace;
            R.at<double>(y, x) = r;
            if (r > maxR) maxR = r;
        }

    // Adaptive threshold: use percentage of max response
    double adaptiveThresh = maxR * threshold / 1e8;

    // 6. Non-maximum suppression + threshold
    std::vector<KeyPoint> corners;
    int margin = std::max(nms_radius, 2);

    for (int y = margin; y < rows - margin; ++y)
        for (int x = margin; x < cols - margin; ++x) {
            double r = R.at<double>(y, x);
            if (r < adaptiveThresh) continue;

            bool is_max = true;
            for (int dy = -nms_radius; dy <= nms_radius && is_max; ++dy)
                for (int dx = -nms_radius; dx <= nms_radius && is_max; ++dx) {
                    if (dy == 0 && dx == 0) continue;
                    if (R.at<double>(y + dy, x + dx) > r) is_max = false;
                }
            if (is_max) {
                corners.push_back({(double)x, (double)y, 1.0, 0.0, r, 0});
            }
        }

    // Sort by response (strongest first)
    std::sort(corners.begin(), corners.end(),
              [](const KeyPoint& a, const KeyPoint& b) { return a.response > b.response; });

    return corners;
}

/* ═══════════════════════════════════════════════════════════════════
 *  2. LAMBDA-MINUS (λ⁻) CORNER DETECTION
 *
 *  Uses the smallest eigenvalue of the structure tensor M:
 *    λ⁻ = (trace - sqrt(trace² - 4·det)) / 2
 *  Corners where λ⁻ > threshold
 * ═══════════════════════════════════════════════════════════════════ */

inline std::vector<KeyPoint> lambda_minus_corners(
    const cv::Mat& gray,
    double threshold = 1e4,
    int nms_radius = 5,
    int gauss_ksize = 5,
    double gauss_sigma = 1.0)
{
    int rows = gray.rows, cols = gray.cols;

    cv::Mat blurred = gaussian_blur(gray, gauss_ksize, gauss_sigma);
    cv::Mat Ix = sobel(blurred, 1, 0);
    cv::Mat Iy = sobel(blurred, 0, 1);

    cv::Mat Ix2(rows, cols, CV_64FC1);
    cv::Mat Iy2(rows, cols, CV_64FC1);
    cv::Mat IxIy(rows, cols, CV_64FC1);

    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x) {
            double ix = Ix.at<double>(y, x);
            double iy = Iy.at<double>(y, x);
            Ix2.at<double>(y, x) = ix * ix;
            Iy2.at<double>(y, x) = iy * iy;
            IxIy.at<double>(y, x) = ix * iy;
        }

    cv::Mat Sx2 = gaussian_blur(Ix2, gauss_ksize, gauss_sigma);
    cv::Mat Sy2 = gaussian_blur(Iy2, gauss_ksize, gauss_sigma);
    cv::Mat Sxy = gaussian_blur(IxIy, gauss_ksize, gauss_sigma);

    // Compute λ⁻ for each pixel
    cv::Mat Lambda(rows, cols, CV_64FC1, cv::Scalar(0));
    double maxL = 0;

    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x) {
            double a = Sx2.at<double>(y, x);
            double b = Sxy.at<double>(y, x);
            double c = Sy2.at<double>(y, x);
            double trace = a + c;
            double det = a * c - b * b;
            double disc = trace * trace - 4.0 * det;
            if (disc < 0) disc = 0;
            double lambda_min = (trace - std::sqrt(disc)) / 2.0;
            Lambda.at<double>(y, x) = lambda_min;
            if (lambda_min > maxL) maxL = lambda_min;
        }

    // Adaptive threshold
    double adaptiveThresh = maxL * threshold / 1e6;

    // NMS + threshold
    std::vector<KeyPoint> corners;
    int margin = std::max(nms_radius, 2);

    for (int y = margin; y < rows - margin; ++y)
        for (int x = margin; x < cols - margin; ++x) {
            double lm = Lambda.at<double>(y, x);
            if (lm < adaptiveThresh) continue;

            bool is_max = true;
            for (int dy = -nms_radius; dy <= nms_radius && is_max; ++dy)
                for (int dx = -nms_radius; dx <= nms_radius && is_max; ++dx) {
                    if (dy == 0 && dx == 0) continue;
                    if (Lambda.at<double>(y + dy, x + dx) > lm) is_max = false;
                }
            if (is_max) {
                corners.push_back({(double)x, (double)y, 1.0, 0.0, lm, 0});
            }
        }

    std::sort(corners.begin(), corners.end(),
              [](const KeyPoint& a, const KeyPoint& b) { return a.response > b.response; });

    return corners;
}

/* ═══════════════════════════════════════════════════════════════════
 *  3. SIFT — Scale Invariant Feature Transform (from scratch)
 *
 *  Pipeline:
 *    a) Build Gaussian scale space (octaves × scales)
 *    b) Compute Difference of Gaussians (DoG)
 *    c) Detect keypoints as DoG extrema
 *    d) Reject low-contrast and edge responses
 *    e) Assign dominant orientation
 *    f) Build 4×4×8 = 128-dim descriptor
 * ═══════════════════════════════════════════════════════════════════ */

// SIFT parameters
struct SIFTParams {
    int num_octaves = 4;
    int scales_per_octave = 3;   // s
    double sigma_base = 1.6;
    double contrast_threshold = 0.03;
    double edge_threshold = 10.0;
    int descriptor_width = 4;    // 4×4 grid
    int descriptor_bins = 8;     // 8 orientation bins
};

// Build Gaussian scale space for one octave
inline std::vector<cv::Mat> build_gaussian_octave(
    const cv::Mat& base, int num_scales, double sigma_base)
{
    // Total images per octave = s + 3 (for DoG we need s+2 DoG images)
    int total = num_scales + 3;
    std::vector<cv::Mat> octave(total);

    // Compute sigma for each scale
    double k = std::pow(2.0, 1.0 / num_scales);
    std::vector<double> sigmas(total);
    sigmas[0] = sigma_base;
    for (int i = 1; i < total; ++i) {
        double prev = sigmas[i-1];
        double curr = prev * k;
        // Incremental sigma: blur prev by delta to get curr
        double delta = std::sqrt(curr * curr - prev * prev);
        sigmas[i] = curr;
        (void)delta; // stored for reference
    }

    // First scale is the base image (already at sigma_base)
    octave[0] = base.clone();

    // Apply incremental Gaussian blur for each subsequent scale
    for (int i = 1; i < total; ++i) {
        double prev_sigma = sigma_base * std::pow(k, i - 1);
        double curr_sigma = sigma_base * std::pow(k, i);
        double delta = std::sqrt(curr_sigma * curr_sigma - prev_sigma * prev_sigma);
        int ksize = std::max(3, (int)(6 * delta + 1) | 1); // ensure odd
        octave[i] = gaussian_blur(octave[i-1], ksize, delta);
    }

    return octave;
}

// Compute DoG from Gaussian octave
inline std::vector<cv::Mat> compute_dog(const std::vector<cv::Mat>& gauss_octave) {
    std::vector<cv::Mat> dogs;
    for (size_t i = 0; i + 1 < gauss_octave.size(); ++i) {
        cv::Mat dog(gauss_octave[i].rows, gauss_octave[i].cols, CV_64FC1);
        for (int y = 0; y < dog.rows; ++y)
            for (int x = 0; x < dog.cols; ++x)
                dog.at<double>(y, x) = gauss_octave[i+1].at<double>(y, x)
                                     - gauss_octave[i].at<double>(y, x);
        dogs.push_back(dog);
    }
    return dogs;
}

// Detect extrema in DoG
inline std::vector<KeyPoint> detect_dog_extrema(
    const std::vector<cv::Mat>& dogs,
    int octave_idx,
    double contrast_thresh,
    double edge_thresh,
    double sigma_base,
    int scales_per_octave)
{
    std::vector<KeyPoint> keypoints;
    int rows = dogs[0].rows, cols = dogs[0].cols;

    // Check middle DoG layers (1 to n-2)
    for (int s = 1; s + 1 < (int)dogs.size(); ++s) {
        for (int y = 2; y < rows - 2; ++y) {
            for (int x = 2; x < cols - 2; ++x) {
                double val = dogs[s].at<double>(y, x);

                // Skip low contrast
                if (std::abs(val) < contrast_thresh * 0.5) continue;

                // Check 26 neighbors (3×3×3 - 1)
                bool is_max = true, is_min = true;
                for (int ds = -1; ds <= 1 && (is_max || is_min); ++ds)
                    for (int dy = -1; dy <= 1 && (is_max || is_min); ++dy)
                        for (int dx = -1; dx <= 1 && (is_max || is_min); ++dx) {
                            if (ds == 0 && dy == 0 && dx == 0) continue;
                            double n = dogs[s + ds].at<double>(y + dy, x + dx);
                            if (n >= val) is_max = false;
                            if (n <= val) is_min = false;
                        }

                if (!is_max && !is_min) continue;

                // Edge response rejection using Hessian
                double dxx = dogs[s].at<double>(y, x+1) + dogs[s].at<double>(y, x-1)
                           - 2.0 * val;
                double dyy = dogs[s].at<double>(y+1, x) + dogs[s].at<double>(y-1, x)
                           - 2.0 * val;
                double dxy = (dogs[s].at<double>(y+1, x+1) - dogs[s].at<double>(y+1, x-1)
                            - dogs[s].at<double>(y-1, x+1) + dogs[s].at<double>(y-1, x-1))
                           * 0.25;

                double tr = dxx + dyy;
                double det = dxx * dyy - dxy * dxy;
                if (det <= 0) continue;

                double ratio = tr * tr / det;
                double threshold = (edge_thresh + 1.0) * (edge_thresh + 1.0) / edge_thresh;
                if (ratio > threshold) continue;

                // Compute scale of this keypoint
                double k = std::pow(2.0, 1.0 / scales_per_octave);
                double scale = sigma_base * std::pow(k, s) * std::pow(2.0, octave_idx);

                // Store in original image coordinates
                double factor = std::pow(2.0, octave_idx);
                KeyPoint kp;
                kp.x = x * factor;
                kp.y = y * factor;
                kp.scale = scale;
                kp.response = std::abs(val);
                kp.orientation = 0;
                kp.octave = octave_idx;
                keypoints.push_back(kp);
            }
        }
    }

    return keypoints;
}

// Assign dominant orientation to each keypoint
inline void assign_orientations(
    std::vector<KeyPoint>& keypoints,
    const std::vector<std::vector<cv::Mat>>& gauss_pyramid,
    int scales_per_octave)
{
    const int num_bins = 36;
    const double bin_width = 2.0 * PI / num_bins;

    for (auto& kp : keypoints) {
        int oct = kp.octave;
        if (oct < 0 || oct >= (int)gauss_pyramid.size()) continue;

        // Find the closest scale in this octave
        double factor = std::pow(2.0, oct);
        int lx = (int)(kp.x / factor + 0.5);
        int ly = (int)(kp.y / factor + 0.5);

        const auto& octave = gauss_pyramid[oct];
        // Use scale index 1 (safe middle)
        int si = std::min(1, (int)octave.size() - 1);
        const cv::Mat& img = octave[si];

        int rows = img.rows, cols = img.cols;
        if (lx < 1 || lx >= cols-1 || ly < 1 || ly >= rows-1) continue;

        double sigma_w = 1.5 * kp.scale / factor;
        int radius = (int)(3.0 * sigma_w + 0.5);
        radius = std::max(1, std::min(radius, std::min(rows, cols) / 2 - 1));

        std::vector<double> hist(num_bins, 0.0);

        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                int ny = ly + dy, nx = lx + dx;
                if (ny < 1 || ny >= rows-1 || nx < 1 || nx >= cols-1) continue;

                double gx = img.at<double>(ny, nx+1) - img.at<double>(ny, nx-1);
                double gy = img.at<double>(ny+1, nx) - img.at<double>(ny-1, nx);
                double mag = std::sqrt(gx*gx + gy*gy);
                double ori = std::atan2(gy, gx);
                if (ori < 0) ori += 2.0 * PI;

                double weight = std::exp(-(dx*dx + dy*dy) / (2.0 * sigma_w * sigma_w));
                int bin = (int)(ori / bin_width) % num_bins;
                hist[bin] += mag * weight;
            }
        }

        // Find peak bin
        int peak_bin = 0;
        for (int i = 1; i < num_bins; ++i)
            if (hist[i] > hist[peak_bin]) peak_bin = i;

        kp.orientation = (peak_bin + 0.5) * bin_width;
    }
}

// Build 128-dim SIFT descriptor for a keypoint
inline std::vector<double> build_sift_descriptor(
    const cv::Mat& gauss_img,
    const KeyPoint& kp,
    double factor,
    int d = 4,   // 4×4 grid
    int n = 8)   // 8 orientation bins
{
    int desc_size = d * d * n;  // 128
    std::vector<double> desc(desc_size, 0.0);

    int rows = gauss_img.rows, cols = gauss_img.cols;
    int lx = (int)(kp.x / factor + 0.5);
    int ly = (int)(kp.y / factor + 0.5);

    double cos_t = std::cos(-kp.orientation);
    double sin_t = std::sin(-kp.orientation);
    double sigma_desc = kp.scale / factor;
    double bins_per_rad = n / (2.0 * PI);

    // Descriptor window radius
    int radius = (int)(sigma_desc * std::sqrt(2.0) * (d + 1) * 0.5 + 0.5);
    radius = std::min(radius, std::min(rows, cols) / 2 - 1);
    radius = std::max(radius, 1);

    double hist_width = sigma_desc * 2.0;
    if (hist_width < 1.0) hist_width = 1.0;

    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            int ny = ly + dy, nx = lx + dx;
            if (ny < 1 || ny >= rows-1 || nx < 1 || nx >= cols-1) continue;

            // Rotate relative to keypoint orientation
            double rx = (cos_t * dx - sin_t * dy) / hist_width;
            double ry = (sin_t * dx + cos_t * dy) / hist_width;

            // Bin indices in 4×4 grid (centered)
            double rbin = ry + d/2.0 - 0.5;
            double cbin = rx + d/2.0 - 0.5;

            if (rbin < -1 || rbin >= d || cbin < -1 || cbin >= d) continue;

            // Gradient magnitude and orientation
            double gx = gauss_img.at<double>(ny, nx+1) - gauss_img.at<double>(ny, nx-1);
            double gy = gauss_img.at<double>(ny+1, nx) - gauss_img.at<double>(ny-1, nx);
            double mag = std::sqrt(gx*gx + gy*gy);
            double ori = std::atan2(gy, gx) - kp.orientation;
            while (ori < 0) ori += 2.0 * PI;
            while (ori >= 2.0 * PI) ori -= 2.0 * PI;

            double weight = std::exp(-(rx*rx + ry*ry) / (2.0 * (d/2.0) * (d/2.0)));
            mag *= weight;

            double obin = ori * bins_per_rad;

            // Trilinear interpolation into descriptor bins
            int r0 = (int)std::floor(rbin);
            int c0 = (int)std::floor(cbin);
            int o0 = (int)std::floor(obin);
            double dr = rbin - r0, dc = cbin - c0, dori = obin - o0;

            for (int ir = 0; ir <= 1; ++ir) {
                int r = r0 + ir;
                if (r < 0 || r >= d) continue;
                double wr = (ir == 0) ? (1.0 - dr) : dr;

                for (int ic = 0; ic <= 1; ++ic) {
                    int c = c0 + ic;
                    if (c < 0 || c >= d) continue;
                    double wc = (ic == 0) ? (1.0 - dc) : dc;

                    for (int io = 0; io <= 1; ++io) {
                        int o = (o0 + io) % n;
                        double wo = (io == 0) ? (1.0 - dori) : dori;

                        int idx = (r * d + c) * n + o;
                        desc[idx] += mag * wr * wc * wo;
                    }
                }
            }
        }
    }

    // Normalize descriptor
    double norm = 0;
    for (double v : desc) norm += v * v;
    norm = std::sqrt(norm) + 1e-7;
    for (double& v : desc) v /= norm;

    // Clamp values > 0.2 and renormalize
    for (double& v : desc) v = std::min(v, 0.2);
    norm = 0;
    for (double v : desc) norm += v * v;
    norm = std::sqrt(norm) + 1e-7;
    for (double& v : desc) v /= norm;

    return desc;
}

// Full SIFT pipeline
inline void sift_detect_and_describe(
    const cv::Mat& gray,
    std::vector<KeyPoint>& keypoints,
    std::vector<std::vector<double>>& descriptors,
    SIFTParams params = SIFTParams())
{
    int rows = gray.rows, cols = gray.cols;

    // Convert to double
    cv::Mat img_f64(rows, cols, CV_64FC1);
    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x)
            img_f64.at<double>(y, x) = get_pix(gray, y, x) / 255.0;

    // Determine number of octaves
    int min_dim = std::min(rows, cols);
    int num_octaves = std::min(params.num_octaves,
                               (int)std::log2(min_dim / 4.0));
    num_octaves = std::max(num_octaves, 1);

    // Build scale space
    std::vector<std::vector<cv::Mat>> gauss_pyramid(num_octaves);
    std::vector<std::vector<cv::Mat>> dog_pyramid(num_octaves);

    cv::Mat current = img_f64.clone();
    // Initial blur to sigma_base
    int init_ksize = std::max(3, (int)(6 * params.sigma_base + 1) | 1);
    current = gaussian_blur(current, init_ksize, params.sigma_base);

    for (int o = 0; o < num_octaves; ++o) {
        gauss_pyramid[o] = build_gaussian_octave(
            current, params.scales_per_octave, params.sigma_base);

        dog_pyramid[o] = compute_dog(gauss_pyramid[o]);

        // Detect extrema
        auto kps = detect_dog_extrema(
            dog_pyramid[o], o,
            params.contrast_threshold, params.edge_threshold,
            params.sigma_base, params.scales_per_octave);

        keypoints.insert(keypoints.end(), kps.begin(), kps.end());

        // Downsample for next octave (take scale s from Gaussian, which is at 2σ)
        int next_idx = params.scales_per_octave;
        if (next_idx < (int)gauss_pyramid[o].size()) {
            const cv::Mat& src = gauss_pyramid[o][next_idx];
            current = resize_image(src, src.rows / 2, src.cols / 2);
        }
    }

    // Assign orientations
    assign_orientations(keypoints, gauss_pyramid, params.scales_per_octave);

    // Build descriptors
    descriptors.resize(keypoints.size());
    for (size_t i = 0; i < keypoints.size(); ++i) {
        int oct = keypoints[i].octave;
        if (oct < 0 || oct >= num_octaves) continue;

        double factor = std::pow(2.0, oct);
        // Use scale 1 from the octave
        int si = std::min(1, (int)gauss_pyramid[oct].size() - 1);
        descriptors[i] = build_sift_descriptor(
            gauss_pyramid[oct][si],
            keypoints[i],
            factor,
            params.descriptor_width,
            params.descriptor_bins);
    }

    // Remove keypoints with empty descriptors
    std::vector<KeyPoint> valid_kps;
    std::vector<std::vector<double>> valid_descs;
    for (size_t i = 0; i < keypoints.size(); ++i) {
        if (!descriptors[i].empty()) {
            // Check descriptor is not all zeros
            double sum = 0;
            for (double v : descriptors[i]) sum += std::abs(v);
            if (sum > 1e-6) {
                valid_kps.push_back(keypoints[i]);
                valid_descs.push_back(descriptors[i]);
            }
        }
    }
    keypoints = valid_kps;
    descriptors = valid_descs;
}

/* ═══════════════════════════════════════════════════════════════════
 *  4. FEATURE MATCHING — SSD (Sum of Squared Differences)
 *
 *  For each descriptor in set 1, find nearest in set 2.
 *  Apply Lowe's ratio test (best/second-best < 0.75)
 * ═══════════════════════════════════════════════════════════════════ */

inline double compute_ssd(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

inline std::vector<MatchResult> match_ssd(
    const std::vector<std::vector<double>>& desc1,
    const std::vector<std::vector<double>>& desc2,
    double ratio_threshold = 0.75)
{
    std::vector<MatchResult> matches;

    for (int i = 0; i < (int)desc1.size(); ++i) {
        double best = 1e30, second_best = 1e30;
        int best_idx = -1;

        for (int j = 0; j < (int)desc2.size(); ++j) {
            double dist = compute_ssd(desc1[i], desc2[j]);
            if (dist < best) {
                second_best = best;
                best = dist;
                best_idx = j;
            } else if (dist < second_best) {
                second_best = dist;
            }
        }

        // Ratio test
        if (best_idx >= 0 && second_best > 1e-10) {
            double ratio = std::sqrt(best) / std::sqrt(second_best);
            if (ratio < ratio_threshold) {
                matches.push_back({i, best_idx, best});
            }
        }
    }

    // Sort by distance (best matches first)
    std::sort(matches.begin(), matches.end(),
              [](const MatchResult& a, const MatchResult& b) { return a.distance < b.distance; });

    return matches;
}

/* ═══════════════════════════════════════════════════════════════════
 *  5. FEATURE MATCHING — NCC (Normalized Cross Correlation)
 *
 *  NCC = (sum(a_i * b_i)) / (||a|| * ||b||)
 *  High NCC (close to 1) = good match
 * ═══════════════════════════════════════════════════════════════════ */

inline double compute_ncc(const std::vector<double>& a, const std::vector<double>& b) {
    double sum_ab = 0, sum_a2 = 0, sum_b2 = 0;
    double mean_a = 0, mean_b = 0;
    int n = (int)std::min(a.size(), b.size());

    for (int i = 0; i < n; ++i) { mean_a += a[i]; mean_b += b[i]; }
    mean_a /= n; mean_b /= n;

    for (int i = 0; i < n; ++i) {
        double da = a[i] - mean_a;
        double db = b[i] - mean_b;
        sum_ab += da * db;
        sum_a2 += da * da;
        sum_b2 += db * db;
    }

    double denom = std::sqrt(sum_a2 * sum_b2);
    if (denom < 1e-10) return 0;
    return sum_ab / denom;
}

inline std::vector<MatchResult> match_ncc(
    const std::vector<std::vector<double>>& desc1,
    const std::vector<std::vector<double>>& desc2,
    double ncc_threshold = 0.7)
{
    std::vector<MatchResult> matches;

    for (int i = 0; i < (int)desc1.size(); ++i) {
        double best_ncc = -2.0, second_best_ncc = -2.0;
        int best_idx = -1;

        for (int j = 0; j < (int)desc2.size(); ++j) {
            double ncc = compute_ncc(desc1[i], desc2[j]);
            if (ncc > best_ncc) {
                second_best_ncc = best_ncc;
                best_ncc = ncc;
                best_idx = j;
            } else if (ncc > second_best_ncc) {
                second_best_ncc = ncc;
            }
        }

        // Threshold test: correlation must be high enough
        if (best_idx >= 0 && best_ncc > ncc_threshold) {
            // Ratio test: best should be significantly better than second
            if (second_best_ncc < 0.9 * best_ncc) {
                matches.push_back({i, best_idx, 1.0 - best_ncc});
            }
        }
    }

    // Sort by correlation (best = lowest distance = highest NCC)
    std::sort(matches.begin(), matches.end(),
              [](const MatchResult& a, const MatchResult& b) { return a.distance < b.distance; });

    return matches;
}

/* ═══════════════════════════════════════════════════════════════════
 *  DRAWING PRIMITIVES (kept from CV_02)
 * ═══════════════════════════════════════════════════════════════════ */

inline void draw_line(cv::Mat& img, cv::Point p1, cv::Point p2,
                      const cv::Scalar& col, int thick)
{
    int x0=p1.x,y0=p1.y,x1=p2.x,y1=p2.y;
    int dx=std::abs(x1-x0), dy=std::abs(y1-y0);
    int sx=(x0<x1)?1:-1, sy=(y0<y1)?1:-1, err=dx-dy;
    int r=std::max(1,thick/2);
    while (true) {
        for (int iy=-r+1;iy<r;++iy)
            for (int ix=-r+1;ix<r;++ix)
                if (ix*ix+iy*iy<r*r)
                    set_pixel_color(img,x0+ix,y0+iy,col);
        if (x0==x1&&y0==y1) break;
        int e2=2*err;
        if (e2>-dy){err-=dy;x0+=sx;}
        if (e2<dx){err+=dx;y0+=sy;}
    }
}

inline void draw_circle(cv::Mat& img, cv::Point cen, int rad,
                        const cv::Scalar& col, int thick)
{
    if (thick < 0) { // filled
        for (int y=-rad;y<=rad;++y)
            for (int x=-rad;x<=rad;++x)
                if (x*x+y*y<=rad*rad)
                    set_pixel_color(img,cen.x+x,cen.y+y,col);
        return;
    }
    int ri=std::max(0,rad-thick/2), ro=rad+(thick+1)/2;
    int ri2=ri*ri, ro2=ro*ro;
    for (int y=-ro;y<=ro;++y)
        for (int x=-ro;x<=ro;++x) {
            int d2=x*x+y*y;
            if (d2>=ri2&&d2<=ro2)
                set_pixel_color(img,cen.x+x,cen.y+y,col);
        }
}

inline void draw_cross(cv::Mat& img, cv::Point cen, int size,
                       const cv::Scalar& col, int thick)
{
    draw_line(img, {cen.x - size, cen.y}, {cen.x + size, cen.y}, col, thick);
    draw_line(img, {cen.x, cen.y - size}, {cen.x, cen.y + size}, col, thick);
}

/* ─── Draw matches between two images side by side ─────────────── */

inline cv::Mat draw_matches(
    const cv::Mat& img1, const std::vector<KeyPoint>& kps1,
    const cv::Mat& img2, const std::vector<KeyPoint>& kps2,
    const std::vector<MatchResult>& matches,
    int max_matches = 100)
{
    int target_rows = std::min(img1.rows, img2.rows);
    int target_cols = std::min(img1.cols, img2.cols);
    int rows = target_rows;
    int cols = target_cols * 2;

    // Create side-by-side canvas (3-channel color)
    cv::Mat canvas(rows, cols, CV_8UC3, cv::Scalar(30, 30, 30));

    auto sample_bgr = [](const cv::Mat& src, int sy, int sx) {
        if (src.channels() == 3) return src.at<cv::Vec3b>(sy, sx);
        uchar v = src.at<uchar>(sy, sx);
        return cv::Vec3b(v, v, v);
    };

    auto copy_normalized = [&](const cv::Mat& src, int dst_offset_x) {
        double x_scale = (src.cols > 1 && target_cols > 1)
            ? (double)(src.cols - 1) / (target_cols - 1)
            : 0.0;
        double y_scale = (src.rows > 1 && target_rows > 1)
            ? (double)(src.rows - 1) / (target_rows - 1)
            : 0.0;

        for (int y = 0; y < target_rows; ++y) {
            int sy = (int)std::round(y * y_scale);
            sy = std::clamp(sy, 0, src.rows - 1);
            for (int x = 0; x < target_cols; ++x) {
                int sx = (int)std::round(x * x_scale);
                sx = std::clamp(sx, 0, src.cols - 1);
                canvas.at<cv::Vec3b>(y, x + dst_offset_x) = sample_bgr(src, sy, sx);
            }
        }
    };

    copy_normalized(img1, 0);
    int offset_x = target_cols;
    copy_normalized(img2, offset_x);

    double kp1_sx = (img1.cols > 1 && target_cols > 1)
        ? (double)(target_cols - 1) / (img1.cols - 1)
        : 0.0;
    double kp1_sy = (img1.rows > 1 && target_rows > 1)
        ? (double)(target_rows - 1) / (img1.rows - 1)
        : 0.0;
    double kp2_sx = (img2.cols > 1 && target_cols > 1)
        ? (double)(target_cols - 1) / (img2.cols - 1)
        : 0.0;
    double kp2_sy = (img2.rows > 1 && target_rows > 1)
        ? (double)(target_rows - 1) / (img2.rows - 1)
        : 0.0;

    auto map_x = [&](double x, double s, int src_cols) {
        if (src_cols <= 1 || target_cols <= 1) return 0;
        int v = (int)std::lround(x * s);
        return std::clamp(v, 0, target_cols - 1);
    };

    auto map_y = [&](double y, double s, int src_rows) {
        if (src_rows <= 1 || target_rows <= 1) return 0;
        int v = (int)std::lround(y * s);
        return std::clamp(v, 0, target_rows - 1);
    };

    // Draw keypoints as small circles
    for (auto& kp : kps1) {
        int x = map_x(kp.x, kp1_sx, img1.cols);
        int y = map_y(kp.y, kp1_sy, img1.rows);
        draw_circle(canvas, {x, y}, 3,
                    cv::Scalar(0, 255, 0), 1);
    }
    for (auto& kp : kps2) {
        int x = map_x(kp.x, kp2_sx, img2.cols);
        int y = map_y(kp.y, kp2_sy, img2.rows);
        draw_circle(canvas, {x + offset_x, y}, 3,
                    cv::Scalar(0, 255, 0), 1);
    }

    // Draw match lines with different colors
    int count = std::min((int)matches.size(), max_matches);
    for (int i = 0; i < count; ++i) {
        auto& m = matches[i];
        if (m.idx1 >= (int)kps1.size() || m.idx2 >= (int)kps2.size()) continue;

        // Generate distinct color for each match
        int hue = (int)(360.0 * i / count);
        // Simple HSV to RGB (hue in degrees, S=V=1)
        double h = hue / 60.0;
        int hi = (int)h % 6;
        double f = h - (int)h;
        double q = 1.0 - f;
        double r = 0, g = 0, b = 0;
        switch(hi) {
            case 0: r=1; g=f; b=0; break;
            case 1: r=q; g=1; b=0; break;
            case 2: r=0; g=1; b=f; break;
            case 3: r=0; g=q; b=1; break;
            case 4: r=f; g=0; b=1; break;
            case 5: r=1; g=0; b=q; break;
        }
        cv::Scalar color((int)(b*255), (int)(g*255), (int)(r*255));

        cv::Point p1(
            map_x(kps1[m.idx1].x, kp1_sx, img1.cols),
            map_y(kps1[m.idx1].y, kp1_sy, img1.rows));
        cv::Point p2(
            map_x(kps2[m.idx2].x, kp2_sx, img2.cols) + offset_x,
            map_y(kps2[m.idx2].y, kp2_sy, img2.rows));
        draw_line(canvas, p1, p2, color, 1);
        draw_circle(canvas, p1, 4, color, -1);
        draw_circle(canvas, p2, 4, color, -1);
    }

    return canvas;
}

/* ─── Draw keypoints on image ──────────────────────────────────── */

inline cv::Mat draw_keypoints(
    const cv::Mat& img,
    const std::vector<KeyPoint>& keypoints,
    const cv::Scalar& color = cv::Scalar(0, 0, 255),
    bool draw_orientation = false)
{
    cv::Mat canvas;
    if (img.channels() == 1) {
        canvas = cv::Mat(img.rows, img.cols, CV_8UC3);
        for (int y = 0; y < img.rows; ++y)
            for (int x = 0; x < img.cols; ++x) {
                uchar v = img.at<uchar>(y, x);
                canvas.at<cv::Vec3b>(y, x) = cv::Vec3b(v, v, v);
            }
    } else {
        canvas = img.clone();
    }

    for (auto& kp : keypoints) {
        int x = (int)kp.x, y = (int)kp.y;
        int r = std::max(2, (int)(kp.scale * 2));
        r = std::min(r, 20);

        draw_circle(canvas, {x, y}, r, color, 1);
        draw_circle(canvas, {x, y}, 1, color, -1);

        if (draw_orientation) {
            int ex = x + (int)(r * std::cos(kp.orientation));
            int ey = y + (int)(r * std::sin(kp.orientation));
            draw_line(canvas, {x, y}, {ex, ey}, color, 1);
        }
    }

    return canvas;
}

} // namespace custom
