/**
 * cv_core.cpp — pybind11 module for CV_04
 *
 * Assignment 4: Image Thresholding & Unsupervised Segmentation
 * All algorithms implemented from scratch in header files.
 * OpenCV used ONLY for cv::Mat, cv::imread, cv::imencode.
 *
 * Results returned as base64-encoded PNG (no intermediate disk writes).
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>
#include <vector>
#include <chrono>
#include <stdexcept>
#include <sstream>

#include "cv_utils.h"
#include "cv_thresholding.h"
#include "cv_segmentation.h"

namespace py = pybind11;

/* ─── Helpers ─── */
namespace {

struct Timer {
    using clock = std::chrono::high_resolution_clock;
    clock::time_point start;
    Timer() : start(clock::now()) {}
    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(clock::now() - start).count();
    }
};

cv::Mat load_image(const std::string& path, const std::string& mode) {
    int flag = cv::IMREAD_COLOR;
    if (mode == "gray") flag = cv::IMREAD_GRAYSCALE;
    else if (mode == "unchanged") flag = cv::IMREAD_UNCHANGED;
    cv::Mat img = cv::imread(path, flag);
    if (img.empty())
        throw std::runtime_error("Could not read image: " + path);
    return img;
}

// Encode cv::Mat to base64 PNG string (no disk write)
std::string mat_to_base64(const cv::Mat& img) {
    std::vector<uchar> buf;
    cv::imencode(".png", img, buf);
    return cv4::base64_encode(buf);
}

// Encode histogram data as comma-separated string
std::string hist_to_string(const std::vector<int>& hist) {
    std::string s;
    for (int i = 0; i < (int)hist.size(); ++i) {
        if (i > 0) s += ",";
        s += std::to_string(hist[i]);
    }
    return s;
}

} // anonymous namespace

/* ═══════════════════════════════════════════════════════════════
 *  THRESHOLDING ENDPOINTS
 * ═══════════════════════════════════════════════════════════════ */

py::dict do_optimal_threshold(const std::string& path) {
    Timer t;
    cv::Mat color = load_image(path, "color");
    cv::Mat gray = cv4::to_grayscale(color);
    auto hist = cv4::compute_histogram(gray);
    auto res = cv4::optimal_threshold(gray);
    double ms = t.elapsed_ms();

    py::dict out;
    out["image_b64"] = mat_to_base64(res.output);
    out["threshold"] = res.threshold;
    out["iterations"] = res.iterations;
    out["time_ms"] = ms;
    out["histogram"] = hist_to_string(hist);
    return out;
}

py::dict do_otsu_threshold(const std::string& path) {
    Timer t;
    cv::Mat color = load_image(path, "color");
    cv::Mat gray = cv4::to_grayscale(color);
    auto hist = cv4::compute_histogram(gray);
    auto res = cv4::otsu_threshold(gray);
    double ms = t.elapsed_ms();

    py::dict out;
    out["image_b64"] = mat_to_base64(res.output);
    out["threshold"] = res.threshold;
    out["time_ms"] = ms;
    out["histogram"] = hist_to_string(hist);
    return out;
}

py::dict do_spectral_threshold(const std::string& path, int num_classes) {
    Timer t;
    cv::Mat color = load_image(path, "color");
    cv::Mat gray = cv4::to_grayscale(color);
    auto hist = cv4::compute_histogram(gray);
    auto res = cv4::spectral_threshold(gray, num_classes);
    double ms = t.elapsed_ms();

    py::dict out;
    out["image_b64"] = mat_to_base64(res.output);
    out["num_classes"] = res.num_classes;
    out["time_ms"] = ms;
    out["histogram"] = hist_to_string(hist);

    // Return thresholds as a list
    py::list thr_list;
    for (int th : res.thresholds) thr_list.append(th);
    out["thresholds"] = thr_list;
    return out;
}

py::dict do_local_threshold(const std::string& path, int block_size) {
    Timer t;
    cv::Mat color = load_image(path, "color");
    cv::Mat gray = cv4::to_grayscale(color);
    auto hist = cv4::compute_histogram(gray);
    auto res = cv4::local_threshold(gray, block_size);
    double ms = t.elapsed_ms();

    py::dict out;
    out["image_b64"] = mat_to_base64(res.output);
    out["block_size"] = res.block_size;
    out["num_blocks"] = res.num_blocks;
    out["time_ms"] = ms;
    out["histogram"] = hist_to_string(hist);
    return out;
}

/* ═══════════════════════════════════════════════════════════════
 *  SEGMENTATION ENDPOINTS
 * ═══════════════════════════════════════════════════════════════ */

py::dict do_kmeans(const std::string& path, int k, int max_iter) {
    Timer t;
    cv::Mat img = load_image(path, "color");
    auto res = cv4::kmeans_segment(img, k, max_iter);
    double ms = t.elapsed_ms();

    py::dict out;
    out["image_b64"] = mat_to_base64(res.output);
    out["num_clusters"] = res.num_clusters;
    out["iterations"] = res.iterations;
    out["time_ms"] = ms;
    return out;
}

py::dict do_region_growing(const std::string& path,
                            const std::vector<std::pair<int,int>>& seeds,
                            double threshold) {
    Timer t;
    cv::Mat img = load_image(path, "color");
    auto res = cv4::region_growing(img, seeds, threshold);
    double ms = t.elapsed_ms();

    py::dict out;
    out["image_b64"] = mat_to_base64(res.output);
    out["num_regions"] = res.num_clusters;
    out["time_ms"] = ms;
    return out;
}

py::dict do_agglomerative(const std::string& path, int num_clusters) {
    Timer t;
    cv::Mat img = load_image(path, "color");
    auto res = cv4::agglomerative_segment(img, num_clusters);
    double ms = t.elapsed_ms();

    py::dict out;
    out["image_b64"] = mat_to_base64(res.output);
    out["num_clusters"] = res.num_clusters;
    out["iterations"] = res.iterations;
    out["time_ms"] = ms;
    return out;
}

py::dict do_mean_shift(const std::string& path,
                        double spatial_radius, double color_radius) {
    Timer t;
    cv::Mat img = load_image(path, "color");
    auto res = cv4::mean_shift_segment(img, spatial_radius, color_radius);
    double ms = t.elapsed_ms();

    py::dict out;
    out["image_b64"] = mat_to_base64(res.output);
    out["num_clusters"] = res.num_clusters;
    out["time_ms"] = ms;
    return out;
}

/* ═══════════════════════════════════════════════════════════════
 *  SAVE ENDPOINT — the ONLY permitted disk write
 * ═══════════════════════════════════════════════════════════════ */

py::dict do_save_result(const std::string& path, const std::string& b64_data) {
    // Decode base64 back to bytes
    // (Simple decoder matching our encoder)
    auto b64_decode = [](const std::string& encoded) -> std::vector<uchar> {
        std::vector<int> T(256, -1);
        for (int i = 0; i < 64; ++i)
            T[(int)"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[i]] = i;

        std::vector<uchar> out;
        int val = 0, bits = -8;
        for (char c : encoded) {
            if (T[(uchar)c] == -1) continue;
            val = (val << 6) + T[(uchar)c];
            bits += 6;
            if (bits >= 0) {
                out.push_back((uchar)((val >> bits) & 0xFF));
                bits -= 8;
            }
        }
        return out;
    };

    std::vector<uchar> png_data = b64_decode(b64_data);
    cv::Mat img = cv::imdecode(png_data, cv::IMREAD_UNCHANGED);
    if (img.empty())
        throw std::runtime_error("Failed to decode image data");

    if (!cv::imwrite(path, img))
        throw std::runtime_error("Failed to write image: " + path);

    py::dict out;
    out["saved_path"] = path;
    out["size_bytes"] = (int)png_data.size();
    return out;
}

/* ═══════════════════════════════════════════════════════════════
 *  PYBIND11 MODULE DEFINITION
 * ═══════════════════════════════════════════════════════════════ */

PYBIND11_MODULE(cv_core, m) {
    m.doc() = "CV_04 — Image Thresholding & Unsupervised Segmentation (from scratch)";

    // Thresholding
    m.def("optimal_threshold", &do_optimal_threshold,
          "Optimal (iterative) global thresholding",
          py::arg("image_path"));

    m.def("otsu_threshold", &do_otsu_threshold,
          "Otsu's thresholding (maximize between-class variance)",
          py::arg("image_path"));

    m.def("spectral_threshold", &do_spectral_threshold,
          "Multi-level spectral thresholding (recursive Otsu)",
          py::arg("image_path"), py::arg("num_classes") = 3);

    m.def("local_threshold", &do_local_threshold,
          "Local adaptive thresholding (block-based Otsu)",
          py::arg("image_path"), py::arg("block_size") = 32);

    // Segmentation
    m.def("kmeans_segment", &do_kmeans,
          "K-means clustering segmentation",
          py::arg("image_path"), py::arg("k") = 4,
          py::arg("max_iter") = 50);

    m.def("region_growing", &do_region_growing,
          "Region growing segmentation from seed points",
          py::arg("image_path"), py::arg("seeds"),
          py::arg("threshold") = 15.0);

    m.def("agglomerative_segment", &do_agglomerative,
          "Agglomerative (bottom-up) clustering segmentation",
          py::arg("image_path"), py::arg("num_clusters") = 4);

    m.def("mean_shift_segment", &do_mean_shift,
          "Mean shift clustering segmentation",
          py::arg("image_path"),
          py::arg("spatial_radius") = 20.0,
          py::arg("color_radius") = 30.0);

    // Save
    m.def("save_result", &do_save_result,
          "Save a base64-encoded result image to disk",
          py::arg("output_path"), py::arg("base64_data"));
}
