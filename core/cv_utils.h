#pragma once
/**
 * cv_utils.h — Shared utility functions for CV_04
 * Pixel access, grayscale conversion, Gaussian blur, drawing primitives.
 * OpenCV used ONLY for cv::Mat types.
 */
#include <opencv2/core.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>

namespace cv4 {

constexpr double PI = 3.14159265358979323846;

/* ─── Pixel access with boundary clamping ─── */
inline double get_pix(const cv::Mat& m, int y, int x) {
    y = std::clamp(y, 0, m.rows - 1);
    x = std::clamp(x, 0, m.cols - 1);
    if (m.type() == CV_8UC1)  return m.at<uchar>(y, x);
    if (m.type() == CV_64FC1) return m.at<double>(y, x);
    if (m.type() == CV_32FC1) return m.at<float>(y, x);
    return 0;
}

inline void set_pixel_color(cv::Mat& img, int x, int y, const cv::Scalar& c) {
    if (x < 0 || x >= img.cols || y < 0 || y >= img.rows) return;
    if (img.channels() == 3)
        img.at<cv::Vec3b>(y, x) = cv::Vec3b((uchar)c[0], (uchar)c[1], (uchar)c[2]);
    else
        img.at<uchar>(y, x) = (uchar)c[0];
}

/* ─── RGB to Grayscale (luminosity formula) ─── */
inline cv::Mat to_grayscale(const cv::Mat& src) {
    if (src.channels() == 1) return src.clone();
    cv::Mat gray(src.rows, src.cols, CV_8UC1);
    for (int y = 0; y < src.rows; ++y)
        for (int x = 0; x < src.cols; ++x) {
            cv::Vec3b bgr = src.at<cv::Vec3b>(y, x);
            gray.at<uchar>(y, x) = (uchar)(0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2]);
        }
    return gray;
}

/* ─── Separable Gaussian Blur (from scratch) ─── */
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
    // Horizontal pass
    cv::Mat tmp(src.rows, src.cols, CV_64FC1, cv::Scalar(0));
    for (int y = 0; y < src.rows; ++y)
        for (int x = 0; x < src.cols; ++x) {
            double v = 0;
            for (int i = -half; i <= half; ++i)
                v += get_pix(src, y, std::clamp(x + i, 0, src.cols - 1)) * k[i + half];
            tmp.at<double>(y, x) = v;
        }
    // Vertical pass
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

/* ─── Drawing: Bresenham line ─── */
inline void draw_line(cv::Mat& img, cv::Point p1, cv::Point p2,
                      const cv::Scalar& col, int thick) {
    int x0=p1.x, y0=p1.y, x1=p2.x, y1=p2.y;
    int dx=std::abs(x1-x0), dy=std::abs(y1-y0);
    int sx=(x0<x1)?1:-1, sy=(y0<y1)?1:-1, err=dx-dy;
    int r=std::max(1, thick/2);
    while (true) {
        for (int iy=-r+1; iy<r; ++iy)
            for (int ix=-r+1; ix<r; ++ix)
                if (ix*ix+iy*iy < r*r)
                    set_pixel_color(img, x0+ix, y0+iy, col);
        if (x0==x1 && y0==y1) break;
        int e2=2*err;
        if (e2>-dy) { err-=dy; x0+=sx; }
        if (e2<dx)  { err+=dx; y0+=sy; }
    }
}

/* ─── Drawing: Circle (Midpoint algorithm) ─── */
inline void draw_circle(cv::Mat& img, cv::Point cen, int rad,
                        const cv::Scalar& col, int thick) {
    if (thick < 0) {
        for (int y=-rad; y<=rad; ++y)
            for (int x=-rad; x<=rad; ++x)
                if (x*x+y*y <= rad*rad)
                    set_pixel_color(img, cen.x+x, cen.y+y, col);
        return;
    }
    int ri=std::max(0, rad-thick/2), ro=rad+(thick+1)/2;
    for (int y=-ro; y<=ro; ++y)
        for (int x=-ro; x<=ro; ++x) {
            int d2=x*x+y*y;
            if (d2>=ri*ri && d2<=ro*ro)
                set_pixel_color(img, cen.x+x, cen.y+y, col);
        }
}

/* ─── Base64 encoding (for in-memory image transfer) ─── */
static const char b64_table[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

inline std::string base64_encode(const std::vector<uchar>& data) {
    std::string out;
    out.reserve(((data.size() + 2) / 3) * 4);
    int val = 0, bits = -6;
    for (uchar c : data) {
        val = (val << 8) + c;
        bits += 8;
        while (bits >= 0) {
            out.push_back(b64_table[(val >> bits) & 0x3F]);
            bits -= 6;
        }
    }
    if (bits > -6) out.push_back(b64_table[((val << 8) >> (bits + 8)) & 0x3F]);
    while (out.size() % 4) out.push_back('=');
    return out;
}

} // namespace cv4
