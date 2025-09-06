#pragma once

#ifndef YOLOP_HPP
#define YOLOP_HPP

#include <chrono>
#include "cuda_utils.h"
#include "logging.h"
#include "utils.h"

#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.45
#define CONF_THRESH 0.25
#define BATCH_SIZE 1

// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;

static const int IMG_H = Yolo::IMG_H;
static const int IMG_W = Yolo::IMG_W;

static const int CLASS_NUM = Yolo::CLASS_NUM;
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1

class YOLOP {
private:
    // for input data
    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    static float prob[BATCH_SIZE * OUTPUT_SIZE];
    static int seg_out[BATCH_SIZE * IMG_H * IMG_W];
    static int lane_out[BATCH_SIZE * IMG_H * IMG_W];

    IRuntime* runtime;
    ICudaEngine* engine;
    IExecutionContext* context;

    int input_index, output_det_index, output_seg_index, output_lane_index;

    void* buffers[4];

    cv::Mat img;

    cudaStream_t stream;

public:
    // for store seg, lane results
    cv::Mat tmp_seg, tmp_lane;

    // for visualization
    std::vector<cv::Vec3b> seg_color, lane_color;

public:
    YOLOP();
    ~YOLOP();
    std::vector<Yolo::Detection> inference();
    void visualization(const std::vector<Yolo::Detection>& res);
    void read_engine_file(const std::string& engine_path);

    void set_img(const cv::Mat& img){
        this->img = img;
    }

    cv::Mat get_img(){
        return this->img;
    }
};

#endif
