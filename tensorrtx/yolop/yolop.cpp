#include "yolop.hpp"

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_DET_NAME = "det";
const char* OUTPUT_SEG_NAME = "seg";
const char* OUTPUT_LANE_NAME = "lane";

static Logger gLogger;

namespace yolop_utils {
    ICudaEngine* build_engine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, std::string& wts_name) {
        INetworkDefinition* network = builder->createNetworkV2(0U);

        // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
        ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
        assert(data);

        std::map<std::string, Weights> weightMap = loadWeights(wts_name);
        Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

        // yolop backbone
        // auto focus0 = focus(network, weightMap, *shuffle->getOutput(0), 3, 32, 3, "model.0");
        auto focus0 = focus(network, weightMap, *data, 3, 32, 3, "model.0");
        auto conv1 = convBlock(network, weightMap, *focus0->getOutput(0), 64, 3, 2, 1, "model.1");
        auto bottleneck_CSP2 = bottleneckCSP(network, weightMap, *conv1->getOutput(0), 64, 64, 1, true, 1, 0.5, "model.2");
        auto conv3 = convBlock(network, weightMap, *bottleneck_CSP2->getOutput(0), 128, 3, 2, 1, "model.3");
        auto bottleneck_csp4 = bottleneckCSP(network, weightMap, *conv3->getOutput(0), 128, 128, 3, true, 1, 0.5, "model.4");
        auto conv5 = convBlock(network, weightMap, *bottleneck_csp4->getOutput(0), 256, 3, 2, 1, "model.5");
        auto bottleneck_csp6 = bottleneckCSP(network, weightMap, *conv5->getOutput(0), 256, 256, 3, true, 1, 0.5, "model.6");
        auto conv7 = convBlock(network, weightMap, *bottleneck_csp6->getOutput(0), 512, 3, 2, 1, "model.7");
        auto spp8 = SPP(network, weightMap, *conv7->getOutput(0), 512, 512, 5, 9, 13, "model.8");

        // yolop head
        auto bottleneck_csp9 = bottleneckCSP(network, weightMap, *spp8->getOutput(0), 512, 512, 1, false, 1, 0.5, "model.9");
        auto conv10 = convBlock(network, weightMap, *bottleneck_csp9->getOutput(0), 256, 1, 1, 1, "model.10");

        float *deval = reinterpret_cast<float*>(malloc(sizeof(float) * 256 * 2 * 2));
        for (int i = 0; i < 256 * 2 * 2; i++) {
            deval[i] = 1.0;
        }
        Weights deconvwts11{ DataType::kFLOAT, deval, 256 * 2 * 2 };
        IDeconvolutionLayer* deconv11 = network->addDeconvolutionNd(*conv10->getOutput(0), 256, DimsHW{ 2, 2 }, deconvwts11, emptywts);
        deconv11->setStrideNd(DimsHW{ 2, 2 });
        deconv11->setNbGroups(256);
        weightMap["deconv11"] = deconvwts11;

        ITensor* inputTensors12[] = { deconv11->getOutput(0), bottleneck_csp6->getOutput(0) };
        auto cat12 = network->addConcatenation(inputTensors12, 2);
        auto bottleneck_csp13 = bottleneckCSP(network, weightMap, *cat12->getOutput(0), 512, 256, 1, false, 1, 0.5, "model.13");
        auto conv14 = convBlock(network, weightMap, *bottleneck_csp13->getOutput(0), 128, 1, 1, 1, "model.14");

        Weights deconvwts15{ DataType::kFLOAT, deval, 128 * 2 * 2 };
        IDeconvolutionLayer* deconv15 = network->addDeconvolutionNd(*conv14->getOutput(0), 128, DimsHW{ 2, 2 }, deconvwts15, emptywts);
        deconv15->setStrideNd(DimsHW{ 2, 2 });
        deconv15->setNbGroups(128);

        ITensor* inputTensors16[] = { deconv15->getOutput(0), bottleneck_csp4->getOutput(0) };
        auto cat16 = network->addConcatenation(inputTensors16, 2);
        auto bottleneck_csp17 = bottleneckCSP(network, weightMap, *cat16->getOutput(0), 256, 128, 1, false, 1, 0.5, "model.17");
        IConvolutionLayer* det0 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);

        auto conv18 = convBlock(network, weightMap, *bottleneck_csp17->getOutput(0), 128, 3, 2, 1, "model.18");
        ITensor* inputTensors19[] = { conv18->getOutput(0), conv14->getOutput(0) };
        auto cat19 = network->addConcatenation(inputTensors19, 2);
        auto bottleneck_csp20 = bottleneckCSP(network, weightMap, *cat19->getOutput(0), 256, 256, 1, false, 1, 0.5, "model.20");
        IConvolutionLayer* det1 = network->addConvolutionNd(*bottleneck_csp20->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);

        auto conv21 = convBlock(network, weightMap, *bottleneck_csp20->getOutput(0), 256, 3, 2, 1, "model.21");
        ITensor* inputTensors22[] = { conv21->getOutput(0), conv10->getOutput(0) };
        auto cat22 = network->addConcatenation(inputTensors22, 2);
        auto bottleneck_csp23 = bottleneckCSP(network, weightMap, *cat22->getOutput(0), 512, 512, 1, false, 1, 0.5, "model.23");
        IConvolutionLayer* det2 = network->addConvolutionNd(*bottleneck_csp23->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);

        auto detect24 = addYoLoLayer(network, weightMap, det0, det1, det2);
        detect24->getOutput(0)->setName(OUTPUT_DET_NAME);

        auto conv25 = convBlock(network, weightMap, *cat16->getOutput(0), 128, 3, 1, 1, "model.25");
        // upsample 26
        Weights deconvwts26{ DataType::kFLOAT, deval, 128 * 2 * 2 };
        IDeconvolutionLayer* deconv26 = network->addDeconvolutionNd(*conv25->getOutput(0), 128, DimsHW{ 2, 2 }, deconvwts26, emptywts);
        deconv26->setStrideNd(DimsHW{ 2, 2 });
        deconv26->setNbGroups(128);

        auto bottleneck_csp27 = bottleneckCSP(network, weightMap, *deconv26->getOutput(0), 128, 64, 1, false, 1, 0.5, "model.27");
        auto conv28 = convBlock(network, weightMap, *bottleneck_csp27->getOutput(0), 32, 3, 1, 1, "model.28");
        // upsample 29
        Weights deconvwts29{ DataType::kFLOAT, deval, 32 * 2 * 2 };
        IDeconvolutionLayer* deconv29 = network->addDeconvolutionNd(*conv28->getOutput(0), 32, DimsHW{ 2, 2 }, deconvwts29, emptywts);
        deconv29->setStrideNd(DimsHW{ 2, 2 });
        deconv29->setNbGroups(32);

        auto conv30 = convBlock(network, weightMap, *deconv29->getOutput(0), 16, 3, 1, 1, "model.30");
        auto bottleneck_csp31 = bottleneckCSP(network, weightMap, *conv30->getOutput(0), 16, 8, 1, false, 1, 0.5, "model.31");

        // upsample32
        Weights deconvwts32{ DataType::kFLOAT, deval, 8 * 2 * 2 };
        IDeconvolutionLayer* deconv32 = network->addDeconvolutionNd(*bottleneck_csp31->getOutput(0), 8, DimsHW{ 2, 2 }, deconvwts32, emptywts);
        deconv32->setStrideNd(DimsHW{ 2, 2 });
        deconv32->setNbGroups(8);

        auto conv33 = convBlock(network, weightMap, *deconv32->getOutput(0), 2, 3, 1, 1, "model.33");
        // segmentation output
        ISliceLayer *slicelayer = network->addSlice(*conv33->getOutput(0), Dims3{ 0, (Yolo::INPUT_H - Yolo::IMG_H) / 2, 0 }, Dims3{ 2, Yolo::IMG_H, Yolo::IMG_W }, Dims3{ 1, 1, 1 });
        auto segout = network->addTopK(*slicelayer->getOutput(0), TopKOperation::kMAX, 1, 1);
        segout->getOutput(1)->setName(OUTPUT_SEG_NAME);

        auto conv34 = convBlock(network, weightMap, *cat16->getOutput(0), 128, 3, 1, 1, "model.34");

        // upsample35
        Weights deconvwts35{ DataType::kFLOAT, deval, 128 * 2 * 2 };
        IDeconvolutionLayer* deconv35 = network->addDeconvolutionNd(*conv34->getOutput(0), 128, DimsHW{ 2, 2 }, deconvwts35, emptywts);
        deconv35->setStrideNd(DimsHW{ 2, 2 });
        deconv35->setNbGroups(128);

        auto bottleneck_csp36 = bottleneckCSP(network, weightMap, *deconv35->getOutput(0), 128, 64, 1, false, 1, 0.5, "model.36");
        auto conv37 = convBlock(network, weightMap, *bottleneck_csp36->getOutput(0), 32, 3, 1, 1, "model.37");

        // upsample38
        Weights deconvwts38{ DataType::kFLOAT, deval, 32 * 2 * 2 };
        IDeconvolutionLayer* deconv38 = network->addDeconvolutionNd(*conv37->getOutput(0), 32, DimsHW{ 2, 2 }, deconvwts38, emptywts);
        deconv38->setStrideNd(DimsHW{ 2, 2 });
        deconv38->setNbGroups(32);

        auto conv39 = convBlock(network, weightMap, *deconv38->getOutput(0), 16, 3, 1, 1, "model.39");
        auto bottleneck_csp40 = bottleneckCSP(network, weightMap, *conv39->getOutput(0), 16, 8, 1, false, 1, 0.5, "model.40");

        // upsample41
        Weights deconvwts41{ DataType::kFLOAT, deval, 8 * 2 * 2 };
        IDeconvolutionLayer* deconv41 = network->addDeconvolutionNd(*bottleneck_csp40->getOutput(0), 8, DimsHW{ 2, 2 }, deconvwts41, emptywts);
        deconv41->setStrideNd(DimsHW{ 2, 2 });
        deconv41->setNbGroups(8);

        auto conv42 = convBlock(network, weightMap, *deconv41->getOutput(0), 2, 3, 1, 1, "model.42");
        // lane-det output
        ISliceLayer *laneSlice = network->addSlice(*conv42->getOutput(0), Dims3{ 0, (Yolo::INPUT_H - Yolo::IMG_H) / 2, 0 }, Dims3{ 2, Yolo::IMG_H, Yolo::IMG_W }, Dims3{ 1, 1, 1 });
        auto laneout = network->addTopK(*laneSlice->getOutput(0), TopKOperation::kMAX, 1, 1);
        laneout->getOutput(1)->setName(OUTPUT_LANE_NAME);

        // detection output
        network->markOutput(*detect24->getOutput(0));
        // segmentation output
        network->markOutput(*segout->getOutput(1));
        // lane output
        network->markOutput(*laneout->getOutput(1));

        assert(false);

        // Build engine
        builder->setMaxBatchSize(maxBatchSize);
        config->setMaxWorkspaceSize(2L * (1L << 30));  // 2GB
    #if defined(USE_FP16)
        config->setFlag(BuilderFlag::kFP16);
    #endif

        std::cout << "Building engine, please wait for a while..." << std::endl;
        ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
        std::cout << "Build engine successfully!" << std::endl;

        // Don't need the network any more
        network->destroy();

        // Release host memory
        for (auto& mem : weightMap)
        {
            free((void*)(mem.second.values));
        }

        return engine;
    }

    void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream, std::string& wts_name) {
        // Create builder
        IBuilder* builder = createInferBuilder(gLogger);
        IBuilderConfig* config = builder->createBuilderConfig();

        // Create model to populate the network, then set the outputs and create an engine
        ICudaEngine* engine = build_engine(maxBatchSize, builder, config, DataType::kFLOAT, wts_name);
        assert(engine != nullptr);

        // Serialize the engine
        (*modelStream) = engine->serialize();

        // Close everything down
        engine->destroy();
        builder->destroy();
        config->destroy();
    }

    void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* det_output, int* seg_output, int* lane_output, int batchSize) {
        // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
        // CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
        context.enqueue(batchSize, buffers, stream, nullptr);
        CUDA_CHECK(cudaMemcpyAsync(det_output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(seg_output, buffers[2], batchSize * IMG_H * IMG_W * sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(lane_output, buffers[3], batchSize * IMG_H * IMG_W * sizeof(int), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
    }

    void doInferenceCpu(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* det_output, int* seg_output, int* lane_output, int batchSize=1) {
        // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
        CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
        context.enqueue(batchSize, buffers, stream, nullptr);
        CUDA_CHECK(cudaMemcpyAsync(det_output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(seg_output, buffers[2], batchSize * IMG_H * IMG_W * sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(lane_output, buffers[3], batchSize * IMG_H * IMG_W * sizeof(int), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
    }

    bool parse_args(int argc, char** argv, std::string& wts, std::string& engine, std::string& img_dir) {
        if (argc < 4) return false;
        if (std::string(argv[1]) == "-s" && argc == 4) {
            wts = std::string(argv[2]);
            engine = std::string(argv[3]);
        } else if (std::string(argv[1]) == "-d" && argc == 4) {
            engine = std::string(argv[2]);
            img_dir = std::string(argv[3]);
        } else {
            return false;
        }
        return true;
    }
}

float YOLOP::data[3 * INPUT_H * INPUT_W];
float YOLOP::prob[OUTPUT_SIZE];
int YOLOP::seg_out[IMG_H * IMG_W];
int YOLOP::lane_out[IMG_H * IMG_W];

YOLOP::YOLOP(): 
    tmp_seg(IMG_H, IMG_W, CV_32S, YOLOP::seg_out), 
    tmp_lane(IMG_H, IMG_W, CV_32S, YOLOP::lane_out) 
{
    cudaSetDevice(DEVICE);

    // predicted segment visualization color
    this->seg_color.emplace_back(cv::Vec3b(0, 0, 0));
    this->seg_color.emplace_back(cv::Vec3b(0, 255, 0));
    this->seg_color.emplace_back(cv::Vec3b(255, 0, 0));

    // predicted lane visualization color
    this->lane_color.emplace_back(cv::Vec3b(0, 0, 0));
    this->lane_color.emplace_back(cv::Vec3b(0, 0, 255));
    this->lane_color.emplace_back(cv::Vec3b(0, 0, 0));
}

void YOLOP::read_engine_file(const std::string& engine_path){
    std::ifstream file(engine_path, std::ios::binary);

    if(file.fail()) {
        IHostMemory* modelStream { nullptr };
        std::string wts_name = "yolop.wts";
        yolop_utils::APIToModel(BATCH_SIZE, &modelStream, wts_name);
        assert(modelStream != nullptr);
        std::ofstream p(engine_path, std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return;
    }

    if (!file.good()) {
        std::cerr << "read " << engine_path << " error!" << std::endl;
        return;
    }
    
    size_t size = 0;
    char *trtModelStream = nullptr;
    
    file.seekg(0, file.end);
    size = file.tellg();

    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    
    file.read(trtModelStream, size);
    file.close();

    this->runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    
    this->engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);

    this->context = engine->createExecutionContext();
    assert(context != nullptr);
    
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 4);

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    this->input_index = this->engine->getBindingIndex(INPUT_BLOB_NAME);
    assert(this->input_index == 0);
    
    this->output_det_index = this->engine->getBindingIndex(OUTPUT_DET_NAME);
    assert(this->output_det_index == 1);
    
    this->output_seg_index = this->engine->getBindingIndex(OUTPUT_SEG_NAME);
    assert(this->output_seg_index == 2);
    
    this->output_lane_index = this->engine->getBindingIndex(OUTPUT_LANE_NAME);
    assert(this->output_lane_index == 3);

    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&this->buffers[this->input_index], 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&this->buffers[this->output_det_index], OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&this->buffers[this->output_seg_index], IMG_H * IMG_W * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&this->buffers[this->output_lane_index], IMG_H * IMG_W * sizeof(int)));

    // Create stream
    CUDA_CHECK(cudaStreamCreate(&this->stream));
}

std::vector<Yolo::Detection> YOLOP::inference(){
    cv::Mat pr_img = preprocess_img(this->img, INPUT_W, INPUT_H);
    
    // BGR to RGB and normalize(-> flatten?)
    for (int i, row = 0; row < INPUT_H; row++) {
        float* uc_pixel = pr_img.ptr<float>(row);
        for (int col = 0; col < INPUT_W; col++) {
            data[i] = uc_pixel[0];
            data[i + INPUT_H * INPUT_W] = uc_pixel[1];
            data[i + 2 * INPUT_H * INPUT_W] = uc_pixel[2];
            uc_pixel += 3;
            i++;
        }
    }

    auto start = std::chrono::system_clock::now();
    yolop_utils::doInferenceCpu(*this->context, this->stream, this->buffers, YOLOP::data, YOLOP::prob, YOLOP::seg_out, YOLOP::lane_out);
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    std::vector<Yolo::Detection> ret;
    nms(ret, &prob[0], CONF_THRESH, NMS_THRESH);

    return ret;
}

void YOLOP::visualization(const std::vector<Yolo::Detection>& res){
    cv::Mat seg_res(this->img.rows, this->img.cols, CV_32S);
    cv::resize(this->tmp_seg, seg_res, seg_res.size(), 0, 0, cv::INTER_NEAREST);
    
    cv::Mat lane_res(this->img.rows, this->img.cols, CV_32S);
    cv::resize(this->tmp_lane, lane_res, lane_res.size(), 0, 0, cv::INTER_NEAREST);
    
    for (int row = 0; row < this->img.rows; row++) {
        uchar* pdata = this->img.data + row * this->img.step;
        for (int col = 0; col < this->img.cols; col++) {
            int seg_idx = seg_res.at<int>(row, col);
            int lane_idx = lane_res.at<int>(row, col);
            
            for (int i = 0; i < 3; ++i) {
                if (lane_idx) {
                    if (i != 2)
                        pdata[i] = pdata[i] / 2 + lane_color[lane_idx][i] / 2;
                }
                else if (seg_idx)
                    pdata[i] = pdata[i] / 2 + seg_color[seg_idx][i] / 2;
            }
            pdata += 3;
        }
    }

    for (size_t j = 0; j < res.size(); j++) {
        cv::Rect r = get_rect(this->img, res[j].bbox);
        cv::rectangle(this->img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
        cv::putText(this->img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
    }

    cv::imshow("result", this->img);
    cv::waitKey(1);
}

YOLOP::~YOLOP(){
    cudaStreamDestroy(this->stream);
    
    CUDA_CHECK(cudaFree(this->buffers[this->input_index]));
    CUDA_CHECK(cudaFree(this->buffers[this->output_det_index]));
    CUDA_CHECK(cudaFree(this->buffers[this->output_seg_index]));
    CUDA_CHECK(cudaFree(this->buffers[this->output_lane_index]));
    
    this->context->destroy();
    this->engine->destroy();
    this->runtime->destroy();
}
