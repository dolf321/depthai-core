#include <csignal>
#include <iostream>

// Includes common necessary includes for development using depthai library
#include "depthai/depthai.hpp"
#include <opencv2/opencv.hpp>

static const std::vector<std::string> labelMap = {
    "person",        "bicycle",      "car",           "motorbike",     "aeroplane",   "bus",         "train",       "truck",        "boat",
    "traffic light", "fire hydrant", "stop sign",     "parking meter", "bench",       "bird",        "cat",         "dog",          "horse",
    "sheep",         "cow",          "elephant",      "bear",          "zebra",       "giraffe",     "backpack",    "umbrella",     "handbag",
    "tie",           "suitcase",     "frisbee",       "skis",          "snowboard",   "sports ball", "kite",        "baseball bat", "baseball glove",
    "skateboard",    "surfboard",    "tennis racket", "bottle",        "wine glass",  "cup",         "fork",        "knife",        "spoon",
    "bowl",          "banana",       "apple",         "sandwich",      "orange",      "broccoli",    "carrot",      "hot dog",      "pizza",
    "donut",         "cake",         "chair",         "sofa",          "pottedplant", "bed",         "diningtable", "toilet",       "tvmonitor",
    "laptop",        "mouse",        "remote",        "keyboard",      "cell phone",  "microwave",   "oven",        "toaster",      "sink",
    "refrigerator",  "book",         "clock",         "vase",          "scissors",    "teddy bear",  "hair drier",  "toothbrush"};

// Keyboard interrupt (Ctrl + C) detected
static std::atomic<bool> alive{true};
static void sigintHandler(int signum) {
    alive = false;
}

int main(int argc, char** argv) {
    using namespace std;
    using namespace std::chrono;
    std::signal(SIGINT, &sigintHandler);

    std::string nnPath(BLOB_PATH);

    // If path to blob specified, use that
    if(argc > 1) {
        nnPath = std::string(argv[1]);
    }

    // Create pipeline
    dai::Pipeline pipeline;

    // Define sources and outputs
    auto camRgb = pipeline.create<dai::node::ColorCamera>();
    auto monoLeft = pipeline.create<dai::node::MonoCamera>();
    auto monoRight = pipeline.create<dai::node::MonoCamera>();
    auto ve1 = pipeline.create<dai::node::VideoEncoder>();
    auto ve2 = pipeline.create<dai::node::VideoEncoder>();
    auto ve3 = pipeline.create<dai::node::VideoEncoder>();
    auto spatialDetectionNetwork = pipeline.create<dai::node::YoloSpatialDetectionNetwork>();
    auto stereo = pipeline.create<dai::node::StereoDepth>();
    auto edgeDetectorLeft = pipeline.create<dai::node::EdgeDetector>();
    auto edgeDetectorRight = pipeline.create<dai::node::EdgeDetector>();
    auto edgeDetectorRgb = pipeline.create<dai::node::EdgeDetector>();

    auto ve1Out = pipeline.create<dai::node::XLinkOut>();
    auto ve2Out = pipeline.create<dai::node::XLinkOut>();
    auto ve3Out = pipeline.create<dai::node::XLinkOut>();
    auto xoutBoundingBoxDepthMapping = pipeline.create<dai::node::XLinkOut>();
    auto xoutDepth = pipeline.create<dai::node::XLinkOut>();
    auto xoutNN = pipeline.create<dai::node::XLinkOut>();
    auto xoutRgb = pipeline.create<dai::node::XLinkOut>();
    auto xoutEdgeLeft = pipeline.create<dai::node::XLinkOut>();
    auto xoutEdgeRight = pipeline.create<dai::node::XLinkOut>();
    auto xoutEdgeRgb = pipeline.create<dai::node::XLinkOut>();
    ve1Out->setStreamName("ve1Out");
    ve2Out->setStreamName("ve2Out");
    ve3Out->setStreamName("ve3Out");
    xoutBoundingBoxDepthMapping->setStreamName("boundingBoxDepthMapping");
    xoutDepth->setStreamName("depth");
    xoutNN->setStreamName("detections");
    xoutRgb->setStreamName("rgb");
    const auto edgeLeftStr = "edge left";
    const auto edgeRightStr = "edge right";
    const auto edgeRgbStr = "edge rgb";
    const auto edgeCfgStr = "edge cfg";
    xoutEdgeLeft->setStreamName(edgeLeftStr);
    xoutEdgeRight->setStreamName(edgeRightStr);
    xoutEdgeRgb->setStreamName(edgeRgbStr);


    // Properties
    camRgb->setBoardSocket(dai::CameraBoardSocket::RGB);
    camRgb->setResolution(dai::ColorCameraProperties::SensorResolution::THE_4_K);
    camRgb->setPreviewSize(416, 416);
    camRgb->setInterleaved(false);
    camRgb->setColorOrder(dai::ColorCameraProperties::ColorOrder::BGR);

    monoLeft->setBoardSocket(dai::CameraBoardSocket::LEFT);
    monoLeft->setResolution(dai::MonoCameraProperties::SensorResolution::THE_400_P);

    monoRight->setBoardSocket(dai::CameraBoardSocket::RIGHT);
    monoRight->setResolution(dai::MonoCameraProperties::SensorResolution::THE_400_P);

    // Setting to 26fps will trigger error
    ve1->setDefaultProfilePreset(25, dai::VideoEncoderProperties::Profile::H264_MAIN);
    ve2->setDefaultProfilePreset(25, dai::VideoEncoderProperties::Profile::H265_MAIN);
    ve3->setDefaultProfilePreset(25, dai::VideoEncoderProperties::Profile::H264_MAIN);

    stereo->setDefaultProfilePreset(dai::node::StereoDepth::PresetMode::HIGH_DENSITY);
    // Align depth map to the perspective of RGB camera, on which inference is done
    stereo->setDepthAlign(dai::CameraBoardSocket::RGB);
    stereo->setOutputSize(monoLeft->getResolutionWidth(), monoLeft->getResolutionHeight());

    spatialDetectionNetwork->setBlobPath(nnPath);
    spatialDetectionNetwork->setConfidenceThreshold(0.5f);
    spatialDetectionNetwork->input.setBlocking(false);
    spatialDetectionNetwork->setBoundingBoxScaleFactor(0.5);
    spatialDetectionNetwork->setDepthLowerThreshold(100);
    spatialDetectionNetwork->setDepthUpperThreshold(5000);

    // yolo specific parameters
    spatialDetectionNetwork->setNumClasses(80);
    spatialDetectionNetwork->setCoordinateSize(4);
    spatialDetectionNetwork->setAnchors({10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319});
    spatialDetectionNetwork->setAnchorMasks({{"side26", {1, 2, 3}}, {"side13", {3, 4, 5}}});
    spatialDetectionNetwork->setIouThreshold(0.5f);

    edgeDetectorRgb->setMaxOutputFrameSize(8294400);

    // Linking
    monoLeft->out.link(ve1->input);
    camRgb->video.link(ve2->input);
    monoRight->out.link(ve3->input);

    ve1->bitstream.link(ve1Out->input);
    ve2->bitstream.link(ve2Out->input);
    ve3->bitstream.link(ve3Out->input);

    monoLeft->out.link(stereo->left);
    monoRight->out.link(stereo->right);

    camRgb->preview.link(spatialDetectionNetwork->input);
    camRgb->preview.link(xoutRgb->input);
    spatialDetectionNetwork->out.link(xoutNN->input);
    spatialDetectionNetwork->boundingBoxMapping.link(xoutBoundingBoxDepthMapping->input);

    stereo->depth.link(spatialDetectionNetwork->inputDepth);
    spatialDetectionNetwork->passthroughDepth.link(xoutDepth->input);

    monoLeft->out.link(edgeDetectorLeft->inputImage);
    monoRight->out.link(edgeDetectorRight->inputImage);
    camRgb->video.link(edgeDetectorRgb->inputImage);
    edgeDetectorLeft->outputImage.link(xoutEdgeLeft->input);
    edgeDetectorRight->outputImage.link(xoutEdgeRight->input);
    // Do not send out rgb edge
    // edgeDetectorRgb->outputImage.link(xoutEdgeRgb->input);

    // Connect to device and start pipeline
    dai::Device device(pipeline);

    // Output queues will be used to get the encoded data from the output defined above
    auto outQ1 = device.getOutputQueue("ve1Out", 30, false);
    auto outQ2 = device.getOutputQueue("ve2Out", 30, false);
    auto outQ3 = device.getOutputQueue("ve3Out", 30, false);
    auto previewQueue = device.getOutputQueue("rgb", 4, false);
    auto xoutBoundingBoxDepthMappingQueue = device.getOutputQueue("boundingBoxDepthMapping", 4, false);
    auto depthQueue = device.getOutputQueue("depth", 4, false);
    auto detectionNNQueue = device.getOutputQueue("detections", 4, false);
    auto edgeLeftQueue = device.getOutputQueue(edgeLeftStr, 8, false);
    auto edgeRightQueue = device.getOutputQueue(edgeRightStr, 8, false);
    auto edgeRgbQueue = device.getOutputQueue(edgeRgbStr, 8, false);

    auto startTime = steady_clock::now();
    int counter = 0;
    float fps = 0;
    auto color = cv::Scalar(255, 255, 255);

    while(alive) {
        auto out1 = outQ1->getAll<dai::ImgFrame>();
        auto out2 = outQ2->getAll<dai::ImgFrame>();
        auto out3 = outQ3->getAll<dai::ImgFrame>();
        auto imgFrame = previewQueue->get<dai::ImgFrame>();
        auto inDet = detectionNNQueue->get<dai::SpatialImgDetections>();
        auto depth = depthQueue->get<dai::ImgFrame>();

        auto edgeLefts = edgeLeftQueue->getAll<dai::ImgFrame>();
        auto edgeRights = edgeRightQueue->getAll<dai::ImgFrame>();
        // auto edgeRgbs = edgeRgbQueue->getAll<dai::ImgFrame>();


        /// DISPLAY & OPENCV Section
        for(const auto& edgeLeft : edgeLefts) {
            cv::Mat edgeLeftFrame = edgeLeft->getFrame();
            cv::imshow(edgeLeftStr, edgeLeftFrame);
        }
        for(const auto& edgeRight : edgeRights) {
            cv::Mat edgeRightFrame = edgeRight->getFrame();
            cv::imshow(edgeRightStr, edgeRightFrame);
        }
        // for(const auto& edgeRgb : edgeRgbs) {
        //     cv::Mat edgeRgbFrame = edgeRgb->getFrame();
        //     cv::imshow(edgeRgbStr, edgeRgbFrame);
        // }

        cv::Mat frame = imgFrame->getCvFrame();
        cv::Mat depthFrame = depth->getFrame();  // depthFrame values are in millimeters

        cv::Mat depthFrameColor;
        cv::normalize(depthFrame, depthFrameColor, 255, 0, cv::NORM_INF, CV_8UC1);
        cv::equalizeHist(depthFrameColor, depthFrameColor);
        cv::applyColorMap(depthFrameColor, depthFrameColor, cv::COLORMAP_HOT);

        counter++;
        auto currentTime = steady_clock::now();
        auto elapsed = duration_cast<duration<float>>(currentTime - startTime);
        if(elapsed > seconds(1)) {
            fps = counter / elapsed.count();
            counter = 0;
            startTime = currentTime;
        }

        auto detections = inDet->detections;
        if(!detections.empty()) {
            auto boundingBoxMapping = xoutBoundingBoxDepthMappingQueue->get<dai::SpatialLocationCalculatorConfig>();
            auto roiDatas = boundingBoxMapping->getConfigData();

            for(auto roiData : roiDatas) {
                auto roi = roiData.roi;
                roi = roi.denormalize(depthFrameColor.cols, depthFrameColor.rows);
                auto topLeft = roi.topLeft();
                auto bottomRight = roi.bottomRight();
                auto xmin = (int)topLeft.x;
                auto ymin = (int)topLeft.y;
                auto xmax = (int)bottomRight.x;
                auto ymax = (int)bottomRight.y;

                cv::rectangle(depthFrameColor, cv::Rect(cv::Point(xmin, ymin), cv::Point(xmax, ymax)), color, cv::FONT_HERSHEY_SIMPLEX);
            }
        }

        for(const auto& detection : detections) {
            int x1 = detection.xmin * frame.cols;
            int y1 = detection.ymin * frame.rows;
            int x2 = detection.xmax * frame.cols;
            int y2 = detection.ymax * frame.rows;

            uint32_t labelIndex = detection.label;
            std::string labelStr = to_string(labelIndex);
            if(labelIndex < labelMap.size()) {
                labelStr = labelMap[labelIndex];
            }
            cv::putText(frame, labelStr, cv::Point(x1 + 10, y1 + 20), cv::FONT_HERSHEY_TRIPLEX, 0.5, 255);
            std::stringstream confStr;
            confStr << std::fixed << std::setprecision(2) << detection.confidence * 100;
            cv::putText(frame, confStr.str(), cv::Point(x1 + 10, y1 + 35), cv::FONT_HERSHEY_TRIPLEX, 0.5, 255);

            std::stringstream depthX;
            depthX << "X: " << (int)detection.spatialCoordinates.x << " mm";
            cv::putText(frame, depthX.str(), cv::Point(x1 + 10, y1 + 50), cv::FONT_HERSHEY_TRIPLEX, 0.5, 255);
            std::stringstream depthY;
            depthY << "Y: " << (int)detection.spatialCoordinates.y << " mm";
            cv::putText(frame, depthY.str(), cv::Point(x1 + 10, y1 + 65), cv::FONT_HERSHEY_TRIPLEX, 0.5, 255);
            std::stringstream depthZ;
            depthZ << "Z: " << (int)detection.spatialCoordinates.z << " mm";
            cv::putText(frame, depthZ.str(), cv::Point(x1 + 10, y1 + 80), cv::FONT_HERSHEY_TRIPLEX, 0.5, 255);

            cv::rectangle(frame, cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)), color, cv::FONT_HERSHEY_SIMPLEX);
        }

        std::stringstream fpsStr;
        fpsStr << std::fixed << std::setprecision(2) << fps;
        cv::putText(frame, fpsStr.str(), cv::Point(2, imgFrame->getHeight() - 4), cv::FONT_HERSHEY_TRIPLEX, 0.4, color);

        cv::imshow("depth", depthFrameColor);
        cv::imshow("rgb", frame);

        int key = cv::waitKey(1);
        if(key == 'q' || key == 'Q') {
            return 0;
        }


    }

    return 0;
}
