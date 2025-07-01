#include "object_detector/object_detection_node.hpp"
#include <vision_msgs/msg/pose2_d.hpp> // 使用 vision_msgs 的 Pose2D

using namespace std::placeholders;

ObjectDetector::ObjectDetector() : Node("object_detection") {
    // 加载参数
    load_parameters();
    
    // 初始化网络
    load_net();
    load_class_list();

    RCLCPP_INFO(get_logger(), "Object detection node started");
    RCLCPP_INFO(get_logger(), "Using model: %s", model_path_.c_str());
    RCLCPP_INFO(get_logger(), "Input size: %dx%d", input_width_, input_height_);
}

ObjectDetector::~ObjectDetector() {
    RCLCPP_INFO(get_logger(), "ObjectDetector destroyed");
}

void ObjectDetector::init() {
    // 初始化图像传输
    it_ = std::make_shared<image_transport::ImageTransport>(shared_from_this());
    
    // 订阅和发布
    image_sub_ = it_->subscribe("camera/image_raw", 1, 
                              std::bind(&ObjectDetector::image_callback, this, _1));
    detection_pub_ = create_publisher<vision_msgs::msg::Detection2DArray>("detections", 10);
    result_image_pub_ = it_->advertise("detection_result", 1);
}

void ObjectDetector::load_parameters() {
    // 声明参数
    declare_parameter<std::string>("model_path", "");
    declare_parameter<std::string>("classes_file", "");
    declare_parameter<bool>("use_cuda", false);
    declare_parameter<float>("conf_threshold", 0.4f);
    declare_parameter<float>("nms_threshold", 0.4f);
    declare_parameter<float>("score_threshold", 0.2f);
    declare_parameter<int>("input_width", 640);
    declare_parameter<int>("input_height", 640);
    
    // 获取参数
    get_parameter("model_path", model_path_);
    get_parameter("classes_file", classes_file_);
    get_parameter("use_cuda", use_cuda_);
    get_parameter("conf_threshold", conf_threshold_);
    get_parameter("nms_threshold", nms_threshold_);
    get_parameter("score_threshold", score_threshold_);
    get_parameter("input_width", input_width_);
    get_parameter("input_height", input_height_);
    
    // 检查必需参数
    if (model_path_.empty()) {
        RCLCPP_ERROR(get_logger(), "model_path parameter is not set!");
        rclcpp::shutdown();
    }
    
    if (classes_file_.empty()) {
        RCLCPP_ERROR(get_logger(), "classes_file parameter is not set!");
        rclcpp::shutdown();
    }
    
    // 打印参数值
    RCLCPP_INFO(get_logger(), "Parameters:");
    RCLCPP_INFO(get_logger(), "  model_path: %s", model_path_.c_str());
    RCLCPP_INFO(get_logger(), "  classes_file: %s", classes_file_.c_str());
    RCLCPP_INFO(get_logger(), "  use_cuda: %s", use_cuda_ ? "true" : "false");
    RCLCPP_INFO(get_logger(), "  conf_threshold: %.2f", conf_threshold_);
    RCLCPP_INFO(get_logger(), "  nms_threshold: %.2f", nms_threshold_);
    RCLCPP_INFO(get_logger(), "  score_threshold: %.2f", score_threshold_);
    RCLCPP_INFO(get_logger(), "  input_size: %dx%d", input_width_, input_height_);
}

void ObjectDetector::load_net() {
    try {
        net_ = cv::dnn::readNet(model_path_);
        if (use_cuda_) {
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
            RCLCPP_INFO(get_logger(), "Using CUDA acceleration");
        } else {
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            RCLCPP_INFO(get_logger(), "Using CPU");
        }
    } catch (...) {
        RCLCPP_ERROR(get_logger(), "Failed to load model: %s", model_path_.c_str());
        rclcpp::shutdown();
    }
}

void ObjectDetector::load_class_list() {
    std::ifstream ifs(classes_file_);
    if (!ifs.is_open()) {
        RCLCPP_ERROR(get_logger(), "Failed to open classes file: %s", classes_file_.c_str());
        rclcpp::shutdown();
    }
    std::string line;
    while (getline(ifs, line)) {
        class_list_.push_back(line);
    }
    RCLCPP_INFO(get_logger(), "Loaded %zu classes", class_list_.size());
}

cv::Mat ObjectDetector::format_yolov5(const cv::Mat& source) {
    int col = source.cols;
    int row = source.rows;
    int _max = std::max(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

void ObjectDetector::detect(const cv::Mat& image, std::vector<Detection>& output) {
    cv::Mat blob;
    auto input_image = format_yolov5(image);

    cv::dnn::blobFromImage(input_image, blob, 1./255., 
                            cv::Size(input_width_, input_height_), 
                            cv::Scalar(), true, false);
    net_.setInput(blob);
    
    std::vector<cv::Mat> outputs;
    net_.forward(outputs, net_.getUnconnectedOutLayersNames());

    float x_factor = input_image.cols / static_cast<float>(input_width_);
    float y_factor = input_image.rows / static_cast<float>(input_height_);

    float* data = (float*)outputs[0].data;
    const int dimensions = 85;
    const int rows = 25200;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i) {
        float confidence = data[4];
        if (confidence >= conf_threshold_) {
            cv::Mat scores(1, class_list_.size(), CV_32FC1, data + 5);
            cv::Point class_id;
            double max_score;
            cv::minMaxLoc(scores, 0, &max_score, 0, &class_id);
            if (max_score > score_threshold_) {
                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                int left = static_cast<int>((x - 0.5 * w) * x_factor);
                int top = static_cast<int>((y - 0.5 * h) * y_factor);
                int width = static_cast<int>(w * x_factor);
                int height = static_cast<int>(h * y_factor);

                boxes.emplace_back(left, top, width, height);
            }
        }
        data += dimensions;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, score_threshold_, nms_threshold_, nms_result);
    for (int idx : nms_result) {
        output.push_back({
            class_ids[idx],
            confidences[idx],
            boxes[idx]
        });
    }
}

void ObjectDetector::draw_boxes(cv::Mat& image, const std::vector<Detection>& output) {
    for (const auto& detection : output) {
        const auto& color = colors_[detection.class_id % colors_.size()];
        const auto& box = detection.box;

        cv::rectangle(image, box, color, 2);
        std::string label = cv::format("%s: %.2f", 
            class_list_[detection.class_id].c_str(), 
            detection.confidence);

        int baseline;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 
                                            0.5, 1, &baseline);

        cv::rectangle(image,
            cv::Point(box.x, box.y - text_size.height - 5),
            cv::Point(box.x + text_size.width, box.y),
            color, cv::FILLED);

        cv::putText(image, label,
            cv::Point(box.x, box.y - 5),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}

vision_msgs::msg::Detection2DArray ObjectDetector::to_detection_msg(
    const std::vector<Detection>& detections, 
    const std_msgs::msg::Header& header) 
{
    vision_msgs::msg::Detection2DArray msg;
    msg.header = header;
    
    for (const auto& det : detections) {
        vision_msgs::msg::Detection2D detection;
        detection.header = header;
        
        // 边界框尺寸
        detection.bbox.size_x = det.box.width;
        detection.bbox.size_y = det.box.height;
        
        // 修复：正确设置中心点（使用不带下划线的字段名）
        detection.bbox.center.position.x = det.box.x + det.box.width / 2.0;
        detection.bbox.center.position.y = det.box.y + det.box.height / 2.0;
        detection.bbox.center.theta = 0.0; // 无旋转
        
        // 类别和置信度
        vision_msgs::msg::ObjectHypothesisWithPose hypothesis;
        hypothesis.hypothesis.class_id = std::to_string(det.class_id);
        hypothesis.hypothesis.score = det.confidence;
        detection.results.push_back(hypothesis);
        
        msg.detections.push_back(detection);
    }
    
    return msg;
}

void ObjectDetector::image_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
    try {
        cv::Mat frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
        cv::Mat frame2 = frame.clone();
        
        // 检测目标
        std::vector<Detection> detections;
        detect(frame, detections);
        
        if (!detections.empty()) {
            RCLCPP_INFO(get_logger(), "Detected %zu objects", detections.size());
            
            // 发布检测结果消息
            auto detections_msg = to_detection_msg(detections, msg->header);
            detection_pub_->publish(detections_msg);
            
            // 绘制边界框
            draw_boxes(frame2, detections);
        }
        
        // 发布带检测结果的图像
        auto result_msg = cv_bridge::CvImage(msg->header, "bgr8", frame2).toImageMsg();
        result_image_pub_.publish(result_msg);
    } catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
    } catch (...) {
        RCLCPP_ERROR(get_logger(), "Unknown error in image callback");
    }
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ObjectDetector>();
    node->init(); // 调用初始化方法
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
