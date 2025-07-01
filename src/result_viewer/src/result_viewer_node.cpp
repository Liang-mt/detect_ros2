#include "result_viewer/result_viewer_node.hpp"

using namespace std::chrono_literals;
using namespace std::placeholders; // 修复：添加占位符命名空间

ResultViewer::ResultViewer() : Node("result_viewer") {
    // 声明参数
    declare_parameter("image_topic", "detection_result");
    declare_parameter("detection_topic", "detections");
    declare_parameter("classes_file", "");
    
    // 获取参数
    get_parameter("image_topic", image_topic_);
    get_parameter("detection_topic", detection_topic_);
    get_parameter("classes_file", classes_file_);
    
    // 打印参数值
    RCLCPP_INFO(get_logger(), "Result viewer parameters:");
    RCLCPP_INFO(get_logger(), "  image_topic: %s", image_topic_.c_str());
    RCLCPP_INFO(get_logger(), "  detection_topic: %s", detection_topic_.c_str());
    
    // 检查类文件
    if (classes_file_.empty()) {
        RCLCPP_ERROR(get_logger(), "classes_file parameter is not set!");
        rclcpp::shutdown();
    }
    load_class_list();
    RCLCPP_INFO(get_logger(), "  classes_file: %s", classes_file_.c_str());
    RCLCPP_INFO(get_logger(), "Result viewer node started");
}

ResultViewer::~ResultViewer() {
    // 关闭OpenCV窗口
    cv::destroyAllWindows();
    RCLCPP_INFO(get_logger(), "OpenCV windows closed");
    RCLCPP_INFO(get_logger(), "ResultViewer node stopped");
}

void ResultViewer::init() {
    // 初始化图像传输
    it_ = std::make_shared<image_transport::ImageTransport>(shared_from_this());
    
    // 订阅图像和检测结果
    image_sub_ = it_->subscribe(image_topic_, 1, 
                              std::bind(&ResultViewer::image_callback, this, _1));
    detection_sub_ = create_subscription<vision_msgs::msg::Detection2DArray>(
        detection_topic_, 10,
        std::bind(&ResultViewer::detection_callback, this, _1));
}

void ResultViewer::run() {
    rclcpp::Rate rate(30); // 30Hz刷新率
    while (rclcpp::ok()) {
        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            if (new_data_available_ && !current_frame_.empty()) {
                // 绘制检测结果
                cv::Mat display_frame = current_frame_.clone();
                draw_detections(display_frame);
                
                // 显示图像
                cv::imshow("Object Detection Results", display_frame);
                cv::waitKey(1);
                
                new_data_available_ = false;
            }
        }
        
        rclcpp::spin_some(shared_from_this());
        rate.sleep();
    }
    cv::destroyAllWindows();
}

void ResultViewer::load_class_list() {
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

void ResultViewer::image_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
    try {
        std::lock_guard<std::mutex> lock(data_mutex_);
        current_frame_ = cv_bridge::toCvCopy(msg, "bgr8")->image;
        new_data_available_ = true;
    } catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
    }
}

void ResultViewer::detection_callback(const vision_msgs::msg::Detection2DArray::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    current_detections_ = *msg;
}

void ResultViewer::draw_detections(cv::Mat& image) {
    const std::vector<cv::Scalar> colors = {
        cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0),
        cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0),
        cv::Scalar(255, 0, 255), cv::Scalar(0, 0, 255)
    };
    
    for (const auto& detection : current_detections_.detections) {
        if (detection.results.empty()) continue;
        
        // 获取类ID和置信度
        const std::string& class_id_str = detection.results[0].hypothesis.class_id;
        const float score = detection.results[0].hypothesis.score;
        
        // 将字符串类ID转换为整数
        int class_id = -1;
        try {
            class_id = std::stoi(class_id_str);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "Invalid class ID format: %s", class_id_str.c_str());
            continue;
        }
        
        // 检查类ID是否有效
        if (class_id < 0 || static_cast<size_t>(class_id) >= class_list_.size()) {
            RCLCPP_WARN(get_logger(), "Class ID out of range: %d", class_id);
            continue;
        }
        
        const cv::Scalar color = colors[class_id % colors.size()];
        
        // 提取边界框信息（使用不带下划线的字段名）
        const float center_x = detection.bbox.center.position.x;
        const float center_y = detection.bbox.center.position.y;
        const float width = detection.bbox.size_x;
        const float height = detection.bbox.size_y;
        
        // 转换为OpenCV矩形
        cv::Rect box(
            static_cast<int>(center_x - width / 2),
            static_cast<int>(center_y - height / 2),
            static_cast<int>(width),
            static_cast<int>(height)
        );
        
        // 确保边界框在图像范围内
        if (box.x < 0) box.x = 0;
        if (box.y < 0) box.y = 0;
        if (box.x + box.width > image.cols) box.width = image.cols - box.x;
        if (box.y + box.height > image.rows) box.height = image.rows - box.y;
        
        // 绘制边界框
        cv::rectangle(image, box, color, 2);
        
        // 创建标签文本
        std::string label = cv::format("%s: %.2f", class_list_[class_id].c_str(), score);
        
        // 绘制标签背景
        int baseline;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,  0.5, 1, &baseline);
        
        // 确保标签不会超出图像顶部
        int text_y = box.y - 5;
        if (text_y < text_size.height + 5) {
            text_y = box.y + box.height + text_size.height + 5;
        }
        
        cv::rectangle(image, 
            cv::Point(box.x, text_y - text_size.height - 5),
            cv::Point(box.x + text_size.width, text_y),
            color, cv::FILLED);
        
        // 绘制标签文本
        cv::putText(image, label,
            cv::Point(box.x, text_y - 5),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ResultViewer>();
    node->init(); // 调用初始化方法
    node->run();
    rclcpp::shutdown();
    return 0;
}
