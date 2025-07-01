#include "result_viewer/result_viewer_node.hpp"  // 包含自定义头文件

using namespace std::chrono_literals;  // 使用时间字面量
using namespace std::placeholders;    // 使用占位符

// ResultViewer类的构造函数
ResultViewer::ResultViewer() : Node("result_viewer") {
    // 声明节点参数
    declare_parameter("image_topic", "detection_result");  // 图像话题名称
    declare_parameter("detection_topic", "detections");    // 检测结果话题名称
    declare_parameter("classes_file", "");                 // 类别文件路径
    
    // 获取参数值
    get_parameter("image_topic", image_topic_);
    get_parameter("detection_topic", detection_topic_);
    get_parameter("classes_file", classes_file_);
    
    // 打印参数信息
    RCLCPP_INFO(get_logger(), "Result viewer parameters:");
    RCLCPP_INFO(get_logger(), "  image_topic: %s", image_topic_.c_str());
    RCLCPP_INFO(get_logger(), "  detection_topic: %s", detection_topic_.c_str());
    
    // 检查类别文件路径是否有效
    if (classes_file_.empty()) {
        RCLCPP_ERROR(get_logger(), "classes_file parameter is not set!");
        rclcpp::shutdown();  // 如果未设置，则关闭节点
    }
    load_class_list();  // 加载类别列表
    RCLCPP_INFO(get_logger(), "  classes_file: %s", classes_file_.c_str());
    RCLCPP_INFO(get_logger(), "Result viewer node started");
}

// 析构函数
ResultViewer::~ResultViewer() {
    // 关闭所有OpenCV窗口
    cv::destroyAllWindows();
    RCLCPP_INFO(get_logger(), "OpenCV windows closed");
    RCLCPP_INFO(get_logger(), "ResultViewer node stopped");
}

// 初始化函数
void ResultViewer::init() {
    // 创建图像传输对象
    it_ = std::make_shared<image_transport::ImageTransport>(shared_from_this());
    
    // 订阅图像话题
    //image_sub_ = it_->subscribe(image_topic_, 1, std::bind(&ResultViewer::image_callback, this, _1));
    // 订阅相机话题
    image_sub_ = it_->subscribe("camera/image_raw", 1, std::bind(&ResultViewer::image_callback, this, _1));
    
    // 订阅检测结果话题
    detection_sub_ = create_subscription<vision_msgs::msg::Detection2DArray>(detection_topic_, 10, std::bind(&ResultViewer::detection_callback, this, _1));
}

// 主运行循环
void ResultViewer::run() {
    rclcpp::Rate rate(30); // 设置刷新率为30Hz
    
    while (rclcpp::ok()) {  // 节点正常运行循环
        {
            // 使用互斥锁保护共享数据
            std::lock_guard<std::mutex> lock(data_mutex_);
            
            // 当有新数据且当前帧不为空时
            if (new_data_available_ && !current_frame_.empty()) {
                // 克隆当前帧用于显示
                cv::Mat display_frame = current_frame_.clone();
                
                // 在图像上绘制检测结果
                draw_detections(display_frame);
                
                // 显示结果图像
                cv::imshow("Object Detection Results", display_frame);
                cv::waitKey(1);  // 必要的OpenCV事件处理
                
                new_data_available_ = false;  // 重置新数据标志
            }
        }
        
        // 处理ROS回调
        rclcpp::spin_some(shared_from_this());
        rate.sleep();  // 控制循环频率
    }
    
    cv::destroyAllWindows();  // 退出时关闭所有窗口
}

// 加载类别列表
void ResultViewer::load_class_list() {
    std::ifstream ifs(classes_file_);  // 打开类别文件
    if (!ifs.is_open()) {
        RCLCPP_ERROR(get_logger(), "Failed to open classes file: %s", classes_file_.c_str());
        rclcpp::shutdown();  // 文件打开失败则关闭节点
    }
    
    std::string line;
    while (getline(ifs, line)) {  // 逐行读取文件
        class_list_.push_back(line);  // 添加到类别列表
    }
    RCLCPP_INFO(get_logger(), "Loaded %zu classes", class_list_.size());
}

// 图像回调函数
void ResultViewer::image_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
    try {
        std::lock_guard<std::mutex> lock(data_mutex_);  // 加锁保护共享数据
        
        // 将ROS图像消息转换为OpenCV格式
        current_frame_ = cv_bridge::toCvCopy(msg, "bgr8")->image;
        new_data_available_ = true;  // 设置新数据标志
    } catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
    }
}

// 检测结果回调函数
void ResultViewer::detection_callback(const vision_msgs::msg::Detection2DArray::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(data_mutex_);  // 加锁保护共享数据
    current_detections_ = *msg;  // 保存检测结果
}

// 在图像上绘制检测结果
void ResultViewer::draw_detections(cv::Mat& image) {
    // 预定义的颜色列表，用于不同类别的边界框
    const std::vector<cv::Scalar> colors = {
        cv::Scalar(255, 255, 0),  // 黄色
        cv::Scalar(0, 255, 0),     // 绿色
        cv::Scalar(0, 255, 255),   // 青色
        cv::Scalar(255, 0, 0),     // 蓝色
        cv::Scalar(255, 0, 255),   // 紫色
        cv::Scalar(0, 0, 255)      // 红色
    };
    
    // 遍历所有检测结果
    for (const auto& detection : current_detections_.detections) {
        if (detection.results.empty()) continue;  // 跳过无结果的检测
        
        // 获取类别ID和置信度
        const std::string& class_id_str = detection.results[0].hypothesis.class_id;
        const float score = detection.results[0].hypothesis.score;
        
        // 将字符串形式的类别ID转换为整数
        int class_id = -1;
        try {
            class_id = std::stoi(class_id_str);  // 字符串转整数
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "Invalid class ID format: %s", class_id_str.c_str());
            continue;  // 转换失败则跳过当前检测
        }
        
        // 验证类别ID是否在有效范围内
        if (class_id < 0 || static_cast<size_t>(class_id) >= class_list_.size()) {
            RCLCPP_WARN(get_logger(), "Class ID out of range: %d", class_id);
            continue;  // 无效ID则跳过
        }
        
        // 根据类别ID选择颜色（循环使用颜色列表）
        const cv::Scalar color = colors[class_id % colors.size()];
        
        // 提取边界框信息（中心坐标和尺寸）
        const float center_x = detection.bbox.center.position.x;
        const float center_y = detection.bbox.center.position.y;
        const float width = detection.bbox.size_x;
        const float height = detection.bbox.size_y;
        
        // 转换为OpenCV矩形表示（左上角坐标 + 宽高）
        cv::Rect box(
            static_cast<int>(center_x - width / 2),  // 左上角x坐标
            static_cast<int>(center_y - height / 2),  // 左上角y坐标
            static_cast<int>(width),                 // 宽度
            static_cast<int>(height)                 // 高度
        );
        
        // 确保边界框在图像范围内（防止越界）
        if (box.x < 0) box.x = 0;
        if (box.y < 0) box.y = 0;
        if (box.x + box.width > image.cols) box.width = image.cols - box.x;
        if (box.y + box.height > image.rows) box.height = image.rows - box.y;
        
        // 绘制边界框（2像素宽度）
        cv::rectangle(image, box, color, 2);
        
        // 创建标签文本（类别名 + 置信度）
        std::string label = cv::format("%s: %.2f", class_list_[class_id].c_str(), score);
        
        // 计算文本尺寸
        int baseline;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,  0.5, 1, &baseline);
        
        // 确定标签位置（避免超出图像顶部）
        int text_y = box.y - 5;  // 默认在框上方
        if (text_y < text_size.height + 5) {
            text_y = box.y + box.height + text_size.height + 5;  // 如果超出顶部则放在框下方
        }
        
        // 绘制标签背景（填充矩形）
        cv::rectangle(image, 
            cv::Point(box.x, text_y - text_size.height - 5),  // 左上角
            cv::Point(box.x + text_size.width, text_y),       // 右下角
            color, cv::FILLED);                              // 填充颜色
        
        // 在背景上绘制标签文本
        cv::putText(image, label,
            cv::Point(box.x, text_y - 5),  // 文本位置
            cv::FONT_HERSHEY_SIMPLEX,       // 字体类型
            0.5,                           // 字体大小
            cv::Scalar(255, 255, 255),     // 白色文本
            1);                            // 线宽
    }
}

// 主函数
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);  // 初始化ROS2
    
    // 创建节点实例
    auto node = std::make_shared<ResultViewer>();
    
    node->init();  // 初始化节点
    node->run();   // 运行主循环
    
    rclcpp::shutdown();  // 关闭ROS2
    return 0;
}
