#include "video_publisher/video_publisher_node.hpp"

using namespace std::chrono_literals;

VideoPublisher::VideoPublisher() : Node("video_publisher") {
    // 使用参数声明
    declare_parameter("video_source", "0");
    declare_parameter("is_video_file", false);
    declare_parameter("frame_rate", 30);

    // 获取参数
    get_parameter("video_source", video_source_);
    get_parameter("is_video_file", is_video_file_);
    get_parameter("frame_rate", frame_rate_);

    // 打印参数值
    RCLCPP_INFO(this->get_logger(), "Loaded parameters:");
    RCLCPP_INFO(this->get_logger(), "  video_source: %s", video_source_.c_str());
    RCLCPP_INFO(this->get_logger(), "  is_video_file: %s", is_video_file_ ? "true" : "false");
    RCLCPP_INFO(this->get_logger(), "  frame_rate: %d Hz", frame_rate_);

    // 检查视频文件是否存在
    if (is_video_file_) {
        if (access(video_source_.c_str(), F_OK) == -1) {
            RCLCPP_ERROR(this->get_logger(), "Video file does not exist: %s", video_source_.c_str());
            rclcpp::shutdown();
            return;
        }
    }

    // 初始化视频源
    if (!init_video_source()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to initialize video source");
        rclcpp::shutdown();
        return;
    }

    RCLCPP_INFO(this->get_logger(), "Video publisher node started");
}

VideoPublisher::~VideoPublisher() {
    // 释放视频捕获资源
    if (cap_.isOpened()) {
        cap_.release();
        RCLCPP_INFO(this->get_logger(), "Video capture released");
    }

    RCLCPP_INFO(this->get_logger(), "VideoPublisher node stopped");
}

void VideoPublisher::run() {
    // 在 run 方法中创建图像传输接口
    it_ = std::make_shared<image_transport::ImageTransport>(shared_from_this());
    image_pub_ = it_->advertise("camera/image_raw", 1);

    rclcpp::Rate rate(frame_rate_);
    cv::Mat frame;

    while (rclcpp::ok()) {
        if (!cap_.read(frame)) {
            if (is_video_file_) {
                // 重置视频位置并跳过当前循环
                cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
                continue;
            } else {
                RCLCPP_ERROR(this->get_logger(), "Failed to read frame from video source");
                break;
            }
        }

        // 发布图像消息
        auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame).toImageMsg();
        image_pub_.publish(*msg);

        rclcpp::spin_some(shared_from_this());
        rate.sleep();
    }
}

bool VideoPublisher::init_video_source() {
    if (is_video_file_) {
        // 打开视频文件
        cap_.open(video_source_);
        if (!cap_.isOpened()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open video file: %s", video_source_.c_str());
            return false;
        }
        RCLCPP_INFO(this->get_logger(), "Successfully opened video file: %s", video_source_.c_str());
        return true;
    } else {
        // 尝试解析为摄像头索引
        try {
            int camera_index = std::stoi(video_source_);
            if (cap_.open(camera_index, cv::CAP_V4L2)) {
                // 设置摄像头参数
                cap_.set(cv::CAP_PROP_FRAME_WIDTH, 640);
                cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
                cap_.set(cv::CAP_PROP_FPS, frame_rate_);
                RCLCPP_INFO(this->get_logger(), "Successfully opened camera device: %d", camera_index);
                return true;
            }
        } catch (...) {
            // 不是数字，继续尝试其他方式
        }
        
        // 尝试直接打开设备路径
        if (cap_.open(video_source_, cv::CAP_V4L2)) {
            // 设置摄像头参数
            cap_.set(cv::CAP_PROP_FRAME_WIDTH, 640);
            cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
            cap_.set(cv::CAP_PROP_FPS, frame_rate_);
            RCLCPP_INFO(this->get_logger(), "Successfully opened video device: %s", video_source_.c_str());
            return true;
        }
        
        // 所有尝试都失败
        RCLCPP_ERROR(this->get_logger(), "Failed to open video source: %s", video_source_.c_str());
        return false;
    }
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<VideoPublisher>();
    node->run();
    rclcpp::shutdown();
    return 0;
}
