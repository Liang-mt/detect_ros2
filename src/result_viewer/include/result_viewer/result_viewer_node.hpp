#ifndef RESULT_VIEWER_NODE_HPP
#define RESULT_VIEWER_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <mutex>
#include <fstream>
#include <cstdlib> // 添加 for std::stoi

class ResultViewer : public rclcpp::Node {
public:
    ResultViewer();
    ~ResultViewer();
    void init();
    void run();
private:
    std::shared_ptr<image_transport::ImageTransport> it_;
    image_transport::Subscriber image_sub_;
    rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr detection_sub_;
    
    // 参数
    std::string image_topic_;
    std::string detection_topic_;
    std::string classes_file_;
    std::vector<std::string> class_list_;

    cv::Mat current_frame_;
    vision_msgs::msg::Detection2DArray current_detections_;
    std::mutex data_mutex_;
    bool new_data_available_ = false;

    void load_class_list();

    void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg);

    void detection_callback(const vision_msgs::msg::Detection2DArray::SharedPtr msg);

    void draw_detections(cv::Mat& image);
};

#endif
