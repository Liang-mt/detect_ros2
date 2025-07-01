#ifndef OBJECT_DETECTOR_H
#define OBJECT_DETECTOR_H

#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <vision_msgs/msg/detection2_d.hpp>
#include <vision_msgs/msg/object_hypothesis_with_pose.hpp>

class ObjectDetector : public rclcpp::Node {
public:
    ObjectDetector();
    ~ObjectDetector();
    void init();
private:
    struct Detection {
        int class_id;
        float confidence;
        cv::Rect box;
    };
    
    std::shared_ptr<image_transport::ImageTransport> it_;
    image_transport::Subscriber image_sub_;
    rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr detection_pub_;
    image_transport::Publisher result_image_pub_;
    
    // 参数
    std::string model_path_;
    std::string classes_file_;
    bool use_cuda_;
    float conf_threshold_;
    float nms_threshold_;
    float score_threshold_;
    int input_width_;
    int input_height_;
    
    // 网络和类列表
    cv::dnn::Net net_;
    std::vector<std::string> class_list_;
    
    // 颜色列表
    const std::vector<cv::Scalar> colors_ = {
        cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0),
        cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0)
    };

    void load_parameters();

    void load_net();

    void load_class_list();

    cv::Mat format_yolov5(const cv::Mat& source);

    void detect(const cv::Mat& image, std::vector<Detection>& output);

    void draw_boxes(cv::Mat& image, const std::vector<Detection>& output);

    vision_msgs::msg::Detection2DArray to_detection_msg(
        const std::vector<Detection>& detections, 
        const std_msgs::msg::Header& header);

    void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg);
};

#endif
