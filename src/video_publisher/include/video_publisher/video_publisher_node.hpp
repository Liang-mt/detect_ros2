#ifndef VIDEO_PUBLISHER_NODE_HPP
#define VIDEO_PUBLISHER_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <unistd.h>

class VideoPublisher : public rclcpp::Node {
public:
    VideoPublisher();
    ~VideoPublisher();
    void run();

private:
    std::shared_ptr<image_transport::ImageTransport> it_;
    image_transport::Publisher image_pub_;
    cv::VideoCapture cap_;
    std::string video_source_;
    bool is_video_file_;
    int frame_rate_;

    bool init_video_source();
};

#endif
