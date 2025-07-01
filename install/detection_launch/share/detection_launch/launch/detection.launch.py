import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # 获取object_detector包的配置文件路径
    config_dir = "/home/lemon/Documents/detect_ros2_ws/src/object_detector/config_files"
    
    # 定义可配置参数
    use_cuda_arg = DeclareLaunchArgument(
        'use_cuda',
        default_value='false',
        description='Use CUDA acceleration for object detection'
    )
    
    video_source_arg = DeclareLaunchArgument(
        'video_source',
        default_value=os.path.join(config_dir, 'test.mp4'),
        description='Path to video file or camera device'
    )
    
    return LaunchDescription([
        use_cuda_arg,
        video_source_arg,
        
        # 视频发布节点
        Node(
            package='video_publisher',
            executable='video_publisher_node',
            name='video_publisher',
            output='screen',
            parameters=[{
                'video_source': LaunchConfiguration('video_source'),
                'is_video_file': True,
                'frame_rate': 30
            }]
        ),
        
        # 目标检测节点
        Node(
            package='object_detector',
            executable='object_detection_node',
            name='object_detection',
            output='screen',
            parameters=[{
                'model_path': os.path.join(config_dir, 'yolov5s.onnx'),
                'classes_file': os.path.join(config_dir, 'classes.txt'),
                'use_cuda': LaunchConfiguration('use_cuda'),
                'conf_threshold': 0.4,
                'nms_threshold': 0.4,
                'score_threshold': 0.2,
                'input_width': 640,
                'input_height': 640
            }]
        ),
        
        # 结果查看器
        Node(
            package='result_viewer',
            executable='result_viewer_node',
            name='result_viewer',
            output='screen',
            parameters=[{
                'classes_file': os.path.join(config_dir, 'classes.txt'),
                'image_topic': 'detection_result',
                'detection_topic': 'detections'
            }]
        )
    ])
