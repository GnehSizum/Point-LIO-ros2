// #include <so3_math.h>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <tf2/time.h>
#include <tf2/convert.h>
#include <tf2/transform_datatypes.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include "li_initialization.h"
#include <malloc.h>
// #include <cv_bridge/cv_bridge.h>
// #include "matplotlibcpp.h"

using namespace std;     

#define PUBFRAME_PERIOD     (20)

const float MOV_THRESHOLD = 1.5f;

string root_dir = ROOT_DIR;

int time_log_counter = 0; //, publish_count = 0;

bool init_map = false, flg_first_scan = true;

// Time Log Variables
double match_time = 0, solve_time = 0, propag_time = 0, update_time = 0;

bool  flg_reset = false, flg_exit = false;

//surf feature in map
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body_space(new PointCloudXYZI());
PointCloudXYZI::Ptr init_feats_world(new PointCloudXYZI());
std::deque<PointCloudXYZI::Ptr> depth_feats_world;
pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

V3D euler_cur;

nav_msgs::msg::Path path;
nav_msgs::msg::Odometry odomAftMapped;
geometry_msgs::msg::PoseStamped msg_body_pose;

void SigHandle(int sig)
{
    flg_exit = true;
    printf("catch sig %d", sig);
    sig_buffer.notify_all();
}

inline void dump_lio_state_to_log(FILE *fp)  
{
    V3D rot_ang;
    if (!use_imu_as_input)
    {
        rot_ang = SO3ToEuler(kf_output.x_.rot);
    }
    else
    {
        rot_ang = SO3ToEuler(kf_input.x_.rot);
    }
    
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                   // Angle
    if (use_imu_as_input)
    {
        fprintf(fp, "%lf %lf %lf ", kf_input.x_.pos(0), kf_input.x_.pos(1), kf_input.x_.pos(2)); // Pos  
        fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega  
        fprintf(fp, "%lf %lf %lf ", kf_input.x_.vel(0), kf_input.x_.vel(1), kf_input.x_.vel(2)); // Vel  
        fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc  
        fprintf(fp, "%lf %lf %lf ", kf_input.x_.bg(0), kf_input.x_.bg(1), kf_input.x_.bg(2));    // Bias_g  
        fprintf(fp, "%lf %lf %lf ", kf_input.x_.ba(0), kf_input.x_.ba(1), kf_input.x_.ba(2));    // Bias_a  
        fprintf(fp, "%lf %lf %lf ", kf_input.x_.gravity(0), kf_input.x_.gravity(1), kf_input.x_.gravity(2)); // Bias_a  
    }
    else
    {
        fprintf(fp, "%lf %lf %lf ", kf_output.x_.pos(0), kf_output.x_.pos(1), kf_output.x_.pos(2)); // Pos  
        fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega  
        fprintf(fp, "%lf %lf %lf ", kf_output.x_.vel(0), kf_output.x_.vel(1), kf_output.x_.vel(2)); // Vel  
        fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc  
        fprintf(fp, "%lf %lf %lf ", kf_output.x_.bg(0), kf_output.x_.bg(1), kf_output.x_.bg(2));    // Bias_g  
        fprintf(fp, "%lf %lf %lf ", kf_output.x_.ba(0), kf_output.x_.ba(1), kf_output.x_.ba(2));    // Bias_a  
        fprintf(fp, "%lf %lf %lf ", kf_output.x_.gravity(0), kf_output.x_.gravity(1), kf_output.x_.gravity(2)); // Bias_a  
    }
    fprintf(fp, "\r\n");  
    fflush(fp);
}

void pointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu;
    if (extrinsic_est_en)
    {
        if (!use_imu_as_input)
        {
            p_body_imu = kf_output.x_.offset_R_L_I * p_body_lidar + kf_output.x_.offset_T_L_I;
        }
        else
        {
            p_body_imu = kf_input.x_.offset_R_L_I * p_body_lidar + kf_input.x_.offset_T_L_I;
        }
    }
    else
    {
        p_body_imu = Lidar_R_wrt_IMU * p_body_lidar + Lidar_T_wrt_IMU;
    }
    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void MapIncremental() {
    PointVector points_to_add;
    int cur_pts = feats_down_world->size();
    points_to_add.reserve(cur_pts);

    for (size_t i = 0; i < cur_pts; ++i) {
        /* decide if need add to map */
        PointType &point_world = feats_down_world->points[i];
        if (!Nearest_Points[i].empty()) {
            const PointVector &points_near = Nearest_Points[i];

            Eigen::Vector3f center =
                ((point_world.getVector3fMap() / filter_size_map_min).array().floor() + 0.5) * filter_size_map_min;
            bool need_add = true;
            for (int readd_i = 0; readd_i < points_near.size(); readd_i++) {
                Eigen::Vector3f dis_2_center = points_near[readd_i].getVector3fMap() - center;
                if (fabs(dis_2_center.x()) < 0.5 * filter_size_map_min &&
                    fabs(dis_2_center.y()) < 0.5 * filter_size_map_min &&
                    fabs(dis_2_center.z()) < 0.5 * filter_size_map_min) {
                    need_add = false;
                    break;
                }
            }
            if (need_add) {
                points_to_add.emplace_back(point_world);
            }
        } else {
            points_to_add.emplace_back(point_world);
        }
    }
    ivox_->AddPoints(points_to_add);
}

void publish_init_map(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFullRes)
{
    int size_init_map = init_feats_world->size();

    // *pcl_wait_pub += *init_feats_world;
    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*init_feats_world, laserCloudmsg);
    // pcl::toROSMsg(*pcl_wait_pub, laserCloudmsg);
        
    laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
    laserCloudmsg.header.frame_id = "odom";
    pubLaserCloudFullRes->publish(laserCloudmsg);
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());

void publish_map(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFullRes)
{
    PointCloudXYZI::Ptr laserCloudFullRes(feats_down_body);
    int size = laserCloudFullRes->points.size();

    PointCloudXYZI::Ptr   laserCloudWorld(new PointCloudXYZI(size, 1));
    
    for (int i = 0; i < size; i++)
    {
        laserCloudWorld->points[i].x = feats_down_world->points[i].x;
        laserCloudWorld->points[i].y = feats_down_world->points[i].y;
        laserCloudWorld->points[i].z = feats_down_world->points[i].z;
        laserCloudWorld->points[i].intensity = feats_down_world->points[i].intensity; // feats_down_world->points[i].y; //
    }
    *pcl_wait_pub += *laserCloudWorld;
    sensor_msgs::msg::PointCloud2 laserCloudmap;
    pcl::toROSMsg(*pcl_wait_pub, laserCloudmap);
    
    laserCloudmap.header.stamp = get_ros_time(lidar_end_time);
    laserCloudmap.header.frame_id = "odom";
    pubLaserCloudFullRes->publish(laserCloudmap);
}

void publish_frame_world(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFullRes)
{
    if (scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(feats_down_body);
        // PointCloudXYZI::Ptr laserCloudFullRes(feats_undistort);
        int size = laserCloudFullRes->points.size();

        PointCloudXYZI::Ptr   laserCloudWorld(new PointCloudXYZI(size, 1));
        
        for (int i = 0; i < size; i++)
        {
            // laserCloudWorld->points[i].x = feats_undistort->points[i].x;
            // laserCloudWorld->points[i].y = feats_undistort->points[i].y;
            // laserCloudWorld->points[i].z = feats_undistort->points[i].z;
            // laserCloudWorld->points[i].intensity = feats_undistort->points[i].intensity;
            laserCloudWorld->points[i].x = feats_down_world->points[i].x;
            laserCloudWorld->points[i].y = feats_down_world->points[i].y;
            laserCloudWorld->points[i].z = feats_down_world->points[i].z;
            laserCloudWorld->points[i].intensity = feats_down_world->points[i].intensity; // feats_down_world->points[i].y; //
        }
        sensor_msgs::msg::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        
        laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
        laserCloudmsg.header.frame_id = "odom";
        pubLaserCloudFullRes->publish(laserCloudmsg);
        // publish_count -= PUBFRAME_PERIOD;
    }
    
    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en)
    {
        int size = feats_down_world->points.size();
        PointCloudXYZI::Ptr   laserCloudWorld(new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            laserCloudWorld->points[i].x = feats_down_world->points[i].x;
            laserCloudWorld->points[i].y = feats_down_world->points[i].y;
            laserCloudWorld->points[i].z = feats_down_world->points[i].z;
            laserCloudWorld->points[i].intensity = feats_down_world->points[i].intensity;
        }

        *pcl_wait_save += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num ++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0  && scan_wait_num >= pcd_save_interval)
        {
            pcd_index ++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
}

void publish_frame_body(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        pointBodyLidarToIMU(&feats_undistort->points[i], \
                            &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body->publish(laserCloudmsg);
    // publish_count -= PUBFRAME_PERIOD;
}

template<typename T>
void set_posestamp(T & out)
{
    if (!use_imu_as_input)
    {
        out.position.x = kf_output.x_.pos(0);
        out.position.y = kf_output.x_.pos(1);
        out.position.z = kf_output.x_.pos(2);
        Eigen::Quaterniond q(kf_output.x_.rot);
        out.orientation.x = q.coeffs()[0];
        out.orientation.y = q.coeffs()[1];
        out.orientation.z = q.coeffs()[2];
        out.orientation.w = q.coeffs()[3];
    }
    else
    {
        out.position.x = kf_input.x_.pos(0);
        out.position.y = kf_input.x_.pos(1);
        out.position.z = kf_input.x_.pos(2);
        Eigen::Quaterniond q(kf_input.x_.rot);
        out.orientation.x = q.coeffs()[0];
        out.orientation.y = q.coeffs()[1];
        out.orientation.z = q.coeffs()[2];
        out.orientation.w = q.coeffs()[3];
    }
}

void publish_odometry(const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped, std::unique_ptr<tf2_ros::TransformBroadcaster> & tf_br)
{
    odomAftMapped.header.frame_id = "odom";
    odomAftMapped.child_frame_id = "livox_frame";
    odomAftMapped.header.stamp = get_ros_time(lidar_end_time);
    set_posestamp(odomAftMapped.pose.pose);
    pubOdomAftMapped->publish(odomAftMapped);

    tf2::Quaternion q;
    tf2::Transform tf;
    tf.setOrigin(tf2::Vector3(odomAftMapped.pose.pose.position.x,
                              odomAftMapped.pose.pose.position.y,
                              odomAftMapped.pose.pose.position.z));
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    q.setW(odomAftMapped.pose.pose.orientation.w);
    tf.setRotation(q);

    tf2::TimePoint tp = tf2_ros::fromRclcpp(get_ros_time(lidar_end_time));
    tf2::Stamped<tf2::Transform> stamptf(tf, tp, "odom");
    geometry_msgs::msg::TransformStamped tf_stamped;
    tf2::convert(stamptf, tf_stamped);
    tf_stamped.child_frame_id = "livox_frame";
    tf_br->sendTransform(tf_stamped);
}

void publish_path(rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath)
{
    set_posestamp(msg_body_pose.pose);
    msg_body_pose.header.stamp = get_ros_time(lidar_end_time);
    msg_body_pose.header.frame_id = "odom";
    static int jjj = 0;
    jjj++;
    // if (jjj % 2 == 0) // if path is too large, the rvis will crash
    {
        path.poses.emplace_back(msg_body_pose);
        pubPath->publish(path);
    }
}        

class LaserMappingNode : public rclcpp::Node
{
public:
    LaserMappingNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions()) : Node("laser_mapping", options)
    {
        // 布尔型参数
        declare_and_get_parameter<bool>("prop_at_freq_of_imu", prop_at_freq_of_imu, true);
        declare_and_get_parameter<bool>("use_imu_as_input", use_imu_as_input, true);
        declare_and_get_parameter<bool>("check_satu", check_satu, true);
        declare_and_get_parameter<bool>("space_down_sample", space_down_sample, true);
        declare_and_get_parameter<bool>("common.con_frame", con_frame, false);
        declare_and_get_parameter<bool>("common.cut_frame", cut_frame, false);
        declare_and_get_parameter<bool>("mapping.imu_en", imu_en, true);
        declare_and_get_parameter<bool>("mapping.extrinsic_est_en", extrinsic_est_en, true);
        declare_and_get_parameter<bool>("publish.path_en", path_en, true);
        declare_and_get_parameter<bool>("publish.scan_publish_en", scan_pub_en, true);
        declare_and_get_parameter<bool>("publish.scan_bodyframe_pub_en", scan_body_pub_en, true);
        declare_and_get_parameter<bool>("runtime_pos_log_enable", runtime_pos_log, false);
        declare_and_get_parameter<bool>("pcd_save.pcd_save_en", pcd_save_en, false);

        // 整数型参数
        declare_and_get_parameter<int>("init_map_size", init_map_size, 100);
        declare_and_get_parameter<int>("point_filter_num", p_pre->point_filter_num, 2);
        declare_and_get_parameter<int>("common.con_frame_num", con_frame_num, 1);
        declare_and_get_parameter<int>("preprocess.lidar_type", lidar_type, 1);
        // declare_and_get_parameter<int>("preprocess.lidar_type", p_pre->lidar_type, 1);
        declare_and_get_parameter<int>("preprocess.scan_line", p_pre->N_SCANS, 16);
        declare_and_get_parameter<int>("preprocess.scan_rate", p_pre->SCAN_RATE, 10);
        declare_and_get_parameter<int>("preprocess.timestamp_unit", p_pre->time_unit, 1);
        declare_and_get_parameter<int>("pcd_save.interval", pcd_save_interval, -1);
        declare_and_get_parameter<int>("ivox_nearby_type", ivox_nearby_type, 18);

        // 浮点型参数
        declare_and_get_parameter<double>("mapping.satu_acc", satu_acc, 3.0);
        declare_and_get_parameter<double>("mapping.satu_gyro", satu_gyro, 35.0);
        declare_and_get_parameter<double>("mapping.acc_norm", acc_norm, 1.0);
        declare_and_get_parameter<float>("mapping.plane_thr", plane_thr, 0.05f);
        declare_and_get_parameter<double>("common.cut_frame_time_interval", cut_frame_time_interval, 0.1);
        declare_and_get_parameter<double>("common.time_diff_lidar_to_imu",time_diff_lidar_to_imu,0.0);
        declare_and_get_parameter<double>("filter_size_surf", filter_size_surf_min, 0.5);
        declare_and_get_parameter<double>("filter_size_map", filter_size_map_min, 0.5);
        declare_and_get_parameter<float>("mapping.det_range", DET_RANGE, 300.f);
        declare_and_get_parameter<double>("mapping.fov_degree", fov_deg, 180);
        declare_and_get_parameter<double>("mapping.imu_time_inte", imu_time_inte, 0.005);
        declare_and_get_parameter<double>("mapping.lidar_meas_cov", laser_point_cov, 0.1);
        declare_and_get_parameter<double>("mapping.acc_cov_input", acc_cov_input, 0.1);
        declare_and_get_parameter<double>("mapping.vel_cov", vel_cov, 20);
        declare_and_get_parameter<double>("mapping.gyr_cov_input", gyr_cov_input, 0.1);
        declare_and_get_parameter<double>("mapping.gyr_cov_output", gyr_cov_output, 0.1);
        declare_and_get_parameter<double>("mapping.acc_cov_output", acc_cov_output, 0.1);
        declare_and_get_parameter<double>("mapping.b_gyr_cov", b_gyr_cov, 0.0001);
        declare_and_get_parameter<double>("mapping.b_acc_cov", b_acc_cov, 0.0001);
        declare_and_get_parameter<double>("mapping.imu_meas_acc_cov", imu_meas_acc_cov, 0.1);
        declare_and_get_parameter<double>("mapping.imu_meas_omg_cov", imu_meas_omg_cov, 0.1);
        declare_and_get_parameter<double>("mapping.match_s", match_s, 81.0);
        declare_and_get_parameter<double>("preprocess.blind", p_pre->blind, 1.0);
        declare_and_get_parameter<double>("mapping.lidar_time_inte",lidar_time_inte,0.1);
        declare_and_get_parameter<float>("mapping.ivox_grid_resolution", ivox_options_.resolution_, 0.2);

        // 字符串型参数
        declare_and_get_parameter<std::string>("common.lid_topic", lid_topic, "/livox/lidar");
        declare_and_get_parameter<std::string>("common.imu_topic", imu_topic, "/livox/imu");

        // 向量型参数
        declare_and_get_parameter<std::vector<double>>("mapping.gravity", gravity, std::vector<double>());
        declare_and_get_parameter<std::vector<double>>("mapping.gravity_init", gravity_init, std::vector<double>());
        declare_and_get_parameter<std::vector<double>>("mapping.extrinsic_T", extrinT, std::vector<double>());
        declare_and_get_parameter<std::vector<double>>("mapping.extrinsic_R", extrinR, std::vector<double>());

        if (ivox_nearby_type == 0) {
            ivox_options_.nearby_type_ = IVoxType::NearbyType::CENTER;
        } else if (ivox_nearby_type == 6) {
            ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY6;
        } else if (ivox_nearby_type == 18) {
            ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
        } else if (ivox_nearby_type == 26) {
            ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY26;
        } else {
            ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
        }
        p_imu->gravity_ << VEC_FROM_ARRAY(gravity);

        /*** ROS subscription initialization ***/
        if (p_pre->lidar_type == AVIA) {
            sub_pcl_livox = this->create_subscription<livox_ros_driver2::msg::CustomMsg>(
                lid_topic, rclcpp::QoS(200000),
                [this](livox_ros_driver2::msg::CustomMsg::UniquePtr msg) {
                    livox_pcl_cbk(std::move(msg));
                });
        } else {
            sub_pcl_pc = this->create_subscription<sensor_msgs::msg::PointCloud2>(
                lid_topic, rclcpp::QoS(200000),
                [this](sensor_msgs::msg::PointCloud2::UniquePtr msg) {
                    standard_pcl_cbk(std::move(msg));
                });
        }
        // sub_imu = this->create_subscription<sensor_msgs::msg::Imu>(
        //     imu_topic, rclcpp::QoS(200000),
        //     [this](const sensor_msgs::msg::Imu::SharedPtr msg) {
        //         imu_cbk(msg);
        //     });
        sub_imu = this->create_subscription<sensor_msgs::msg::Imu>(imu_topic, 10, imu_cbk);

        /*** ROS publisher initialization ***/
        pubLaserCloudFullRes = this->create_publisher<sensor_msgs::msg::PointCloud2>
            ("/cloud_registered", rclcpp::QoS(100000));       
        pubLaserCloudFullRes_body = this->create_publisher<sensor_msgs::msg::PointCloud2>
            ("/cloud_registered_body", rclcpp::QoS(100000));
        pubLaserCloudEffect = this->create_publisher<sensor_msgs::msg::PointCloud2>
            ("/cloud_effect", rclcpp::QoS(100000));
        pubLaserCloudMap = this->create_publisher<sensor_msgs::msg::PointCloud2>
            ("/Laser_map", rclcpp::QoS(100000));
        pubOdomAftMapped = this->create_publisher<nav_msgs::msg::Odometry> 
            ("/base_link_to_init", rclcpp::QoS(100000));
        pubPath = this->create_publisher<nav_msgs::msg::Path> 
            ("/path", rclcpp::QoS(100000));
        plane_pub = this->create_publisher<visualization_msgs::msg::Marker>
            ("/planner_normal", rclcpp::QoS(1000));
        // auto period_ms = std::chrono::milliseconds(static_cast<int64_t>(1000.0 / 100.0));
        auto period_ms = std::chrono::milliseconds(static_cast<int64_t>(500.0 / 100.0));
        timer = rclcpp::create_timer(this, this->get_clock(), period_ms, std::bind(&LaserMappingNode::timer_callback, this));
        tf_buffer = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        transform_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

        ivox_ = std::make_shared<IVoxType>(ivox_options_);
        path.header.stamp = this->get_clock()->now();
        path.header.frame_id ="odom";

        /*** variables definition for counting ***/
        frame_num = 0;
        aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_propag = 0;

        // /*** initialize variables ***/
        // FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
        // HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

        memset(point_selected_surf, true, sizeof(point_selected_surf));
        downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
        downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
        Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
        Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);

        if (extrinsic_est_en)
        {
            if (!use_imu_as_input)
            {
                kf_output.x_.offset_R_L_I = Lidar_R_wrt_IMU;
                kf_output.x_.offset_T_L_I = Lidar_T_wrt_IMU;
            }
            else
            {
                kf_input.x_.offset_R_L_I = Lidar_R_wrt_IMU;
                kf_input.x_.offset_T_L_I = Lidar_T_wrt_IMU;
            }
        }
        p_imu->lidar_type = p_pre->lidar_type = lidar_type;
        p_imu->imu_en = imu_en;
        printf("p_pre->lidar_type: %d", p_pre->lidar_type);

        kf_input.init_dyn_share_modified_2h(get_f_input, df_dx_input, h_model_input);
        kf_output.init_dyn_share_modified_3h(get_f_output, df_dx_output, h_model_output, h_model_IMU_output);
        P_init = Eigen::Matrix<double, 24, 24>::Zero();
        reset_cov(P_init);
        kf_input.change_P(P_init);
        P_init_output = Eigen::Matrix<double, 30, 30>::Zero();
        reset_cov_output(P_init_output);
        kf_output.change_P(P_init_output);
        Q_input = Eigen::Matrix<double, 24, 24>::Zero();
        Q_output = Eigen::Matrix<double, 30, 30>::Zero();
        Q_input = process_noise_cov_input();
        Q_output = process_noise_cov_output();
        /*** debug record ***/
        string pos_log_dir = root_dir + "/Log/pos_log.txt";
        fp = fopen(pos_log_dir.c_str(),"w");
        open_file();
    }

    ~LaserMappingNode()
    {
        fout_out.close();
        fout_imu_pbp.close();
        fclose(fp);
    }

    template<typename T>
    void declare_and_get_parameter(const std::string& name, T& variable, const T& default_value) {
        this->declare_parameter<T>(name, default_value);
        if (!this->get_parameter(name, variable)) {
            RCLCPP_WARN(this->get_logger(), "Failed to get parameter: %s, using default: %s", name.c_str(), to_string(default_value).c_str());
        } else {
            RCLCPP_INFO(this->get_logger(), "Parameter: %s, value: %s", name.c_str(), to_string(variable).c_str());
        }
    }

    std::string to_string(const std::string& value) {
        return value;
    }

    std::string to_string(const bool& value) {
        return value ? "true" : "false";
    }

    std::string to_string(const double& value) {
        return std::to_string(value);
    }

    std::string to_string(const float& value) {
        return std::to_string(value);
    }

    std::string to_string(const int& value) {
        return std::to_string(value);
    }

    template<typename T>
    std::string to_string(const std::vector<T>& value) {
        std::string result = "[";
        for (const auto& elem : value) {
            if (result != "[") result += ", ";
            result += to_string(elem);
        }
        return result + "]";
    }

private:
    void timer_callback()
    {
        if (flg_exit) return;
        if(sync_packages(Measures)) 
        {
            if (flg_reset)
            {
                printf("reset when rosbag play back");
                p_imu->Reset();
                feats_undistort.reset(new PointCloudXYZI());
                if (use_imu_as_input)
                {
                    // state_in = kf_input.get_x();
                    state_in = state_input();
                    kf_input.change_P(P_init);
                }
                else
                {
                    // state_out = kf_output.get_x();
                    state_out = state_output();
                    kf_output.change_P(P_init_output);
                }
                flg_first_scan = true;
                is_first_frame = true;
                flg_reset = false;
                init_map = false;
                
                {
                    ivox_.reset(new IVoxType(ivox_options_));
                }
            }

            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;
                flg_first_scan = false;
                if (first_imu_time < 1)
                {
                    first_imu_time = get_time_sec(imu_next.header.stamp);
                    printf("first imu time: %f\n", first_imu_time);
                }
                time_current = 0.0;
                if(imu_en)
                {
                    // imu_next = *(imu_deque.front());
                    kf_input.x_.gravity << VEC_FROM_ARRAY(gravity);
                    kf_output.x_.gravity << VEC_FROM_ARRAY(gravity);
                    // kf_output.x_.acc << VEC_FROM_ARRAY(gravity);
                    // kf_output.x_.acc *= -1; 

                    {
                        while (Measures.lidar_beg_time > get_time_sec(imu_next.header.stamp)) // if it is needed for the new map?
                        {
                            imu_deque.pop_front();
                            if (imu_deque.empty())
                            {
                                return;
                            }
                            imu_last = imu_next;
                            imu_next = *(imu_deque.front());
                            // imu_deque.pop();
                        }
                    }
                }
                else
                {
                    kf_input.x_.gravity << VEC_FROM_ARRAY(gravity_init);
                    kf_output.x_.gravity << VEC_FROM_ARRAY(gravity_init);
                    kf_output.x_.acc << VEC_FROM_ARRAY(gravity_init);
                    kf_output.x_.acc *= -1; 
                    p_imu->imu_need_init_ = false;
                    // p_imu->after_imu_init_ = true;
                }        
            }

            double t0,t1,t2,t3,t4,t5,match_start, solve_start;
            match_time = 0;
            solve_time = 0;
            propag_time = 0;
            update_time = 0;
            t0 = omp_get_wtime();
            
            /*** downsample the feature points in a scan ***/
            t1 = omp_get_wtime();
            p_imu->Process(Measures, feats_undistort);
            if(space_down_sample)
            {
                downSizeFilterSurf.setInputCloud(feats_undistort);
                downSizeFilterSurf.filter(*feats_down_body);
                sort(feats_down_body->points.begin(), feats_down_body->points.end(), time_list); 
            }
            else
            {
                feats_down_body = Measures.lidar;
                sort(feats_down_body->points.begin(), feats_down_body->points.end(), time_list); 
            }
            {
                time_seq = time_compressing<int>(feats_down_body);
                feats_down_size = feats_down_body->points.size();
            }
            if (!p_imu->after_imu_init_)
            {
                if (!p_imu->imu_need_init_)
                { 
                    V3D tmp_gravity;
                    if (imu_en)
                    {tmp_gravity = - p_imu->mean_acc / acc_norm * G_m_s2;}
                    else
                    {tmp_gravity << VEC_FROM_ARRAY(gravity_init);
                    p_imu->after_imu_init_ = true;
                    }
                    // V3D tmp_gravity << VEC_FROM_ARRAY(gravity_init);
                    M3D rot_init;
                    p_imu->Set_init(tmp_gravity, rot_init);  
                    kf_input.x_.rot = rot_init;
                    kf_output.x_.rot = rot_init;
                    // kf_input.x_.rot; //.normalize();
                    // kf_output.x_.rot; //.normalize();
                    kf_output.x_.acc = - rot_init.transpose() * kf_output.x_.gravity;
                }
                else{
                return;}
            }
            /*** initialize the map ***/
            if(!init_map)
            {
                feats_down_world->resize(feats_undistort->size());
                for(int i = 0; i < feats_undistort->size(); i++)
                {
                    {
                        pointBodyToWorld(&(feats_undistort->points[i]), &(feats_down_world->points[i]));
                    }
                }
                for (size_t i = 0; i < feats_down_world->size(); i++) 
                {
                    init_feats_world->points.emplace_back(feats_down_world->points[i]);
                }
                if(init_feats_world->size() < init_map_size) 
                {init_map = false;}
                else
                {   
                    ivox_->AddPoints(init_feats_world->points);
                    publish_init_map(pubLaserCloudMap); //(pubLaserCloudFullRes);
                    
                    init_feats_world.reset(new PointCloudXYZI());
                    init_map = true;
                }
                return;
            }

            /*** ICP and Kalman filter update ***/
            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);

            Nearest_Points.resize(feats_down_size);

            t2 = omp_get_wtime();
            
            /*** iterated state estimation ***/
            crossmat_list.reserve(feats_down_size);
            pbody_list.reserve(feats_down_size);
            // pbody_ext_list.reserve(feats_down_size);
                          
            for (size_t i = 0; i < feats_down_body->size(); i++)
            {
                V3D point_this(feats_down_body->points[i].x,
                            feats_down_body->points[i].y,
                            feats_down_body->points[i].z);
                pbody_list[i]=point_this;
                if (!extrinsic_est_en)
                {
                    point_this = Lidar_R_wrt_IMU * point_this + Lidar_T_wrt_IMU;
                    M3D point_crossmat;
                    point_crossmat << SKEW_SYM_MATRX(point_this);
                    crossmat_list[i]=point_crossmat;
                }
            }
            if (!use_imu_as_input)
            {     
                bool imu_upda_cov = false;
                effct_feat_num = 0;
                /**** point by point update ****/
                if (time_seq.size() > 0)
                {
                double pcl_beg_time = Measures.lidar_beg_time;
                idx = -1;
                for (k = 0; k < time_seq.size(); k++)
                {
                    PointType &point_body  = feats_down_body->points[idx+time_seq[k]];

                    time_current = point_body.curvature / 1000.0 + pcl_beg_time;

                    if (is_first_frame)
                    {
                        if(imu_en)
                        {
                            while (time_current > get_time_sec(imu_next.header.stamp))
                            {
                                imu_deque.pop_front();
                                if (imu_deque.empty()) return;
                                imu_last = imu_next;
                                imu_next = *(imu_deque.front());
                            }
                            angvel_avr<<imu_last.angular_velocity.x, imu_last.angular_velocity.y, imu_last.angular_velocity.z;
                            acc_avr   <<imu_last.linear_acceleration.x, imu_last.linear_acceleration.y, imu_last.linear_acceleration.z;
                        }
                        is_first_frame = false;
                        imu_upda_cov = true;
                        time_update_last = time_current;
                        time_predict_last_const = time_current;
                    }
                    if(imu_en && !imu_deque.empty())
                    {
                        bool last_imu = get_time_sec(imu_next.header.stamp) == get_time_sec(imu_deque.front()->header.stamp);
                        while (get_time_sec(imu_next.header.stamp) < time_predict_last_const && !imu_deque.empty())
                        {
                            if (!last_imu)
                            {
                                imu_last = imu_next;
                                imu_next = *(imu_deque.front());
                                return;
                            }
                            else
                            {
                                imu_deque.pop_front();
                                if (imu_deque.empty()) return;
                                imu_last = imu_next;
                                imu_next = *(imu_deque.front());
                            }
                        }
                        bool imu_comes = time_current > get_time_sec(imu_next.header.stamp);
                        while (imu_comes) 
                        {
                            imu_upda_cov = true;
                            angvel_avr<<imu_next.angular_velocity.x, imu_next.angular_velocity.y, imu_next.angular_velocity.z;
                            acc_avr   <<imu_next.linear_acceleration.x, imu_next.linear_acceleration.y, imu_next.linear_acceleration.z;

                            /*** covariance update ***/
                            double dt = get_time_sec(imu_next.header.stamp) - time_predict_last_const;
                            kf_output.predict(dt, Q_output, input_in, true, false);
                            time_predict_last_const = get_time_sec(imu_next.header.stamp); // big problem
                            
                            {
                                double dt_cov = get_time_sec(imu_next.header.stamp) - time_update_last; 

                                if (dt_cov > 0.0)
                                {
                                    time_update_last = get_time_sec(imu_next.header.stamp);
                                    double propag_imu_start = omp_get_wtime();

                                    kf_output.predict(dt_cov, Q_output, input_in, false, true);

                                    propag_time += omp_get_wtime() - propag_imu_start;
                                    double solve_imu_start = omp_get_wtime();
                                    kf_output.update_iterated_dyn_share_IMU();
                                    solve_time += omp_get_wtime() - solve_imu_start;
                                }
                            }
                            imu_deque.pop_front();
                            if (imu_deque.empty()) return;
                            imu_last = imu_next;
                            imu_next = *(imu_deque.front());
                            imu_comes = time_current > get_time_sec(imu_next.header.stamp);
                        }
                    }
                    if (flg_reset)
                    {
                        return;
                    }

                    double dt = time_current - time_predict_last_const;
                    double propag_state_start = omp_get_wtime();
                    if(!prop_at_freq_of_imu)
                    {
                        double dt_cov = time_current - time_update_last;
                        if (dt_cov > 0.0)
                        {
                            kf_output.predict(dt_cov, Q_output, input_in, false, true);
                            time_update_last = time_current;   
                        }
                    }
                    kf_output.predict(dt, Q_output, input_in, true, false);
                    propag_time += omp_get_wtime() - propag_state_start;
                    time_predict_last_const = time_current;
                    double t_update_start = omp_get_wtime();

                    if (feats_down_size < 1)
                    {
                        printf("No point, skip this scan!\n");
                        idx += time_seq[k];
                        return;
                    }
                    if (!kf_output.update_iterated_dyn_share_modified()) 
                    {
                        idx = idx+time_seq[k];
                        return;
                    }
                    solve_start = omp_get_wtime();
                        
                    if (publish_odometry_without_downsample)
                    {
                        /******* Publish odometry *******/

                        // publish_odometry(pubOdomAftMapped, tf_broadcaster, tf_buffer, this->get_logger());
                        publish_odometry(pubOdomAftMapped, tf_broadcaster);
                        if (runtime_pos_log)
                        {
                            euler_cur = SO3ToEuler(kf_output.x_.rot);
                            fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << kf_output.x_.pos.transpose() << " " << kf_output.x_.vel.transpose() \
                            <<" "<<kf_output.x_.omg.transpose()<<" "<<kf_output.x_.acc.transpose()<<" "<<kf_output.x_.gravity.transpose()<<" "<<kf_output.x_.bg.transpose()<<" "<<kf_output.x_.ba.transpose()<<" "<<feats_undistort->points.size()<<endl;
                        }
                    }

                    for (int j = 0; j < time_seq[k]; j++)
                    {
                        PointType &point_body_j  = feats_down_body->points[idx+j+1];
                        PointType &point_world_j = feats_down_world->points[idx+j+1];
                        pointBodyToWorld(&point_body_j, &point_world_j);
                    }
                
                    solve_time += omp_get_wtime() - solve_start;
    
                    update_time += omp_get_wtime() - t_update_start;
                    idx += time_seq[k];
                    // cout << "pbp output effect feat num:" << effct_feat_num << endl;
                }
                }
                else
                {
                    if (!imu_deque.empty())
                    { 
                        imu_last = imu_next;
                        imu_next = *(imu_deque.front());

                    while (get_time_sec(imu_next.header.stamp) > time_current && ((get_time_sec(imu_next.header.stamp) < Measures.lidar_beg_time + lidar_time_inte )))
                    { // >= ?
                        if (is_first_frame)
                        {
                            {
                                {
                                    while (get_time_sec(imu_next.header.stamp) < Measures.lidar_beg_time + lidar_time_inte)
                                    {
                                        // meas.imu.emplace_back(imu_deque.front()); should add to initialization
                                        imu_deque.pop_front();
                                        if(imu_deque.empty()) return;
                                        imu_last = imu_next;
                                        imu_next = *(imu_deque.front());
                                    }
                                }
                                return;
                            }
                            angvel_avr<<imu_last.angular_velocity.x, imu_last.angular_velocity.y, imu_last.angular_velocity.z;
                                            
                            acc_avr   <<imu_last.linear_acceleration.x, imu_last.linear_acceleration.y, imu_last.linear_acceleration.z;

                            imu_upda_cov = true;
                            time_update_last = time_current;
                            time_predict_last_const = time_current;

                                is_first_frame = false;
                        }
                        time_current = get_time_sec(imu_next.header.stamp);

                        if (!is_first_frame)
                        {
                        double dt = time_current - time_predict_last_const;
                        {
                            double dt_cov = time_current - time_update_last;
                            if (dt_cov > 0.0)
                            {
                                kf_output.predict(dt_cov, Q_output, input_in, false, true);
                                time_update_last = time_current;
                            }
                            kf_output.predict(dt, Q_output, input_in, true, false);
                        }

                        time_predict_last_const = time_current;

                        angvel_avr<<imu_next.angular_velocity.x, imu_next.angular_velocity.y, imu_next.angular_velocity.z;
                        acc_avr   <<imu_next.linear_acceleration.x, imu_next.linear_acceleration.y, imu_next.linear_acceleration.z; 
                        acc_avr_norm = acc_avr * G_m_s2 / acc_norm;
                        kf_output.update_iterated_dyn_share_IMU();
                        imu_deque.pop_front();
                        if (imu_deque.empty()) return;
                        imu_last = imu_next;
                        imu_next = *(imu_deque.front());
                    }
                    else
                    {
                        imu_deque.pop_front();
                        if (imu_deque.empty()) return;
                        imu_last = imu_next;
                        imu_next = *(imu_deque.front());
                    }
                    }
                    }
                }
            }
            else
            {
                bool imu_prop_cov = false;
                effct_feat_num = 0;
                if (time_seq.size() > 0)
                {
                double pcl_beg_time = Measures.lidar_beg_time;
                idx = -1;
                for (k = 0; k < time_seq.size(); k++)
                {
                    PointType &point_body  = feats_down_body->points[idx+time_seq[k]];
                    time_current = point_body.curvature / 1000.0 + pcl_beg_time;
                    if (is_first_frame)
                    {
                        while (time_current > get_time_sec(imu_next.header.stamp)) 
                        {
                            imu_deque.pop_front();
                            if (imu_deque.empty()) 
                            {
                                return;}
                            imu_last = imu_next;
                            imu_next = *(imu_deque.front());
                        }
                        imu_prop_cov = true;

                        is_first_frame = false;
                        t_last = time_current;
                        time_update_last = time_current; 
                        {
                            input_in.gyro<<imu_last.angular_velocity.x, imu_last.angular_velocity.y, imu_last.angular_velocity.z;                 
                            input_in.acc<<imu_last.linear_acceleration.x, imu_last.linear_acceleration.y, imu_last.linear_acceleration.z;
                            input_in.acc = input_in.acc * G_m_s2 / acc_norm;
                        }
                    }
                    
                    while (time_current > get_time_sec(imu_next.header.stamp)) // && !imu_deque.empty())
                    {
                        imu_deque.pop_front();
                        
                        input_in.gyro<<imu_last.angular_velocity.x, imu_last.angular_velocity.y, imu_last.angular_velocity.z;
                        input_in.acc <<imu_last.linear_acceleration.x, imu_last.linear_acceleration.y, imu_last.linear_acceleration.z; 
                        input_in.acc    = input_in.acc * G_m_s2 / acc_norm; 
                        double dt = get_time_sec(imu_last.header.stamp) - t_last;

                        double dt_cov = get_time_sec(imu_last.header.stamp) - time_update_last;
                        if (dt_cov > 0.0)
                        {
                            kf_input.predict(dt_cov, Q_input, input_in, false, true); 
                            time_update_last = get_time_sec(imu_last.header.stamp); //time_current;
                        }
                        kf_input.predict(dt, Q_input, input_in, true, false); 
                        t_last = get_time_sec(imu_last.header.stamp);
                        imu_prop_cov = true;

                        if (imu_deque.empty()) {
                                return;}
                        imu_last = imu_next;
                        imu_next = *(imu_deque.front());
                        // imu_upda_cov = true;
                    }     
                    if (flg_reset)
                    {
                        return;
                    }     
                    double dt = time_current - t_last;
                    t_last = time_current;
                    double propag_start = omp_get_wtime();
                    
                    if(!prop_at_freq_of_imu)
                    {     
                        double dt_cov = time_current - time_update_last;
                        if (dt_cov > 0.0)
                        {    
                            kf_input.predict(dt_cov, Q_input, input_in, false, true); 
                            time_update_last = time_current; 
                        }
                    }
                    kf_input.predict(dt, Q_input, input_in, true, false); 

                    propag_time += omp_get_wtime() - propag_start;

                    double t_update_start = omp_get_wtime();
                    
                    if (feats_down_size < 1)
                    {
                        printf("No point, skip this scan!\n");

                        idx += time_seq[k];
                        return;
                    } 
                    if (!kf_input.update_iterated_dyn_share_modified()) 
                    {
                        idx = idx+time_seq[k];
                        return;
                    }

                    solve_start = omp_get_wtime();

                    if (publish_odometry_without_downsample)
                    {
                        /******* Publish odometry *******/
                        // publish_odometry(pubOdomAftMapped, tf_broadcaster, tf_buffer, this->get_logger());
                        publish_odometry(pubOdomAftMapped, tf_broadcaster);
                        if (runtime_pos_log)
                        {
                            euler_cur = SO3ToEuler(kf_input.x_.rot);
                            fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << kf_input.x_.pos.transpose() << " " << kf_input.x_.vel.transpose() \
                            <<" "<<kf_input.x_.bg.transpose()<<" "<<kf_input.x_.ba.transpose()<<" "<<kf_input.x_.gravity.transpose()<<" "<<feats_undistort->points.size()<<endl;
                        }
                    }

                    for (int j = 0; j < time_seq[k]; j++)
                    {
                        PointType &point_body_j  = feats_down_body->points[idx+j+1];
                        PointType &point_world_j = feats_down_world->points[idx+j+1];
                        pointBodyToWorld(&point_body_j, &point_world_j); 
                    }
                    solve_time += omp_get_wtime() - solve_start;
                
                    update_time += omp_get_wtime() - t_update_start;
                    idx = idx + time_seq[k];
                }  
                }
                else
                {
                    if (!imu_deque.empty())
                    {  
                    imu_last = imu_next;
                    imu_next = *(imu_deque.front());
                    while (get_time_sec(imu_next.header.stamp) > time_current && ((get_time_sec(imu_next.header.stamp) < Measures.lidar_beg_time + lidar_time_inte)))
                    { // >= ?  
                        if (is_first_frame)
                        {
                            {
                                {
                                    while (get_time_sec(imu_next.header.stamp) < Measures.lidar_beg_time + lidar_time_inte)
                                    {
                                        imu_deque.pop_front();
                                        if(imu_deque.empty()) 
                                        {
                                            return;
                                        }
                                        imu_last = imu_next;
                                        imu_next = *(imu_deque.front());
                                    }
                                }
                                return;
                            }
                            imu_prop_cov = true;
                            
                            t_last = time_current;
                            time_update_last = time_current; 
                            input_in.gyro<<imu_last.angular_velocity.x, imu_last.angular_velocity.y, imu_last.angular_velocity.z;
                            input_in.acc   <<imu_last.linear_acceleration.x, imu_last.linear_acceleration.y, imu_last.linear_acceleration.z;
                            input_in.acc = input_in.acc * G_m_s2 / acc_norm;
                            
                                is_first_frame = false;
                            
                        }
                        time_current = get_time_sec(imu_next.header.stamp);

                        if (!is_first_frame)
                        {
                        double dt = time_current - t_last;

                        double dt_cov = time_current - time_update_last;
                        if (dt_cov > 0.0)
                        {        
                            // kf_input.predict(dt_cov, Q_input, input_in, false, true);
                            time_update_last = get_time_sec(imu_next.header.stamp); //time_current;
                        }
                        // kf_input.predict(dt, Q_input, input_in, true, false);

                        t_last = get_time_sec(imu_next.header.stamp);
                    
                        input_in.gyro<<imu_next.angular_velocity.x, imu_next.angular_velocity.y, imu_next.angular_velocity.z;
                        input_in.acc<<imu_next.linear_acceleration.x, imu_next.linear_acceleration.y, imu_next.linear_acceleration.z; 
                        input_in.acc = input_in.acc * G_m_s2 / acc_norm;
                        imu_deque.pop_front();
                        if (imu_deque.empty()) 
                        {
                            return;
                        }
                        imu_last = imu_next;
                        imu_next = *(imu_deque.front());
                        }
                        else
                        {
                            imu_deque.pop_front();
                            if (imu_deque.empty()) return;
                            imu_last = imu_next;
                            imu_next = *(imu_deque.front());
                        }
                    }
                    }
                }
            }

            /******* Publish odometry downsample *******/
            if (!publish_odometry_without_downsample)
            {
                // publish_odometry(pubOdomAftMapped, tf_broadcaster, tf_buffer, this->get_logger());
                publish_odometry(pubOdomAftMapped, tf_broadcaster);
            }

            /*** add the feature points to map ***/
            t3 = omp_get_wtime();
            if(feats_down_size > 4)
            {
                MapIncremental();
            }
            t5 = omp_get_wtime();

            /******* Publish points *******/
            if (path_en)                         publish_path(pubPath);
            if (scan_pub_en || pcd_save_en)      publish_frame_world(pubLaserCloudFullRes);
            if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFullRes_body);
            // publish_map(pubLaserCloudMap);
            
            /*** Debug variables Logging ***/
            if (runtime_pos_log)
            {
                frame_num ++;
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
                {aver_time_icp = aver_time_icp * (frame_num - 1)/frame_num + update_time/frame_num;}
                aver_time_match = aver_time_match * (frame_num - 1)/frame_num + (match_time)/frame_num;
                aver_time_solve = aver_time_solve * (frame_num - 1)/frame_num + solve_time/frame_num;
                aver_time_propag = aver_time_propag * (frame_num - 1)/frame_num + propag_time / frame_num;
                T1[time_log_counter] = Measures.lidar_beg_time;
                s_plot[time_log_counter] = t5 - t0;
                s_plot2[time_log_counter] = feats_undistort->points.size();
                s_plot3[time_log_counter] = aver_time_consu;
                time_log_counter ++;
                printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f propogate: %0.6f \n",t1-t0,aver_time_match,aver_time_solve,t3-t1,t5-t3,aver_time_consu, aver_time_icp, aver_time_propag); 
                if (!publish_odometry_without_downsample)
                {
                    if (!use_imu_as_input)
                    {
                        fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << kf_output.x_.pos.transpose() << " " << kf_output.x_.vel.transpose() \
                        <<" "<<kf_output.x_.omg.transpose()<<" "<<kf_output.x_.acc.transpose()<<" "<<kf_output.x_.gravity.transpose()<<" "<<kf_output.x_.bg.transpose()<<" "<<kf_output.x_.ba.transpose()<<" "<<feats_undistort->points.size()<<endl;
                    }
                    else
                    {
                        fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << kf_input.x_.pos.transpose() << " " << kf_input.x_.vel.transpose() \
                        <<" "<<kf_input.x_.bg.transpose()<<" "<<kf_input.x_.ba.transpose()<<" "<<kf_input.x_.gravity.transpose()<<" "<<feats_undistort->points.size()<<endl;
                    }
                }
                dump_lio_state_to_log(fp);
            }
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl_pc;
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_pcl_livox;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFullRes;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFullRes_body;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudEffect;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMap;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr plane_pub;

    rclcpp::TimerBase::SharedPtr timer;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;
    std::shared_ptr<tf2_ros::TransformListener> transform_listener;

    int frame_num = 0;
    double FOV_DEG, HALF_FOV_COS, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_propag = 0;
    std::time_t startTime, endTime;
    Eigen::Matrix<double, 24, 24> P_init;
    Eigen::Matrix<double, 30, 30> P_init_output;
    Eigen::Matrix<double, 24, 24> Q_input;
    Eigen::Matrix<double, 30, 30> Q_output;
    ofstream fout_out, fout_imu_pbp;
    FILE *fp;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    signal(SIGINT, SigHandle);
    rclcpp::spin(std::make_shared<LaserMappingNode>());

    if (rclcpp::ok())
        rclcpp::shutdown();
    
    //--------------------------save map-----------------------------------
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        string file_name = string("scans.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }

    return 0;
}
