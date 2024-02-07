#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <rclcpp/rclcpp.hpp>
#include <Eigen/Core>
#include "IMU_Processing.hpp"
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2/transform_datatypes.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/time.h>
#include <tf2_ros/buffer.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <geometry_msgs/msg/vector3.hpp>
#include <geometry_msgs/msg/transform.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include "livox_ros_driver2/msg/custom_msg.hpp"
#include "parameters.h"
#include "Estimator.h"


#define MAXN                (720000)
#define PUBFRAME_PERIOD     (20)

const float MOV_THRESHOLD = 1.5f;
string root_dir = ROOT_DIR;

mutex mtx_buffer;
condition_variable sig_buffer;

int feats_down_size = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int frame_ct = 0;
double time_update_last = 0.0, time_current = 0.0, time_predict_last_const = 0.0, t_last = 0.0;
bool init_map = false, flg_first_scan = true;

// Time Log Variables
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, propag_time = 0, update_time = 0;

bool   lidar_pushed = false, flg_reset = false, flg_exit = false;

bool is_first_frame = true;
double lidar_end_time = 0.0, first_lidar_time = 0.0, time_con = 0.0;
double last_timestamp_lidar = -1.0, last_timestamp_imu = -1.0;
int pcd_index = 0;

std::string lid_topic, imu_topic;
bool prop_at_freq_of_imu, check_satu, con_frame, cut_frame;
bool use_imu_as_input, space_down_sample, publish_odometry_without_downsample;
int  init_map_size, con_frame_num;
double match_s, satu_acc, satu_gyro, cut_frame_time_interval;
float  plane_thr;
double filter_size_surf_min, filter_size_map_min, fov_deg;
double cube_len; 
float  DET_RANGE;
bool   imu_en, gravity_align, non_station_start;
double imu_time_inte;
double laser_point_cov, acc_norm;
double vel_cov, acc_cov_input, gyr_cov_input;
double gyr_cov_output, acc_cov_output, b_gyr_cov, b_acc_cov;
double imu_meas_acc_cov, imu_meas_omg_cov; 
int    lidar_type, pcd_save_interval;
std::vector<double> gravity_init, gravity;
std::vector<double> extrinT;
std::vector<double> extrinR;
bool   runtime_pos_log, pcd_save_en, path_en, extrinsic_est_en = true;
bool   scan_pub_en, scan_body_pub_en;
double time_lag_imu_to_lidar = 0.0;

std::shared_ptr<ImuProcess> p_imu(new ImuProcess());
std::shared_ptr<Preprocess> p_pre(new Preprocess());
PointCloudXYZI::Ptr  ptr_con(new PointCloudXYZI());
vector<BoxPointType> cub_needrm;

deque<PointCloudXYZI::Ptr>  lidar_buffer;
deque<double>               time_buffer;
deque<std::shared_ptr<const sensor_msgs::msg::Imu>> imu_deque;

//surf feature in map
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body_space(new PointCloudXYZI());
PointCloudXYZI::Ptr init_feats_world(new PointCloudXYZI());

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

V3D euler_cur;
MeasureGroup Measures;

sensor_msgs::msg::Imu imu_last, imu_next;
std::shared_ptr<const sensor_msgs::msg::Imu> imu_last_ptr;
nav_msgs::msg::Path path;
nav_msgs::msg::Odometry odomAftMapped;
geometry_msgs::msg::PoseStamped msg_body_pose;



void SigHandle(int sig)
{
    flg_exit = true;
    printf("catch sig %d \n", sig);
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

int points_cache_size = 0;

void points_cache_collect() // seems for debug
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
    points_cache_size = points_history.size();
}

BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;
void lasermap_fov_segment()
{
    cub_needrm.shrink_to_fit();

    V3D pos_LiD;
    if (use_imu_as_input)
    {
        pos_LiD = kf_input.x_.pos + kf_input.x_.rot * Lidar_T_wrt_IMU;
    }
    else
    {
        pos_LiD = kf_output.x_.pos + kf_output.x_.rot * Lidar_T_wrt_IMU;
    }
    if (!Localmap_Initialized){
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
    }
    if (!need_move) return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.emplace_back(tmp_boxpoints);
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.emplace_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    if(cub_needrm.size() > 0) int kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
}

void standard_pcl_cbk(const std::shared_ptr<const sensor_msgs::msg::PointCloud2> &msg) 
{
    mtx_buffer.lock();
    scan_count ++;
    double preprocess_start_time = omp_get_wtime();
    if (get_time_sec(msg->header.stamp) < last_timestamp_lidar)
    {
        printf("lidar loop back, clear buffer");
        // lidar_buffer.shrink_to_fit();

        mtx_buffer.unlock();
        sig_buffer.notify_all();
        return;
    }

    last_timestamp_lidar = get_time_sec(msg->header.stamp);

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    PointCloudXYZI::Ptr  ptr_div(new PointCloudXYZI());
    double time_div = get_time_sec(msg->header.stamp);
    p_pre->process(msg, ptr);
    if (cut_frame)
    {
        sort(ptr->points.begin(), ptr->points.end(), time_list);

        for (int i = 0; i < ptr->size(); i++)
        {
            ptr_div->push_back(ptr->points[i]);
            // cout << "check time:" << ptr->points[i].curvature << endl;
            if (ptr->points[i].curvature / double(1000) + get_time_sec(msg->header.stamp) - time_div > cut_frame_time_interval)
            {
                if(ptr_div->size() < 1) continue;
                PointCloudXYZI::Ptr  ptr_div_i(new PointCloudXYZI());
                *ptr_div_i = *ptr_div;
                lidar_buffer.push_back(ptr_div_i);
                time_buffer.push_back(time_div);
                time_div += ptr->points[i].curvature / double(1000);
                ptr_div->clear();
            }
        }
        if (!ptr_div->empty())
        {
            lidar_buffer.push_back(ptr_div);
            // ptr_div->clear();
            time_buffer.push_back(time_div);
        }
    }
    else if (con_frame)
    {
        if (frame_ct == 0)
        {
            time_con = last_timestamp_lidar; //msg->header.stamp.toSec();
        }
        if (frame_ct < con_frame_num)
        {
            for (int i = 0; i < ptr->size(); i++)
            {
                ptr->points[i].curvature += (last_timestamp_lidar - time_con) * 1000;
                ptr_con->push_back(ptr->points[i]);
            }
            frame_ct ++;
        }
        else
        {
            PointCloudXYZI::Ptr  ptr_con_i(new PointCloudXYZI());
            *ptr_con_i = *ptr_con;
            lidar_buffer.push_back(ptr_con_i);
            double time_con_i = time_con;
            time_buffer.push_back(time_con_i);
            ptr_con->clear();
            frame_ct = 0;
        }
    }
    else
    { 
        lidar_buffer.emplace_back(ptr);
        time_buffer.emplace_back(get_time_sec(msg->header.stamp));
    }
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void imu_cbk(const std::shared_ptr<const sensor_msgs::msg::Imu> &msg_in) 
{
    publish_count ++;
    std::shared_ptr<sensor_msgs::msg::Imu> msg(new sensor_msgs::msg::Imu(*msg_in));

    msg->header.stamp = get_ros_time(get_time_sec(msg_in->header.stamp) - time_lag_imu_to_lidar);
    if (abs(time_lag_imu_to_lidar) > 0.1)
    {
        msg->header.stamp = \
        rclcpp::Time(time_lag_imu_to_lidar + get_time_sec(msg_in->header.stamp));
    }
    double timestamp = get_time_sec(msg->header.stamp);

    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)
    {
        printf("imu loop back, clear deque\n");
        std::cout << "last_timestamp_imu " << last_timestamp_imu << std::endl;
        std::cout << "timestamp " << timestamp << std::endl;
        // imu_deque.shrink_to_fit();
        mtx_buffer.unlock();
        sig_buffer.notify_all();
        return;
    }
    
    imu_deque.emplace_back(msg);
    last_timestamp_imu = timestamp;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

bool sync_packages(MeasureGroup &meas)
{
    if (!imu_en)
    {
        if (!lidar_buffer.empty())
        {
            meas.lidar = lidar_buffer.front();
            meas.lidar_beg_time = time_buffer.front();
            time_buffer.pop_front();
            lidar_buffer.pop_front();
            if(meas.lidar->points.size() < 1) 
            {
                cout << "lose lidar" << std::endl;
                return false;
            }
            double end_time = meas.lidar->points.back().curvature;
            for (auto pt: meas.lidar->points)
            {
                if (pt.curvature > end_time)
                {
                    end_time = pt.curvature;
                }
            }
            lidar_end_time = meas.lidar_beg_time + end_time / double(1000);
            meas.lidar_last_time = lidar_end_time;
            return true;
        }
        return false;
    }

    if (lidar_buffer.empty() || imu_deque.empty())
    {
        return false;
    }

    /*** push a lidar scan ***/
    if(!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        if(meas.lidar->points.size() < 1) 
        {
            cout << "lose lidar" << endl;
            lidar_buffer.pop_front();
            time_buffer.pop_front();
            return false;
        }
        meas.lidar_beg_time = time_buffer.front();
        double end_time = meas.lidar->points.back().curvature;
        for (auto pt: meas.lidar->points)
        {
            if (pt.curvature > end_time)
            {
                end_time = pt.curvature;
            }
        }
        lidar_end_time = meas.lidar_beg_time + end_time / double(1000);
        
        meas.lidar_last_time = lidar_end_time;
        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }
    /*** push imu data, and pop from imu buffer ***/
    if (p_imu->imu_need_init_)
    {
        double imu_time = get_time_sec(imu_deque.front()->header.stamp);
        meas.imu.shrink_to_fit();
        while ((!imu_deque.empty()) && (imu_time < lidar_end_time))
        {
            imu_time = get_time_sec(imu_deque.front()->header.stamp); 
            if(imu_time > lidar_end_time) break;
            meas.imu.emplace_back(imu_deque.front());
            imu_last = imu_next;
            imu_last_ptr = imu_deque.front();
            imu_next = *(imu_deque.front());
            imu_deque.pop_front();
        }
    }
    else if(!init_map)
    {
        double imu_time = get_time_sec(imu_deque.front()->header.stamp);
        meas.imu.shrink_to_fit();
        meas.imu.emplace_back(imu_last_ptr);

        while ((!imu_deque.empty()) && (imu_time < lidar_end_time))
        {
            imu_time = get_time_sec(imu_deque.front()->header.stamp); 
            if(imu_time > lidar_end_time) break;
            meas.imu.emplace_back(imu_deque.front());
            imu_last = imu_next;
            imu_last_ptr = imu_deque.front();
            imu_next = *(imu_deque.front());
            imu_deque.pop_front();
        }
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

int process_increments = 0;
void map_incremental()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    
        for(int i = 0; i < feats_down_size; i++)
        {            
            if (!Nearest_Points[i].empty())
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            PointType downsample_result, mid_point; 
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
                /* If the nearest points is definitely outside the downsample box */
                if (fabs(points_near[0].x - mid_point.x) > 0.866 * filter_size_map_min || fabs(points_near[0].y - mid_point.y) > 0.866 * filter_size_map_min || fabs(points_near[0].z - mid_point.z) > 0.866 * filter_size_map_min){
                    PointNoNeedDownsample.emplace_back(feats_down_world->points[i]);
                continue;
            }
                /* Check if there is a point already in the downsample box */
                float dist  = calc_dist<float>(feats_down_world->points[i],mid_point);
                for (int readd_i = 0; readd_i < points_near.size(); readd_i ++)
            {
                    /* Those points which are outside the downsample box should not be considered. */
                    if (fabs(points_near[readd_i].x - mid_point.x) < 0.5 * filter_size_map_min && fabs(points_near[readd_i].y - mid_point.y) < 0.5 * filter_size_map_min && fabs(points_near[readd_i].z - mid_point.z) < 0.5 * filter_size_map_min) {
                    need_add = false;
                    break;
                }
            }
                if (need_add) PointToAdd.emplace_back(feats_down_world->points[i]);
        }
        else
        {
                // PointToAdd.emplace_back(feats_down_world->points[i]);
                PointNoNeedDownsample.emplace_back(feats_down_world->points[i]);
        }
        }
    int add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false);
}

void publish_init_kdtree(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFullRes)
{
    int size_init_ikdtree = ikdtree.size();
    PointCloudXYZI::Ptr laserCloudInit(new PointCloudXYZI(size_init_ikdtree, 1));

    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    PointVector().swap(ikdtree.PCL_Storage);
    ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                
    laserCloudInit->points = ikdtree.PCL_Storage;
    pcl::toROSMsg(*laserCloudInit, laserCloudmsg);
        
    laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
    laserCloudmsg.header.frame_id = "odom";
    pubLaserCloudFullRes->publish(laserCloudmsg);

}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
void publish_frame_world(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFullRes)
{
    if (scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(feats_down_body);
        int size = laserCloudFullRes->points.size();

        PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));
        
        for (int i = 0; i < size; i++)
        {
            // if (i % 3 == 0)
            // {
            laserCloudWorld->points[i].x = feats_down_world->points[i].x;
            laserCloudWorld->points[i].y = feats_down_world->points[i].y;
            laserCloudWorld->points[i].z = feats_down_world->points[i].z;
            laserCloudWorld->points[i].intensity = feats_down_world->points[i].intensity; // feats_down_world->points[i].y; // 
            // }
        }
        sensor_msgs::msg::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        
        laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
        laserCloudmsg.header.frame_id = "odom";
        pubLaserCloudFullRes->publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }
    
    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en)
    {
        int size = feats_down_world->points.size();
        PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

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
    publish_count -= PUBFRAME_PERIOD;
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

void publish_odometry(const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped,
                                std::unique_ptr<tf2_ros::TransformBroadcaster> & tf_br,
                                std::unique_ptr<tf2_ros::Buffer> & tf_buffer_,
                                rclcpp::Logger logger_)
{
    odomAftMapped.header.frame_id = "odom";
    odomAftMapped.child_frame_id = "base_link";
    if (publish_odometry_without_downsample)
    {
        odomAftMapped.header.stamp = get_ros_time(time_current);
    }
    else
    {
        odomAftMapped.header.stamp = get_ros_time(lidar_end_time);
    }
    set_posestamp(odomAftMapped.pose.pose);

    string odometryFrame = "odom";
    string baselinkFrame = "base_link";
    string lidarFrame = "livox_frame";
    rclcpp::Time stamp = get_ros_time(lidar_end_time);
    // Publish TF
    tf2::Quaternion quat_tf(odomAftMapped.pose.pose.orientation.x,
                            odomAftMapped.pose.pose.orientation.y,
                            odomAftMapped.pose.pose.orientation.z,
                            odomAftMapped.pose.pose.orientation.w);

    tf2::Transform t_odom_to_lidar = tf2::Transform(quat_tf, tf2::Vector3(odomAftMapped.pose.pose.position.x,
                                                                            odomAftMapped.pose.pose.position.y,
                                                                            odomAftMapped.pose.pose.position.z));
    tf2::TimePoint time_point = tf2_ros::fromRclcpp(get_ros_time(lidar_end_time));
    tf2::Stamped<tf2::Transform> temp_odom_to_base(t_odom_to_lidar, time_point, odometryFrame);
    tf2::Stamped<tf2::Transform> lidar2Baselink;
    if(lidarFrame != baselinkFrame)
    {
        try
        {
            tf2::fromMsg(tf_buffer_->lookupTransform(
                lidarFrame, baselinkFrame, get_ros_time(lidar_end_time), rclcpp::Duration::from_seconds(0.01)), lidar2Baselink);
        }
        catch (tf2::TransformException ex)
        {
            RCLCPP_ERROR(logger_, "%s", ex.what());
        }
        tf2::Stamped<tf2::Transform> tb(
            temp_odom_to_base * lidar2Baselink, time_point, odometryFrame);
        temp_odom_to_base = tb;
    }
    geometry_msgs::msg::TransformStamped trans_odom_to_base;
    tf2::convert(temp_odom_to_base, trans_odom_to_base);
    trans_odom_to_base.child_frame_id = baselinkFrame;
    tf_br->sendTransform(trans_odom_to_base);

    // geometry_msgs::msg::TransformStamped trans;
    // trans.header.frame_id = "lidar_odom";
    // trans.child_frame_id = "body";
    // trans.transform.translation.x = odomAftMapped.pose.pose.position.x;
    // trans.transform.translation.y = odomAftMapped.pose.pose.position.y;
    // trans.transform.translation.z = odomAftMapped.pose.pose.position.z;
    // trans.transform.rotation.w = odomAftMapped.pose.pose.orientation.w;
    // trans.transform.rotation.x = odomAftMapped.pose.pose.orientation.x;
    // trans.transform.rotation.y = odomAftMapped.pose.pose.orientation.y;
    // trans.transform.rotation.z = odomAftMapped.pose.pose.orientation.z;
    // tf_br->sendTransform(trans);
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
        declare_and_get_parameter<bool>("mapping.start_in_aggressive_motion", non_station_start, false);
        declare_and_get_parameter<bool>("mapping.extrinsic_est_en", extrinsic_est_en, true);
        declare_and_get_parameter<bool>("publish.path_en", path_en, true);
        declare_and_get_parameter<bool>("publish.scan_publish_en", scan_pub_en, true);
        declare_and_get_parameter<bool>("publish.scan_bodyframe_pub_en", scan_body_pub_en, true);
        declare_and_get_parameter<bool>("runtime_pos_log_enable", runtime_pos_log, false);
        declare_and_get_parameter<bool>("pcd_save.pcd_save_en", pcd_save_en, false);
        declare_and_get_parameter<bool>("mapping.gravity_align", gravity_align, true);

        // 整数型参数
        declare_and_get_parameter<int>("init_map_size", init_map_size, 100);
        declare_and_get_parameter<int>("point_filter_num", p_pre->point_filter_num, 2);
        declare_and_get_parameter<int>("common.con_frame_num", con_frame_num, 1);
        declare_and_get_parameter<int>("preprocess.lidar_type", p_pre->lidar_type, 1);
        declare_and_get_parameter<int>("preprocess.scan_line", p_pre->N_SCANS, 16);
        declare_and_get_parameter<int>("preprocess.scan_rate", p_pre->SCAN_RATE, 10);
        declare_and_get_parameter<int>("preprocess.timestamp_unit", p_pre->time_unit, 1);
        declare_and_get_parameter<int>("pcd_save.interval", pcd_save_interval, -1);

        // 浮点型参数
        declare_and_get_parameter<double>("mapping.satu_acc", satu_acc, 3.0);
        declare_and_get_parameter<double>("mapping.satu_gyro", satu_gyro, 35.0);
        declare_and_get_parameter<double>("mapping.acc_norm", acc_norm, 1.0);
        declare_and_get_parameter<float>("mapping.plane_thr", plane_thr, 0.05f);
        declare_and_get_parameter<double>("common.cut_frame_time_interval", cut_frame_time_interval, 0.1);
        declare_and_get_parameter<double>("common.time_lag_imu_to_lidar", time_lag_imu_to_lidar, 0.0);
        declare_and_get_parameter<double>("filter_size_surf", filter_size_surf_min, 0.5);
        declare_and_get_parameter<double>("filter_size_map", filter_size_map_min, 0.5);
        declare_and_get_parameter<double>("cube_side_length", cube_len, 200);
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

        // 字符串型参数
        declare_and_get_parameter<std::string>("common.lid_topic", lid_topic, "/livox/lidar");
        declare_and_get_parameter<std::string>("common.imu_topic", imu_topic, "/livox/imu");

        // 向量型参数
        declare_and_get_parameter<std::vector<double>>("mapping.gravity", gravity, std::vector<double>());
        declare_and_get_parameter<std::vector<double>>("mapping.gravity_init", gravity_init, std::vector<double>());
        declare_and_get_parameter<std::vector<double>>("mapping.extrinsic_T", extrinT, std::vector<double>());
        declare_and_get_parameter<std::vector<double>>("mapping.extrinsic_R", extrinR, std::vector<double>());

        printf("p_pre->lidar_type: %d", p_pre->lidar_type);

        path.header.stamp = this->get_clock()->now();
        path.header.frame_id ="lidar_odom";

         /*** variables definition for counting ***/
        frame_num = 0;
        aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_propag = 0;

        /*** initialize variables ***/
        FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
        HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

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

        kf_input.init_dyn_share_modified(get_f_input, df_dx_input, h_model_input);
        kf_output.init_dyn_share_modified_2h(get_f_output, df_dx_output, h_model_output, h_model_IMU_output);
        P_init = Eigen::Matrix<double, 24, 24>::Zero();
        P_init = MD(24,24)::Identity() * 0.01;
        P_init.block<3, 3>(21, 21) = MD(3,3)::Identity() * 0.0001;
        P_init.block<6, 6>(15, 15) = MD(6,6)::Identity() * 0.001;
        P_init.block<6, 6>(6, 6) = MD(6,6)::Identity() * 0.0001;
        kf_input.change_P(P_init);
        P_init_output = Eigen::Matrix<double, 30, 30>::Zero();
        P_init_output = MD(30,30)::Identity() * 0.01;
        P_init_output.block<3, 3>(21, 21) = MD(3,3)::Identity() * 0.0001;
        P_init_output.block<6, 6>(6, 6) = MD(6,6)::Identity() * 0.0001;
        P_init_output.block<6, 6>(24, 24) = MD(6,6)::Identity() * 0.001;
        kf_input.change_P(P_init);
        kf_output.change_P(P_init_output);
        Q_input = Eigen::Matrix<double, 24, 24>::Zero();
        Q_output = Eigen::Matrix<double, 30, 30>::Zero();
        Q_input = process_noise_cov_input();
        Q_output = process_noise_cov_output();
        /*** debug record ***/
        string pos_log_dir = root_dir + "/Log/pos_log.txt";
        fp = fopen(pos_log_dir.c_str(),"w");

        fout_out.open(DEBUG_FILE_DIR("mat_out.txt"),ios::out);
        fout_imu_pbp.open(DEBUG_FILE_DIR("imu_pbp.txt"),ios::out);
        if (fout_out && fout_imu_pbp)
            cout << "~~~~"<<ROOT_DIR<<" file opened" << endl;
        else
            cout << "~~~~"<<ROOT_DIR<<" doesn't exist" << endl;

        /*** ROS subscribe initialization ***/
        sub_pcl_pc = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            lid_topic, rclcpp::QoS(200000),
            [this](sensor_msgs::msg::PointCloud2::UniquePtr msg) {
                standard_pcl_cbk(std::move(msg));
            });
        
        sub_imu = this->create_subscription<sensor_msgs::msg::Imu>(
            imu_topic, rclcpp::QoS(200000),
            [this](const sensor_msgs::msg::Imu::SharedPtr msg) {
                imu_cbk(msg);
            });
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

        auto period_ms = std::chrono::milliseconds(static_cast<int64_t>(1000.0 / 100.0));
        timer = rclcpp::create_timer(this, this->get_clock(), period_ms, std::bind(&LaserMappingNode::timer_callback, this));
        tf_buffer = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        transform_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);
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
    void timer_callback(){
        if(sync_packages(Measures)) 
        {
            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;
                flg_first_scan = false;
                cout << "first lidar time" << first_lidar_time << endl;
                return;
            }
            if (flg_reset)
            {
                printf("reset when rosbag play back");
                p_imu->Reset();
                flg_reset = false;
                return;
            }
            double t0,t1,t2,t3,t4,t5,match_start, solve_start;
            match_time = 0;
            solve_time = 0;
            propag_time = 0;
            update_time = 0;
            t0 = omp_get_wtime();
            
            p_imu->Process(Measures, feats_undistort);

            // if (feats_undistort->empty() || feats_undistort == NULL)
            if (p_imu->imu_need_init_)
            {
                return;
            }
            if(imu_en)
            {
                if (!p_imu->gravity_align_)
                {
                    while (Measures.lidar_beg_time > get_time_sec(imu_next.header.stamp))
                    {
                        imu_last = imu_next;
                        imu_next = *(imu_deque.front());
                        imu_deque.pop_front();
                        // imu_deque.pop();
                    }
                    if (non_station_start)
                    {
                        state_in.gravity << VEC_FROM_ARRAY(gravity_init);
                        state_out.gravity << VEC_FROM_ARRAY(gravity_init);
                        state_out.acc << VEC_FROM_ARRAY(gravity_init);
                        state_out.acc *= -1;
                    }
                    else
                    {
                        state_in.gravity =  -1 * p_imu->mean_acc * G_m_s2 / acc_norm; 
                        state_out.gravity = -1 * p_imu->mean_acc * G_m_s2 / acc_norm; 
                        state_out.acc = p_imu->mean_acc * G_m_s2 / acc_norm;
                    }
                    if (gravity_align)
                    {
                        Eigen::Matrix3d rot_init;
                        p_imu->gravity_ << VEC_FROM_ARRAY(gravity);
                        p_imu->Set_init(state_in.gravity, rot_init);
                        state_in.gravity = p_imu->gravity_;
                        state_out.gravity = p_imu->gravity_;
                        state_in.rot = rot_init;
                        state_out.rot = rot_init;
                        // state_in.rot.normalize();
                        // state_out.rot.normalize();
                        state_out.acc = -rot_init.transpose() * state_out.gravity;
                    }
                    kf_input.change_x(state_in);
                    kf_output.change_x(state_out);
                    p_imu->gravity_align_ = true;
                }
            }
            else
            {
                if (!p_imu->gravity_align_)
                {
                    state_in.gravity << VEC_FROM_ARRAY(gravity_init);
                    if (gravity_align)
                    {
                        Eigen::Matrix3d rot_init;
                        p_imu->gravity_ << VEC_FROM_ARRAY(gravity);
                        p_imu->Set_init(state_in.gravity, rot_init);
                        state_out.gravity = p_imu->gravity_;
                        state_out.rot = rot_init;
                        // state_in.rot.normalize();
                        // state_out.rot.normalize();
                        state_out.acc = -rot_init.transpose() * state_out.gravity;
                    }
                    else
                    {
                        state_out.gravity << VEC_FROM_ARRAY(gravity_init);
                        state_out.acc << VEC_FROM_ARRAY(gravity_init);
                        state_out.acc *= -1;
                    }
                    // kf_input.change_x(state_in);
                    kf_output.change_x(state_out);
                    p_imu->gravity_align_ = true;
                }
            }
            /*** Segment the map in lidar FOV ***/
            lasermap_fov_segment();
            /*** downsample the feature points in a scan ***/
            t1 = omp_get_wtime();
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
            time_seq = time_compressing<int>(feats_down_body);
            feats_down_size = feats_down_body->points.size();
            
            /*** initialize the map kdtree ***/
            if(!init_map)
            {
                if(ikdtree.Root_Node == nullptr) //
                // if(feats_down_size > 5)
                {
                    ikdtree.set_downsample_param(filter_size_map_min);
                }
                    
                feats_down_world->resize(feats_down_size);
                for(int i = 0; i < feats_down_size; i++)
                {
                    pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                }
                for (size_t i = 0; i < feats_down_world->size(); i++) {
                init_feats_world->points.emplace_back(feats_down_world->points[i]);}
                if(init_feats_world->size() < init_map_size) return;
                ikdtree.Build(init_feats_world->points); 
                init_map = true;
                publish_init_kdtree(pubLaserCloudMap); //(pubLaserCloudFullRes);
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
                if (extrinsic_est_en)
                {
                    if (!use_imu_as_input)
                    {
                        point_this = kf_output.x_.offset_R_L_I * point_this + kf_output.x_.offset_T_L_I;
                    }
                    else
                    {
                        point_this = kf_input.x_.offset_R_L_I * point_this + kf_input.x_.offset_T_L_I;
                    }
                }
                else
                {
                    point_this = Lidar_R_wrt_IMU * point_this + Lidar_T_wrt_IMU;
                }
                M3D point_crossmat;
                point_crossmat << SKEW_SYM_MATRX(point_this);
                crossmat_list[i]=point_crossmat;
            }
            
            if (!use_imu_as_input)
            {     
                // bool imu_upda_cov = false;
                effct_feat_num = 0;
                /**** point by point update ****/

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
                                imu_last = imu_next;
                                imu_next = *(imu_deque.front());
                                imu_deque.pop_front();
                                // imu_deque.pop();
                            }

                            angvel_avr<<imu_last.angular_velocity.x, imu_last.angular_velocity.y, imu_last.angular_velocity.z;
                            acc_avr   <<imu_last.linear_acceleration.x, imu_last.linear_acceleration.y, imu_last.linear_acceleration.z;
                        }
                        is_first_frame = false;
                        // imu_upda_cov = true;
                        time_update_last = time_current;
                        time_predict_last_const = time_current;
                    }
                    if(imu_en)
                    {
                        bool imu_comes = time_current > get_time_sec(imu_next.header.stamp);
                        while (imu_comes) 
                        {
                            // imu_upda_cov = true;
                            angvel_avr<<imu_next.angular_velocity.x, imu_next.angular_velocity.y, imu_next.angular_velocity.z;
                            acc_avr   <<imu_next.linear_acceleration.x, imu_next.linear_acceleration.y, imu_next.linear_acceleration.z;

                            /*** covariance update ***/
                            imu_last = imu_next;
                            imu_next = *(imu_deque.front());
                            imu_deque.pop_front();
                            double dt = get_time_sec(imu_last.header.stamp) - time_predict_last_const;
                                kf_output.predict(dt, Q_output, input_in, true, false);
                            time_predict_last_const = get_time_sec(imu_last.header.stamp); // big problem
                            imu_comes = time_current > get_time_sec(imu_next.header.stamp);
                            // if (!imu_comes)
                            {
                                double dt_cov = get_time_sec(imu_last.header.stamp) - time_update_last; 

                                if (dt_cov > 0.0)
                                {
                                    time_update_last = get_time_sec(imu_last.header.stamp);
                                    double propag_imu_start = omp_get_wtime();

                                    kf_output.predict(dt_cov, Q_output, input_in, false, true);

                                    propag_time += omp_get_wtime() - propag_imu_start;
                                    double solve_imu_start = omp_get_wtime();
                                    kf_output.update_iterated_dyn_share_IMU();
                                    solve_time += omp_get_wtime() - solve_imu_start;
                                }
                            }
                        }
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
                    // if(k == 0)
                    // {
                    //     fout_imu_pbp << Measures.lidar_last_time - first_lidar_time << " " << imu_last.angular_velocity.x << " " << imu_last.angular_velocity.y << " " << imu_last.angular_velocity.z \
                    //             << " " << imu_last.linear_acceleration.x << " " << imu_last.linear_acceleration.y << " " << imu_last.linear_acceleration.z << endl;
                    // }

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

                    // if(prop_at_freq_of_imu)
                    // {
                        // double dt_cov = time_current - time_update_last;
                        // if ((dt_cov >= imu_time_inte)) // (point_cov_not_prop && imu_prop_cov)
                        // {
                        //     double propag_cov_start = omp_get_wtime();
                        //     kf_output.predict(dt_cov, Q_output, input_in, false, true);
                        //     // imu_upda_cov = false;
                        //     time_update_last = time_current;
                        //     propag_time += omp_get_wtime() - propag_cov_start;
                        // }
                    // }

                    solve_start = omp_get_wtime();
                        
                    if (publish_odometry_without_downsample)
                    {
                        /******* Publish odometry *******/

                        // publish_odometry(pubOdomAftMapped);
                        publish_odometry(pubOdomAftMapped, tf_broadcaster, tf_buffer, this->get_logger());   
                        if (runtime_pos_log)
                        {
                            state_out = kf_output.x_;
                            euler_cur = SO3ToEuler(state_out.rot);
                            fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_out.pos.transpose() << " " << state_out.vel.transpose() \
                            <<" "<<state_out.omg.transpose()<<" "<<state_out.acc.transpose()<<" "<<state_out.gravity.transpose()<<" "<<state_out.bg.transpose()<<" "<<state_out.ba.transpose()<<" "<<feats_undistort->points.size()<<endl;
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
                bool imu_prop_cov = false;
                effct_feat_num = 0;

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
                            imu_last = imu_next;
                            imu_next = *(imu_deque.front());
                            imu_deque.pop_front();
                            // imu_deque.pop();
                        }
                        imu_prop_cov = true;
                        // imu_upda_cov = true;

                        is_first_frame = false;
                        t_last = time_current;
                        time_update_last = time_current; 
                        // if(prop_at_freq_of_imu)
                        {
                            input_in.gyro<<imu_last.angular_velocity.x,
                                        imu_last.angular_velocity.y,
                                        imu_last.angular_velocity.z;
                                            
                            input_in.acc<<imu_last.linear_acceleration.x,
                                        imu_last.linear_acceleration.y,
                                        imu_last.linear_acceleration.z;
                            // angvel_avr<<0.5 * (imu_last.angular_velocity.x + imu_next.angular_velocity.x),
                            //             0.5 * (imu_last.angular_velocity.y + imu_next.angular_velocity.y),
                            //             0.5 * (imu_last.angular_velocity.z + imu_next.angular_velocity.z);
                                            
                            // acc_avr   <<0.5 * (imu_last.linear_acceleration.x + imu_next.linear_acceleration.x),
                            //             0.5 * (imu_last.linear_acceleration.y + imu_next.linear_acceleration.y),
                                        // 0.5 * (imu_last.linear_acceleration.z + imu_next.linear_acceleration.z);

                            // angvel_avr -= state.bias_g;
                            input_in.acc = input_in.acc * G_m_s2 / acc_norm;
                        }
                    }
                    
                    while (time_current > get_time_sec(imu_next.header.stamp)) // && !imu_deque.empty())
                    {
                        imu_last = imu_next;
                        imu_next = *(imu_deque.front());
                        imu_deque.pop_front();
                        input_in.gyro<<imu_last.angular_velocity.x, imu_last.angular_velocity.y, imu_last.angular_velocity.z;
                        input_in.acc <<imu_last.linear_acceleration.x, imu_last.linear_acceleration.y, imu_last.linear_acceleration.z; 

                        // angvel_avr<<0.5 * (imu_last.angular_velocity.x + imu_next.angular_velocity.x),
                        //             0.5 * (imu_last.angular_velocity.y + imu_next.angular_velocity.y),
                        //             0.5 * (imu_last.angular_velocity.z + imu_next.angular_velocity.z);
                                        
                        // acc_avr   <<0.5 * (imu_last.linear_acceleration.x + imu_next.linear_acceleration.x),
                        //             0.5 * (imu_last.linear_acceleration.y + imu_next.linear_acceleration.y),
                        //             0.5 * (imu_last.linear_acceleration.z + imu_next.linear_acceleration.z);
                        input_in.acc    = input_in.acc * G_m_s2 / acc_norm; 
                        double dt = get_time_sec(imu_last.header.stamp) - t_last;

                        // if(!prop_at_freq_of_imu)
                        // {       
                        double dt_cov = get_time_sec(imu_last.header.stamp) - time_update_last;
                        if (dt_cov > 0.0)
                        {
                            kf_input.predict(dt_cov, Q_input, input_in, false, true); 
                            time_update_last = get_time_sec(imu_last.header.stamp); //time_current;
                        }
                            kf_input.predict(dt, Q_input, input_in, true, false); 
                        t_last = get_time_sec(imu_last.header.stamp);
                        imu_prop_cov = true;
                        // imu_upda_cov = true;
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

                    // if(k == 0)
                    // {
                    //     fout_imu_pbp << Measures.lidar_last_time - first_lidar_time << " " << imu_last.angular_velocity.x << " " << imu_last.angular_velocity.y << " " << imu_last.angular_velocity.z \
                    //             << " " << imu_last.linear_acceleration.x << " " << imu_last.linear_acceleration.y << " " << imu_last.linear_acceleration.z << endl;
                    // }

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

                    // if(prop_at_freq_of_imu)
                    // {
                    //     double dt_cov = time_current - time_update_last;
                    //     if ((imu_prop_cov && dt_cov > 0.0) || (dt_cov >= imu_time_inte * 1.2)) 
                    //     {
                    //         double propag_cov_start = omp_get_wtime();
                    //         kf_input.predict(dt_cov, Q_input, input_in, false, true); 
                    //         propag_time += omp_get_wtime() - propag_cov_start;
                    //         time_update_last = time_current;
                    //         imu_prop_cov = false;
                    //     }
                    // }
                    if (publish_odometry_without_downsample)
                    {
                        /******* Publish odometry *******/

                        // publish_odometry(pubOdomAftMapped);
                        publish_odometry(pubOdomAftMapped, tf_broadcaster, tf_buffer, this->get_logger());   
                        if (runtime_pos_log)
                        {
                            state_in = kf_input.x_;
                            euler_cur = SO3ToEuler(state_in.rot);
                            fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_in.pos.transpose() << " " << state_in.vel.transpose() \
                            <<" "<<state_in.bg.transpose()<<" "<<state_in.ba.transpose()<<" "<<state_in.gravity.transpose()<<" "<<feats_undistort->points.size()<<endl;
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

            /******* Publish odometry downsample *******/
            if (!publish_odometry_without_downsample)
            {
                publish_odometry(pubOdomAftMapped, tf_broadcaster, tf_buffer, this->get_logger());   
            }

            /*** add the feature points to map kdtree ***/
            t3 = omp_get_wtime();
            
            if(feats_down_size > 4)
            {
                map_incremental();
            }

            t5 = omp_get_wtime();
            /******* Publish points *******/
            if (path_en)                         publish_path(pubPath);
            if (scan_pub_en || pcd_save_en)      publish_frame_world(pubLaserCloudFullRes);
            if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFullRes_body);
            
            /*** Debug variables Logging ***/
            if (runtime_pos_log)
            {
                frame_num ++;
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;

                aver_time_icp = aver_time_icp * (frame_num - 1)/frame_num + update_time/frame_num;
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
                        state_out = kf_output.x_;
                        euler_cur = SO3ToEuler(state_out.rot);
                        fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_out.pos.transpose() << " " << state_out.vel.transpose() \
                        <<" "<<state_out.omg.transpose()<<" "<<state_out.acc.transpose()<<" "<<state_out.gravity.transpose()<<" "<<state_out.bg.transpose()<<" "<<state_out.ba.transpose()<<" "<<feats_undistort->points.size()<<endl;
                    }
                    else
                    {
                        state_in = kf_input.x_;
                        euler_cur = SO3ToEuler(state_in.rot);
                        fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_in.pos.transpose() << " " << state_in.vel.transpose() \
                        <<" "<<state_in.bg.transpose()<<" "<<state_in.ba.transpose()<<" "<<state_in.gravity.transpose()<<" "<<feats_undistort->points.size()<<endl;
                    }
                }
                dump_lio_state_to_log(fp);
            }
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl_pc;
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
