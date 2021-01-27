#include <iostream>
#include <fstream>
#include <thread>
#include <string>
#include <vector>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/console/parse.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/common/pca.h>

#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/crop_box.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/region_growing_rgb.h>

using namespace std::chrono_literals;

//global visualizer
pcl::visualization::PCLVisualizer::Ptr viewer;
pcl::visualization::PCLVisualizer::Ptr detail_viewer;
int viewport_1, viewport_2, viewport_3, viewport_4;

//global clouds
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr input_cloud;
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr alligned_cloud;
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr clustered_cloud;
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr slab_cloud;
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr extracted_cloud;
//pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pier_cloud;
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr clustered_extraction_cloud;
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr bridge_cloud;



//global normals
pcl::PointCloud<pcl::Normal>::Ptr normals;
pcl::PointCloud<pcl::Normal>::Ptr estimated_normals;

int bbox_number = 0;
unsigned int show_normals = 0;
bool detail_view = false;
std::string current_detail_cloud_name = "";
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr current_detail_cloud;


//reads txt cloud file 
int read_ASCII(std::string filename, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr c, pcl::PointCloud <pcl::Normal>::Ptr n = nullptr) {
    std::cout << "Loading ASCII point cloud with normals ..." << std::endl;
    FILE* file = fopen(filename.c_str(), "r");

    if (file == NULL)
    {
        std::cout << "ERROR: failed to open file: " << filename << endl;
        return -1;
    }

    float x, y, z;
    float r, g, b;
    float nx, ny, nz;

    while (!feof(file))
    {
        int n_args = fscanf(file, "%f %f %f %f %f %f %f %f %f", &x, &y, &z, &r, &g, &b, &nx, &ny, &nz);
        if (n_args != 9)
            continue;

        pcl::PointXYZRGBNormal point;
        point.x = x;
        point.y = y;
        point.z = z;
        point.r = (char)(r * 255.0);
        point.g = (char)(g * 255.0);
        point.b = (char)(b * 255.0);
        point.normal_x = nx;
        point.normal_y = ny;
        point.normal_z = nz;
        c->push_back(point);

        pcl::Normal normal;
        normal.normal_x = nx;
        normal.normal_y = ny;
        normal.normal_z = nz;
        n->push_back(normal);
    }

    fclose(file);

    std::cout << "Loaded cloud with " << c->size() << " points." << std::endl;
    return 1;
}

//alligns the point cloud by the center of mass and adds a bounding box
void allign_cloud(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr in_cloud, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr alligned_cloud) {
    // http://codextechnicanum.blogspot.com/2015/04/find-minimum-oriented-bounding-box-of.html

     //============== compute the principal direction of the cloud by PCA ============
    //compute centroid
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*in_cloud, centroid);

    //compute covariance matrix
    Eigen::Matrix3f covariance_mat;
    computeCovarianceMatrixNormalized(*in_cloud, centroid, covariance_mat);

    //get eigenvectors
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance_mat, Eigen::ComputeEigenvectors);
    Eigen::Matrix3f eigen_vecs = eigen_solver.eigenvectors();

    //ensure right orientation for eigen_vec 3 due to signs
    eigen_vecs.col(2) = eigen_vecs.col(0).cross(eigen_vecs.col(1));

    // PCL Variant
    /*
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr projected_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PCA<pcl::PointXYZRGB> pca;
    pca.setInputCloud(in_cloud);
    pca.project(*in_cloud, *projected_cloud);
    Eigen::Matrix3f eigen_vecs = pca.getEigenVectors();
    */

    //============= transform the whole point cloud to the origin and allign the cloud paralell to the global coordinate axis by using the bounding box ===========
    // project cloud to the eigen vectors / build projection matrix
    Eigen::Matrix4f projection(Eigen::Matrix4f::Identity());
    projection.block<3, 3>(0, 0) = eigen_vecs.transpose();
    projection.block<3, 1>(0, 3) = -1.0f * (projection.block<3, 3>(0, 0) * centroid.head<3>());

    //transform to origin
    pcl::transformPointCloudWithNormals(*in_cloud, *alligned_cloud, projection);

    //const Eigen::Quaternionf bboxQuaternion(eigen_vecs);
    //const Eigen::Vector3f bboxTransform = eigen_vecs * meanDiagonal + centroid.head<3>();
}

//adds a simple bounding box to cloud
void add_bounding_box(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr in_cloud, int viewport, float color_r = 1.0, float color_g = 0.0, float color_b = 0.0, pcl::visualization::PCLVisualizer::Ptr in_viewer = viewer) {
    
    //get min, max and diagonal center for bounding box
    pcl::PointXYZRGBNormal minPoint, maxPoint;
    pcl::getMinMax3D(*in_cloud, minPoint, maxPoint);
    const Eigen::Vector3f meanDiagonal = 0.5f * (maxPoint.getVector3fMap() + minPoint.getVector3fMap());

    in_viewer->addCube(minPoint.x, maxPoint.x, minPoint.y, maxPoint.y, minPoint.z, maxPoint.z, color_r, color_g, color_b, "bounding_box_"+bbox_number, viewport);
    bbox_number++;
}

//estimates normals by fitting local tangent plane
void estimate_normals(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr normals) {
    pcl::search::Search<pcl::PointXYZRGB>::Ptr search_tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> normal_estimator;
    normal_estimator.setSearchMethod(search_tree);
    normal_estimator.setInputCloud(cloud);
    normal_estimator.setKSearch(50);
    normal_estimator.setViewPoint(-600.589, 1108.48, 1479.04);
    normal_estimator.compute(*normals);
}

//region growing depenant on planarity constraints determines the slab/deck
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr extract_slab(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr in_cloud, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr colored_cloud)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*in_cloud, *cloud);

    pcl::search::Search<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);

    pcl::IndicesPtr indices(new std::vector <int>);
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(0.0, 1.0);
    pass.filter(*indices);

    pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
    reg.setMinClusterSize(200);
    reg.setMaxClusterSize(100000);
    reg.setSearchMethod(tree);
    reg.setNumberOfNeighbours(500);
    reg.setInputCloud(cloud);
    //reg.setIndices (indices);
    reg.setInputNormals(normals);
    reg.setSmoothnessThreshold(0.45 / 180.0 * M_PI);
    reg.setCurvatureThreshold(1.7);

    std::vector <pcl::PointIndices> clusters;
    reg.extract(clusters);

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr slab_cluster;
    float maxDiagLength = 0;
    for (auto c = 0; c < clusters.size(); c++) {
        if (clusters[c].indices.size() < 2000) {
            continue;
        }

        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr ext_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        pcl::copyPointCloud(*in_cloud, clusters[c].indices, *ext_cloud);
        pcl::PointXYZRGBNormal minPoint, maxPoint;
        pcl::getMinMax3D(*ext_cloud, minPoint, maxPoint);

        Eigen::Vector3f diag = maxPoint.getVector3fMap() - minPoint.getVector3fMap();
        float l = diag.norm();
        if (l > maxDiagLength) {
            maxDiagLength = l;
            slab_cluster = ext_cloud;
        }
    }
    //std::cout << "\n" << longest_cluster->size() << " -> longest\n" << input_cloud->size() << " -> input\n";
    pcl::copyPointCloud(*reg.getColoredCloud(), *colored_cloud);

    return slab_cluster;
}

//colored region growing
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr filter_surrounding(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr in_cloud, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr colored_cloud)
{
    pcl::search::Search <pcl::PointXYZRGBNormal>::Ptr rg_tree(new pcl::search::KdTree<pcl::PointXYZRGBNormal>);

    //color test
    pcl::RegionGrowingRGB<pcl::PointXYZRGBNormal, pcl::Normal> reg;
    reg.setInputCloud(in_cloud);
    //reg.setIndices(indices);
    reg.setSearchMethod(rg_tree);
    reg.setDistanceThreshold(10);
    reg.setPointColorThreshold(9);
    reg.setRegionColorThreshold(10);
    reg.setMinClusterSize(30);

    //normal test
    /*reg.setInputNormals(normals);
    reg.setNumberOfNeighbours(20);
    reg.setSmoothnessThreshold(0.2 / 180.0 * M_PI);
    reg.setCurvatureThreshold(0.4);*/

    std::vector <pcl::PointIndices> clusters;
    reg.extract(clusters);
    pcl::copyPointCloud(*reg.getColoredCloud(), *colored_cloud);

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr ext_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    ////////////find biggest cluster
    pcl::copyPointCloud(*in_cloud, clusters[0].indices, *ext_cloud);

    return ext_cloud;
}

// keyboard callback
void keyboardEventOccurred(const pcl::visualization::KeyboardEvent& event, void* viewer_void) {
    pcl::visualization::PCLVisualizer* viewer = static_cast<pcl::visualization::PCLVisualizer*> (viewer_void);

    //std::cout << "keycode: " << event.getKeyCode()<<"\n";
    //std::cout << "keysymbol: " << event.getKeySym()<<"\n";

    //toggles between no-, estimated- and original normals
    if (event.getKeySym() == "n" && event.keyDown())
    {
        if (show_normals == 0) {
            if (detail_view && (current_detail_cloud_name == "alligned_cloud" || current_detail_cloud_name == "clustered_cloud")) {
                detail_viewer->addPointCloudNormals<pcl::PointXYZRGBNormal, pcl::Normal>(current_detail_cloud, normals, 10, 4.5, "detail_normals", 0);
                detail_viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 1.0, 0.0, "detail_normals", 0);
                detail_viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, current_detail_cloud_name);
            }
            viewer->addPointCloudNormals<pcl::PointXYZRGBNormal, pcl::Normal>(alligned_cloud, normals, 10, 4.5, "normals", 1);
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 1.0, 0.0, "normals", 1);
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "alligned_cloud");

            std::cout << "enable normals vis\n";
        }
        else if (show_normals == 1) {
            if (detail_view && (current_detail_cloud_name == "alligned_cloud" || current_detail_cloud_name == "clustered_cloud")) {
                detail_viewer->removePointCloud("detail_normals");
                detail_viewer->addPointCloudNormals<pcl::PointXYZRGBNormal, pcl::Normal>(current_detail_cloud, estimated_normals, 10, 4.5, "detail_estimated_normals", 0);
                detail_viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 1.0, "detail_estimated_normals");
            }
            viewer->removePointCloud("normals");
            viewer->addPointCloudNormals<pcl::PointXYZRGBNormal, pcl::Normal>(alligned_cloud, estimated_normals, 10, 4.5, "estimated_normals", 1);
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 1.0, "estimated_normals");

            std::cout << "enable estimated normals vis\n";
        }
        else {
            if (detail_view && (current_detail_cloud_name == "alligned_cloud" || current_detail_cloud_name == "clustered_cloud")) {
                detail_viewer->removePointCloud("detail_estimated_normals");
                detail_viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, current_detail_cloud_name);
            }
            viewer->removePointCloud("estimated_normals");
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "alligned_cloud");

            std::cout << "disable estimated normals vis\n";
        }
        show_normals = (show_normals + 1) % 3;
    }

    //enables and sets the detail view according to input
    if (event.getKeySym() == "F1" && event.keyDown()) {
        if (current_detail_cloud_name == "alligned_cloud") {
            return;
        }
        detail_viewer->removePointCloud(current_detail_cloud_name);
        detail_viewer->removeAllShapes();
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal> rgb(alligned_cloud);
        detail_viewer->addPointCloud<pcl::PointXYZRGBNormal>(alligned_cloud, rgb, "alligned_cloud");
        add_bounding_box(alligned_cloud, 0, 1.0, 0.0, 0.0, detail_viewer);
        add_bounding_box(slab_cloud, 0, 0.0, 1.0, 1.0, detail_viewer);
        detail_view = true;
        current_detail_cloud_name = "alligned_cloud";
        current_detail_cloud = alligned_cloud;
        detail_viewer->setRepresentationToWireframeForAllActors();
    }

    if (event.getKeySym() == "F2" && event.keyDown()) {
        if (current_detail_cloud_name == "clustered_cloud") {
            return;
        }
        detail_viewer->removePointCloud(current_detail_cloud_name);
        detail_viewer->removeAllShapes();
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal> rgb(clustered_cloud);
        detail_viewer->addPointCloud<pcl::PointXYZRGBNormal>(clustered_cloud, rgb, "clustered_cloud");
        add_bounding_box(clustered_cloud, 0, 1.0, 0.0, 0.0, detail_viewer);
        detail_view = true;
        current_detail_cloud_name = "clustered_cloud";
        current_detail_cloud = clustered_cloud;
        detail_viewer->setRepresentationToWireframeForAllActors();
    }

    if (event.getKeySym() == "F3" && event.keyDown()) {
        if (current_detail_cloud_name == "slab_cloud") {
            return;
        }
        detail_viewer->removePointCloud(current_detail_cloud_name);
        detail_viewer->removeAllShapes();
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal> rgb(extracted_cloud);
        detail_viewer->addPointCloud<pcl::PointXYZRGBNormal>(extracted_cloud, rgb, "extracted_cloud");
        add_bounding_box(extracted_cloud, 0, 1.0, 0.0, 0.0, detail_viewer);
        detail_view = true;
        current_detail_cloud_name = "extracted_cloud";
        current_detail_cloud = extracted_cloud;
        detail_viewer->setRepresentationToWireframeForAllActors();
    }

    if (event.getKeySym() == "F4" && event.keyDown()) {
        if (current_detail_cloud_name == "clustered_extraction_cloud") {
            return;
        }
        detail_viewer->removePointCloud(current_detail_cloud_name);
        detail_viewer->removeAllShapes();
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal> rgb(clustered_extraction_cloud);
        detail_viewer->addPointCloud<pcl::PointXYZRGBNormal>(clustered_extraction_cloud, rgb, "clustered_extraction_cloud");
        //add_bounding_box(clustered_extraction, 0, 1.0, 0.0, 0.0, detail_viewer);
        detail_view = true;
        current_detail_cloud_name = "clustered_extraction_cloud";
        current_detail_cloud = clustered_extraction_cloud;
        detail_viewer->setRepresentationToWireframeForAllActors();
    }

    if (event.getKeySym() == "F5" && event.keyDown()) {
        if (current_detail_cloud_name == "slab_cloud") {
            return;
        }
        detail_viewer->removePointCloud(current_detail_cloud_name);
        detail_viewer->removeAllShapes();
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal> rgb(slab_cloud);
        detail_viewer->addPointCloud<pcl::PointXYZRGBNormal>(slab_cloud, rgb, "slab_cloud");
        add_bounding_box(slab_cloud, 0, 1.0, 0.0, 0.0, detail_viewer);
        detail_view = true;
        current_detail_cloud_name = "slab_cloud";
        current_detail_cloud = slab_cloud;
        detail_viewer->setRepresentationToWireframeForAllActors();
    }

    if (event.getKeySym() == "F6" && event.keyDown()) {
        if (current_detail_cloud_name == "bridge_cloud") {
            return;
        }
        detail_viewer->removePointCloud(current_detail_cloud_name);
        detail_viewer->removeAllShapes();
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal> rgb(bridge_cloud);
        detail_viewer->addPointCloud<pcl::PointXYZRGBNormal>(bridge_cloud, rgb, "slab_cloud");
        add_bounding_box(bridge_cloud, 0, 1.0, 0.0, 0.0, detail_viewer);
        detail_view = true;
        current_detail_cloud_name = "bridge_cloud";
        current_detail_cloud = bridge_cloud;
        detail_viewer->setRepresentationToWireframeForAllActors();
    }

}

//sets visualization and camera up
void setup_pointcloud_viewer(std::string camera_file = "") {

    //creates visualisation with 4 windows
    viewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("PointCloud viewer"));
    viewer->createViewPort(0.0, 0.5, 0.5, 1.0, viewport_1);
    viewer->createViewPort(0.5, 0.5, 1.0, 1.0, viewport_2);
    viewer->createViewPort(0.0, 0.0, 0.5, 0.5, viewport_3);
    viewer->createViewPort(0.5, 0.0, 1.0, 0.5, viewport_4);

    viewer->setBackgroundColor(1, 1, 1);
    viewer->addCoordinateSystem(30.0, "coordinate_system", 0);
    viewer->setRepresentationToWireframeForAllActors();

    //loads camera file if provided
    viewer->initCameraParameters();
    if (!camera_file.empty()) {

        //read .cam file
        std::string in = "";
        std::ifstream camera_filestream(camera_file);
        if (camera_filestream.is_open()) {
            getline(camera_filestream, in, ',');
            float near_dist = std::stof(in);
            getline(camera_filestream, in, '/');
            float far_dist = std::stof(in);

            getline(camera_filestream, in, ',');
            float view_x = std::stof(in);
            getline(camera_filestream, in, ',');
            float view_y = std::stof(in);
            getline(camera_filestream, in, '/');
            float view_z = std::stof(in);

            getline(camera_filestream, in, ',');
            float pos_x = std::stof(in);
            getline(camera_filestream, in, ',');
            float pos_y = std::stof(in);
            getline(camera_filestream, in, '/');
            float pos_z = std::stof(in);

            getline(camera_filestream, in, ',');
            float up_x = std::stof(in);
            getline(camera_filestream, in, ',');
            float up_y = std::stof(in);
            getline(camera_filestream, in, '/');
            float up_z = std::stof(in);

            getline(camera_filestream, in, '/');
            float fov = std::stof(in);

            //set camera settings
            viewer->setCameraClipDistances(near_dist, far_dist);
            viewer->setCameraPosition(pos_x, pos_y, pos_z, view_x, view_y, view_z, up_x, up_y, up_z);
            viewer->setCameraFieldOfView(fov);
        }
        else
        {
            std::cout << "Failed to open file: " << camera_file;
        }
        camera_filestream.close();
    }

    //additionally add detail view
    detail_viewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("PointCloud detail viewer"));
    detail_viewer->setBackgroundColor(1, 1, 1);
    detail_viewer->addCoordinateSystem(30.0, "coordinate_system", 0);
    detail_viewer->setRepresentationToWireframeForAllActors();
    detail_viewer->initCameraParameters();
    std::vector<pcl::visualization::Camera> cam;
    viewer->getCameras(cam);
    detail_viewer->setCameraParameters(cam[0]);
    detail_viewer->registerKeyboardCallback(keyboardEventOccurred, (void*)detail_viewer.get());

    //enables keyboard and mouse input
    viewer->registerKeyboardCallback(keyboardEventOccurred, (void*)viewer.get());
    //viewer->registerMouseCallback(mouseEventOccurred, (void*)viewer.get());
}

//main
int main(int argc, char** argv)
{
    //parse arguments
    if (argc < 2) {
        std::cout << "Usage: startPipeline <pcdfile> / <ASCII .txt> [CAMERA SETTINGS]";
        return 0;
    }

    //read input pointcloud 
    input_cloud = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    normals = pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud<pcl::Normal>);
    std::string file_name = argv[1];
    std::string format = file_name.substr(file_name.find_last_of(".") + 1);
    if (format == "pcd") {
        std::cout << "Read pcd file." << std::endl;
        if (pcl::io::loadPCDFile <pcl::PointXYZRGBNormal>(file_name, *input_cloud) == -1){
            std::cout << "Cloud reading failed." << std::endl;
            return (-1);
        }
        pcl::copyPointCloud(*input_cloud, *normals);
    }
    else if(format == "txt"){
        std::cout << "Read txt file." << std::endl;
        if (read_ASCII(file_name, input_cloud, normals) == -1) {
            std::cout << "Cloud reading failed." << std::endl;
            return (-1);
        }
    }
    else {
        std::cout << "File format ." << format << " of " << file_name << " not supported. Valid formats: .pcd .txt";
        return(-1);
    }

    //check for camera file and setup viewer + camera
    if (argc > 2) {
        file_name = argv[2];
        format = file_name.substr(file_name.find_last_of(".") + 1);
        if (format == "cam") {
            std::cout<<"Load camera file and setup viewer." << std::endl;
            setup_pointcloud_viewer(file_name);

        }else {
            std::cout << "Camera file has the wrong format. Use .cam file generated by PCL Viewer." << endl;
            std::cout << "Load default camera settings." << std::endl;
            setup_pointcloud_viewer();
        }
    }
    else {
        std::cout << "Setup viewer." << std::endl;
        setup_pointcloud_viewer();
    }

    //allign input cloud
    std::cout << "Allign point cloud" << std::endl;
    alligned_cloud = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr centered_cloud = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    allign_cloud(input_cloud, centered_cloud);
    //additionally rotate cloud
    Eigen::Affine3f rot = Eigen::Affine3f::Identity();
    rot.rotate(Eigen::AngleAxisf(-M_PI/2, Eigen::Vector3f::UnitZ()));
    pcl::transformPointCloudWithNormals(*centered_cloud, *alligned_cloud, rot);
    centered_cloud.reset();

    //copy alligned normals
    pcl::copyPointCloud(*alligned_cloud, *normals);

    //vis. alligned input cloud
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal> rgb(alligned_cloud);
    viewer->addPointCloud<pcl::PointXYZRGBNormal>(alligned_cloud, rgb, "alligned_cloud", viewport_1);
    add_bounding_box(alligned_cloud, viewport_1);

    //generate estimated normals
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_without_normals(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud(*alligned_cloud, *cloud_without_normals);
    estimated_normals = pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud<pcl::Normal>);
    estimate_normals(cloud_without_normals, estimated_normals);

    //================== slab extraction and bounding box creation ==============
    //extract slab
    std::cout << "Extract slab." << std::endl;
    clustered_cloud = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    slab_cloud = extract_slab(alligned_cloud, clustered_cloud);
    
    //show clustering
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal> clustered_rgb(clustered_cloud);
    viewer->addPointCloud<pcl::PointXYZRGBNormal>(clustered_cloud, clustered_rgb, "clustered_cloud", viewport_2);
    add_bounding_box(clustered_cloud, viewport_2);

    //show slab extraction
    //pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal> extracted_slab_rgb(slab_cloud);
    //viewer->addPointCloud<pcl::PointXYZRGBNormal>(slab_cloud, extracted_slab_rgb, "slab_cloud", viewport_3);
    add_bounding_box(slab_cloud, viewport_3, 0.0, 1.0, 1.0);
    add_bounding_box(slab_cloud, viewport_1, 0.0, 1.0, 1.0);

    //================== expand and crop bounding box ==============

    //construct bounds 
    pcl::PointXYZRGBNormal minPoint_cloud, maxPoint_cloud, minPoint_extract, maxPoint_extract;
    pcl::getMinMax3D(*alligned_cloud, minPoint_cloud, maxPoint_cloud);
    pcl::getMinMax3D(*slab_cloud, minPoint_extract, maxPoint_extract);
    Eigen::Vector4f maxBound = Eigen::Vector4f(maxPoint_extract.x, maxPoint_extract.y, maxPoint_extract.z, 1.0);
    Eigen::Vector4f minBound = Eigen::Vector4f(minPoint_extract.x, minPoint_cloud.y, minPoint_extract.z, 1.0);
    
    //filter by box filter containing occlusions
    std::cout << "Extract bridge with surrounding." << std::endl;
    extracted_cloud = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::CropBox<pcl::PointXYZRGBNormal> boxFilter;
    boxFilter.setMax(maxBound);
    boxFilter.setMin(minBound);
    boxFilter.setInputCloud(alligned_cloud);
    boxFilter.filter(*extracted_cloud);

    //visualize extracted bridge with surrounding
    std::cout << "Filter bridge." << std::endl;
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal> extracted_rgb(extracted_cloud);
    viewer->addPointCloud<pcl::PointXYZRGBNormal>(extracted_cloud, extracted_rgb, "extracted_cloud", viewport_3);
    add_bounding_box(extracted_cloud, viewport_3, 1.0, 0.0, 1.0);
    add_bounding_box(extracted_cloud, viewport_1, 1.0, 0.0, 1.0);

    //filter surrounding
    clustered_extraction_cloud = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pier_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pier_cloud = filter_surrounding(extracted_cloud, clustered_extraction_cloud);

    //visualize clustered extraction
    std::cout << "Show final bridge extraction." << std::endl;
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal> clust_extracted_rgb(clustered_extraction_cloud);
    viewer->addPointCloud<pcl::PointXYZRGBNormal>(clustered_extraction_cloud, clust_extracted_rgb, "clustered_extraction", viewport_4);

    viewer->setRepresentationToWireframeForAllActors();

    //main viewer loop
    while (!viewer->wasStopped()){
        viewer->spinOnce(100);
        std::this_thread::sleep_for(100ms);
    }
}