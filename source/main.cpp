#include<omp.h>
#include <direct.h>

#include <pcl/io/obj_io.h>
#include<pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/crop_hull.h>
#include <pcl/filters/random_sample.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/ml/kmeans.h>

#include "viewer.h"
#include "utils.h"

using namespace std::chrono_literals; 

std::string bridge_name = "";
bool close_flag = false;

//global visualizer
Viewer* viewer;
Viewer* detail_viewer;

//input mesh
pcl::PolygonMesh::Ptr mesh;

//global clouds and indices for viewer
//input and alignment clouds
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr input_cloud;
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr subsampled_cloud;
pcl::PointIndices::Ptr subsampled_indices;

//slab and cplanar cluster clouds
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr clustered_cloud;
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr slab_cloud;
pcl::PointIndices::Ptr slab_indices;
int config_num_neightbours = 500;
float config_smoothness = 0.62;
float config_curvature = 1.6;

//clipping cloud and indices
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr clipped_cloud;
pcl::PointIndices::Ptr clipped_indices;

//first color clustering clouds and indices
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr first_clustered_extraction_cloud;
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr first_filtered_cloud;
pcl::PointIndices::Ptr first_filtered_indices;

//second color clustering clouds and indices
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr second_clustered_extraction_cloud;
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr second_filtered_cloud;
pcl::PointIndices::Ptr second_filtered_indices;

//final output cloud and indices after outlier removal
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr bridge_cloud;
pcl::PointIndices::Ptr bridge_indices;
int config_point_neightbourhood = 200;
float config_std_deviation = 0.8;
int config_num_cluster_first = 4;
int config_num_cluster_second = 3;

//global normals
pcl::PointCloud<pcl::Normal>::Ptr normals;
pcl::PointCloud<pcl::Normal>::Ptr subsampled_normals;
pcl::PointCloud<pcl::Normal>::Ptr estimated_normals;
int show_normals = false; //normal toggle

//subsamples input to 2mio points
pcl::PointIndices subsample_pointcloud(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr in_cloud, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr out_cloud, pcl::PointIndices::Ptr removed_indices = nullptr) {

    pcl::RandomSample<pcl::PointXYZRGBNormal> rand_sampling;
    rand_sampling.setInputCloud(in_cloud);
    rand_sampling.setSeed(std::rand());
    rand_sampling.setSample((unsigned int)(2000000));
    if (removed_indices != nullptr) {
        rand_sampling.getRemovedIndices(*removed_indices);
    }
    pcl::PointIndices indices;
    rand_sampling.filter(indices.indices);
    pcl::copyPointCloud(*in_cloud, indices, *out_cloud);

    return indices;
}

//aligns pointcloud by its oriented bounding box
void align_pointcloud(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr in_cloud) {

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr aligned_cloud;
    aligned_cloud = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);

    //use the axis aligned bounding box to transform to origin
    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();
    BoundingBox aligned_bbox = create_oriented_boundingbox(in_cloud, 1.0, 1.0, 0.0, projection);
    pcl::transformPointCloudWithNormals(*in_cloud, *aligned_cloud, projection);

    //rotate by align average normal along +z
    pcl::VectorAverage3f avNormal;
    for (auto i = aligned_cloud->points.begin(); i != aligned_cloud->points.end(); i++) {
        avNormal.add(i->getNormalVector3fMap());
    }

    Eigen::Vector3f averageNormal = avNormal.getMean(); //calculate average normal
    float cosin = (averageNormal.dot(Eigen::Vector3f::UnitZ())) / averageNormal.norm(); //calculate cosin to unit z

    //rotate 180° if average normal is negaitve
    if (cosin < 0) {
        in_cloud->clear();
        Eigen::Affine3f rot = Eigen::Affine3f::Identity();
        rot.rotate(Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitX()));
        pcl::transformPointCloudWithNormals(*aligned_cloud, *in_cloud, rot);
        std::cout << "ALIGNMENT:: Rotated Cloud!" << std::endl;
    }
    else {
        in_cloud->clear();
        pcl::copyPointCloud(*aligned_cloud, *in_cloud);
    }
}

//realign cloud by slab
void realign_by_slab(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr in_cloud, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr slab){
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr aligned_cloud;
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr aligned_slab;
    aligned_cloud = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    aligned_slab = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);

    //use the axis aligned bounding box to transform to origin
    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();
    BoundingBox aligned_bbox = create_oriented_boundingbox(slab, 1.0, 1.0, 0.0, projection);
    pcl::transformPointCloudWithNormals(*in_cloud, *aligned_cloud, projection);
    pcl::transformPointCloudWithNormals(*slab, *aligned_slab, projection);

    //rotate by align average normal along +z
    pcl::VectorAverage3f avNormal;
    for (auto i = aligned_slab->points.begin(); i != aligned_slab->points.end(); i++) {
        avNormal.add(i->getNormalVector3fMap());
    }

    Eigen::Vector3f averageNormal = avNormal.getMean(); //calculate average normal
    float cosin = (averageNormal.dot(Eigen::Vector3f::UnitZ())) / averageNormal.norm(); //calculate cosin to unit z

    //rotate 180° if average normal is negaitve
    if (cosin < 0) {
        in_cloud->clear();
        slab->clear();
        Eigen::Affine3f rot = Eigen::Affine3f::Identity();
        rot.rotate(Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitX()));
        pcl::transformPointCloudWithNormals(*aligned_cloud, *in_cloud, rot);
        pcl::transformPointCloudWithNormals(*aligned_slab, *slab, rot);
        std::cout << "ALIGNMENT:: Rotated Cloud!" << std::endl;
    }
    else {
        in_cloud->clear();
        slab->clear();
        pcl::copyPointCloud(*aligned_cloud, *in_cloud);
        pcl::copyPointCloud(*aligned_slab, *slab);
    }
}

//region growing depenant on planarity constraints determines the slab/deck
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr extract_slab(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr in_cloud, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr colored_cloud)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*in_cloud, *cloud);

    pcl::search::Search<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
    reg.setMinClusterSize(5000);
    reg.setMaxClusterSize(in_cloud->points.size()/2);
    reg.setSearchMethod(tree);
    reg.setNumberOfNeighbours(config_num_neightbours);
    reg.setInputCloud(cloud);
    reg.setInputNormals(estimated_normals);
    reg.setSmoothnessThreshold(config_smoothness / 180.0 * M_PI);
    reg.setCurvatureThreshold(config_curvature);

    std::vector <pcl::PointIndices> clusters;
    reg.extract(clusters);

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr slab_cluster;
    float maxDiagLength = 0;
    float height_condition = 0;
    slab_indices = pcl::PointIndices::Ptr(new pcl::PointIndices);
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
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*ext_cloud, centroid);
        float h = centroid[2];
        if (l > maxDiagLength && h > height_condition){
            maxDiagLength = l;
            slab_cluster = ext_cloud;
            *slab_indices = clusters[c];
            height_condition = h;
        }
    }
    //std::cout << "\n" << longest_cluster->size() << " -> longest\n" << subsampled_cloud->size() << " -> input\n";
    pcl::copyPointCloud(*reg.getColoredCloud(), *colored_cloud);

    return slab_cluster;
}

//creates polygonal prism for bridge bounding volume 
void clip_bounding_volume(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr in_cloud, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr s_cloud, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr out_cloud) {
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_on_hull(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr slab_cloud_spatial(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud_spatial(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::PointXYZRGBNormal minPoint_cloud, maxPoint_cloud, minPoint_extract, maxPoint_extract;
    pcl::getMinMax3D(*in_cloud, minPoint_cloud, maxPoint_cloud);
    pcl::getMinMax3D(*s_cloud, minPoint_extract, maxPoint_extract);
    
    //create prism
    pcl::copyPointCloud(*s_cloud, *slab_cloud_spatial);
    pcl::copyPointCloud(*in_cloud, *input_cloud_spatial);
    std::vector<pcl::Vertices> hull_indices;
    pcl::ConcaveHull<pcl::PointXYZ> hull;
    //pcl::ConvexHull<pcl::PointXYZ> hull;
    hull.setInputCloud(slab_cloud_spatial);
    hull.setAlpha(20.0);
    hull.setDimension(2);
    hull.reconstruct(*cloud_on_hull, hull_indices);

    //extract pointcloud
    if (hull.getDimension() == 2) {
        pcl::ExtractPolygonalPrismData<pcl::PointXYZ> prism;
        prism.setInputCloud(input_cloud_spatial);
        prism.setInputPlanarHull(cloud_on_hull);
        prism.setHeightLimits(-(maxPoint_extract.z-minPoint_cloud.z),maxPoint_cloud.z-minPoint_cloud.z);
        pcl::PointIndices::Ptr extracted_cloud_indices(new pcl::PointIndices);

        prism.segment(*extracted_cloud_indices);

        pcl::ExtractIndices<pcl::PointXYZRGBNormal> extract;
        extract.setInputCloud(in_cloud);
        extract.setIndices(extracted_cloud_indices);
        clipped_indices = extracted_cloud_indices;
        extract.filter(*out_cloud);

    }
    else {
        std::cout << "Hull dimensions not 2D / Hull not planar." << std::endl;
    }
}

//filter by color with kmeans
pcl::PointCloud <pcl::PointXYZRGBNormal>::Ptr colored_kmeans(pcl::PointCloud <pcl::PointXYZRGBNormal>::Ptr const in_cloud, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr colored_cloud, unsigned int num_cluster, pcl::PointIndices &out_indices) {

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_rgbnormalized(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    normalize_RGB(in_cloud, cloud_rgbnormalized);

    cloud_rgbnormalized = in_cloud;

    detail_viewer->visualize_pointcloud(cloud_rgbnormalized, "rgb_norm_cloud");

    pcl::Kmeans means(static_cast<int>(cloud_rgbnormalized->points.size()), 3);
    means.setClusterSize(num_cluster);
    std::vector<std::vector<float>> formated_points;

    //format rgb space for kmeans
    for (size_t i = 0; i < cloud_rgbnormalized->points.size(); i++) {
        std::vector<float> data(3);
        data[0] = float(cloud_rgbnormalized->points[i].r) / 255.0;
        data[1] = float(cloud_rgbnormalized->points[i].g) / 255.0;
        data[2] = float(cloud_rgbnormalized->points[i].b) / 255.0;
        means.addDataPoint(data);
        formated_points.push_back(data);
    }
    std::cout << "kmeans : Start kmeans " << std::endl;
    means.kMeans();
    pcl::Kmeans::Centroids centroids = means.get_centroids();
    std::cout << "Kmeans : Points in total Cloud : " << formated_points.size() << std::endl;
    std::cout << "Kmeans : Centroid count: " << centroids.size() << std::endl;

    //assign each point to cluster
    std::cout << "Kmeans : Extract kmeans cluster" << std::endl;
    std::vector<pcl::PointIndices> cluster_indices;
    for (size_t i = 0; i < formated_points.size(); i++) {
        unsigned int closest_centroid = 0;
        float distance = std::numeric_limits<float>::max();
        float closest_distance = distance;
        for (size_t c = 0; c < centroids.size(); c++) {
            distance = means.distance(centroids[c], formated_points[i]);
            if (distance < closest_distance) {
                closest_distance = distance;
                closest_centroid = c;
            }
        }
        while (cluster_indices.size() <= closest_centroid) {
            cluster_indices.push_back(pcl::PointIndices());
        }
        cluster_indices[closest_centroid].indices.push_back(i);
    }
    std::cout << "kmeans : cluster sizes: " << std::endl;

    int biggest_cluster = 0;
    int biggest_cluster_size = 0;
    int size = 0;

    //extract cluster
    for (size_t j = 0; j < centroids.size(); j++) {
        size = cluster_indices[j].indices.size();
        std::cout << "kmeans : Centroid " << j << ": " << size << std::endl;

        //condition which cluster is determined as pile/bridge cluster
        if (size > biggest_cluster_size) {
            biggest_cluster = j;
            biggest_cluster_size = size;
        }

        //construct colored cloud
        int r = __int16(rand() % 255);
        int g = __int16(rand() % 255);
        int b = __int16(rand() % 255);
        int32_t rgb = (static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
        pcl::PointCloud<pcl::PointXYZRGBNormal> cloud;
        pcl::copyPointCloud(*cloud_rgbnormalized, cluster_indices[j], cloud);
        for (auto& p : cloud.points) p.rgb = rgb;
        *colored_cloud += cloud;
    }
    std::cout << "kmeans : Extract cluster " << biggest_cluster << " : " << biggest_cluster_size << std::endl;
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr extraction(new pcl::PointCloud<pcl::PointXYZRGBNormal>);

    out_indices = cluster_indices[biggest_cluster];
    pcl::copyPointCloud(*in_cloud, cluster_indices[biggest_cluster], *extraction);
    return extraction;
}

// keyboard callback
void keyboardEventOccurred(const pcl::visualization::KeyboardEvent& event, void* viewer_void) {
    pcl::visualization::PCLVisualizer* v = static_cast<pcl::visualization::PCLVisualizer*> (viewer_void);

    //show normals toggle
    if (event.getKeySym() == "n" && event.keyDown()) {
        if (show_normals == 0) {
            pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr current_cloud = detail_viewer->get_current_pointcloud().second;
            if (current_cloud) {
                detail_viewer->addPointCloudNormals<pcl::PointXYZRGBNormal, pcl::Normal>(current_cloud, subsampled_normals, 10, 0.5, "detail_subsampled_normals", 0);
                detail_viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 1.0, 0.0, "detail_subsampled_normals", 0);
            }
            else { return; }
        }
        else if (show_normals == 1)
        {
            detail_viewer->removePointCloud("detail_subsampled_normals");
            pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr current_cloud = detail_viewer->get_current_pointcloud().second;
            if (current_cloud && estimated_normals != nullptr) {
                detail_viewer->addPointCloudNormals<pcl::PointXYZRGBNormal, pcl::Normal>(current_cloud, estimated_normals, 10, 0.5, "detail_estimated_normals", 0);
                detail_viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 1.0, "detail_estimated_normals", 0);
            }
            else {
                show_normals = 0;
                return;
            }
        }
        else {
            detail_viewer->removePointCloud("detail_estimated_normals");
        }
        show_normals = (show_normals + 1) % 3;
    }

    //show viewport in detail view
    if (event.getKeySym() == "F1") {
        detail_viewer->remove_pointcloud(detail_viewer->get_current_pointcloud(0).first);
        detail_viewer->visualize_pointcloud(viewer->get_current_pointcloud(0).second, viewer->get_current_pointcloud(0).first, 0);
    }
    else if (event.getKeySym() == "F2") {
        detail_viewer->remove_pointcloud(detail_viewer->get_current_pointcloud(0).first);
        detail_viewer->visualize_pointcloud(viewer->get_current_pointcloud(1).second, viewer->get_current_pointcloud(1).first, 0);
    }
    else if (event.getKeySym() == "F3") {
        detail_viewer->remove_pointcloud(detail_viewer->get_current_pointcloud(0).first);
        detail_viewer->visualize_pointcloud(viewer->get_current_pointcloud(2).second, viewer->get_current_pointcloud(2).first, 0);
    }
    else if (event.getKeySym() == "F4") {
        detail_viewer->remove_pointcloud(detail_viewer->get_current_pointcloud(0).first);
        detail_viewer->visualize_pointcloud(viewer->get_current_pointcloud(3).second, viewer->get_current_pointcloud(3).first, 0);
    }
    else if (event.getKeySym() == "F5") {
        detail_viewer->remove_pointcloud(detail_viewer->get_current_pointcloud(0).first);
        detail_viewer->visualize_pointcloud(viewer->get_current_pointcloud(4).second, viewer->get_current_pointcloud(4).first, 0);
    }
    else if (event.getKeySym() == "F6") {
        detail_viewer->remove_pointcloud(detail_viewer->get_current_pointcloud(0).first);
        detail_viewer->visualize_pointcloud(viewer->get_current_pointcloud(5).second, viewer->get_current_pointcloud(5).first, 0);
    }
    else if (event.getKeySym() == "F7") {
        detail_viewer->remove_pointcloud(detail_viewer->get_current_pointcloud(0).first);
        detail_viewer->visualize_pointcloud(viewer->get_current_pointcloud(6).second, viewer->get_current_pointcloud(6).first, 0);
    }
    else if (event.getKeySym() == "F8") {
        detail_viewer->remove_pointcloud(detail_viewer->get_current_pointcloud(0).first);
        detail_viewer->visualize_pointcloud(viewer->get_current_pointcloud(7).second, viewer->get_current_pointcloud(7).first, 0);

    }
}

//executes the pipline without gui and writes data
void write_evaluation_data(std::vector<double>& time_measurements) {
    std::cout << "\n MAIN:: Evaluate pipeline for: " << bridge_name << std::endl;

    std::string directory = "E:/OneDrive/Dokumente/Uni/Bachelorarbeit/code/bridge_segmentation/evaluation/pipeline_output/" + bridge_name;

    if (!IsPathExist(directory)) {
        if (mkdir(directory.c_str()) == 0) {
            std::cout << "Directory "<< bridge_name << " created." << std::endl;
        }
        else {
            std::cout << "Could not create evaluation directory." << std::endl;
            return;
        }
    }

    //write time spend
    fstream out_time_measurements;
    out_time_measurements.open(directory+ "/time_measurements.txt", ios::out | ios::trunc);
    if (out_time_measurements.is_open()) {
        out_time_measurements << "Pointcloud subsampling: " << time_measurements[0];
        out_time_measurements << "Pointcloud alignment: " << time_measurements[1];
        out_time_measurements << std::endl << "Normal estimation: " << time_measurements[2];
        out_time_measurements << std::endl << "Slab extraction: " << time_measurements[3];
        out_time_measurements << std::endl << "Clipping: " << time_measurements[4];
        out_time_measurements << std::endl << "First kmeans: " << time_measurements[5];
        out_time_measurements << std::endl << "Second kmeans: " << time_measurements[6];
        out_time_measurements << std::endl << "outlier removal: " << time_measurements[7];
        out_time_measurements << std::endl << "Whole pipeline: " << time_measurements[8];
        out_time_measurements.close();
    }
    else {
        std::cout << "Could not write time measurements."<<std::endl;
    }

    fstream out_slab_indices;
    out_slab_indices.open(directory + "/slab_indices.txt", ios::out | ios::trunc);
    if (out_slab_indices.is_open()) {
        out_slab_indices << slab_indices->indices.size() <<";"<< subsampled_cloud->points.size() << std::endl;
        for (auto i = slab_indices->indices.begin(); i < slab_indices->indices.end()-1; i++) {
            out_slab_indices << *i << "; ";
        }
        out_slab_indices << *(slab_indices->indices.end() - 1);
        out_slab_indices.close();
    }
    else {
        std::cout << "Could not write slab indices."<<std::endl;
    }

    ofstream out_bridge_indices;
    out_bridge_indices.open(directory + "/bridge_indices.txt", ios::out | ios::trunc);
    if (out_bridge_indices.is_open()) {
        out_bridge_indices << bridge_indices->indices.size() << ";" << subsampled_cloud->points.size() << std::endl;
        for (auto i = bridge_indices->indices.begin(); i < bridge_indices->indices.end()-1; i++) {
            out_bridge_indices << *i << "; ";
        }
        out_bridge_indices << *(bridge_indices->indices.end() - 1);
        out_bridge_indices.close();
    }
    else {
        std::cout << "Could not write slab indices."<<std::endl;
    }
    std::cout << "Evaluation done." << std::endl;
}

//===================== MAIN ========================
//main
int main(int argc, char** argv) {

    //parse arguments
    if (argc < 2) {
        std::cout << "Usage: startPipeline [-e Evaluation option] <pcdfile> / <ASCII .txt> / <Mesh .obj> [CAMERA SETTINGS] [CONFIG FILE]";
        return 0;
    }
    int arg = 1;

    //read input pointcloud 
    input_cloud = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    normals = pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud<pcl::Normal>);
    subsampled_normals = pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud<pcl::Normal>);
    
    std::string file_name = argv[arg];
    bool evaluation_mode = false;
    if (file_name == "-e") {
        evaluation_mode = true;
        arg++;
    }

    file_name = argv[arg];
    bridge_name = file_name.substr(file_name.find_last_of("/\\")+1);
    std::string::size_type const p(bridge_name.find_first_of("_"));
    bridge_name = bridge_name.substr(0, p);
    arg++;

    
    std::string format = file_name.substr(file_name.find_last_of(".") + 1);
    if (format == "pcd") {
        std::cout << "Read pcd file." << std::endl;
        if (pcl::io::loadPCDFile <pcl::PointXYZRGBNormal>(file_name, *input_cloud) == -1) {
            std::cout << "MAIN:: Cloud reading failed." << std::endl;
            return (-1);
        }
        pcl::copyPointCloud(*input_cloud, *normals);
    }
    else if (format == "ply") {
        pcl::PLYReader Reader;
        Reader.read(file_name, *input_cloud);
        pcl::copyPointCloud(*input_cloud, *normals);
    }
    else if (format == "txt") {
        if (!read_ASCII(file_name, input_cloud, normals)) {
            return -1;
        }
    }
    else if (format == "obj") {
        //mesh = pcl::PolygonMesh::Ptr(new pcl::PolygonMesh());
        //pcl::io::loadOBJFile(file_name, *mesh);
        subsampled_cloud = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        sample_mesh(file_name, 2000000, subsampled_cloud);
        pcl::copyPointCloud(*subsampled_cloud, *subsampled_normals);
    }
    else {
        std::cout << "MAIN:: File format ." << format << " of " << file_name << " not supported. Valid formats: .pcd .txt .obj" << std::endl;
        return(-1);
    }

    //temp show ground cloud from indices
    /*if (argc > 4) {
        bridge_indices = pcl::PointIndices::Ptr(new pcl::PointIndices);
        std::string index_file = argv[4];
        ifstream in_test(index_file);
        if (in_test.is_open())
        {
            std::string line = "";
            while (std::getline(in_test, line)) {
                std::stringstream strstr(line);
                std::string value = "";
                while (std::getline(strstr, value, ';')) {
                    if (!value.empty()){
                        bridge_indices->indices.push_back(std::stoi(value));
                    }
                }
            }
        }
        else {
            std::cout << "LUUUUUUUUUUUUUUUUUUUUUUUUUUL" << std::endl;
        }
    }  */

    //check for camera file and config file
    //create viewer
    viewer = new Viewer("Split overview viewer");
    bool config_file_exist = false;
    if (argc > 4)
    {
        file_name = argv[arg];
        arg++;
        viewer->setup_viewer(file_name, 9);
        config_file_exist = true;
    }
    else if (argc > 3) 
    {
        file_name = argv[arg];
        std::string format = file_name.substr(file_name.find_last_of(".") + 1);
        if (format == "cam") {
            viewer->setup_viewer(file_name, 9);
            if (!evaluation_mode) {
                config_file_exist = true;
                arg++;
            }
        }
        else {
            config_file_exist = true;
            viewer->setup_viewer(9);
        }
    }
    else if (argc > 2 && !evaluation_mode) {
        file_name = argv[arg];
        std::string format = file_name.substr(file_name.find_last_of(".") + 1);
        if (format == "cam") {
            viewer->setup_viewer(file_name, 9);
        }
        else {
            config_file_exist = true;
            viewer->setup_viewer(9);
        }
    }
    else {
        viewer->setup_viewer(9);
    }

    //loads config file if exists
    if(config_file_exist) {
        std::cout << "MAIN:: Read config file."<<std::endl;
        file_name = argv[arg];
        FILE* config_file = fopen(file_name.c_str(), "r");
        if (config_file != NULL) {
            fscanf_s(config_file, "%i", &config_num_neightbours);
            fscanf_s(config_file, "%f", &config_smoothness);
            fscanf_s(config_file, "%f", &config_curvature);
            fscanf_s(config_file, "%i", &config_num_cluster_first);
            fscanf_s(config_file, "%i", &config_num_cluster_second);
            fscanf_s(config_file, "%i", &config_point_neightbourhood);
            fscanf_s(config_file, "%f", &config_std_deviation);
            std::printf("MAIN:: Readed config file: \n%i number of neighbours, \n%f smoothness, \n%f curvature, \n%i number of cluster first, \n%i number of cluster second, \n%i point neighbourhood, \n%f std. deviation",config_num_neightbours, config_smoothness, config_curvature, config_num_cluster_first, config_num_cluster_second, config_point_neightbourhood, config_std_deviation);
        }
        else {
            std::cout << "MAIN:: ERROR: failed to open config file.\n";
        }
    }

    viewer->registerKeyboardCallback(keyboardEventOccurred, (void*)viewer);

    //setup detail viewer
    detail_viewer = new Viewer("Detail viewer");
    detail_viewer->setup_viewer(1);
    detail_viewer->registerKeyboardCallback(keyboardEventOccurred, (void*)detail_viewer);

    /*pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr temp_cloud = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::copyPointCloud(*subsampled_cloud, *bridge_indices, *temp_cloud);
    detail_viewer->visualize_pointcloud(temp_cloud, "temp");
    */
    double complete_start_timestamp = omp_get_wtime();
    double complete_time;

#pragma omp parallel
    {
#pragma omp master
        {
            std::printf("MAIN:: Start visualizer on thread %d of %d \n", omp_get_thread_num(), omp_get_num_threads());
            //visualizer loop
            while (!viewer->wasStopped()&&!close_flag) {
                viewer->spinOnce(100);
                std::this_thread::sleep_for(100ms);
            }
        }

        //use separate visualizer thread
#pragma omp sections nowait// starts a new team
        {

            //main pipeline
#pragma omp section
            {
                std::vector<double> time_measurements;
                double time = 0;
                double start_timestamp = complete_start_timestamp;

                //=============== GENERAL =================
                std::printf("MAIN:: Start Pipeline on thread %d of %d \n", omp_get_thread_num(), omp_get_num_threads());

                //=============== SUBSAMPLING ================
                std::cout << "\nMAIN:: ========= SUBSAMPLE POINTCLOUD =========" << std::endl;
                start_timestamp = omp_get_wtime();
                if (subsampled_cloud == nullptr) {
                    subsampled_cloud = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
                    subsampled_indices = pcl::PointIndices::Ptr(new pcl::PointIndices);
                    *subsampled_indices = subsample_pointcloud(input_cloud, subsampled_cloud);
                }
                time = omp_get_wtime() - start_timestamp;
                printf("MAIN:: Pointcloud subsampling took %f seconds\n", time);
                time_measurements.push_back(time);
                //detail_viewer->visualize_pointcloud(input_cloud, "input_cloud");

                std::cout << "\nMAIN:: ========= ALIGN POINTCLOUD =========" << std::endl;
                //=============== ALIGN POINTCLOUD ===============
                start_timestamp = omp_get_wtime();
                align_pointcloud(subsampled_cloud);

                time = omp_get_wtime() - start_timestamp;
                printf("MAIN:: Pointcloud alignment took %f seconds\n", time);
                time_measurements.push_back(time);

                //vis. input
                std::cout << "PIPELINE:: subsampled_cloud-> "<<subsampled_cloud->size()<<" size."<<std::endl;
                viewer->visualize_pointcloud(subsampled_cloud, "subsampled_cloud");
                //detail_viewer->visualize_pointcloud(aligned_cloud, "aligned_cloud");
                viewer->assign_oriented_bounding_box("subsampled_cloud", 1.0, 0.0, 0.0);

                std::cout << "\nMAIN:: ========= NORMAL ESTIMATION =========" << std::endl;
                //=============== NORMAL ESTIMATION ==============
                start_timestamp = omp_get_wtime();
                std::cout << "MAIN:: Estimate normals." << std::endl;
                estimated_normals = pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud<pcl::Normal>);
                estimate_normals(subsampled_cloud, estimated_normals, true);

                time = omp_get_wtime() - start_timestamp;
                printf("MAIN:: Normal estimation took %f seconds\n", time);
                time_measurements.push_back(time);

                //viewer->remove_pointcloud("subsampled_cloud");

                std::cout << "\nMAIN:: ========= SLAB EXTRACTION =========" << std::endl;
                //================ SLAB EXTRACTION ==============
                std::cout << "MAIN:: Extract slab." << std::endl;
                start_timestamp = omp_get_wtime();
                clustered_cloud = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
                slab_cloud = extract_slab(subsampled_cloud, clustered_cloud);

                time = omp_get_wtime() - start_timestamp;
                printf("MAIN:: Slab extraction took %f seconds\n", time);
                time_measurements.push_back(time);

                //visualize clustered cloud and slab
                viewer->visualize_pointcloud(clustered_cloud, "clustered_cloud", 2);
                viewer->visualize_pointcloud(slab_cloud, "slab_cloud", 1);
                BoundingBox bbox = viewer->assign_oriented_bounding_box("slab_cloud", 1.0, 0.0, 0.0);
                viewer->add_oriented_bounding_box(bbox, 0.0, 0.0, 1.0, viewer->get_viewport(0));

                //temp add eigen vec vis
                /*Eigen::Vector3f p1;
                Eigen::Vector3f p2;

                Eigen::Vector4f centroid;
                pcl::compute3DCentroid(*slab_cloud, centroid);

                //compute the principal direction of the cloud by PCA
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr projected_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
                pcl::PCA<pcl::PointXYZRGBNormal> pca;
                pca.setInputCloud(slab_cloud);
                pca.project(*slab_cloud, *projected_cloud);
                Eigen::Matrix3f eigen_vecs = pca.getEigenVectors();
                p1 = centroid.head<3>() + 10.0 * eigen_vecs.col(0);
                p2 = centroid.head<3>() + 2.0 * eigen_vecs.col(1);

                detail_viewer->addArrow(pcl::PointXYZ(p1[0], p1[1], p1[2]), pcl::PointXYZ(centroid[0], centroid[1], centroid[2]), 1.0, 0.0, 0.0, false, "eigen_arrow_1");
                detail_viewer->addArrow(pcl::PointXYZ(p2[0], p2[1], p2[2]), pcl::PointXYZ(centroid[0], centroid[1], centroid[2]), 0.0, 1.0, 0.0, false, "eigen_arrow_2");
                */

                //=============== REALIGN POINTCLOUD ===============
                realign_by_slab(subsampled_cloud, slab_cloud);

                std::cout << "\nMAIN:: ========= CLIP POINTCLOUD =========" << std::endl;
                //============== CLIP CLOUD ================
                std::cout << "MAIN:: Clip cloud." << std::endl;
                start_timestamp = omp_get_wtime();
                clipped_cloud = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
                clip_bounding_volume(subsampled_cloud, slab_cloud, clipped_cloud);
                //clip_bounding_volume(subsampled_cloud, slab_cloud, clipped_cloud);

                time = omp_get_wtime() - start_timestamp;
                printf("MAIN:: Cloud clipping took %f seconds\n", time);
                time_measurements.push_back(time);

                viewer->visualize_pointcloud(clipped_cloud, "extracted_cloud", 3);
                viewer->assign_bounding_box("extracted_cloud", 1.0f,1.0f,0.0f);

                //============== SUBSAMPLE CLIP VOLUME ===========

                /*while (!subsampled_flag) {
                    #pragma omp flush(subsampled_flag);
                }
                #pragma omp flush(subsampled_cloud)
                */

                std::cout << "\nMAIN:: ========= FILTER SURROUNDINGS =========" << std::endl;

                //============== FILTER SURROUNDINGS =============
                //rough kmeans
                std::cout << "MAIN:: Filter surroundings." << std::endl;
                first_filtered_indices = pcl::PointIndices::Ptr(new pcl::PointIndices);
                pcl::PointIndices ind1;
                start_timestamp = omp_get_wtime();
                first_clustered_extraction_cloud = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
                first_filtered_cloud = colored_kmeans(clipped_cloud, first_clustered_extraction_cloud, config_num_cluster_first, ind1);

                time = omp_get_wtime() - start_timestamp;
                printf("MAIN:: First kmeans took %f seconds\n", time);
                time_measurements.push_back(time);
                
                //extract indices
                for (auto ind = ind1.indices.begin(); ind != ind1.indices.end(); ind++) {
                    first_filtered_indices->indices.push_back(clipped_indices->indices[*ind]);
                }

                viewer->visualize_pointcloud(first_clustered_extraction_cloud, "first_clustered_extraction_cloud", 4);
                viewer->visualize_pointcloud(first_filtered_cloud, "first_filtered_cloud", 5);

                //detailed kmeans
                second_filtered_indices = pcl::PointIndices::Ptr(new pcl::PointIndices);
                pcl::PointIndices ind2;
                start_timestamp = omp_get_wtime();
                second_clustered_extraction_cloud = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
                second_filtered_cloud = colored_kmeans(first_filtered_cloud, second_clustered_extraction_cloud, config_num_cluster_second, ind2);
                
                time = omp_get_wtime() - start_timestamp;
                printf("MAIN:: Second kmeans took %f seconds\n", time);
                time_measurements.push_back(time);

                //extract indices
                for (auto ind = ind2.indices.begin(); ind != ind2.indices.end(); ind++) {
                    second_filtered_indices->indices.push_back(first_filtered_indices->indices[*ind]);
                }
                viewer->visualize_pointcloud(second_filtered_cloud, "second_filtered_cloud", 6);

                //outlier removal
                pcl::PointIndices ind3;
                start_timestamp = omp_get_wtime();
                bridge_cloud = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
                pcl::StatisticalOutlierRemoval<pcl::PointXYZRGBNormal> sor(true);
                sor.setInputCloud(second_filtered_cloud);
                sor.setMeanK(config_point_neightbourhood);
                sor.setStddevMulThresh(config_std_deviation);
                //sor.filter(*bridge_cloud);
                sor.filter(ind3.indices);

                time = omp_get_wtime() - start_timestamp;
                printf("MAIN:: Outlier filtering took %f seconds\n", time);
                time_measurements.push_back(time);

                pcl::copyPointCloud(*second_filtered_cloud, ind3, *bridge_cloud);

                bridge_indices = pcl::PointIndices::Ptr(new pcl::PointIndices);
                //extract indices
                for (auto ind = ind3.indices.begin(); ind != ind3.indices.end(); ind++) {
                    bridge_indices->indices.push_back(second_filtered_indices->indices[*ind]);
                }

                viewer->visualize_pointcloud(bridge_cloud, "bridge_cloud", 7);

                /*pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr temp = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
                pcl::copyPointCloud(*subsampled_cloud, *slab_indices, *temp);
                viewer->visualize_pointcloud(temp,"temp",8);*/

                std::cout << "\nMAIN:: ========= RESAMPLE CLOUD =========" << std::endl;
                
                //================ Resample ================
                /*std::cout << "MAIN:: Resample Cloud." << std::endl;
               
                start_timestamp = omp_get_wtime();
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr resampled_cloud = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
                std::vector<pcl::Vertices> convHull_indices;
                //pcl::PointCloud<pcl::PointXYZ>::Ptr bridge_cloud_xyz = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_on_conhull = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
                int dim = 3;
                
                pcl::ConcaveHull<pcl::PointXYZRGBNormal> conHull;
                conHull.setInputCloud(bridge_cloud);
                conHull.setAlpha(1.0);
                conHull.setDimension(3);
                conHull.reconstruct(*cloud_on_conhull, convHull_indices);
                dim = conHull.getDimension();

                std::cout << "RESAMPLING:: Concave hull created." << std::endl;

                pcl::CropHull<pcl::PointXYZRGBNormal> crop_filter;
                crop_filter.setInputCloud(clipped_cloud);
                crop_filter.setHullCloud(cloud_on_conhull);
                crop_filter.setHullIndices(convHull_indices);
                crop_filter.setDim(dim);

                crop_filter.filter(*resampled_cloud);
                viewer->visualize_pointcloud(resampled_cloud, "resampled_cloud",viewer->get_viewport(8));

                /*pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr tree;
                pcl::MovingLeastSquares<pcl::PointXYZRGBNormal, pcl::PointNormal> mls_upsampling;

                pcl::PointCloud<pcl::PointNormal>::Ptr temp_cloud_res(new pcl::PointCloud<pcl::PointNormal>());

                // Set parameters
                mls_upsampling.setInputCloud(bridge_cloud);
                mls_upsampling.setComputeNormals(false);
                mls_upsampling.setPolynomialOrder(2);
                mls_upsampling.setSearchMethod(tree);
                mls_upsampling.setSearchRadius(50);
                mls_upsampling.setUpsamplingMethod(pcl::MovingLeastSquares<pcl::PointXYZRGBNormal, pcl::PointNormal>::VOXEL_GRID_DILATION);
                mls_upsampling.setDilationIterations(5);
                mls_upsampling.setDilationVoxelSize(0.005f);
                mls_upsampling.setNumberOfThreads(4);
                mls_upsampling.process(*temp_cloud_res);

                time = omp_get_wtime() - start_timestamp;
                printf("MAIN:: Cloud resampling took %f seconds\n", time);

                pcl::copyPointCloud(*temp_cloud_res, *resampled_cloud);
                
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr resampled_cloud = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
                *resampled_cloud += *bridge_cloud;
                *resampled_cloud += *slab_cloud;
                viewer->visualize_pointcloud(resampled_cloud, "resampled_cloud", 8);
                //std::cout << "Number of points resampled: " << temp_cloud_res->points.size() << std::endl;*/
                
                //================ END ===============
                time = omp_get_wtime() - complete_start_timestamp;
                printf("\nMAIN:: Pipline took %f seconds\n", time);
                time_measurements.push_back(time);

                if (evaluation_mode) {
                    write_evaluation_data(time_measurements);          
                }
                std::cout << "DONE." << std::endl;

                //write result persistent
                pcl::io::savePCDFile("output_bridge.pcd", *bridge_cloud, true);

                /*if (evaluation_mode) {
                    close_flag = true;
                    #pragma omp flush(close_flag)
                }*/
            }

            //additional thread for work with full res
            /*#pragma omp section
            {
                while (!clipping_flag) {
                    std::this_thread::sleep_for(100ms);
                }

                pcl::

            }*/
        }
    }
    /*delete viewer;
    viewer = nullptr;
    delete detail_viewer;
    detail_viewer = nullptr;
    return 0;*/
}