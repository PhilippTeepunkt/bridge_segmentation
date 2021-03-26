#include<omp.h>
#include <direct.h>
#include <algorithm>
#include <numeric>

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
#include <pcl/filters/uniform_sampling.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/ml/kmeans.h>
#include <pcl/search/search.h>

#include "viewer.h"
#include "utils.h"

using namespace std::chrono_literals; 

std::string bridge_name = "";

//explicit barier flags
bool close_flag = false;
bool split_flag = false; 

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
float config_sampling_radius = 0.027;
int config_sampling_density = 2000000;
int config_num_neightbours = 550;
float config_smoothness = 0.82;
float config_curvature = 0.008; 
int config_point_neightbourhood = 450;
float config_std_deviation = 0.7;
int config_num_cluster_first = 3;
int config_num_cluster_second = 4;
//float config_residuals_threshold = 0.36; 

//clipping cloud and indices
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr clipped_cloud;
pcl::PointIndices::Ptr clipped_indices;

//color normalization
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_rgbnormalized(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
Eigen::Vector3f average_deck_color = Eigen::Vector3f(0.0, 0.0, 0.0); //average deck color used for color filtering

//color filtered clouds and indices
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr color_quantized_cloud;
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr color_filtered_cloud;
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr filtered_cloud;

pcl::PointIndices::Ptr color_filtered_indices;
pcl::PointIndices::Ptr outlier_indices;

//splitting
std::vector<BoundingBox> splitted_bounds;
std::vector<bool> labels;

//final output cloud
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr bridge_cloud;
pcl::PointIndices::Ptr bridge_indices;

//resampling
std::vector<pcl::PointIndices::Ptr> part_indices; //clipping parts for each thread to crop
int dim = 3;
std::vector<pcl::Vertices> concave_indices;
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_on_conhull;
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr resampled_cloud;
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr concave_sampled_cloud;
int ready = 0;

//global normals
pcl::PointCloud<pcl::Normal>::Ptr normals;
pcl::PointCloud<pcl::Normal>::Ptr subsampled_normals;
pcl::PointCloud<pcl::Normal>::Ptr estimated_normals;
int show_normals = false; //normal toggle

//time measurement
double timestamp = 0;
double start_timestamp = 0;
double complete_start_timestamp = 0;
std::vector<double> time_measurements;


//subsamples input to 2mio points

/*pcl::PointIndices subsample_pointcloud(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr in_cloud, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr out_cloud, pcl::PointIndices::Ptr removed_indices = nullptr) {

    pcl::RandomSample<pcl::PointXYZRGBNormal> rand_sampling;
    rand_sampling.setInputCloud(in_cloud);
    rand_sampling.setSeed(std::rand());
    rand_sampling.setSample((unsigned int)(config_sampling_density));
    if (removed_indices != nullptr) {
        rand_sampling.getRemovedIndices(*removed_indices);
    }
    pcl::PointIndices indices;
    rand_sampling.filter(indices.indices);
    pcl::copyPointCloud(*in_cloud, indices, *out_cloud);

    return indices;
}*/

//subsamples input by distance
pcl::PointIndices subsample_pointcloud(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr const in_cloud, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr out_cloud) {

    //pcl::RandomSample<pcl::PointXYZRGBNormal> sampling(true);
    pcl::UniformSampling<pcl::PointXYZRGBNormal> sampling(true);
    sampling.setInputCloud(in_cloud);
    //sampling.setSeed(std::rand());
    //sampling.setSample((unsigned int)(config_sampling_density));
    sampling.setRadiusSearch(config_sampling_radius);

    pcl::PointIndices::Ptr removed_indices(new pcl::PointIndices());
    sampling.filter(*out_cloud);
    sampling.getRemovedIndices(*removed_indices);

    pcl::ExtractIndices<pcl::PointXYZRGBNormal> extract;
    extract.setInputCloud(in_cloud);
    extract.setIndices(removed_indices);
    extract.setNegative(true);
    pcl::PointIndices::Ptr indices(new pcl::PointIndices());
    extract.filter(indices->indices);

    std::printf("MAIN:: Subsampled pointcloud to %d points.\n", indices->indices.size());
    return *indices;
}

//aligns pointcloud by its oriented bounding box
std::pair<Eigen::Matrix4f, Eigen::Affine3f> align_pointcloud(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr in_cloud) {
    std::pair<Eigen::Matrix4f, Eigen::Affine3f> transformation;

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
    Eigen::Affine3f rot = Eigen::Affine3f::Identity();
    if (cosin < 0) {
        in_cloud->clear();
        rot.rotate(Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitX()));
        pcl::transformPointCloudWithNormals(*aligned_cloud, *in_cloud, rot);
        std::cout << "ALIGNMENT:: Rotated Cloud!" << std::endl;
    }
    else {
        in_cloud->clear();
        pcl::copyPointCloud(*aligned_cloud, *in_cloud);
    }
    transformation.first = projection;
    transformation.second = rot;
    return transformation;
}

//realign cloud by slab
std::pair<Eigen::Matrix4f, Eigen::Affine3f> realign_by_slab(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr in_cloud, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr slab){
    std::pair<Eigen::Matrix4f, Eigen::Affine3f> transformation;
    
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
    Eigen::Affine3f rot = Eigen::Affine3f::Identity();
    if (cosin < 0) {
        in_cloud->clear();
        slab->clear();
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
    transformation.first = projection;
    transformation.second = rot;
    return transformation;
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
    //reg.setResidualTestFlag(true);
    reg.setCurvatureTestFlag(true);
    //reg.setResidualThreshold(config_residuals_threshold);


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

        if (l > maxDiagLength) {
            if (h > height_condition || l*0.5 > maxDiagLength)
            {
                maxDiagLength = l;
                slab_cluster = ext_cloud;
                *slab_indices = clusters[c];
                height_condition = h;
            }
        }
    }

    /*
    //visualize knearest neighbourhood 
    pcl::PointIndices neighbourhood;
    int query_point = slab_indices->indices[5230];
    std::vector<float> distances;
    tree->nearestKSearch(query_point, 500 , neighbourhood.indices, distances);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr neighbourhood_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::copyPointCloud(*in_cloud, neighbourhood, *neighbourhood_cloud);
    BoundingBox neighbourhood_box = create_boundingbox(neighbourhood_cloud, 1.0, 0.0, 0.0);
    detail_viewer->add_bounding_box(neighbourhood_box);
    detail_viewer->visualize_pointcloud(input_cloud, "input_cloud");
    detail_viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "input_cloud");
    std::cout << "MAIN: KNEAREST Neighbourhood, dimensions: \nDim x: " << neighbourhood_box.dim_x << "\nDim y:" << neighbourhood_box.dim_y << "\nDim z:" << neighbourhood_box.dim_z;
    std::cout << "\nMAIN: KNEAREST Neighbourhood, diagonal: " << pcl::euclideanDistance(neighbourhood_box.minPoint, neighbourhood_box.maxPoint) << std::endl;
    */
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
void color_filtering(pcl::PointCloud <pcl::PointXYZRGBNormal>::Ptr const in_cloud, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr colored_cloud, unsigned int num_cluster, pcl::PointIndices& out_indices, Eigen::Vector3f reference_color, int number_output_clusters = 2) {
//void color_filtering(pcl::PointCloud <pcl::PointXYZRGBNormal>::Ptr const in_cloud, unsigned int num_cluster, pcl::PointIndices &out_indices) {

    pcl::Kmeans means(static_cast<int>(in_cloud->points.size()), 3);
    means.setClusterSize(num_cluster);
    std::vector<std::vector<float>> formated_points;

    //format rgb space for kmeans
    for (size_t i = 0; i < in_cloud->points.size(); i++) {
        std::vector<float> data(3);
        data[0] = float(in_cloud->points[i].r) / 255.0;
        data[1] = float(in_cloud->points[i].g) / 255.0;
        data[2] = float(in_cloud->points[i].b) / 255.0;
        means.addDataPoint(data);
        formated_points.push_back(data);
    }
    //std::cout << "kmeans : Start kmeans " << std::endl;
    means.kMeans();
    pcl::Kmeans::Centroids centroids = means.get_centroids();
    //std::cout << "Kmeans : Centroid count: " << centroids.size() << std::endl;

    //assign each point to cluster
    //std::cout << "Kmeans : Extract kmeans cluster" << std::endl;
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

    //sort cluster by centroid distance to reference color
    std::vector<size_t> centroid_indexes(centroids.size());
    std::iota(centroid_indexes.begin(), centroid_indexes.end(), 0);

    std::sort(std::begin(centroid_indexes), std::end(centroid_indexes), [&centroids, &reference_color, &means](size_t idx1, size_t idx2) {
        std::vector<float> ref{ reference_color[0],reference_color[1],reference_color[2] };
        float distance_idx1 = means.distance(centroids[idx1], ref);
        float distance_idx2 = means.distance(centroids[idx2], ref);
        return distance_idx1 < distance_idx1;
    });

    for (int i = 0; i < number_output_clusters; i++) {
        out_indices.indices.insert(out_indices.indices.end(), cluster_indices[centroid_indexes[i]].indices.begin(), cluster_indices[centroid_indexes[i]].indices.end());
    }

    //color clusters
    for (size_t j = 0; j < centroids.size(); j++) {
        //int size = cluster_indices[j].indices.size();
        //std::cout << "kmeans : Centroid " << j << ": " << size << std::endl;

        //construct colored cloud
        int r = __int16(rand() % 255);
        int g = __int16(rand() % 255);
        int b = __int16(rand() % 255);
        int32_t rgb = (static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
        pcl::PointCloud<pcl::PointXYZRGBNormal> cloud;
        pcl::copyPointCloud(*in_cloud, cluster_indices[j], cloud);
        for (auto& p : cloud.points) p.rgb = rgb;
        *colored_cloud += cloud;
    }
}

//semantic decisioning if deck or not
bool classify_and_correct(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr in_cloud, pcl::PointIndices const& i_upper, pcl::PointIndices const& i_lower, pcl::PointIndices& result, float full_height) {
    
    //get boundings for both parts
    BoundingBox upperBox;
    Eigen::Vector4f min_u, max_u;
    pcl::getMinMax3D(*in_cloud, i_upper, min_u, max_u);
    float dimz_u = std::abs(max_u[2] - min_u[2]);

    BoundingBox lowerBox;
    Eigen::Vector4f min_l, max_l;
    pcl::getMinMax3D(*in_cloud, i_lower, min_l, max_l);
    float dimz_l = std::abs(max_l[2] - min_l[2]);

    //determine if the subbox uses full height
    //upper low -> slab, below noise
    if (dimz_u < full_height-0.1f) {
        result = i_upper;
        return true;
    }
    //upper full -> check low
    else {
        //lower low -> upper slab, below noise
        if(dimz_l < full_height-0.1f) {
            result = i_upper;
            return true;
        }
        //lower full -> pier
        else {
            result.indices.insert(result.indices.end(), i_upper.indices.begin(), i_upper.indices.end());
            result.indices.insert(result.indices.end(), i_lower.indices.begin(), i_lower.indices.end());
            return false;
        }
    }   
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
        out_time_measurements << std::endl << "Realignment: " << time_measurements[4];
        out_time_measurements << std::endl << "Clipping: " << time_measurements[5];
        out_time_measurements << std::endl << "Color filtering: " << time_measurements[6];
        out_time_measurements << std::endl << "Outlier removal: " << time_measurements[7];
        out_time_measurements << std::endl << "Pointcloud splitting: " << time_measurements[8];
        out_time_measurements << std::endl << "Classification and semantic removal: " << time_measurements[9];
        out_time_measurements << std::endl << "Reassembling: " << time_measurements[10];
        out_time_measurements << std::endl << "Whole pipeline: " << time_measurements[11];
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
        std::cout << "Read ply file." << std::endl;
        pcl::PLYReader Reader;
        Reader.read(file_name, *input_cloud);
        pcl::copyPointCloud(*input_cloud, *normals);
    }
    else if (format == "txt") {
        std::cout << "Read txt file." << std::endl;
        if (!read_ASCII(file_name, input_cloud, normals)) {
            return -1;
        }
    }
    else if (format == "obj") {
        //mesh = pcl::PolygonMesh::Ptr(new pcl::PolygonMesh());
        //pcl::io::loadOBJFile(file_name, *mesh);
        std::cout << "Read mesh file." << std::endl;
        subsampled_cloud = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        sample_mesh(file_name, 2000000, subsampled_cloud);
        pcl::copyPointCloud(*subsampled_cloud, *subsampled_normals);
        pcl::copyPointCloud(*subsampled_cloud, *input_cloud);
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
            fscanf_s(config_file, "%i", &config_sampling_density);
            fscanf_s(config_file, "%i", &config_num_neightbours);
            fscanf_s(config_file, "%f", &config_smoothness);
            fscanf_s(config_file, "%f", &config_curvature);
            //fscanf_s(config_file, "%f", &config_residuals_threshold);
            fscanf_s(config_file, "%i", &config_num_cluster_first);
            fscanf_s(config_file, "%i", &config_num_cluster_second);
            fscanf_s(config_file, "%i", &config_point_neightbourhood);
            fscanf_s(config_file, "%f", &config_std_deviation);
            std::printf("MAIN:: Readed config file: \n%i number of neighbours, \n%f smoothness, \n%f curvature_threshold, \n%i number of cluster first, \n%i number of cluster second, \n%i point neighbourhood, \n%f std. deviation", config_num_neightbours, config_smoothness, config_curvature, config_num_cluster_first, config_num_cluster_second, config_point_neightbourhood, config_std_deviation);
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
    complete_start_timestamp = omp_get_wtime();
    const int rest_num_threads = omp_get_num_threads()-1;
    omp_set_nested(true);
    omp_set_dynamic(1);

    #pragma omp parallel num_threads(2)
    {
        //use separate visualizer thread
        #pragma omp master
        {
            std::printf("\nMAIN:: Start visualizer on thread %d of %d \n", omp_get_thread_num(), omp_get_num_threads());
            //visualizer loop
            while (!viewer->wasStopped() && !close_flag) {
                viewer->spinOnce(100);
                std::this_thread::sleep_for(100ms);
            }
        }

        //nested parallelism
        #pragma omp parallel num_threads(omp_get_max_threads()-1) shared(ready, part_indices, splitted_bounds, split_flag, config_num_cluster_first) // starts a new team
        {
            //main pipeline
            #pragma omp single nowait
            {
                timestamp = 0;
                start_timestamp = complete_start_timestamp;

                //=============== GENERAL =================
                std::printf("\nMAIN:: Start Pipeline on thread %d of %d \n", omp_get_thread_num(), omp_get_num_threads());


                //=============== SUBSAMPLING ================
                std::cout << "\nMAIN:: ========= SUBSAMPLE POINTCLOUD =========" << std::endl;
                start_timestamp = omp_get_wtime();
                if (subsampled_cloud == nullptr) {
                    subsampled_cloud = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
                    //subsampled_indices = pcl::PointIndices::Ptr(new pcl::PointIndices);
                    //*subsampled_indices = subsample_pointcloud(input_cloud, subsampled_cloud);
                    //pcl::copyPointCloud(*input_cloud, *subsampled_indices ,*subsampled_cloud);
                    pcl::copyPointCloud(*input_cloud, *subsampled_cloud);
                }
                std::cout << "MAIN:: subsampled_cloud-> " << subsampled_cloud->size() << " size." << std::endl;
                timestamp = omp_get_wtime() - start_timestamp;
                printf("MAIN:: Pointcloud subsampling took %f seconds\n", timestamp);
                time_measurements.push_back(timestamp);
                //detail_viewer->visualize_pointcloud(input_cloud, "input_cloud");

                std::cout << "\nMAIN:: ========= ALIGN POINTCLOUD =========" << std::endl;
                //=============== ALIGN POINTCLOUD ===============
                start_timestamp = omp_get_wtime();
                std::pair<Eigen::Matrix4f, Eigen::Affine3f> transform = align_pointcloud(subsampled_cloud);

                //also align dense cloud
                pcl::PointCloud<pcl::PointXYZRGBNormal> al_cloud;
                pcl::transformPointCloudWithNormals(*input_cloud, al_cloud, transform.first);
                //input_cloud->clear();
                pcl::transformPointCloudWithNormals(al_cloud, *input_cloud, transform.second);

                timestamp = omp_get_wtime() - start_timestamp;
                printf("MAIN:: Pointcloud alignment took %f seconds\n", timestamp);
                time_measurements.push_back(timestamp);

                viewer->visualize_pointcloud(subsampled_cloud, "subsampled_cloud");
                //detail_viewer->visualize_pointcloud(aligned_cloud, "aligned_cloud");
                viewer->assign_oriented_bounding_box("subsampled_cloud", 1.0, 0.0, 0.0);

                std::cout << "\nMAIN:: ========= NORMAL ESTIMATION =========" << std::endl;
                //=============== NORMAL ESTIMATION ==============
                start_timestamp = omp_get_wtime();
                std::cout << "MAIN:: Estimate normals." << std::endl;
                estimated_normals = pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud<pcl::Normal>);
                estimate_normals(subsampled_cloud, estimated_normals, true);

                timestamp = omp_get_wtime() - start_timestamp;
                printf("MAIN:: Normal estimation took %f seconds\n", timestamp);
                time_measurements.push_back(timestamp);

                //viewer->remove_pointcloud("subsampled_cloud");

                std::cout << "\nMAIN:: ========= SLAB EXTRACTION =========" << std::endl;
                //================ SLAB EXTRACTION ==============
                std::cout << "MAIN:: Extract slab." << std::endl;
                start_timestamp = omp_get_wtime();
                clustered_cloud = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
                slab_cloud = extract_slab(subsampled_cloud, clustered_cloud);

                timestamp = omp_get_wtime() - start_timestamp;
                printf("MAIN:: Slab extraction took %f seconds\n", timestamp);
                time_measurements.push_back(timestamp);

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
                std::cout << "\nMAIN:: ========= REALIGN POINTCLOUD BY SLAB =========" << std::endl;
                start_timestamp = omp_get_wtime();

                transform = realign_by_slab(subsampled_cloud, slab_cloud);
                al_cloud.clear();
                pcl::transformPointCloudWithNormals(*input_cloud, al_cloud, transform.first);
                input_cloud->clear();
                pcl::transformPointCloudWithNormals(al_cloud, *input_cloud, transform.second);

                timestamp = omp_get_wtime() - start_timestamp;
                printf("MAIN:: Realignment took %f seconds\n", timestamp);
                time_measurements.push_back(timestamp);

                //============== CLIP CLOUD ================
                std::cout << "\nMAIN:: ========= CLIP POINTCLOUD =========" << std::endl;
                std::cout << "MAIN:: Clip cloud." << std::endl;
                start_timestamp = omp_get_wtime();
                clipped_cloud = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
                clip_bounding_volume(subsampled_cloud, slab_cloud, clipped_cloud);
                //clip_bounding_volume(subsampled_cloud, slab_cloud, clipped_cloud);

                timestamp = omp_get_wtime() - start_timestamp;
                printf("MAIN:: Cloud clipping took %f seconds\n", timestamp);
                time_measurements.push_back(timestamp);

                //=============== COLOR FILTERING ==============
                std::cout << "\nMAIN::  ========= COLOR FILTERING ===========" << std::endl;

                start_timestamp = omp_get_wtime();
                std::cout << "MAIN::  Normalize colors." << std::endl;
                normalize_RGB(clipped_cloud, cloud_rgbnormalized);
                //cloud_rgbnormalized = clipped_cloud;

                std::cout << "MAIN::  Extract reference deck color." << std::endl;
                
                //extract average color from slab neighbourhood
                pcl::PointIndices neighbourhood;
                //int query_point = slab_indices->indices[slab_indices->indices.size()/2];
                std::vector<float> distances;
                pcl::search::Search<pcl::PointXYZRGBNormal>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGBNormal>);
                tree->setInputCloud(cloud_rgbnormalized);
                tree->nearestKSearchT(pcl::PointXYZRGBNormal(0.0,0.0,0.0,0,0,0), 12000, neighbourhood.indices, distances);
                distances.clear();

                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr neighbourhood_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
                pcl::copyPointCloud(*cloud_rgbnormalized, neighbourhood, *neighbourhood_cloud);

                pcl::CentroidPoint<pcl::PointXYZRGBNormal> color_average;
                for (auto i = neighbourhood_cloud->points.begin(); i != neighbourhood_cloud->points.end(); i++) { color_average.add(*i); };
                pcl::PointXYZRGBNormal av;
                color_average.get(av);
                Eigen::Vector3i rgb_int = av.getRGBVector3i();
                average_deck_color = { float(rgb_int[0]) / 255.0f, float(rgb_int[1]) / 255.0f, float(rgb_int[2]) / 255.0f };
                std::cout << "MAIN:: Average deck color extracted: \n" << rgb_int <<std::endl;

                BoundingBox neighbourhood_box = create_boundingbox(neighbourhood_cloud, 1.0, 0.0, 0.0);
                neighbourhood_cloud->clear();
                neighbourhood_cloud = nullptr;

                viewer->visualize_pointcloud(clipped_cloud, "clipped_cloud_splitted", 3);
                viewer->assign_bounding_box("clipped_cloud_splitted", 1.0f, 1.0f, 0.0f);
                viewer->visualize_pointcloud(cloud_rgbnormalized, "cloud_rgbnormalized", 4);
                viewer->add_bounding_box(neighbourhood_box, 0.0, 0.0, 1.0, viewer->get_viewport(4));

                std::cout << "MAIN::  Filter by color quantization." << std::endl;
                
                //use color quantization by kmeans to filter surroundings
                color_quantized_cloud = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
                color_filtered_indices = pcl::PointIndices::Ptr(new pcl::PointIndices);
                color_filtered_cloud = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);

                color_filtering(cloud_rgbnormalized, color_quantized_cloud, config_num_cluster_first, *color_filtered_indices, average_deck_color, 2);
                pcl::copyPointCloud(*clipped_cloud, *color_filtered_indices, *color_filtered_cloud);

                timestamp = omp_get_wtime() - start_timestamp;
                printf("MAIN:: Color filtering took %f seconds\n", timestamp);
                time_measurements.push_back(timestamp);

                viewer->visualize_pointcloud(color_quantized_cloud, "color_quantized_cloud", 5);
                viewer->visualize_pointcloud(color_filtered_cloud, "color_filtered_cloud", 6);

                //=============== OUTLIER FILTERING ==============
                std::cout << "\nMAIN::  ========= OUTLIER FILTERING ===========" << std::endl;
                //filter outliers
                std::cout << "MAIN:: Outlier filtering."<<std::endl;
                start_timestamp = omp_get_wtime();
                pcl::StatisticalOutlierRemoval<pcl::PointXYZRGBNormal> sor(true);
                sor.setInputCloud(color_filtered_cloud);
                outlier_indices = pcl::PointIndices::Ptr(new pcl::PointIndices);
                sor.setMeanK(config_point_neightbourhood);
                sor.setStddevMulThresh(config_std_deviation);
                sor.filter(outlier_indices->indices);

                timestamp = omp_get_wtime() - start_timestamp;
                printf("MAIN:: Outlier filtering took %f seconds\n", timestamp);
                time_measurements.push_back(timestamp);
                
                filtered_cloud = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
                pcl::copyPointCloud(*color_filtered_cloud, *outlier_indices, *filtered_cloud);
                viewer->visualize_pointcloud(filtered_cloud, "filtered_cloud", 7);

                //============== SUBSAMPLE CLIP VOLUME ===========

                /*while (!subsampled_flag) {
                    #pragma omp flush(subsampled_flag);
                }
                #pragma omp flush(subsampled_cloud)
                */

                std::cout << "\nMAIN:: ========= SPLIT CLIPPED VOLUME ========" << std::endl;
                start_timestamp = omp_get_wtime();

                BoundingBox bBox = create_boundingbox(filtered_cloud, 1.0, 0.0, 0.0);
                int splits = 150;
                float splitting_width = bBox.dim_x / splits;
                splitted_bounds.reserve(splits);
                for (int i = 0; i < splits; i++) {
                    pcl::CropBox<pcl::PointXYZRGBNormal> crop_box;
                    float x = bBox.minPoint.x + i * splitting_width;
                    Eigen::Vector3f color = Eigen::Vector3f(0.0, 0.0, 1.0);
                    BoundingBox bb{pcl::PointXYZRGBNormal(x, bBox.minPoint.y, bBox.minPoint.z, 255,255,0,0.0,0.0,0.0), pcl::PointXYZRGBNormal(x + splitting_width, bBox.maxPoint.y, bBox.maxPoint.z, 255, 255, 0, 0.0, 0.0, 0.0), color[0], color[1], color[2] };
                    crop_box.setMin(bb.minPoint.getVector4fMap());
                    crop_box.setMax(bb.maxPoint.getVector4fMap());
                    crop_box.setInputCloud(filtered_cloud);
                    pcl::PointIndices::Ptr pi = pcl::PointIndices::Ptr(new pcl::PointIndices);
                    crop_box.filter(pi->indices);
                    part_indices.push_back(pi);
                    splitted_bounds.push_back(bb);
                }
                split_flag = true;
                #pragma omp flush(split_flag)

                timestamp = omp_get_wtime() - start_timestamp;
                printf("MAIN:: Volume splitting took %f seconds\n", timestamp);
                time_measurements.push_back(timestamp);

                //init label vector
                labels = std::vector<bool>(splits);

                std::cout << "\nMAIN:: ========= CLASSIFICATION AND SEMANTIC NOISE REMOVAL ========" << std::endl;
                start_timestamp = omp_get_wtime();
            }

            while (!split_flag) {
                std::this_thread::sleep_for(100ms);
                #pragma omp flush(split_flag)
            }

            #pragma omp for
            for (int i = 0; i < part_indices.size(); i++) {
                if (part_indices[i]->indices.empty()) {
                    continue;
                }

                //create fitting bounding boxes
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_part = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
                pcl::copyPointCloud(*filtered_cloud, *part_indices[i], *cloud_part);
                BoundingBox box = create_boundingbox(cloud_part, 0.0, 0.0, 1.0);
                float new_max_z = box.maxPoint.z;
                float new_min_z = box.minPoint.z;
                float new_dim_z = box.dim_z;
                box.maxPoint = pcl::PointXYZRGBNormal(splitted_bounds[i].maxPoint);
                box.maxPoint.z = new_max_z;
                box.dim_z = new_dim_z;
                box.minPoint = pcl::PointXYZRGBNormal(splitted_bounds[i].minPoint);
                box.minPoint.z = new_min_z;
                box.dim_z = new_dim_z;

                //check if the box is already cropping a slab part
                if (new_dim_z < splitted_bounds[i].dim_z / 4) {
                    labels[i] = true;
                }
                else {

                    //crop upper and lower part
                    pcl::PointIndices output_ind;
                    pcl::PointIndices upper_ind;
                    pcl::PointIndices lower_ind;
                    Eigen::Vector4f middle_vec = box.minPoint.getVector4fMap();
                    middle_vec[2] = middle_vec[2] + (box.dim_z / 2);
                    pcl::CropBox<pcl::PointXYZRGBNormal> crop_box(true);
                    crop_box.setInputCloud(filtered_cloud);
                    crop_box.setIndices(part_indices[i]);
                    crop_box.setMin(middle_vec);
                    crop_box.setMax(box.maxPoint.getVector4fMap());
                    crop_box.filter(upper_ind.indices);
                    crop_box.getRemovedIndices(lower_ind);

                    //classify part
                    bool classified_slab = classify_and_correct(filtered_cloud, upper_ind, lower_ind, output_ind, box.dim_z/2);
                    *part_indices[i] = output_ind;

                    //color of box corresponds to label
                    if (classified_slab) {
                        labels[i] = true;
                    }
                    else {
                        labels[i] = false;
                    }
                }
                /*if (i == 95) {
                    detail_viewer->visualize_pointcloud(cloud_part, "cloud_part95");
                }
                else {
                    cloud_part->clear();
                    cloud_part = nullptr;
                }*/
            }

            #pragma omp single
            {
                timestamp = omp_get_wtime() - start_timestamp;
                printf("MAIN:: Classification and semantic noise removal took %f seconds\n", timestamp);
                time_measurements.push_back(timestamp);

                //================== REASSEMBLING ===============
                std::cout << "\nMAIN:: ========= REASSEMBLE CLOUD =========";
                start_timestamp = omp_get_wtime();

                bridge_cloud = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
                bridge_indices = pcl::PointIndices::Ptr(new pcl::PointIndices);
                
                for (size_t i = 0; i != part_indices.size(); i++) {
                    pcl::PointCloud<pcl::PointXYZRGBNormal> cl;
                    pcl::copyPointCloud(*filtered_cloud, *part_indices[i], cl);
                    *bridge_cloud += cl;
                    bridge_indices->indices.insert(bridge_indices->indices.end(), *part_indices[i]->indices.begin(), *part_indices[i]->indices.begin());

                    BoundingBox bb = create_boundingbox(cl, 1.0, 0.0, 0.0);
                    if (labels[i]) {
                        viewer->add_bounding_box(bb, 1.0, 0.0, 0.0, viewer->get_viewport(7));
                    }
                    else {
                        viewer->add_bounding_box(bb, 0.0, 1.0, 0.0, viewer->get_viewport(7));
                    }
                }

                timestamp = omp_get_wtime() - start_timestamp;
                printf("\nMAIN:: Reassembling took %f seconds\n", timestamp);
                time_measurements.push_back(timestamp);

                std::sort(bridge_indices->indices.begin(), bridge_indices->indices.end(), [](int &a, int &b) {
                    return a < b;
                });
                viewer->visualize_pointcloud(bridge_cloud, "bridge_cloud", 8);

                //================ END ===============
                timestamp = omp_get_wtime() - complete_start_timestamp;
                printf("\nMAIN:: Pipline took %f seconds\n", timestamp);
                time_measurements.push_back(timestamp);

                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr temp_cloud = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
                pcl::copyPointCloud(*input_cloud, *bridge_indices, *temp_cloud);
                detail_viewer->visualize_pointcloud(temp_cloud, "temp");
                

                if (evaluation_mode) {
                    write_evaluation_data(time_measurements);
                }
                std::cout << "DONE." << std::endl;
            }

            //std::cout << "\nMAIN:: ========= RESAMPLE CLOUD =========" << std::endl;

            //================ Resample ================
            /*
            std::cout << "MAIN:: Resample Cloud." << std::endl;

            start_timestamp = omp_get_wtime();
            //pcl::PointCloud<pcl::PointXYZ>::Ptr bridge_cloud_xyz = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
            cloud_on_conhull = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>());

            pcl::ConcaveHull<pcl::PointXYZRGBNormal> concave_bridge_hull;
            concave_bridge_hull.setInputCloud(bridge_cloud);
            concave_bridge_hull.setAlpha(0.6);
            concave_bridge_hull.setDimension(3);
            concave_bridge_hull.reconstruct(*cloud_on_conhull, concave_indices);

            pcl::PolygonMesh mesh;
            mesh.polygons = concave_indices;
            pcl::PCLPointCloud2::Ptr cloud_blob(new pcl::PCLPointCloud2);
            pcl::toPCLPointCloud2(*cloud_on_conhull, *cloud_blob);
            mesh.cloud = *cloud_blob;

            //sample concave points from hull
            concave_sampled_cloud = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
            sample_mesh(mesh, bridge_cloud->points.size() * 2, concave_sampled_cloud);

            viewer->addPolygonMesh(mesh, "concave_polygon", viewer->get_viewport(8));
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 1.0, 0.0, "concave_polygon", viewer->get_viewport(8));

            dim = concave_bridge_hull.getDimension();

            std::cout << "RESAMPLING:: Concave hull created." << std::endl;

            //
            crop_flag = true;

            //write result persistent
            pcl::io::savePCDFile("output_bridge.pcd", *bridge_cloud, true);

            /*if (evaluation_mode) {
                close_flag = true;
                #pragma omp flush(close_flag)
            }*/

            /*#pragma omp section
                        {
                            while (!clipped_flag) {
                                std::this_thread::sleep_for(100ms);
                            }

                            std::printf("MAIN:: Thread %d prepares cropping clouds.\n", omp_get_thread_num());

                            BoundingBox bBox = create_boundingbox(clipped_cloud, 1.0, 0.0, 0.0);
                            int nt = omp_get_num_threads() - 1;
                            float width_part = bBox.dim_x / nt;
                            std::vector<Eigen::Vector3f> colors;
                            colors.push_back(Eigen::Vector3f(1.0, 0.0, 0.0));
                            colors.push_back(Eigen::Vector3f(0.0, 1.0, 0.0));
                            colors.push_back(Eigen::Vector3f(0.0, 0.0, 1.0));
                            colors.push_back(Eigen::Vector3f(1.0, 0.0, 1.0));
                            for (int i = 0; i < nt; i++) {
                                pcl::CropBox<pcl::PointXYZRGBNormal> crop_box;
                                float x = bBox.minPoint.x + i * width_part;
                                Eigen::Vector3f color = colors[i];
                                BoundingBox bb{ pcl::PointXYZRGBNormal(x, bBox.minPoint.y, bBox.minPoint.z, 255,255,0,0.0,0.0,0.0), pcl::PointXYZRGBNormal(x + width_part, bBox.maxPoint.y, bBox.maxPoint.z, 255, 255, 0, 0.0, 0.0, 0.0), color[0], color[1], color[2] };
                                viewer->add_bounding_box(bb, bb.r, bb.g, bb.b, viewer->get_viewport(7));

                                crop_box.setMin(bb.minPoint.getVector4fMap());
                                crop_box.setMax(bb.maxPoint.getVector4fMap());
                                crop_box.setInputCloud(clipped_cloud);
                                pcl::PointIndices::Ptr pi = pcl::PointIndices::Ptr(new pcl::PointIndices);
                                crop_box.filter(pi->indices);
                                part_indices.push_back(pi);
                            }
                            prepared_flag = true;
                            std::printf("DEBUG:: NUM parts: %d.\n", part_indices.size());
                        }
                    }

                    while (!crop_flag || !prepared_flag) {
                        std::this_thread::sleep_for(500ms);
                    }*/

                    /*int data_index = omp_get_thread_num() - 1;
                    std::printf("MAIN:: THREAD %d starts cropping.\n", omp_get_thread_num());

                    //Generate octree for clipped volume
                    float resolution = 0.05;
                    pcl::octree::OctreePointCloudSearch<pcl::PointXYZRGBNormal> search_octree(resolution);
                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr part_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
                    pcl::copyPointCloud(*clipped_cloud, *part_indices[data_index], *part_cloud);
                    BoundingBox bb = create_boundingbox(part_cloud, 1.0, 0.0,0.0);
                    search_octree.setInputCloud(part_cloud);
                    search_octree.addPointsFromInputCloud();
                    search_octree.defineBoundingBox(bb.minPoint.x, bb.minPoint.y, bb.minPoint.z, bb.maxPoint.x, bb.maxPoint.y, bb.maxPoint.z);

                    //add point neightbours for concave sampled points
                    std::set<int> part;
                    std::vector<int> contained_indices;
                    for (auto p : concave_sampled_cloud->points) {
                        if (search_octree.voxelSearch(p, contained_indices)) {
                            for (size_t i = 0; i < contained_indices.size(); ++i) {
                               part.insert(contained_indices[i]);
                            }
                        }
                    }
                    pcl::PointIndices ind;
                    std::copy(part.begin(), part.end(), std::back_inserter(ind.indices));
                    *part_indices[data_index] = ind;


                    //crop part of the cloud
                    /*pcl::CropHull<pcl::PointXYZRGBNormal> crop_filter;
                    crop_filter.setInputCloud(clipped_cloud);
                    crop_filter.setIndices(part_indices[data_index]);
                    std::printf("DEBUG:: NUM INDICES: %d.\n", part_indices[data_index]->indices.size());
                    crop_filter.setHullCloud(cloud_on_conhull);
                    crop_filter.setHullIndices(concave_indices);
                    crop_filter.setDim(dim);

                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr part = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
                    crop_filter.filter(*part);
                    std::printf("DEBUG:: NUM CLOUD PART: %d.\n", part->points.size());
                    resampled_parts.push_back(part);
                    */
                    /*ready++;
                    std::printf("MAIN:: THREAD %d has successfully croped the part. Ready counter: %d of %d.\n", omp_get_thread_num(), ready, (int)omp_get_num_threads() - 1);

                    #pragma omp single
                    {
                        while(ready < (int)omp_get_num_threads() - 1) {
                            std::this_thread::sleep_for(100ms);
                        }

                        std::printf("MAIN:: THREAD %d assembles the final cloud.\n", omp_get_thread_num());
                        resampled_cloud = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
                        for (auto idx = part_indices.begin(); idx != part_indices.end(); idx++) {
                            pcl::PointCloud<pcl::PointXYZRGBNormal> cloud;
                            pcl::copyPointCloud(*clipped_cloud, *(*idx), cloud);
                            *resampled_cloud += cloud;
                        }
                        std::printf("MAIN:: Assembled cloud has %d points.\n", resampled_cloud->points.size());
                        //detail_viewer->visualize_pointcloud(resampled_cloud, "resampled_cloud");

                        //================ END ===============
                        timestamp = omp_get_wtime() - complete_start_timestamp;
                        printf("\nMAIN:: Pipline took %f seconds\n", timestamp);
                        time_measurements.push_back(timestamp);

                        if (evaluation_mode) {
                            write_evaluation_data(time_measurements);*/

                            //std::cout << "DONE." << std::endl;

        }
    }

    /*delete viewer;
    viewer = nullptr;
    delete detail_viewer;
    detail_viewer = nullptr;
    return 0;*/
}