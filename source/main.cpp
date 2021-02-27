#include<omp.h>

#include <pcl/io/obj_io.h>
#include<pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/crop_hull.h>
#include <pcl/ml/kmeans.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include "viewer.h"
#include "utils.h"

using namespace std::chrono_literals; 


//global visualizer
Viewer* viewer;
Viewer* detail_viewer;

//input mesh
pcl::PolygonMesh::Ptr mesh;

//global clouds
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr input_cloud;
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr aligned_cloud;
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr clustered_cloud;
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr slab_cloud;
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr extracted_cloud;

pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr first_clustered_extraction_cloud;
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr second_clustered_extraction_cloud;

pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr bridge_cloud;

//global normals
pcl::PointCloud<pcl::Normal>::Ptr normals;
pcl::PointCloud<pcl::Normal>::Ptr estimated_normals;
int show_normals = false;//normal toggle

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
    reg.setNumberOfNeighbours(500);
    reg.setInputCloud(cloud);
    reg.setInputNormals(estimated_normals);
    reg.setSmoothnessThreshold(0.65 / 180.0 * M_PI);
    reg.setCurvatureThreshold(1.7);

    std::vector <pcl::PointIndices> clusters;
    reg.extract(clusters);

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr slab_cluster;
    float maxDiagLength = 0;
    float height_condition = 0;
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
            height_condition = h;
        }
    }
    //std::cout << "\n" << longest_cluster->size() << " -> longest\n" << input_cloud->size() << " -> input\n";
    pcl::copyPointCloud(*reg.getColoredCloud(), *colored_cloud);

    return slab_cluster;
}

//creates polygonal prism for bridge bounding volume 
void clip_bounding_volume(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr input_cloud, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr slab_cloud, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr out_cloud) {
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_on_hull(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr slab_cloud_spatial(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud_spatial(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::PointXYZRGBNormal minPoint_cloud, maxPoint_cloud, minPoint_extract, maxPoint_extract;
    pcl::getMinMax3D(*input_cloud, minPoint_cloud, maxPoint_cloud);
    pcl::getMinMax3D(*slab_cloud, minPoint_extract, maxPoint_extract);
    
    //create prism
    pcl::copyPointCloud(*slab_cloud, *slab_cloud_spatial);
    pcl::copyPointCloud(*input_cloud, *input_cloud_spatial);
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
        prism.setHeightLimits(-maxPoint_cloud.z, maxPoint_extract.z - minPoint_cloud.z);
        pcl::PointIndices::Ptr extracted_cloud_indices(new pcl::PointIndices);

        prism.segment(*extracted_cloud_indices);

        pcl::ExtractIndices<pcl::PointXYZRGBNormal> extract;
        extract.setInputCloud(input_cloud);
        extract.setIndices(extracted_cloud_indices);
        extract.filter(*out_cloud);

    }
    else {
        std::cout << "Hull dimensions not 2D / Hull not planar." << std::endl;
    }
}

//filter by color with kmeans
pcl::PointCloud <pcl::PointXYZRGBNormal>::Ptr colored_kmeans(pcl::PointCloud <pcl::PointXYZRGBNormal>::Ptr const in_cloud, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr colored_cloud, unsigned int num_cluster) {

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_rgbnormalized(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    normalize_RGB(in_cloud, cloud_rgbnormalized);

    detail_viewer->visualize_pointcloud(cloud_rgbnormalized, "normalized_pointcloud");

    //cloud_rgbnormalized = in_cloud;

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
    std::vector<std::vector<int>> cluster_indices;
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
            cluster_indices.push_back(std::vector<int>());
        }
        cluster_indices[closest_centroid].push_back(i);
    }
    std::cout << "kmeans : cluster sizes: " << std::endl;

    int biggest_cluster = 0;
    int biggest_cluster_size = 0;
    int size = 0;

    //extract cluster
    for (size_t j = 0; j < centroids.size(); j++) {
        size = cluster_indices[j].size();
        std::cout << "kmeans : Centroid " << j << ": " << size << std::endl;

        //condition which cluster is determined as pile cluster
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
                detail_viewer->addPointCloudNormals<pcl::PointXYZRGBNormal, pcl::Normal>(current_cloud, normals, 10, 0.5, "detail_normals", 0);
                detail_viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 1.0, 0.0, "detail_normals", 0);
            }
            else { return; }
        }
        else if (show_normals == 1)
        {
            detail_viewer->removePointCloud("detail_normals");
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

//===================== MAIN ========================
//main
int main(int argc, char** argv) {

    //parse arguments
    if (argc < 2) {
        std::cout << "Usage: startPipeline <pcdfile> / <ASCII .txt> / <Mesh .obj> [CAMERA SETTINGS]";
        return 0;
    }

    //read input pointcloud 
    input_cloud = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    normals = pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud<pcl::Normal>);
    std::string file_name = argv[1];
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
        sample_mesh(file_name, 200000, input_cloud);
        pcl::copyPointCloud(*input_cloud, *normals);
    }
    else {
        std::cout << "MAIN:: File format ." << format << " of " << file_name << " not supported. Valid formats: .pcd .txt .obj" << std::endl;
        return(-1);
    }

    //check for camera file and create viewer
    viewer = new Viewer("Split overview viewer");
    if (argc > 2) {
        file_name = argv[2];
        viewer->setup_viewer(file_name, 9);
    }
    else {
        viewer->setup_viewer(9);
    }
    viewer->registerKeyboardCallback(keyboardEventOccurred, (void*)viewer);

    //setup detail viewer
    detail_viewer = new Viewer("Detail viewer");
    detail_viewer->setup_viewer(1);
    detail_viewer->registerKeyboardCallback(keyboardEventOccurred, (void*)detail_viewer);

    double complete_start_timestamp = omp_get_wtime();
    double complete_time;

#pragma omp parallel 
    {
#pragma omp master
        {
            std::printf("MAIN:: Start visualizer on thread %d of %d \n", omp_get_thread_num(), omp_get_num_threads());
            //visualizer loop
            while (!viewer->wasStopped()) {
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
                //=============== GENERAL =================
                std::printf("MAIN:: Start Pipeline on thread %d of %d \n", omp_get_thread_num(), omp_get_num_threads());
                double start_timestamp = omp_get_wtime();

                std::cout << "\nMAIN:: ========= ALIGN POINTCLOUD =========" << std::endl;

                //=============== ALIGN POINTCLOUD ===============
                //use the axis aligned bounding box to transform to origin
                aligned_cloud = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
                Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();
                BoundingBox aligned_bbox = create_oriented_boundingbox(input_cloud, 1.0, 1.0, 0.0, projection);
                pcl::transformPointCloudWithNormals(*input_cloud, *aligned_cloud, projection);

                //rotate by align average normal along +z
                pcl::VectorAverage3f avNormal;
                for (auto i = aligned_cloud->points.begin(); i != aligned_cloud->points.end(); i++) {
                    avNormal.add(i->getNormalVector3fMap());
                }
                Eigen::Vector3f averageNormal = avNormal.getMean(); //calculate average normal
                float cosin = (averageNormal.dot(Eigen::Vector3f::UnitZ())) / averageNormal.norm(); //calculate cosin to unit z
                
                //rotate 180° if average normal is negaitve
                if (cosin < 0) {
                    input_cloud->clear();
                    Eigen::Affine3f rot = Eigen::Affine3f::Identity();
                    rot.rotate(Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitX()));
                    pcl::transformPointCloudWithNormals(*aligned_cloud, *input_cloud, rot);
                }

                printf("MAIN:: Pointcloud alignment took %f seconds\n", omp_get_wtime() - start_timestamp);

                //vis. input
                viewer->visualize_pointcloud(input_cloud, "input_cloud");
                //detail_viewer->visualize_pointcloud(aligned_cloud, "aligned_cloud");
                viewer->assign_oriented_bounding_box("input_cloud", 1.0, 0.0, 0.0);

                std::cout << "\nMAIN:: ========= NORMAL ESTIMATION =========" << std::endl;

                //=============== NORMAL ESTIMATION ==============
                start_timestamp = omp_get_wtime();
                std::cout << "MAIN:: Estimate normals." << std::endl;
                estimated_normals = pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud<pcl::Normal>);
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr c = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
                pcl::copyPointCloud(*input_cloud, *c);
                estimate_normals(c, estimated_normals);
                printf("MAIN:: Normal estimation took %f seconds\n", omp_get_wtime() - start_timestamp);

                //viewer->remove_pointcloud("input_cloud");

                std::cout << "\nMAIN:: ========= SLAB EXTRACTION =========" << std::endl;

                //================ SLAB EXTRACTION ==============
                std::cout << "MAIN:: Extract slab." << std::endl;
                start_timestamp = omp_get_wtime();
                clustered_cloud = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
                slab_cloud = extract_slab(input_cloud, clustered_cloud);
                printf("MAIN:: Slab extraction took %f seconds\n", omp_get_wtime() - start_timestamp);

                //visualize clustered cloud and slab
                viewer->visualize_pointcloud(clustered_cloud, "clustered_cloud", 2);
                viewer->visualize_pointcloud(slab_cloud, "slab_cloud", 1);
                BoundingBox bbox = viewer->assign_oriented_bounding_box("slab_cloud", 1.0, 0.0, 0.0);
                viewer->add_oriented_box(bbox, 0.0, 0.0, 1.0, viewer->get_viewport(0));


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
                /*projection = Eigen::Matrix4f::Identity();
                aligned_bbox = create_oriented_boundingbox(slab_cloud, 1.0,1.0,0.0, projection);
                aligned_cloud->clear();
                pcl::transformPointCloudWithNormals(*input_cloud, *aligned_cloud, projection);*/

                //viewer->remove_pointcloud("input_cloud");
                //viewer->visualize_pointcloud(input_cloud, "input_cloud");
                //viewer->assign_oriented_bounding_box("input_cloud", 1.0, 0.0, 0.0);

                std::cout << "\nMAIN:: ========= CLIP POINTCLOUD =========" << std::endl;

                //============== CLIP CLOUD ================
                std::cout << "MAIN:: Clip cloud." << std::endl;
                start_timestamp = omp_get_wtime();
                extracted_cloud = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
                clip_bounding_volume(input_cloud, slab_cloud, extracted_cloud);
                printf("MAIN:: Cloud clipping took %f seconds\n", omp_get_wtime() - start_timestamp);
                viewer->visualize_pointcloud(extracted_cloud, "extracted_cloud", 3);

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
                start_timestamp = omp_get_wtime();
                first_clustered_extraction_cloud = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr first_filtered_cloud = colored_kmeans(extracted_cloud, first_clustered_extraction_cloud, 4);
                printf("MAIN:: First kmeans took %f seconds\n", omp_get_wtime() - start_timestamp);

                viewer->visualize_pointcloud(first_clustered_extraction_cloud, "first_clustered_extraction_cloud", 4);
                viewer->visualize_pointcloud(first_filtered_cloud, "first_filtered_cloud", 5);

                //detailed kmeans
                start_timestamp = omp_get_wtime();
                second_clustered_extraction_cloud = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr second_filtered_cloud = colored_kmeans(first_filtered_cloud, second_clustered_extraction_cloud, 3);
                printf("MAIN:: Second kmeans took %f seconds\n", omp_get_wtime() - start_timestamp);

                viewer->visualize_pointcloud(second_filtered_cloud, "second_filtered_cloud", 6);

                //outlier removal
                start_timestamp = omp_get_wtime();
                bridge_cloud = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
                pcl::StatisticalOutlierRemoval<pcl::PointXYZRGBNormal> sor;
                sor.setInputCloud(second_filtered_cloud);
                sor.setMeanK(200);
                sor.setStddevMulThresh(0.8);
                sor.filter(*bridge_cloud);
                printf("MAIN:: Outlier filtering took %f seconds\n", omp_get_wtime() - start_timestamp);

                viewer->visualize_pointcloud(bridge_cloud, "bridge_cloud", 7);

                //================ Resample ================
                /*std::cout << "MAIN:: Resample Cloud." << std::endl;
                start_timestamp = omp_get_wtime();
                std::vector<pcl::Vertices> convHull_indices;
                //pcl::PointCloud<pcl::PointXYZ>::Ptr bridge_cloud_xyz = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_on_hull = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr resampled_cloud = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
                int dim = 3;
                
                pcl::ConcaveHull<pcl::PointXYZRGBNormal> conHull;
                conHull.setInputCloud(bridge_cloud);
                conHull.setAlpha(2.0);
                conHull.setDimension(3);
                conHull.reconstruct(*cloud_on_hull, convHull_indices);
                dim = conHull.getDimension();

                pcl::CropHull<pcl::PointXYZRGBNormal> crop_filter;
                crop_filter.setInputCloud(extracted_cloud);
                crop_filter.setHullCloud(cloud_on_hull);
                crop_filter.setHullIndices(convHull_indices);
                crop_filter.setDim(dim);

                crop_filter.filter(*resampled_cloud);

                printf("MAIN:: Cloud resampling took %f seconds\n", omp_get_wtime() - start_timestamp);

                viewer->visualize_pointcloud(resampled_cloud, "resampled_cloud", 8);*/

                //================ END ===============
                printf("\nMAIN:: Pipline took %f seconds\n", omp_get_wtime() - complete_start_timestamp);
            }

            //additional thread for subsampling
            /*#pragma omp section
            {
                subsample_data(*input_cloud, subsampled_input, 0.1);
            }*/

        }
    }
}