#include "viewer.h"
#include <cmath>

Viewer::Viewer(std::string name) : pcl::visualization::PCLVisualizer(name){
}
Viewer::~Viewer() {};

//=====================SETUP==================================
//initializes a basic viewing setup
void Viewer::setup_viewer(int num_viewports) {
    
    //generate viewports
    if (num_viewports > 1) {
        int cols = (int)ceil(sqrt(num_viewports));
        int rows = (int)ceil(num_viewports / cols);

        for (int r = rows - 1; r > -1; r--) {
            for (int c = 0; c < cols; c++) {
                int vp = 0;
                createViewPort(c * (1.0/cols), r * (1.0/rows), (c+1) * (1.0/cols), (r+1) * (1.0/rows), vp);
                addCoordinateSystem(30.0, "coordinate_system", vp);
                viewports_.push_back(vp);
            }
        }
        std::cout << "VIEWER:: Created " << viewports_.size() << " viewports."<<std::endl;
    }
    else {
        viewports_.push_back(0);
    }

    setBackgroundColor(1, 1, 1);
    //addCoordinateSystem(30.0, "coordinate_system", 0);
    initCameraParameters();
}

//uses a camerafile to init camera parameters
void Viewer::setup_viewer(std::string const& camera_filepath, int num_viewport) {
    setup_viewer(num_viewport);
    
    //set camera settings
    set_camera(load_camera_file(camera_filepath));
}

//=====================POINT CLOUD RENDERING===================
bool Viewer::visualize_pointcloud(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr in_cloud, std::string cloud_name, int viewport) {
    int vp = get_viewport(viewport);
    auto cl = pcl_viewport_mapping_.find(cloud_name);
    if (cl != pcl_viewport_mapping_.end()) {
        std::cout << "VIEWER:: Point cloud with name " << cloud_name << "already visualized on viewport: " << std::distance(viewports_.begin(), std::find(viewports_.begin(), viewports_.end(),cl->second)) << std::endl;
        return false;
    }

    auto it = pointclouds_.find(viewport);
    if (it != pointclouds_.end()) { 
         std::cout << "VIEWER:: Replace pointcloud on " << viewport << " with "<< cloud_name << std::endl;
         pointclouds_.erase(it);
         for (auto i = pcl_viewport_mapping_.begin(); i != pcl_viewport_mapping_.end(); i++)
         {
             if (i->second == viewport) {
                 pcl_viewport_mapping_.erase(i);
                 remove_bounding_box("bounding_box_"+i->first);
             }
         }
    }

    pointclouds_.insert({ viewport, {cloud_name,in_cloud} });
    pcl_viewport_mapping_.insert({cloud_name,viewport});

    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal> rgb(in_cloud);
    addPointCloud<pcl::PointXYZRGBNormal>(in_cloud, rgb, cloud_name, get_viewport(viewport));
    return true;
}

bool Viewer::remove_pointcloud(std::string cloud_name) {
    auto vp = pcl_viewport_mapping_.find(cloud_name);
    if (vp != pcl_viewport_mapping_.end()) {
        removePointCloud(cloud_name, get_viewport(vp->second));
        auto cl = pointclouds_.find(vp->second);
        pointclouds_.erase(cl);
        pcl_viewport_mapping_.erase(vp);
        remove_bounding_box("bounding_box_" + cloud_name);
        return true;
    }
    return false;
}

//=====================BOUNDING BOX============================
//functions to add a bounding box to a point cloud
void Viewer::add_bounding_box(BoundingBox const& boundingBox, float r, float g, float b, int viewport) {
    num_bounding_box_++;
    addCube(boundingBox.minPoint.x, boundingBox.maxPoint.x, boundingBox.minPoint.y, boundingBox.maxPoint.y, boundingBox.minPoint.z, boundingBox.maxPoint.z, boundingBox.r, boundingBox.g, boundingBox.b, "bounding_box_"+num_bounding_box_, viewport);
    setRepresentationToWireframeForAllActors();
}

void Viewer::add_oriented_box(BoundingBox const& boundingBox, float r, float g, float b, int viewport) {
    num_bounding_box_++;
    addCube(boundingBox.transform, boundingBox.rotation, boundingBox.dim_x, boundingBox.dim_y, boundingBox.dim_z, "o_bounding_box_" + num_bounding_box_, viewport);
    setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, boundingBox.r, boundingBox.g, boundingBox.b, "o_bounding_box_" + num_bounding_box_);
    setRepresentationToWireframeForAllActors();
}

BoundingBox Viewer::assign_bounding_box(std::string cloud_name, float r, float g, float b){
    auto pcl = pcl_viewport_mapping_.find(cloud_name);
    if (pcl != pcl_viewport_mapping_.end()) {
        num_bounding_box_++;
        BoundingBox boundingBox = create_boundingbox(pointclouds_.find(pcl->second)->second.second, r, g, b);
        addCube(boundingBox.minPoint.x, boundingBox.maxPoint.x, boundingBox.minPoint.y, boundingBox.maxPoint.y, boundingBox.minPoint.z, boundingBox.maxPoint.z, boundingBox.r, boundingBox.g, boundingBox.b, "bounding_box_" + cloud_name);
        return boundingBox;
    }
    else
    {
        std::cout << "VIEWER:: No cloud found with name: "<<cloud_name<<". Maby add the cloud first."<<std::endl;
        return BoundingBox();
    }
    setRepresentationToWireframeForAllActors();
    
}

BoundingBox Viewer::assign_oriented_bounding_box(std::string cloud_name, float r, float g, float b) {
    auto pcl = pcl_viewport_mapping_.find(cloud_name);
    if (pcl != pcl_viewport_mapping_.end()) {
        num_bounding_box_++;
        BoundingBox boundingBox = create_oriented_boundingbox(pointclouds_.find(pcl->second)->second.second, r, g, b);
        addCube(boundingBox.transform, boundingBox.rotation, boundingBox.dim_x, boundingBox.dim_y, boundingBox.dim_z, "bounding_box_"+cloud_name, get_viewport(pcl->second));
        setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, boundingBox.r, boundingBox.g, boundingBox.b, "bounding_box_"+ cloud_name);
        return boundingBox;
    }
    else
    {
        std::cout << "VIEWER:: No cloud found with name: " << cloud_name << ". Maby add the cloud first." << std::endl;
        return BoundingBox();
    }
    setRepresentationToWireframeForAllActors();
}

void Viewer::remove_bounding_box(std::string name) {
    removeShape(name);
}

//===================GET-SET===================================
//loads a .cam file
pcl::visualization::Camera Viewer::load_camera_file(std::string const& filepath) {
    
    pcl::visualization::Camera cam;
    std::string format = filepath.substr(filepath.find_last_of(".") + 1);
    if (format == "cam") {
        std::cout << "VIEWER:: Read .cam file." << std::endl;

        //read .cam file
        std::string in = "";
        std::ifstream camera_filestream(filepath);
        if (camera_filestream.is_open()) {
            getline(camera_filestream, in, ',');
            cam.clip[0] = std::stof(in);
            getline(camera_filestream, in, '/');
            cam.clip[1] = std::stof(in);

            getline(camera_filestream, in, ',');
            cam.focal[0] = std::stof(in);
            getline(camera_filestream, in, ',');
            cam.focal[1] = std::stof(in);
            getline(camera_filestream, in, '/');
            cam.focal[2] = std::stof(in);

            getline(camera_filestream, in, ',');
            cam.pos[0] = std::stof(in);
            getline(camera_filestream, in, ',');
            cam.pos[1] = std::stof(in);
            getline(camera_filestream, in, '/');
            cam.pos[2] = std::stof(in);

            getline(camera_filestream, in, ',');
            cam.view[0] = std::stof(in);
            getline(camera_filestream, in, ',');
            cam.view[1] = std::stof(in);
            getline(camera_filestream, in, '/');
            cam.view[2] = std::stof(in);

            getline(camera_filestream, in, '/');
            cam.fovy = std::stof(in);

            camera_filestream.close();
        }
    }
    else {
        std::cout << "VIEWER:: Invalid Camera-File ending: ." + format << std::endl;
        std::cout << "VIEWER:: Load default camera."<< std::endl;
        getCameraParameters(cam);
        return cam;
    }
    return cam;
}

//get camera
pcl::visualization::Camera Viewer::get_camera() {
    return camera_;
}

//set new camera
void Viewer::set_camera(pcl::visualization::Camera const& new_camera) {
    camera_ = new_camera;
    setCameraClipDistances(new_camera.clip[0], new_camera.clip[1]);
    setCameraPosition(new_camera.pos[0], new_camera.pos[1], new_camera.pos[2], new_camera.view[0], new_camera.view[1], new_camera.view[2]);
    setCameraFieldOfView(new_camera.fovy);
}

//get viewport
int Viewer::get_viewport(int viewport_number) {
    if (viewport_number > viewports_.size()-1) {
        std::cout << "VIEWER:: viewport out of range. Only " << viewports_.size() << " rendered."<<std::endl;
        return -1;
    }
    return viewports_[viewport_number];
}

//gets the viewport a cloud is currently shown
int Viewer::get_viewport(std::string cloud_name) {
    auto vp = pcl_viewport_mapping_.find(cloud_name);
    if (vp == pcl_viewport_mapping_.end()) {
        return -1;
    }
    return get_viewport(vp->second);
}

//get current pointcloud on viewport
std::pair<std::string, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> Viewer::get_current_pointcloud(int viewport_index) {
    if (!pointclouds_.empty()) {
        auto it = pointclouds_.find(viewport_index);
        if (it != pointclouds_.end()) {
            return it->second;
        }
    }
    return { "empty",nullptr };
}