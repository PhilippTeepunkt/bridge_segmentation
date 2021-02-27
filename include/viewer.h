#ifndef VIEWER_H
#define VIEWER_H

#include <string>
#include <vector>
#include <map>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/common/common.h>

#include "utils.h"

class Viewer : public pcl::visualization::PCLVisualizer
{
public:
	Viewer(std::string name);
	~Viewer();

	void setup_viewer(int num_viewports = 1);
	void setup_viewer(std::string const& camera_filepath, int num_viewports = 1);

	bool visualize_pointcloud(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr in_cloud, std::string cloud_name, int viewport = 0);
	bool remove_pointcloud(std::string cloud_name);

	void add_bounding_box(BoundingBox const& boundingBox, float r = 1.0, float g = 1.0, float b = 1.0, int viewport = 0);
	void add_oriented_box(BoundingBox const& boundingBox, float r = 1.0, float g = 1.0, float b = 1.0, int viewport = 0);
	BoundingBox assign_bounding_box(std::string cloud_name, float r = 1.0, float g = 1.0, float b = 1.0);
	BoundingBox assign_oriented_bounding_box(std::string cloud_name, float r = 1.0, float g = 1.0, float b = 1.0);
	void remove_bounding_box(std::string name);

	pcl::visualization::Camera get_camera();
	int get_viewport(int viewport_number);
	int Viewer::get_viewport(std::string cloud_name);
	std::pair<std::string, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr>get_current_pointcloud(int viewport_index = 0);
	void set_camera(pcl::visualization::Camera const& new_camera);

private:
	pcl::visualization::Camera load_camera_file(std::string const& filepath);
	pcl::visualization::Camera camera_;
	std::vector<int> viewports_;
	size_t num_bounding_box_ = 0;
	
	std::map<int, std::pair<std::string, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr>>pointclouds_;
	std::map<std::string, int> pcl_viewport_mapping_;
};

#endif
