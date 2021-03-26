#ifndef UTILS_H
#define UTILS_H

#include <pcl/point_types.h>
#include <pcl/common/common.h>

#include <pcl/common/centroid.h>
#include <pcl/common/pca.h>

#include <pcl/io/vtk_lib_io.h>
#include <vtkVersion.h>
#include <vtkOBJReader.h>
#include <vtkTriangle.h>
#include <vtkTriangleFilter.h>
#include <vtkPolyDataMapper.h>
#include <pcl/surface/vtk_smoothing/vtk_utils.h>
#include <pcl/features/principal_curvatures.h>

#include <pcl/features/normal_3d_omp.h>

#include <sys/types.h>
#include <sys/stat.h>

struct BoundingBox
{
	pcl::PointXYZRGBNormal minPoint;
	pcl::PointXYZRGBNormal maxPoint;
	float r = 0;
	float g = 0;
	float b = 0;

	float dim_x = maxPoint.x - minPoint.x;
	float dim_y = maxPoint.y - minPoint.y;
	float dim_z = maxPoint.z - minPoint.z;

	const Eigen::Vector3f getMeanDiagonal() {
		return 0.5f * (maxPoint.getVector3fMap() + minPoint.getVector3fMap());
	};

	Eigen::Quaternionf rotation;
	Eigen::Vector3f transform;
};

static Eigen::Matrix4f IDENTITY = Eigen::Matrix4f::Identity();

BoundingBox create_boundingbox(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr in_cloud, float r = 0.0, float g = 0.0, float b = 0.0);
BoundingBox create_boundingbox(pcl::PointCloud<pcl::PointXYZRGBNormal> in_cloud, float r = 0.0, float g = 0.0, float b = 0.0);
BoundingBox create_oriented_boundingbox(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr in_cloud, float r, float g, float b, Eigen::Matrix4f &projection = IDENTITY);

void uniform_sampling(vtkSmartPointer<vtkPolyData> polydata, size_t n_samples, bool calc_normal, bool calc_color, pcl::PointCloud<pcl::PointXYZRGBNormal>& cloud_out);

void sample_mesh(std::string file_name, int number_samples, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr out_cloud);
void sample_mesh(pcl::PolygonMesh mesh, int number_samples, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr out_cloud);

bool read_ASCII(std::string filename, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr c, pcl::PointCloud <pcl::Normal>::Ptr n = nullptr);

void estimate_normals(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr in_cloud, pcl::PointCloud<pcl::Normal>::Ptr normals, bool copy_to_cloud, float v_x = 0.0f, float v_y = 0.0f, float v_z = 100000.0f);
void normalize_RGB(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr const cloud, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr normalized_cloud);

bool IsPathExist(const std::string& s);
#endif