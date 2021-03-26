#include "utils.h"

//================ BOUNDING BOX ==================
//creates boundingbox for a given pointcloud
BoundingBox create_boundingbox(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr in_cloud, float r, float g, float b) {
	BoundingBox boundingBox;
	pcl::getMinMax3D(*in_cloud, boundingBox.minPoint, boundingBox.maxPoint);
	boundingBox.r = r;
	boundingBox.g = g;
	boundingBox.b = b;
    boundingBox.dim_x = std::abs(boundingBox.maxPoint.x - boundingBox.minPoint.x);
    boundingBox.dim_y = std::abs(boundingBox.maxPoint.y - boundingBox.minPoint.y);
    boundingBox.dim_z = std::abs(boundingBox.maxPoint.z - boundingBox.minPoint.z);

	return boundingBox;
}

//creates boundingbox for a given pointcloud
BoundingBox create_boundingbox(pcl::PointCloud<pcl::PointXYZRGBNormal> in_cloud, float r, float g, float b) {
    BoundingBox boundingBox;
    pcl::getMinMax3D(in_cloud, boundingBox.minPoint, boundingBox.maxPoint);
    boundingBox.r = r;
    boundingBox.g = g;
    boundingBox.b = b;
    boundingBox.dim_x = std::abs(boundingBox.maxPoint.x - boundingBox.minPoint.x);
    boundingBox.dim_y = std::abs(boundingBox.maxPoint.y - boundingBox.minPoint.y);
    boundingBox.dim_z = std::abs(boundingBox.maxPoint.z - boundingBox.minPoint.z);

    return boundingBox;
}

//creates an oriented bounding box
BoundingBox create_oriented_boundingbox(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr in_cloud, float r, float g, float b, Eigen::Matrix4f &projection) {

    //assign color
    BoundingBox boundingBox;
    boundingBox.r = r;
    boundingBox.g = g;
    boundingBox.b = b;

    //compute centroid
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*in_cloud, centroid);

    //compute the principal direction of the cloud by PCA
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr projected_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::PCA<pcl::PointXYZRGBNormal> pca;
    pca.setInputCloud(in_cloud);
    pca.project(*in_cloud, *projected_cloud);
    Eigen::Matrix3f eigen_vecs = pca.getEigenVectors();

    // project cloud to the eigen vectors / build projection matrix
    projection = Eigen::Matrix4f::Identity();
    projection.block<3, 3>(0, 0) = eigen_vecs.transpose();
    projection.block<3, 1>(0, 3) = -1.0f * (projection.block<3, 3>(0, 0) * centroid.head<3>());
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_points_projected(new pcl::PointCloud<pcl::PointXYZRGBNormal>);

    //transform to origin
    pcl::transformPointCloudWithNormals(*in_cloud, *cloud_points_projected, projection);

    // Get the minimum and maximum points of the transformed cloud.
    pcl::getMinMax3D(*cloud_points_projected, boundingBox.minPoint, boundingBox.maxPoint);

    boundingBox.dim_x = boundingBox.maxPoint.x - boundingBox.minPoint.x;
    boundingBox.dim_y = boundingBox.maxPoint.y - boundingBox.minPoint.y;
    boundingBox.dim_z = boundingBox.maxPoint.z - boundingBox.minPoint.z;

    //transform to cloud position
    boundingBox.rotation = eigen_vecs;
    boundingBox.transform = eigen_vecs * boundingBox.getMeanDiagonal() + centroid.head<3>();  
    
    return boundingBox;
}

//================ INPUT =================
//uniform sampling based on triangle area
inline double uniform_deviate(int seed) {
    double ran = seed * (1.0 / (RAND_MAX + 1.0));
    return ran;
}
inline void randomPointTriangle(float a1, float a2, float a3, float b1, float b2, float b3, float c1, float c2, float c3,float r1, float r2, Eigen::Vector3f& p)
{
    float r1sqr = std::sqrt(r1);
    float OneMinR1Sqr = (1 - r1sqr);
    float OneMinR2 = (1 - r2);
    a1 *= OneMinR1Sqr;
    a2 *= OneMinR1Sqr;
    a3 *= OneMinR1Sqr;
    b1 *= OneMinR2;
    b2 *= OneMinR2;
    b3 *= OneMinR2;
    c1 = r1sqr * (r2 * c1 + b1) + a1;
    c2 = r1sqr * (r2 * c2 + b2) + a2;
    c3 = r1sqr * (r2 * c3 + b3) + a3;
    p[0] = c1;
    p[1] = c2;
    p[2] = c3;
}
inline void randPSurface(vtkPolyData* polydata, std::vector<double>* cumulativeAreas, double totalArea, Eigen::Vector3f& p, bool calcNormal, Eigen::Vector3f& n, bool calcColor, Eigen::Vector3f& c)
{
    float r = static_cast<float> (uniform_deviate(rand()) * totalArea);

    std::vector<double>::iterator low = std::lower_bound(cumulativeAreas->begin(), cumulativeAreas->end(), r);
    vtkIdType el = vtkIdType(low - cumulativeAreas->begin());

    double A[3], B[3], C[3];
    vtkIdType npts = 0;
    vtkIdType *ptIds = NULL;

    polydata->GetCellPoints(el, npts, ptIds);
    polydata->GetPoint(ptIds[0], A);
    polydata->GetPoint(ptIds[1], B);
    polydata->GetPoint(ptIds[2], C);
    if (calcNormal)
    {
        // OBJ: Vertices are stored in a counter-clockwise order by default
        Eigen::Vector3f v1 = Eigen::Vector3f(A[0], A[1], A[2]) - Eigen::Vector3f(C[0], C[1], C[2]);
        Eigen::Vector3f v2 = Eigen::Vector3f(B[0], B[1], B[2]) - Eigen::Vector3f(C[0], C[1], C[2]);
        n = v1.cross(v2);
        n.normalize();
    }
    float r1 = static_cast<float> (uniform_deviate(rand()));
    float r2 = static_cast<float> (uniform_deviate(rand()));
    randomPointTriangle(float(A[0]), float(A[1]), float(A[2]),
        float(B[0]), float(B[1]), float(B[2]),
        float(C[0]), float(C[1]), float(C[2]), r1, r2, p);

    if (calcColor)
    {
        vtkUnsignedCharArray* const colors = vtkUnsignedCharArray::SafeDownCast(polydata->GetPointData()->GetScalars());
        if (colors && colors->GetNumberOfComponents() == 3)
        {
            double cA[3], cB[3], cC[3];
            colors->GetTuple(ptIds[0], cA);
            colors->GetTuple(ptIds[1], cB);
            colors->GetTuple(ptIds[2], cC);

            randomPointTriangle(float(cA[0]), float(cA[1]), float(cA[2]),
                float(cB[0]), float(cB[1]), float(cB[2]),
                float(cC[0]), float(cC[1]), float(cC[2]), r1, r2, c);
        }
        else
        {
            static bool printed_once = false;
            if (!printed_once)
                PCL_WARN("Mesh has no vertex colors, or vertex colors are not RGB!\n");
            printed_once = true;
        }
    }
}
void uniform_sampling(vtkSmartPointer<vtkPolyData> polydata, size_t n_samples, bool calc_normal, bool calc_color, pcl::PointCloud<pcl::PointXYZRGBNormal>& cloud_out)
{
    polydata->BuildCells();
    vtkSmartPointer<vtkCellArray> cells = polydata->GetPolys();

    double p1[3], p2[3], p3[3], totalArea = 0;
    std::vector<double> cumulativeAreas(cells->GetNumberOfCells(), 0);
    size_t i = 0;
    vtkIdType npts = 0, * ptIds = NULL;
    for (cells->InitTraversal(); cells->GetNextCell(npts, ptIds); i++)
    {
        polydata->GetPoint(ptIds[0], p1);
        polydata->GetPoint(ptIds[1], p2);
        polydata->GetPoint(ptIds[2], p3);
        totalArea += vtkTriangle::TriangleArea(p1, p2, p3);
        cumulativeAreas[i] = totalArea;
    }

    cloud_out.resize(n_samples);
    cloud_out.width = static_cast<std::uint32_t> (n_samples);
    cloud_out.height = 1;

    for (std::size_t i = 0; i < n_samples; i++)
    {
        Eigen::Vector3f p;
        Eigen::Vector3f n(0, 0, 0);
        Eigen::Vector3f c(0, 0, 0);
        randPSurface(polydata, &cumulativeAreas, totalArea, p, calc_normal, n, calc_color, c);
        cloud_out[i].x = p[0];
        cloud_out[i].y = p[1];
        cloud_out[i].z = p[2];
        if (calc_normal)
        {
            cloud_out[i].normal_x = n[0];
            cloud_out[i].normal_y = n[1];
            cloud_out[i].normal_z = n[2];
        }
        if (calc_color)
        {
            cloud_out[i].r = static_cast<std::uint8_t>(c[0]);
            cloud_out[i].g = static_cast<std::uint8_t>(c[1]);
            cloud_out[i].b = static_cast<std::uint8_t>(c[2]);
        }
    }
}

//mesh sampling
void sample_mesh(std::string file_name, int number_samples,pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr out_cloud) {
    vtkSmartPointer<vtkPolyData> polydata1 = vtkSmartPointer<vtkPolyData>::New();
    vtkSmartPointer<vtkOBJReader> readerQuery = vtkSmartPointer<vtkOBJReader>::New();
    readerQuery->SetFileName(file_name.c_str());
    readerQuery->Update();
    polydata1 = readerQuery->GetOutput();

    vtkSmartPointer<vtkTriangleFilter> triangleFilter = vtkSmartPointer<vtkTriangleFilter>::New();
    triangleFilter->SetInputData(polydata1);
    triangleFilter->Update();

    vtkSmartPointer<vtkPolyDataMapper> triangleMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    triangleMapper->SetInputConnection(triangleFilter->GetOutputPort());
    triangleMapper->Update();
    polydata1 = triangleMapper->GetInput();

    uniform_sampling(polydata1, number_samples, true, true, *out_cloud);
}

void sample_mesh(pcl::PolygonMesh mesh, int number_samples, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr out_cloud) {
    vtkSmartPointer<vtkPolyData> polydata1 = vtkSmartPointer<vtkPolyData>::New();
    pcl::VTKUtils::convertToVTK(mesh, polydata1);

    vtkSmartPointer<vtkTriangleFilter> triangleFilter = vtkSmartPointer<vtkTriangleFilter>::New();
    triangleFilter->SetInputData(polydata1);
    triangleFilter->Update();

    vtkSmartPointer<vtkPolyDataMapper> triangleMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    triangleMapper->SetInputConnection(triangleFilter->GetOutputPort());
    triangleMapper->Update();
    polydata1 = triangleMapper->GetInput();

    uniform_sampling(polydata1, number_samples, true, true, *out_cloud);
}

//reads txt cloud file 
bool read_ASCII(std::string filename, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr c, pcl::PointCloud <pcl::Normal>::Ptr n) {
    std::cout << "MAIN:: Loading ASCII point cloud with normals ..." << std::endl;
    FILE* file = fopen(filename.c_str(), "r");

    if (file == NULL)
    {
        std::cout << "MAIN:: ERROR: failed to open file: " << filename << endl;
        return false;
    }

    float x, y, z;
    float r, g, b;
    float nx, ny, nz;
    bool floatmapped = true;
    float format_factor = 1.0;

    while (!feof(file))
    {
        int n_args = fscanf_s(file, "%f %f %f %f %f %f %f %f %f", &x, &y, &z, &r, &g, &b, &nx, &ny, &nz);
        if (n_args != 9)
            continue;

        if (floatmapped && r > 1.1f || g > 1.1f || b > 1.1f) {
            format_factor = 255.0;
            floatmapped != false;
        }

        pcl::PointXYZRGBNormal point;
        point.x = x;
        point.y = y;
        point.z = z;
        point.r = (char)(r * format_factor);
        point.g = (char)(g * format_factor);
        point.b = (char)(b * format_factor);
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

    std::cout << "MAIN:: Loaded cloud with " << c->size() << " points." << std::endl;
    return true;
}

//================ PIPELINE BUILDING BLOCKS ==============
//normal estimation
void estimate_normals(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr in_cloud, pcl::PointCloud<pcl::Normal>::Ptr normals, bool copy_to_cloud, float v_x, float v_y, float v_z) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*in_cloud, *cloud);

    pcl::search::Search<pcl::PointXYZ>::Ptr search_tree(new pcl::search::KdTree<pcl::PointXYZ>);
    
    //estimates normals by fitting local tangent plane
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normal_estimator;
    normal_estimator.setSearchMethod(search_tree);
    normal_estimator.setInputCloud(cloud);
    normal_estimator.setKSearch(50);

    //orient to view point
    normal_estimator.setViewPoint(v_x, v_y, v_z);
    normal_estimator.compute(*normals);
    
    if (copy_to_cloud) {
        pcl::copyPointCloud(*normals, *in_cloud);
    }

    /*pcl::PrincipalCurvaturesEstimation<pcl::PointXYZ, pcl::Normal, pcl::PrincipalCurvatures> principal_curvatures_estimation;
    principal_curvatures_estimation.setInputCloud(cloud);
    principal_curvatures_estimation.setInputNormals(normals);
    principal_curvatures_estimation.setSearchMethod(search_tree);
    principal_curvatures_estimation.setKSearch(50);
    pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr principal_curvatures(new pcl::PointCloud<pcl::PrincipalCurvatures>());
    principal_curvatures_estimation.compute(*principal_curvatures);
    
    if (copy_to_cloud) {
        pcl::copyPointCloud(*principal_curvatures, *in_cloud);
    }*/
}

//color normalization
//normalze RGB values
void normalize_RGB(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr const cloud, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr normalized_cloud) {
    __int16 sum = 0;
    for (pcl::PointCloud<pcl::PointXYZRGBNormal>::iterator it = cloud->begin(); it != cloud->end(); it++) {
        pcl::PointXYZRGBNormal point;
        pcl::copyPoint(*it, point);
        int r = __int16(point.r);
        int g = __int16(point.g);
        int b = __int16(point.b);
        sum = r;
        sum += g;
        sum += b;
        float f_R = float(r) / float(sum) * 255.0;
        float f_G = float(g) / float(sum) * 255.0;
        float f_B = float(b) / float(sum) * 255.0;

        int32_t rgb = (static_cast<uint32_t>(f_R) << 16 | static_cast<uint32_t>(f_G) << 8 | static_cast<uint32_t>(f_B));
        point.rgb = rgb;
        normalized_cloud->push_back(point);
        sum = 0;
    }
}

bool IsPathExist(const std::string& s)
{
    struct stat buffer;
    return (stat(s.c_str(), &buffer) == 0);
}