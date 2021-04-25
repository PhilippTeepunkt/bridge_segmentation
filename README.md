# bridge_segmentation
Implementation of an unsupervised Point Cloud Segmentation Pipeline for Beam-Slap-Bridges.
This implementation follows the proposal in the thesis: 
**Unsupervised Object Extraction for Semantic Segmentation of Bridge Scenes**

## Installation:
**Requirements:**
* Cmake
* C++ Compiler
* [PCL Library](https://pointclouds.org/)
(Windows All in One installer: https://github.com/PointCloudLibrary/pcl/releases + set Path variables)

**Setup:**  
Clone repository  
`git clone https://github.com/PhilippTeepunkt/bridge_segmentation.git`  

create build directory  
`mkdir build; cd build`  

build files  
`cmake ..`  

**Run**  
`./startPipeline [-e Evaluation option] <pcdfile> / <ASCII .txt> / <Mesh .obj> [CAMERA SETTINGS] [CONFIG FILE]"`  

Camera settings (.cam) can be obtained when pressing **j** in the visualization window.  
Example Config File (.txt):  
2000000 _//sampling size_   
550 _//k-neighborhood_  
0.81 _//smoothness threshold_   
0.05 _//curvature threshold_  
5 _//quantized color/num color clusters_  
300 _//neighborhood statOutlierFilter_    
0.7 _//stdv. distance to center for statOutlierFilter_  

![Overview Window](https://github.com/PhilippTeepunkt/bridge_segmentation/blob/2ca34325b6184ef796757b9f58140c6dd515fd94/overview_pclviewer.png)  


