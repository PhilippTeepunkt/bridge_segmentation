import os
import sys
import numpy as np
import time
import subprocess

ground_slab_indices = []
ground_bridge_indices = []

pipe_slab_indices = []
pipe_bridge_indices = []

bridge_name = ""

#def confusion_matrix():

### Calculates IoU by the number indices 
def IoU():
	union_s = np.union1d(ground_slab_indices,pipe_slab_indices)
	union_b = np.union1d(ground_bridge_indices,pipe_bridge_indices)
	intersection_s = np.intersect1d(ground_slab_indices, pipe_slab_indices)
	intersection_b = np.intersect1d(ground_bridge_indices, pipe_bridge_indices)

	print("IoU Slab: "+str(len(intersection_s)/len(union_s))+"\n")
	print("IoU Bridge: "+str(len(intersection_b)/len(union_b))+"\n")

### Formats the cloud compare indices file ###
def format_ground_truth(ground_truth_file, ground_indices):
	f = open(ground_truth_file, "r")
	print(ground_truth_file)
	filename = "./ground_truth/"+bridge_name+"/"+bridge_name+"_formated_"+os.path.splitext(os.path.basename(ground_truth_file))[0]+".txt"
	o = open(filename, "w")

	f.readline()
	lines = f.readlines()
	last = lines[-1]
	for line in lines:
		index = int(line.split(";")[0])
		ground_indices.append(index)
		o.write(str(index))
		if(line != last):
			o.write(";")
	o.close()
	f.close()


if __name__ == "__main__":

	numberFiles = len(sys.argv)-1
	if numberFiles < 4:
		print("Not enough input. <pipeline_program> <pointcloud_file> <input_slab_gt> <input_bridge_gt ")
		sys.exit()

	pipeline_program = sys.argv[1]
	pointcloud_file = sys.argv[2]
	input_slab_gt = sys.argv[3]
	input_bridge_gt = sys.argv[4]

	evaluation_folder = os.path.dirname(os.path.dirname(os.path.dirname(input_slab_gt)))
	bridge_name = os.path.splitext(os.path.basename(pointcloud_file))[0].split("_")[0]

	print("Read and format Ground Truth. \n")
	format_ground_truth(input_slab_gt, ground_slab_indices)
	format_ground_truth(input_bridge_gt, ground_bridge_indices)

	print("Start Pipeline. \n");
	subprocess.Popen([pipeline_program, "-e", pointcloud_file])

	pipline_slab_output_path = evaluation_folder+"\\pipeline_output\\"+bridge_name+"\\"+bridge_name+"_slab_indices.txt"

	while not os.path.exists(pipline_slab_output_path):
		time.sleep(1)
	
	print("Start Evaluation. \n")
	p_s = open(pipline_output_path)
	p_b = open(evaluation_folder+"\\ipeline_output\\"+bridge_name+"\\"+bridge_name+"_bridge_indices.txt")
	for line in p_s.readlines():
		for value in line.split(";"):
			index = int(value)
			pipe_slab_indices.append(index)
	p_s.close()

	for line in p_b.readlines():
		for value in line.split(";"):
			index = int(value)
			pipe_bridge_indices.append(index)
	p_b.close()

	IoU()