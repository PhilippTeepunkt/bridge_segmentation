#!/usr/bin/python3

import os
import sys
import numpy as np
import subprocess

### calculates evaluation metrics
def confusion_matrix(sampling_size, ground_slab_indices, ground_bridge_indices, pipe_slab_indices, pipe_bridge_indices):
	union_s = np.union1d(ground_slab_indices,pipe_slab_indices)
	union_b = np.union1d(ground_bridge_indices,pipe_bridge_indices)
	intersection_s = np.intersect1d(ground_slab_indices, pipe_slab_indices)
	intersection_b = np.intersect1d(ground_bridge_indices, pipe_bridge_indices)
	print("Number of correctly labeled points (s,b):")
	print(len(intersection_s))
	print(len(intersection_b))
	print("Number of points in predicted cluster + missing points (s,b):")
	print(len(union_s))
	print(len(union_b))

	# IoU / TP/(TP+FP+FN)
	iou_slab = len(intersection_s)/len(union_s)
	iou_bridge = len(intersection_b)/len(union_b)
	print("\n////////////////// IoU /////////////////")
	print("IoU Slab: "+str(iou_slab))
	print("IoU Bridge: "+str(iou_bridge))

	# Recall :: Out of all points part of the slab/bridge, how many were predicted
	recall_slab = len(intersection_s)/len(ground_slab_indices)
	recall_bridge = len(intersection_b)/len(ground_bridge_indices)
	print("\n////////////////// RECALL /////////////////")
	print("Recall Slab: "+str(recall_slab))
	print("Recall Bridge: "+str(recall_bridge))

	# Precision :: Out of all points predicted to be part of the slab/bridge, how many were right predicted
	precision_slab = len(intersection_s)/len(pipe_slab_indices)
	precision_bridge = len(intersection_b)/len(pipe_bridge_indices)
	print("\n////////////////// PRECISION /////////////////")
	print("Precision Slab: "+str(precision_slab))
	print("Precision Bridge: "+str(precision_bridge))

	# Accuracy :: Out of all classes, how much were predicted correctly
	accuracy_slab = (len(intersection_s)+sampling_size-len(union_s))/sampling_size
	accuracy_bridge = (len(intersection_b)+sampling_size-len(union_b))/sampling_size
	print("\n////////////////// ACCURACY /////////////////")
	print("Accuracy Slab: "+str(accuracy_slab))
	print("Accuracy Bridge: "+str(accuracy_bridge))

	# F-Measure :: harmonic mean to measure recall and precision at the same time
	if recall_slab == 0.0 or precision_slab == 0.0:
		f_measure_slab = 0
	else:
		f_measure_slab = (2*recall_slab*precision_slab)/(recall_slab+precision_slab)

	if recall_bridge == 0.0 or precision_bridge == 0.0:
		f_measure_bridge = 0
	else:
		f_measure_bridge = (2*recall_bridge*precision_bridge)/(recall_bridge+precision_bridge)
	print("\n////////////////// F-MEASURE /////////////////")
	print("F-Measure Slab: "+str(f_measure_slab))
	print("F-Measure Bridge: "+str(f_measure_bridge))
	return [str(iou_slab),str(iou_bridge),str(recall_slab),str(recall_bridge),str(precision_slab),str(precision_bridge),str(accuracy_slab),str(accuracy_bridge),str(f_measure_slab),str(f_measure_bridge)]

### Formats the cloud compare indices file ###
def format_ground_truth(ground_truth_file, ground_indices, bridge_name):
	f = open(ground_truth_file, "r")
	filename = "./ground_truth/"+bridge_name+"/"+"formated_"+os.path.splitext(os.path.basename(ground_truth_file))[0]+".txt"
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

### write results persistend
def write_results(bridge_name, config_number, results):
	if not os.path.exists("./evaluation_data/"+bridge_name):
		os.makedirs("./evaluation_data/"+bridge_name)
	if config_number<0:
		filename = "./evaluation_data/"+bridge_name+"/"+bridge_name+"_results.txt"
	else:
		filename = "./evaluation_data/"+bridge_name+"/"+bridge_name+"_results_config_" + str(config_number)+".txt"
	f = open(filename, "w")
	f.write("IoU Slab: "+results[0]+"\n")
	f.write("IoU Bridge: "+results[1]+"\n")
	f.write("Recall Slab: "+results[2]+"\n")
	f.write("Recall Bridge: "+results[3]+"\n")
	f.write("Precision Slab: "+results[4]+"\n")
	f.write("Precision Bridge: "+results[5]+"\n")
	f.write("Accuracy Slab: "+results[6]+"\n")
	f.write("Accuracy Bridge: "+results[7]+"\n")
	f.write("F-Measure Slab: "+results[8]+"\n")
	f.write("F-Measure Bridge: "+results[9]+"\n")
	f.close()

def main(args):
	numberFiles = len(args)
	if numberFiles < 4:
		print("Not enough input. <pipeline_program> <pointcloud_file> <input_slab_gt> <input_bridge_gt ")
		sys.exit()

	pipeline_program = args[0]
	pointcloud_file = args[1]
	input_slab_gt = args[2]
	input_bridge_gt = args[3]
	
	# assign a configuration to pipline if available
	config_number = -1
	config_file = "DEFAULT"
	if len(args)>4:
		config_number = int(args[4])
		config_file = args[5]

	evaluation_folder = os.path.dirname(os.path.dirname(os.path.dirname(input_slab_gt)))
	bridge_name = os.path.splitext(os.path.basename(pointcloud_file))[0].split("_")[0]

	print("Read and format Ground Truth. \n")
	ground_slab_indices = []
	ground_bridge_indices = []
	format_ground_truth(input_slab_gt, ground_slab_indices, bridge_name)
	format_ground_truth(input_bridge_gt, ground_bridge_indices,bridge_name)

	print("Start Pipeline. \n");
	if(config_number>-1):
		print(config_file)
		subprocess.call([pipeline_program, "-e", pointcloud_file, config_file])
	else:
		subprocess.call([pipeline_program, "-e", pointcloud_file])

	pipline_slab_output_path = evaluation_folder+"\\pipeline_output\\"+bridge_name+"_"+os.path.splitext(os.path.basename(config_file))[0]+"\\slab_indices.txt"
	
	print("Start Evaluation. \n")
	p_s = open(pipline_slab_output_path)
	p_b = open(evaluation_folder+"\\pipeline_output\\"+bridge_name+"_"+os.path.splitext(os.path.basename(config_file))[0]+"\\bridge_indices.txt")
	
	pipe_slab_indices = []
	pipe_bridge_indices = []

	sampling_size = int(p_s.readline().split(";")[1])
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

	conf_mat_results = confusion_matrix(sampling_size, ground_slab_indices, ground_bridge_indices, pipe_slab_indices, pipe_bridge_indices)
	write_results(bridge_name, config_number, conf_mat_results)

if __name__ == "__main__":
	main(sys.argv[1:])