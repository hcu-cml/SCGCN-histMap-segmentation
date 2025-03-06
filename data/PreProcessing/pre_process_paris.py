# crop the images of the paris dataset to 500x500

#Iteriere über test, val und traing dir
#Hier müsste es egal sein, dass ich nicht
#in der richtigen Reihenfolge über die 
#Bilder iteriere weil ja vorher alle in 
#eine Liste gesteckt werden und ein index
#für train und val und test erzeugt wird

import cv2
import os
import numpy as np
import random

# array_ofones_3d = np.ones((500,500,3))
# array_ofones_1d = np.ones((500,500))

# val gt (labels)
directory = '/home_domuser/s6luarzo/paris/train/labels'

for filename in os.listdir(directory):
	f = os.path.join(directory, filename)
	print(f, " is being processed")
	# grayscale_image_og = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
	rgb_image_og = cv2.imread(f)

	rgb_image = cv2.cvtColor(rgb_image_og, cv2.COLOR_BGR2RGB)

	grayscale_image = np.zeros((1000, 1000), dtype=int)
	for i in range(np.shape(grayscale_image)[1]):
		for k in range(np.shape(grayscale_image)[0]):

			# 0 = frame
			if rgb_image[i, k, 0] == 0 and rgb_image[i, k, 1] == 0 and rgb_image[i, k, 2] == 0:
				grayscale_image[i, k] = 0;
			# 1 = water
			elif rgb_image[i, k, 0] == 0 and rgb_image[i, k, 1] == 0 and rgb_image[i, k, 2] == 255:
				grayscale_image[i, k] = 1;
			# 2 = blocks
			elif rgb_image[i, k, 0] == 255 and rgb_image[i, k, 1] == 0 and rgb_image[i, k, 2] == 255:
				grayscale_image[i, k] = 2;
			# 3 = non-build
			elif rgb_image[i, k, 0] == 0 and rgb_image[i, k, 1] == 255 and rgb_image[i, k, 2] == 255:
				grayscale_image[i, k] = 3;
			# 4 = streets
			elif rgb_image[i, k, 0] == 255 and rgb_image[i, k, 1] == 255 and rgb_image[i, k, 2] == 255:
				grayscale_image[i, k] = 4;

	cv2.imwrite('historicalMaps/paris/train/gt/' + str(filename), grayscale_image)

directory = 'historicalMaps/paris/val/labels'

for filename in os.listdir(directory):
	f = os.path.join(directory, filename)
	print(f, " is being processed")

	rgb_image_og = cv2.imread(f)

	rgb_image = cv2.cvtColor(rgb_image_og, cv2.COLOR_BGR2RGB)

	grayscale_image = np.zeros((1000, 1000), dtype=int)
	for i in range(np.shape(grayscale_image)[1]):
		for k in range(np.shape(grayscale_image)[0]):

			# 0 = frame
			if rgb_image[i, k, 0] == 0 and rgb_image[i, k, 1] == 0 and rgb_image[i, k, 2] == 0:
				grayscale_image[i, k] = 0;
			# 1 = water
			elif rgb_image[i, k, 0] == 0 and rgb_image[i, k, 1] == 0 and rgb_image[i, k, 2] == 255:
				grayscale_image[i, k] = 1;
			# 2 = blocks
			elif rgb_image[i, k, 0] == 255 and rgb_image[i, k, 1] == 0 and rgb_image[i, k, 2] == 255:
				grayscale_image[i, k] = 2;
			# 3 = non-build
			elif rgb_image[i, k, 0] == 0 and rgb_image[i, k, 1] == 255 and rgb_image[i, k, 2] == 255:
				grayscale_image[i, k] = 3;
			# 4 = streets
			elif rgb_image[i, k, 0] == 255 and rgb_image[i, k, 1] == 255 and rgb_image[i, k, 2] == 255:
				grayscale_image[i, k] = 4;

	cv2.imwrite('historicalMaps/paris/val/gt/' + str(filename), grayscale_image)


# # val gt (labels)
# directory = '/historicalMaps/paris/val/labels'
#
# for filename in os.listdir(directory):
# 	f = os.path.join(directory, filename)
# 	print(f, " is being processed")
# 	# grayscale_image_og = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
# 	rgb_image_og = cv2.imread(f)
#
# 	rgb_image = cv2.cvtColor(rgb_image_og, cv2.COLOR_BGR2RGB)
#
# 	rgb_image_ul = rgb_image[:500,:500]
# 	grayscale_image_ul = np.zeros((500, 500), dtype=int)
# 	for i in range(np.shape(grayscale_image_ul)[1]):
# 		for k in range(np.shape(grayscale_image_ul)[0]):
#
# 			# 0 = frame
# 			if rgb_image_ul[i, k, 0] == 0 and rgb_image_ul[i, k, 1] == 0 and rgb_image_ul[i, k, 2] == 0:
# 				grayscale_image_ul[i, k] = 0;
# 			# 1 = water
# 			elif rgb_image_ul[i, k, 0] == 0 and rgb_image_ul[i, k, 1] == 0 and rgb_image_ul[i, k, 2] == 255:
# 				grayscale_image_ul[i, k] = 1;
# 			# 2 = blocks
# 			elif rgb_image_ul[i, k, 0] == 255 and rgb_image_ul[i, k, 1] == 0 and rgb_image_ul[i, k, 2] == 255:
# 				grayscale_image_ul[i, k] = 2;
# 			# 3 = non-build
# 			elif rgb_image_ul[i, k, 0] == 0 and rgb_image_ul[i, k, 1] == 255 and rgb_image_ul[i, k, 2] == 255:
# 				grayscale_image_ul[i, k] = 3;
# 			# 4 = streets
# 			elif rgb_image_ul[i, k, 0] == 255 and rgb_image_ul[i, k, 1] == 255 and rgb_image_ul[i, k, 2] == 255:
# 				grayscale_image_ul[i, k] = 4;
#
# 	# grayscale_image_ul = grayscale_image_og[:500,:500]
# 	# grayscale_image_ul  = np.subtract(grayscale_image_ul , array_ofones_1d)
# 	# grayscale_image_ul_mask = np.where((grayscale_image_ul[:,:]==-1))
# 	# grayscale_image_ul[grayscale_image_ul_mask]=(255)
#
# 	rgb_image_ll = rgb_image[500:1000,:500]
# 	grayscale_image_ll = np.zeros((500, 500), dtype=int)
# 	for i in range(np.shape(grayscale_image_ll)[1]):
# 		for k in range(np.shape(grayscale_image_ll)[0]):
#
# 			# 0 = frame
# 			if rgb_image_ll[i, k, 0] == 0 and rgb_image_ll[i, k, 1] == 0 and rgb_image_ll[i, k, 2] == 0:
# 				grayscale_image_ll[i, k] = 0;
# 			# 1 = water
# 			elif rgb_image_ll[i, k, 0] == 0 and rgb_image_ll[i, k, 1] == 0 and rgb_image_ll[i, k, 2] == 255:
# 				grayscale_image_ll[i, k] = 1;
# 			# 2 = blocks
# 			elif rgb_image_ll[i, k, 0] == 255 and rgb_image_ll[i, k, 1] == 0 and rgb_image_ll[i, k, 2] == 255:
# 				grayscale_image_ll[i, k] = 2;
# 			# 3 = non-build
# 			elif rgb_image_ll[i, k, 0] == 0 and rgb_image_ll[i, k, 1] == 255 and rgb_image_ll[i, k, 2] == 255:
# 				grayscale_image_ll[i, k] = 3;
# 			# 4 = streets
# 			elif rgb_image_ll[i, k, 0] == 255 and rgb_image_ll[i, k, 1] == 255 and rgb_image_ll[i, k, 2] == 255:
# 				grayscale_image_ll[i, k] = 4;
# 	#
# 	# grayscale_image_ll = grayscale_image_og[500:1000,:500]
# 	# grayscale_image_ll  = np.subtract(grayscale_image_ll , array_ofones_1d)
# 	# grayscale_image_ll_mask = np.where((grayscale_image_ll[:,:]==-1))
# 	# grayscale_image_ll[grayscale_image_ll_mask]=(255)
#
# 	rgb_image_ur = rgb_image[:500,500:1000]
# 	grayscale_image_ur = np.zeros((500, 500), dtype=int)
# 	for i in range(np.shape(grayscale_image_ur)[1]):
# 		for k in range(np.shape(grayscale_image_ur)[0]):
#
# 			# 0 = frame
# 			if rgb_image_ur[i, k, 0] == 0 and rgb_image_ur[i, k, 1] == 0 and rgb_image_ur[i, k, 2] == 0:
# 				grayscale_image_ur[i, k] = 0;
# 			# 1 = water
# 			elif rgb_image_ur[i, k, 0] == 0 and rgb_image_ur[i, k, 1] == 0 and rgb_image_ur[i, k, 2] == 255:
# 				grayscale_image_ur[i, k] = 1;
# 			# 2 = blocks
# 			elif rgb_image_ur[i, k, 0] == 255 and rgb_image_ur[i, k, 1] == 0 and rgb_image_ur[i, k, 2] == 255:
# 				grayscale_image_ur[i, k] = 2;
# 			# 3 = non-build
# 			elif rgb_image_ur[i, k, 0] == 0 and rgb_image_ur[i, k, 1] == 255 and rgb_image_ur[i, k, 2] == 255:
# 				grayscale_image_ur[i, k] = 3;
# 			# 4 = streets
# 			elif rgb_image_ur[i, k, 0] == 255 and rgb_image_ur[i, k, 1] == 255 and rgb_image_ur[i, k, 2] == 255:
# 				grayscale_image_ur[i, k] = 4;
# 	#
# 	# grayscale_image_ur = grayscale_image_og[:500,500:1000]
# 	# grayscale_image_ur  = np.subtract(grayscale_image_ur , array_ofones_1d)
# 	# grayscale_image_ur_mask = np.where((grayscale_image_ur[:,:]==-1))
# 	# grayscale_image_ur[grayscale_image_ur_mask]=(255)
#
# 	rgb_image_lr = rgb_image[500:1000,500:1000]
# 	grayscale_image_lr = np.zeros((500, 500), dtype=int)
# 	for i in range(np.shape(grayscale_image_lr)[1]):
# 		for k in range(np.shape(grayscale_image_lr)[0]):
#
# 			# 0 = frame
# 			if rgb_image_lr[i, k, 0] == 0 and rgb_image_lr[i, k, 1] == 0 and rgb_image_lr[i, k, 2] == 0:
# 				grayscale_image_lr[i, k] = 0;
# 			# 1 = water
# 			elif rgb_image_lr[i, k, 0] == 0 and rgb_image_lr[i, k, 1] == 0 and rgb_image_lr[i, k, 2] == 255:
# 				grayscale_image_lr[i, k] = 1;
# 			# 2 = blocks
# 			elif rgb_image_lr[i, k, 0] == 255 and rgb_image_lr[i, k, 1] == 0 and rgb_image_lr[i, k, 2] == 255:
# 				grayscale_image_lr[i, k] = 2;
# 			# 3 = non-build
# 			elif rgb_image_lr[i, k, 0] == 0 and rgb_image_lr[i, k, 1] == 255 and rgb_image_lr[i, k, 2] == 255:
# 				grayscale_image_lr[i, k] = 3;
# 			# 4 = streets
# 			elif rgb_image_lr[i, k, 0] == 255 and rgb_image_lr[i, k, 1] == 255 and rgb_image_lr[i, k, 2] == 255:
# 				grayscale_image_lr[i, k] = 4;
# 	#
# 	# grayscale_image_lr = grayscale_image_og[500:1000,500:1000]
# 	# grayscale_image_lr  = np.subtract(grayscale_image_lr , array_ofones_1d)
# 	# grayscale_image_lr_mask = np.where((grayscale_image_lr[:,:]==-1))
# 	# grayscale_image_lr[grayscale_image_lr_mask]=(255)
#
# 	cv2.imwrite('/home_domuser/s6luarzo/paris/val/gt/' + str(filename[:(len(filename)-4)]) + '_ul' + '.png', grayscale_image_ul)
# 	cv2.imwrite('/home_domuser/s6luarzo/paris/val/gt/' + str(filename[:(len(filename)-4)]) + '_ll' + '.png', grayscale_image_ll)
# 	cv2.imwrite('/home_domuser/s6luarzo/paris/val/gt/' + str(filename[:(len(filename)-4)]) + '_ur' + '.png', grayscale_image_ur)
# 	cv2.imwrite('/home_domuser/s6luarzo/paris/val/gt/' + str(filename[:(len(filename)-4)]) + '_lr' + '.png', grayscale_image_lr)
#
# # # val images
# directory = '/home_domuser/s6luarzo/paris/val/images_1000'
#
# for filename in os.listdir(directory):
# 	f = os.path.join(directory, filename)
# 	print(f, " is being processed")
# 	image_BGR = cv2.imread(f, cv2.IMREAD_COLOR)
#
# 	rgb_image_ul = image_BGR[:500,:500,:]
# 	rgb_image_ll = image_BGR[500:1000,:500,:]
# 	rgb_image_ur = image_BGR[:500,500:1000,:]
# 	rgb_image_lr = image_BGR[500:1000,500:1000,:]
#
#
# 	cv2.imwrite('/home_domuser/s6luarzo/paris/val/images/' + str(filename[:(len(filename)-4)]) + '_ul' + '.png', rgb_image_ul)
# 	cv2.imwrite('/home_domuser/s6luarzo/paris/val/images/' + str(filename[:(len(filename)-4)]) + '_ll' + '.png', rgb_image_ll)
# 	cv2.imwrite('/home_domuser/s6luarzo/paris/val/images/' + str(filename[:(len(filename)-4)]) + '_ur' + '.png', rgb_image_ur)
# 	cv2.imwrite('/home_domuser/s6luarzo/paris/val/images/' + str(filename[:(len(filename)-4)]) + '_lr' + '.png', rgb_image_lr)
#
#
# # train gt (labels) -> es muss in grayscale images umgewandelt werden
# directory = '/home_domuser/s6luarzo/paris/train/labels'
#
# for filename in os.listdir(directory):
# 	f = os.path.join(directory, filename)
# 	print(f, " is being processed")
# 	rgb_image_og = cv2.imread(f)
#
# 	rgb_image = cv2.cvtColor(rgb_image_og, cv2.COLOR_BGR2RGB)
#
#
# 	rgb_image_ul = rgb_image[:500,:500]
# 	grayscale_image_ul = np.zeros((500, 500), dtype=int)
# 	for i in range(np.shape(grayscale_image_ul)[1]):
# 		for k in range(np.shape(grayscale_image_ul)[0]):
#
# 			# 0 = frame
# 			if rgb_image_ul[i, k, 0] == 0 and rgb_image_ul[i, k, 1] == 0 and rgb_image_ul[i, k, 2] == 0:
# 				grayscale_image_ul[i, k] = 0;
# 			# 1 = water
# 			elif rgb_image_ul[i, k, 0] == 0 and rgb_image_ul[i, k, 1] == 0 and rgb_image_ul[i, k, 2] == 255:
# 				grayscale_image_ul[i, k] = 1;
# 			# 2 = blocks
# 			elif rgb_image_ul[i, k, 0] == 255 and rgb_image_ul[i, k, 1] == 0 and rgb_image_ul[i, k, 2] == 255:
# 				grayscale_image_ul[i, k] = 2;
# 			# 3 = non-build
# 			elif rgb_image_ul[i, k, 0] == 0 and rgb_image_ul[i, k, 1] == 255 and rgb_image_ul[i, k, 2] == 255:
# 				grayscale_image_ul[i, k] = 3;
# 			# 4 = streets
# 			elif rgb_image_ul[i, k, 0] == 255 and rgb_image_ul[i, k, 1] == 255 and rgb_image_ul[i, k, 2] == 255:
# 				grayscale_image_ul[i, k] = 4;
# 	#Für ohne Frame
# 	# grayscale_image_ul  = np.subtract(grayscale_image_ul , array_ofones_1d)
# 	# grayscale_image_ul_mask = np.where((grayscale_image_ul[:,:]==-1))
# 	# grayscale_image_ul[grayscale_image_ul_mask]=(255)
#
# 	rgb_image_ll = rgb_image[500:1000,:500]
# 	grayscale_image_ll = np.zeros((500, 500), dtype=int)
# 	for i in range(np.shape(grayscale_image_ll)[1]):
# 		for k in range(np.shape(grayscale_image_ll)[0]):
#
# 			# 0 = frame
# 			if rgb_image_ll[i, k, 0] == 0 and rgb_image_ll[i, k, 1] == 0 and rgb_image_ll[i, k, 2] == 0:
# 				grayscale_image_ll[i, k] = 0;
# 			# 1 = water
# 			elif rgb_image_ll[i, k, 0] == 0 and rgb_image_ll[i, k, 1] == 0 and rgb_image_ll[i, k, 2] == 255:
# 				grayscale_image_ll[i, k] = 1;
# 			# 2 = blocks
# 			elif rgb_image_ll[i, k, 0] == 255 and rgb_image_ll[i, k, 1] == 0 and rgb_image_ll[i, k, 2] == 255:
# 				grayscale_image_ll[i, k] = 2;
# 			# 3 = non-build
# 			elif rgb_image_ll[i, k, 0] == 0 and rgb_image_ll[i, k, 1] == 255 and rgb_image_ll[i, k, 2] == 255:
# 				grayscale_image_ll[i, k] = 3;
# 			# 4 = streets
# 			elif rgb_image_ll[i, k, 0] == 255 and rgb_image_ll[i, k, 1] == 255 and rgb_image_ll[i, k, 2] == 255:
# 				grayscale_image_ll[i, k] = 4;
# 	# grayscale_image_ll  = np.subtract(grayscale_image_ll , array_ofones_1d)
# 	# grayscale_image_ll_mask = np.where((grayscale_image_ll[:,:]==-1))
# 	# grayscale_image_ll[grayscale_image_ll_mask]=(255)
#
# 	rgb_image_ur = rgb_image[:500,500:1000]
# 	grayscale_image_ur = np.zeros((500, 500), dtype=int)
# 	for i in range(np.shape(grayscale_image_ur)[1]):
# 		for k in range(np.shape(grayscale_image_ur)[0]):
#
# 			# 0 = frame
# 			if rgb_image_ur[i, k, 0] == 0 and rgb_image_ur[i, k, 1] == 0 and rgb_image_ur[i, k, 2] == 0:
# 				grayscale_image_ur[i, k] = 0;
# 			# 1 = water
# 			elif rgb_image_ur[i, k, 0] == 0 and rgb_image_ur[i, k, 1] == 0 and rgb_image_ur[i, k, 2] == 255:
# 				grayscale_image_ur[i, k] = 1;
# 			# 2 = blocks
# 			elif rgb_image_ur[i, k, 0] == 255 and rgb_image_ur[i, k, 1] == 0 and rgb_image_ur[i, k, 2] == 255:
# 				grayscale_image_ur[i, k] = 2;
# 			# 3 = non-build
# 			elif rgb_image_ur[i, k, 0] == 0 and rgb_image_ur[i, k, 1] == 255 and rgb_image_ur[i, k, 2] == 255:
# 				grayscale_image_ur[i, k] = 3;
# 			# 4 = streets
# 			elif rgb_image_ur[i, k, 0] == 255 and rgb_image_ur[i, k, 1] == 255 and rgb_image_ur[i, k, 2] == 255:
# 				grayscale_image_ur[i, k] = 4;
# 	# grayscale_image_ur  = np.subtract(grayscale_image_ur , array_ofones_1d)
# 	# grayscale_image_ur_mask = np.where((grayscale_image_ur[:,:]==-1))
# 	# grayscale_image_ur[grayscale_image_ur_mask]=(255)
#
# 	rgb_image_lr = rgb_image[500:1000,500:1000]
# 	grayscale_image_lr = np.zeros((500, 500), dtype=int)
# 	for i in range(np.shape(grayscale_image_lr)[1]):
# 		for k in range(np.shape(grayscale_image_lr)[0]):
#
# 			# 0 = frame
# 			if rgb_image_lr[i, k, 0] == 0 and rgb_image_lr[i, k, 1] == 0 and rgb_image_lr[i, k, 2] == 0:
# 				grayscale_image_lr[i, k] = 0;
# 			# 1 = water
# 			elif rgb_image_lr[i, k, 0] == 0 and rgb_image_lr[i, k, 1] == 0 and rgb_image_lr[i, k, 2] == 255:
# 				grayscale_image_lr[i, k] = 1;
# 			# 2 = blocks
# 			elif rgb_image_lr[i, k, 0] == 255 and rgb_image_lr[i, k, 1] == 0 and rgb_image_lr[i, k, 2] == 255:
# 				grayscale_image_lr[i, k] = 2;
# 			# 3 = non-build
# 			elif rgb_image_lr[i, k, 0] == 0 and rgb_image_lr[i, k, 1] == 255 and rgb_image_lr[i, k, 2] == 255:
# 				grayscale_image_lr[i, k] = 3;
# 			# 4 = streets
# 			elif rgb_image_lr[i, k, 0] == 255 and rgb_image_lr[i, k, 1] == 255 and rgb_image_lr[i, k, 2] == 255:
# 				grayscale_image_lr[i, k] = 4;
# 	# grayscale_image_lr  = np.subtract(grayscale_image_lr , array_ofones_1d)
# 	# grayscale_image_lr_mask = np.where((grayscale_image_lr[:,:]==-1))
# 	# grayscale_image_lr[grayscale_image_lr_mask]=(255)
#
#
# 	cv2.imwrite('/home_domuser/s6luarzo/paris/train/gt/' + str(filename[:(len(filename)-4)]) + '_ul' + '.png', grayscale_image_ul)
# 	cv2.imwrite('/home_domuser/s6luarzo/paris/train/gt/' + str(filename[:(len(filename)-4)]) + '_ll' + '.png', grayscale_image_ll)
# 	cv2.imwrite('/home_domuser/s6luarzo/paris/train/gt/' + str(filename[:(len(filename)-4)]) + '_ur' + '.png', grayscale_image_ur)
# 	cv2.imwrite('/home_domuser/s6luarzo/paris/train/gt/' + str(filename[:(len(filename)-4)]) + '_lr' + '.png', grayscale_image_lr)
#
#
# # train images
# directory = '/home_domuser/s6luarzo/paris/train/images_1000'
#
# for filename in os.listdir(directory):
# 	if filename[0:2] == "._":
# 		filename = filename[2:]
#
# 	f = os.path.join(directory, filename)
# 	print(f, " is being processed")
# 	image_BGR = cv2.imread(f, cv2.IMREAD_COLOR)
#
# 	rgb_image_ul = image_BGR[:500,:500,:]
# 	rgb_image_ll = image_BGR[500:1000,:500,:]
# 	rgb_image_ur = image_BGR[:500,500:1000,:]
# 	rgb_image_lr = image_BGR[500:1000,500:1000,:]
#
#
# 	cv2.imwrite('/home_domuser/s6luarzo/paris/train/images/' + str(filename[:(len(filename)-4)]) + '_ul' + '.png', rgb_image_ul)
# 	cv2.imwrite('/home_domuser/s6luarzo/paris/train/images/' + str(filename[:(len(filename)-4)]) + '_ll' + '.png', rgb_image_ll)
# 	cv2.imwrite('/home_domuser/s6luarzo/paris/train/images/' + str(filename[:(len(filename)-4)]) + '_ur' + '.png', rgb_image_ur)
# 	cv2.imwrite('/home_domuser/s6luarzo/paris/train/images/' + str(filename[:(len(filename)-4)]) + '_lr' + '.png', rgb_image_lr)


# test gt (labels)
# directory = '/home_domuser/s6luarzo/paris/test/gt_1000'

# for filename in os.listdir(directory):
# 	f = os.path.join(directory, filename)
# 	print(f, " is being processed")
# 	grayscale_image_og = cv2.imread(f)

# 	grayscale_image_ul = grayscale_image_og[:500,:500]

# 	grayscale_image_ll = grayscale_image_og[500:1000,:500]


# 	grayscale_image_ur = grayscale_image_og[:500,500:1000]


# 	grayscale_image_lr = grayscale_image_og[500:1000,500:1000]


# 	cv2.imwrite('/home_domuser/s6luarzo/paris/test/gt/' + str(filename[:(len(filename)-4)]) + '_ul' + '.png', grayscale_image_ul)
# 	cv2.imwrite('/home_domuser/s6luarzo/paris/test/gt/' + str(filename[:(len(filename)-4)]) + '_ll' + '.png', grayscale_image_ll)
# 	cv2.imwrite('/home_domuser/s6luarzo/paris/test/gt/' + str(filename[:(len(filename)-4)]) + '_ur' + '.png', grayscale_image_ur)
# 	cv2.imwrite('/home_domuser/s6luarzo/paris/test/gt/' + str(filename[:(len(filename)-4)]) + '_lr' + '.png', grayscale_image_lr)


# test images 
# directory = '/home_domuser/s6luarzo/paris/test/images_1000'

# for filename in os.listdir(directory):
# 	if filename[0:2] == "._":
# 		filename = filename[2:]

# 	f = os.path.join(directory, filename)
# 	print(f, " is being processed")
# 	image_BGR = cv2.imread(f, cv2.IMREAD_COLOR)

# 	rgb_image_ul = image_BGR[:500,:500,:]
# 	rgb_image_ll = image_BGR[500:1000,:500,:]
# 	rgb_image_ur = image_BGR[:500,500:1000,:]
# 	rgb_image_lr = image_BGR[500:1000,500:1000,:]


# 	cv2.imwrite('/home_domuser/s6luarzo/paris/test/images/' + str(filename[:(len(filename)-4)]) + '_ul' + '.png', rgb_image_ul)
# 	cv2.imwrite('/home_domuser/s6luarzo/paris/test/images/' + str(filename[:(len(filename)-4)]) + '_ll' + '.png', rgb_image_ll)
# 	cv2.imwrite('/home_domuser/s6luarzo/paris/test/images/' + str(filename[:(len(filename)-4)]) + '_ur' + '.png', rgb_image_ur)
# 	cv2.imwrite('/home_domuser/s6luarzo/paris/test/images/' + str(filename[:(len(filename)-4)]) + '_lr' + '.png', rgb_image_lr)


# Create train and val set
# directory_train = '/home_domuser/s6luarzo/paris/train/images/'

# # directory_val = '/home_domuser/s6luarzo/paris/val/images/'

# k = 0
# while k <= 100:
# 	file = random.choice(os.listdir(directory_train))
	
# 	if file[0:2] == '._':
# 		file = file[2:]

# 	print(file)

# 	if os.path.exists('/home_domuser/s6luarzo/paris_100_1000_1000/train/images/'+ file) == False:
# 		k = k + 1

# 		image = cv2.imread("/home_domuser/s6luarzo/paris/train/images/"+ file, cv2.IMREAD_COLOR)
# 		cv2.imwrite("/home_domuser/s6luarzo/paris_100_1000_1000/train/images/" + file, image)
# 		label = cv2.imread("/home_domuser/s6luarzo/paris/train/gt/"+ file, cv2.IMREAD_GRAYSCALE)
# 		cv2.imwrite("/home_domuser/s6luarzo/paris_100_1000_1000/train/gt/" + file, label)

# k = 0
# while k <= 20:
# 	file = random.choice(os.listdir(directory_val))
# 	print(file)

# 	if os.path.exists('/home_domuser/s6luarzo/paris_test200_1000_1000/val/images/'+ file) == False:
# 		k = k + 1
# 		image = cv2.imread("/home_domuser/s6luarzo/paris/val/images/"+ file, cv2.IMREAD_COLOR)
# 		cv2.imwrite("/home_domuser/s6luarzo/paris_test200_1000_1000/val/images/" + file, image)
# 		label = cv2.imread("/home_domuser/s6luarzo/paris/val/gt/"+ file, cv2.IMREAD_GRAYSCALE)
# 		cv2.imwrite("/home_domuser/s6luarzo/paris_test200_1000_1000/val/gt/" + file, label)


