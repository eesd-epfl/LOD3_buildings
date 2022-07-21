#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 10:46:06 2020

This script contains the codes to generate LOD3 models.
The codes are based on "Generation LOD3 models from structure-from-motion and semantic segmentation" 
by Pantoja-Rosero et., al.
https://doi.org/10.1016/j.autcon.2022.104430

@author: pantoja
"""


import glob
import os
import numpy as np
import cv2 as cv
from skimage.measure import label, regionprops
from skimage.morphology import disk, erosion
import torch
from torch.utils.data import DataLoader
from math import sqrt as sqrt
from cnn.corner.src.data_loader import get_loader
from cnn.corner.src.network import U_Net16
#from cnn.ssd.ssd_project import ssd
#from cnn.ssd.ssd_project.detection import *
from cnn.semantic.src.data_set import open_dataset
from sklearn.linear_model import LinearRegression
import scipy.spatial
from facade_segmentation import fac_segmentation

def line_adjustor(x_lr,y_lr, smooth=1e-13):
    """
    Adjusts line in a set of points and project them later
    to the line
    Args:
        x_lr (array): x coordinates for points
        y_lr (array): x coordinates for points
        smooth (float, optional): smoothing value to avoid zero division. Defaults to 1e-13.

    Returns:
        X_proj (array): x and y projected coordinates to line
    """

    x_lr = x_lr.reshape((-1,1))
    model_lr = LinearRegression()
    model_lr.fit(x_lr,y_lr)
    
    y_adj = model_lr.predict(x_lr)
    
    yu = model_lr.predict(x_lr)
    yv = model_lr.predict(x_lr+.2)
    
        
    X_proj = np.zeros((len(x_lr),2))
    for i, x in enumerate(x_lr):
        u = np.array([x_lr[i], yu[i]])
        v = np.array([x_lr[i]+.2, yv[i]])
        X = np.array([x_lr[i], y_lr[i]])
        
        e1 = (v-u)/(np.linalg.norm(v-u)+smooth)
        e2 = (X-u)/(np.linalg.norm(X-u)+smooth)
        
        Pu = (np.dot(e1,e2)) * np.linalg.norm(X-u)
        P = u + Pu*e1
        X_proj[i,:] = P
    
    return X_proj[:,0], X_proj[:,1]


def op_detector_2(data_folder, images_path, bound_box, two_views):
    """
    Uses unet architecture to detect opening with semantic segmentation.

    Args:
        data_folder (str): folder name containing building data
        images_path (str): path to the input images
        bound_box (str): region or poly. Method how to generate bound box
        two_views (bool, optional): if true computes detection for two facade views. Defaults to False.
    Returns:
        opening_information (dict): dictionary with information of detected openings
    """

    ##CORNERS DETECTOR
    print("Predicting corners with corner detector CNN....")
    model = 'model_p2'
    image_size = 256
    
    # model path
    model_path = '../weights/corners/models/' + model + '/' 
    model_path = model_path + os.listdir(model_path)[0]
    
    if two_views:
        im_fold="im/"
    else:
        im_fold="im1/"

    #To use the CNN model 
    test_loader = get_loader(image_path=images_path+im_fold,
                             image_size=image_size,
                             batch_size=1,
                             num_workers=0,
                             mode='test',
                             augmentation_prob=0.,
                             shuffle_flag=False)
    
    # load a trained model
    model = U_Net16()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.train(False)
    model.eval()
    
    #Dictionaries to storage prediction and images information    
    prediction_bin = {}
    prediction_rgb = {}
    progress = 0
    for ni, image in enumerate(test_loader):
        progress+=1
        print("Corner prediction process in {}%".format(progress*100/len(test_loader)))
        SR = model(image)
        SR_probs = torch.sigmoid(SR)
        SR_probs_arr = SR_probs.detach().numpy().reshape(image_size, image_size)
        binary_result = SR_probs_arr > .5
        image_numpy = image.detach().numpy()
        image_numpy = image_numpy[0, 0, :, :]
        image_name = test_loader.dataset.image_paths[ni].split('/')
        image_name = image_name[-1].split(".")
        image_name = image_name[0]
    
        corner = np.array(binary_result, dtype='uint8')
        prediction_bin[image_name] = corner*255
        
        cshp = corner.shape
        corner_rgb = np.zeros((cshp[0], cshp[1], 3), dtype='uint8')
        corner_rgb[:,:,2] = corner*255
        prediction_rgb[image_name] = corner_rgb
    
    #Getting original images
    list_images = os.listdir(images_path+im_fold)
    images = {}
    for img in list_images:
        images[img[:-4]] = cv.imread(images_path + im_fold + img)
    
    
    #Resizing predictions to the original size
    prediction_bin_resized = {}
    prediction_rgb_resized = {}
    for key in images:
        prediction_bin_resized[key] = cv.resize(prediction_bin[key], (images[key].shape[1],images[key].shape[0]), interpolation = cv.INTER_CUBIC)
        prediction_rgb_resized[key] = cv.resize(prediction_rgb[key], (images[key].shape[1],images[key].shape[0]), interpolation = cv.INTER_CUBIC)
    
    #Overlaying prediction_rgb with original image
    overlayed_prediction = {}
    for key in images:
        overlayed_prediction[key] = cv.addWeighted(images[key], 1.0, prediction_rgb_resized[key], 0.7, 0)
    
    #saving binary and overlayed predictions
    #Check if directory exists, if not, create it
    check_dir = os.path.isdir('../results/' + data_folder)
    if not check_dir:
        os.makedirs('../results/' + data_folder)
    progress=0
    for key in images:
        progress+=1
        print("Saving corner prediction process in {}%".format(progress*100/len(images)))
        cv.imwrite('../results/' + data_folder + '/' + key + '.png', images[key])
        cv.imwrite('../results/' + data_folder + '/corners_' + key + '_overlayed.png', overlayed_prediction[key])
    
    #Making prediction_bin_resized as binary
    progress=0
    for key in prediction_bin_resized:
        progress+=1
        print("Morphological operations in masks {}%".format(progress*100/len(prediction_bin_resized)))
        prediction_bin_resized[key] = prediction_bin_resized[key]>0
        prediction_bin_resized[key] = erosion(prediction_bin_resized[key], disk(2)) #paiano fails with 6
    
    #Extracting information of centroids of the segmentated objects as the opening corners
    opening_information = {}
    progress=0
    for key in images:
        progress+=1
        print("labeling and regionprops. Saving final corners process in {}%".format(progress*100/len(images)))
        label_corners = label(prediction_bin_resized[key])
        regions_corn = regionprops(label_corners)
        opening_information[key]={}
        opening_information[key]['corner_centroids'] = [region.centroid for region in regions_corn]
        
        #Drawing keypoints
        cent = opening_information[key]['corner_centroids']
        radius = int(overlayed_prediction[key].shape[0]/100)
        for c in cent:
            cv.circle(overlayed_prediction[key], (int(c[1]),int(c[0])), radius=radius, color=(0,0,255), thickness=-1)
        cv.imwrite('../results/' + data_folder + '/corners_' + key + '.png', overlayed_prediction[key])
        
    
    ##SEMANTIC SEGMENTATION OPENINNG DETECTOR    
    print("Predicting openings with semantic segmentation...")
    architecture = 'U_Net16'
       
    # model path
    model = 'model_p2/'
    model_path = '../weights/semantic_openings/' + model
    model_path = model_path + os.listdir(model_path)[0]
    image_size = 256

    # model path
    test_ds=open_dataset(images_path + im_fold, transform='test')
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)
    # load a trained model
    model = U_Net16()
    device = torch.device('cpu')
    model=model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.train(False)
    model.eval()
    
    #Dictionaries to storage prediction and images information    
    prediction_bin = {}
    prediction_rgb = {}
    
    for ni, image in enumerate(test_dl):
        image = image.to(device)
        SR = model(image)
        SR_probs = torch.sigmoid(SR)
        SR_probs_arr = SR_probs.detach().numpy().reshape(image_size, image_size)
        binary_result = SR_probs_arr > .5
        image_numpy = image.detach().numpy()
        image_numpy = image_numpy[0, 0, :, :]
        image_name = test_dl.dataset.path2imgs[ni].split('/')
        image_name = image_name[-1].split(".")
        image_name = image_name[0]
        
        corner = np.array(binary_result, dtype='uint8')
        prediction_bin[image_name] = corner*255
        
        cshp = corner.shape
        corner_rgb = np.zeros((cshp[0], cshp[1], 3), dtype='uint8')
        corner_rgb[:,:,2] = corner*255
        prediction_rgb[image_name] = corner_rgb

        #Resizing predictions to the original size
        prediction_bin_resized = {}
        prediction_rgb_resized = {}
    
    for key in images:
        prediction_bin_resized[key] = cv.resize(prediction_bin[key], (images[key].shape[1],images[key].shape[0]), interpolation = cv.INTER_CUBIC)
        prediction_rgb_resized[key] = cv.resize(prediction_rgb[key], (images[key].shape[1],images[key].shape[0]), interpolation = cv.INTER_CUBIC)
    
    #Overlaying prediction_rgb with original image
    overlayed_prediction = {}
    for key in images:
        overlayed_prediction[key] = cv.addWeighted(images[key], 1.0, prediction_rgb_resized[key], 0.7, 0)
    
    progress=0
    for key in images:
        progress+=1
        print("Saving opening prediction process in {}%".format(progress*100/len(images)))
        cv.imwrite('../results/' + data_folder + '/pred_op_bin_' + key + '.png', prediction_bin_resized[key])
        cv.imwrite('../results/' + data_folder + '/' + key + '_op_overlayed.png', overlayed_prediction[key])
        
    #Making prediction_bin_resized as binary
    progress = 0
    for key in prediction_bin_resized:
        progress+=1
        print("Morphological operations in masks {}%".format(progress*100/len(prediction_bin_resized)))
        prediction_bin_resized[key] = prediction_bin_resized[key]>0
    
    #To extract boundig boxes there are two options. 1) Region props (max and min coord)
    #2) Finding a 4 side polygon with polygoning (ransac)
    
    if bound_box=="region":
        #Extracting information of regions to extract boundic boxes
        for key in images:
            label_op = label(prediction_bin_resized[key])
            regions_op = regionprops(label_op)
            areas = np.array([region.area for region in regions_op])
            areas.sort()
            area_min = .1*areas[-2] 
            regions_op_f = []
            bboxes = []
            labels = []
            boxes_im = np.copy(images[key])
            for reg in regions_op:
                if reg.area>=area_min:
                    regions_op_f.append(reg)
            
            label_op_f = np.zeros(label_op.shape) #to save just regions of interest. Helps to skeleton_ransac. Find border
            for reg in regions_op_f:
                bboxes.append(reg.bbox)
                labels.append('window') #In this architecture just are identified openings as windows
                cv.rectangle(boxes_im, (reg.bbox[1],reg.bbox[0]), (reg.bbox[3],reg.bbox[2]), (0,0,255), 10)
                label_op_f += label_op==reg.label
                
            bboxes = np.array(bboxes)
            bboxes[:,[0,1,2,3]] = bboxes[:,[1,0,3,2]]
            
            opening_information[key]["bboxes"] = (bboxes, labels, bound_box)
            cv.imwrite("../results/" + data_folder + "/boxes_" + key + ".png", boxes_im)
            
            #Check if directory exists, if not, create it
            check_dir = os.path.isdir('../results/' + data_folder + '/skeleton/')
            if not check_dir:
                os.makedirs('../results/' + data_folder + '/skeleton/')
            cv.imwrite("../results/" + data_folder + "/skeleton/for_skeleton_" + key + ".png", label_op_f*255)
    
    return opening_information


def op_detector_1(data_folder, images_path, two_views):
    """Uses SSD architecture to detect opening as bouning boxes.

    Args:
        data_folder (str): folder name containing building data
        images_path (str): path to the input images
        two_views (bool, optional): if true computes detection for two facade views. Defaults to False.
    Returns:
        opening_information (dict): dictionary with information of detected openings

    """
    
    #Using SDD to detect openings
    #CORNERS DETECTOR#
    print("Predicting corners with corner detector CNN....")
    model = 'model_p2'
    architecture = 'U_Net16'
    image_size = 256
    
    # model path
    model_path = '../weights/corners/models/' + model + '/' 
    model_path = model_path + os.listdir(model_path)[0]
    
    if two_views:
        im_fold="im/"
    else:
        im_fold="im1/"
    
    #For inference of CNN
    test_loader = get_loader(image_path=images_path+im_fold,
                             image_size=image_size,
                             batch_size=1,
                             num_workers=0,
                             mode='test',
                             augmentation_prob=0.,
                             shuffle_flag=False)
    
    #load a trained model
    model = U_Net16()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.train(False)
    model.eval()
    
    #Dictionaries to storage prediction and images information    
    prediction_bin = {}
    prediction_rgb = {}
    progress = 0
    for ni, image in enumerate(test_loader):
        progress+=1
        print("Corner prediction process in {}%".format(progress*100/len(test_loader)))
        SR = model(image)
        SR_probs = torch.sigmoid(SR)
        SR_probs_arr = SR_probs.detach().numpy().reshape(image_size, image_size)
        binary_result = SR_probs_arr > .5
        image_numpy = image.detach().numpy()
        image_numpy = image_numpy[0, 0, :, :]
        image_name = test_loader.dataset.image_paths[ni].split('/')
        image_name = image_name[-1].split(".")
        image_name = image_name[0]
        
        corner = np.array(binary_result, dtype='uint8')
        prediction_bin[image_name] = corner*255
        
        cshp = corner.shape
        corner_rgb = np.zeros((cshp[0], cshp[1], 3), dtype='uint8')
        corner_rgb[:,:,2] = corner*255
        prediction_rgb[image_name] = corner_rgb
    
    #Getting original images
    list_images = os.listdir(images_path+im_fold)
    images = {}
    for img in list_images:
        images[img[:-4]] = cv.imread(images_path+ im_fold + img)
    
    #Resizing predictions to the original size
    prediction_bin_resized = {}
    prediction_rgb_resized = {}
    for key in images:
        prediction_bin_resized[key] = cv.resize(prediction_bin[key], (images[key].shape[1],images[key].shape[0]), interpolation = cv.INTER_CUBIC)
        prediction_rgb_resized[key] = cv.resize(prediction_rgb[key], (images[key].shape[1],images[key].shape[0]), interpolation = cv.INTER_CUBIC)
    
    #Overlaying prediction_rgb with original image
    overlayed_prediction = {}
    for key in images:
        overlayed_prediction[key] = cv.addWeighted(images[key], 1.0, prediction_rgb_resized[key], 0.7, 0)
    
    #saving binary and overlayed predictions
    #Check if directory exists, if not, create it
    check_dir = os.path.isdir('../results/' + data_folder)
    if not check_dir:
        os.makedirs('../results/' + data_folder)
    progress=0
    for key in images:
        progress+=1
        print("Saving corner prediction process in {}%".format(progress*100/len(images)))
        cv.imwrite('../results/' + data_folder + '/' + key + '.png', images[key])
    
    #Making prediction_bin_resized as binary
    progress=0
    for key in prediction_bin_resized:
        progress+=1
        print("Morphological operations in masks {}%".format(progress*100/len(prediction_bin_resized)))
        prediction_bin_resized[key] = prediction_bin_resized[key]>0
        prediction_bin_resized[key] = erosion(prediction_bin_resized[key], disk(6))
    
    #Extracting information of centroids of the segmentated objects as the opening corners
    opening_information = {}
    progress=0
    for key in images:
        progress+=1
        print("labeling and regionprops. Saving final corners process in {}%".format(progress*100/len(images)))
        label_corners = label(prediction_bin_resized[key])
        regions_corn = regionprops(label_corners)
        opening_information[key]={}
        opening_information[key]['corner_centroids'] = [region.centroid for region in regions_corn]
        
        #Drawing keypoints
        cent = opening_information[key]['corner_centroids']
        for c in cent:
            radius = int(overlayed_prediction[key].shape[0]/100)
            cv.circle(overlayed_prediction[key], (int(c[1]),int(c[0])), radius=radius, color=(0,0,255), thickness=-1)
        cv.imwrite('../results/' + data_folder + '/corners_' + key + '.png', overlayed_prediction[key])
        
    
    #SSD DETECTOR
    print("Parsing with SSD NN....")
    #Load pretrained model
    best_model = "../weights/ssd/saved_models/BEST_model_ssd300.pth.tar"
    model = ssd.build_ssd(num_classes = 4)
    model.load_state_dict(best_model["model_state_dict"])
    device = "cuda"
    model = model.to(device)
    
    imgs_or = glob.glob("../data/" + data_folder + "/images/im1/*")
    imgs_or.sort()
    imgs = glob.glob("../results/" + data_folder  + "/corners*") 
    imgs.sort()
    background = glob.glob("cnn/ssd/background.png")[0]
    img_names = list([])
    for im in imgs:
        img_names.append(im.split('/')[-1].split('.')[0])
    
    #SDD predictions
    predictions = list([])
    progress=0
    for i, im in enumerate(imgs):
        progress+=1
        print("SDD prediction process in {}%".format(progress*100/len(imgs)))
        img, bboxes, labels, scores = predict_objects(model, imgs_or[i], min_score=0.2, max_overlap = 0.01, top_k=200)
        predictions.append((im,bboxes,labels,scores))
        annotated_img = draw_detected_objects(im, bboxes, labels, scores)
        labels_img = draw_detected_objects(background, bboxes, labels, scores)
        cv.imwrite("../results/" + data_folder + "/ssd_{}.png".format(img_names[i]),cv.cvtColor(annotated_img,cv.COLOR_BGR2RGB))
        key = im.split('/')[-1].replace("corners_","")[:-4]
        opening_information[key]["bboxes"] = (bboxes, labels, "region")
    
    
    return opening_information


def op_detector_labeler(opening_information):
    """ 
    Deletes keypoints detected outside the building and adds a label to each 
    keypoint if belong to an opening (if it is used boundig boxes approach).
    Box 10% bigger to fit some points that can be outside.

    Args:
        opening_information (dict): information of detected openings detected by deep learning models
    
    Returns:
        opening_labeled_corners (dict): labeled opening corners 

    """

    opening_labeled_corners = {}
    progress=0
    for key in opening_information:
        progress+=1
        print("labeling opening corners process in {}%".format(progress*100/len(opening_information)))
        label_name = 0
        corner_centroids_cp = list([])
        label_corn_key = list([])
        for i, box in enumerate(opening_information[key]['bboxes'][0]):
            if type(box)==torch.Tensor:
                box = box.detach().numpy()
            
            centroid = .5*np.array([box[0]+box[2], box[1]+box[3]])
            height = 1.3*np.array(box[3] - box[1]) 
            width = 1.3*np.array(box[2] - box[0])
            minr = centroid[1] - .5*height
            maxr = centroid[1] + .5*height
            minc = centroid[0] - .5*width
            maxc = centroid[0] + .5*width
            
            #deleting points outside of the building 
            if opening_information[key]['bboxes'][1][i] =='building':
                for point in opening_information[key]['corner_centroids']:
                    corner_centroids_cp.append(np.array(point)) 
            else:
                if i==0:
                    corner_centroids_cp = opening_information[key]['corner_centroids'].copy()
                c = 0 #to check if a square has or not corners inside. if so, label_name actualizes
                a = 0 #constant to vary size of box
                while a<.5 and c<=4:
                    #c<=4 to be sure at least 4 corners are taken
                    #a<4 to guarantee only 4 increments
                    temp_corn = list([])
                    c = 0
                    minr = minr - .5*(a*height) #if there are no points, increment size in the box to try to catch some
                    maxr = maxr + .5*(a*height)
                    minc = minc - .5*(a*width)
                    maxc = maxc + .5*(a*width)
                    for point in corner_centroids_cp:
                        cond1 = minr < point[0] < maxr 
                        cond2 = minc < point[1] < maxc
                        if cond1 and cond2:
                            temp_corn.append(np.array(point))
                            c+=1
                    a+=.1
                if len(temp_corn)!=0:
                    for ccorn in temp_corn:
                        label_corn_key.append((ccorn, label_name))
                label_name +=1 #in case of opening does not have any points inside
                    
        opening_labeled_corners[key] = label_corn_key
    
    
    return opening_labeled_corners


def corn_filt_1_1(box, current_open_corners):
    """
        Filter1: The corner points selected are the ones closest to the box corners.
        box corners (top,bottom - left,right). For ssd.
    Args:
        box (array): bounding box corners
        current_open_corners (array): corners that correspond to each opening

    Returns:
        current_open_corners_filtered (array): modified opening corners
        box_corners (array) : bounding box containing corners
    """
    
    if type(box)==torch.Tensor:
        box = box.detach().numpy()
    box_tl = np.array((box[1],box[0]))
    box_tr = np.array((box[1],box[2]))
    box_bl = np.array((box[3],box[0]))
    box_br = np.array((box[3],box[2]))                                  
    box_corners = np.array([box_tl,box_tr,box_bl,box_br])

    current_open_corners_filtered = list([])
    #Filter1: The corner points selected are the ones closest to the box corners.
    for c in box_corners:
        distances = [np.linalg.norm(c-p[0]) for p in current_open_corners] 
        min_index = np.where(distances == np.amin(distances))
        current_open_corners_filtered.append(current_open_corners[min_index][0][0])

    return current_open_corners_filtered, box_corners


def corn_filt_1_2(box, building_corners, bound_box='region'):
    """
    Filter1:Initial opening corners based in distance between full corners and boxes. For unet.
    
    Args:
        box (array): bounding box corners
        building_corners (array): opening corners
        bound_box (str, optional): region or poly. Method to define bounding box. Defaults to 'region'.

    Returns:
        current_open_corners_filtered (array): modified opening corners
    """

    if bound_box=='region':
        if type(box)==torch.Tensor:
            box = box.detach().numpy()                    
        box_tl = np.array((box[1],box[0]))
        box_tr = np.array((box[1],box[2]))
        box_bl = np.array((box[3],box[0]))
        box_br = np.array((box[3],box[2]))                                  
        box_corners = np.array([box_tl,box_tr,box_bl,box_br])
            
    thresh_1 = 0.5*np.min([[np.linalg.norm(box_tr-box_tl)],[np.linalg.norm(box_tl-box_bl)],\
                           [np.linalg.norm(box_bl-box_br)],[np.linalg.norm(box_br-box_tr)]]) #!
    
    
    current_open_corners_filtered = list([])
    distances_full = np.abs(scipy.spatial.distance.cdist(box_corners,building_corners)) #calculate distances between points in two points sets
    distances = np.min(distances_full,axis=1)
    for ii, d in enumerate(distances):
        if d<thresh_1:
            ind = np.where(distances_full[ii]==d)
            current_open_corners_filtered.append(building_corners[ind][0])
        else:
            current_open_corners_filtered.append(box_corners[ii])    

    return current_open_corners_filtered

def corn_filt_2(current_open_corners_filtered, box_corners):
    """
        Filter2: when two box corners select a single poing. The furthest is leaving as opening corner.
        this filter is passed twice in case there are two cases of repetead points

    Args:
        current_open_corners_filtered (array): modified opening corners
        box_corners (array) : bounding box containing corners

    Returns:
        current_open_corners_filtered (array): modified opening corners
    """    
    
    for ii in range(2):    
        equal = 0
        counter = 0
        while equal==0 and counter!=4:
            for p1 in current_open_corners_filtered:
                comparition = np.array([np.linalg.norm(p1-p2) for p2 in current_open_corners_filtered])
                comp_idx = np.where(comparition==0.0)
                if len(comp_idx[0])==2:
                    equal=1
                    break
            counter+=1
        if equal == 1:
            dist1 = np.linalg.norm(box_corners[comp_idx[0][0]] - current_open_corners_filtered[comp_idx[0][0]])
            dist2 = np.linalg.norm(box_corners[comp_idx[0][1]] - current_open_corners_filtered[comp_idx[0][1]])
            if dist1<dist2:
                current_open_corners_filtered[comp_idx[0][1]] = box_corners[comp_idx[0][1]]
            else:
                current_open_corners_filtered[comp_idx[0][0]] = box_corners[comp_idx[0][0]]
        
    return current_open_corners_filtered

def corn_filt_3(current_open_corners_filtered):
    """
        Filter3: if from the 4 points there is an outlier, rectify it
        to decide if is outlier or not. check=0 - outlier, =1 help to relocate outlier
        check if x and y coordinates are so diferent for a point. If yes, it is outlier
        op: opening t:top b:bottom l:left r:right

    Args:
        current_open_corners_filtered (array): modified opening corners

    Returns:
        current_open_corners_filtered (array): modified opening corners
    """
        
    op_tl_check = 0
    op_tr_check = 0
    op_bl_check = 0
    op_br_check = 0
    op_tl = current_open_corners_filtered[0]
    op_tr = current_open_corners_filtered[1]
    op_bl = current_open_corners_filtered[2]
    op_br = current_open_corners_filtered[3]
    op_corners = np.array([op_tl,op_tr,op_bl,op_br])
    side1 = np.linalg.norm(op_tl-op_tr)
    side2 = np.linalg.norm(op_tl-op_bl)
    side3 = np.linalg.norm(op_br-op_tr)
    side4 = np.linalg.norm(op_br-op_bl)
    
    threshold = .2*np.abs(np.min((side1,side2,side3,side4))) #!
    
    op_tl_check = np.sum(np.abs(op_tl - op_corners) < threshold) - 2
    op_tr_check = np.sum(np.abs(op_tr - op_corners) < threshold) - 2
    op_bl_check = np.sum(np.abs(op_bl - op_corners) < threshold) - 2
    op_br_check = np.sum(np.abs(op_br - op_corners) < threshold) - 2
    check = np.array([(op_tl_check,op_tr_check,op_bl_check,op_br_check)])
    
    if np.sum(check) == 4:  
        if op_tl_check==0:
            current_open_corners_filtered[0][0] = op_tr[0]
            current_open_corners_filtered[0][1] = op_bl[1]
        elif op_tr_check==0:
            current_open_corners_filtered[1][0] = op_tl[0]
            current_open_corners_filtered[1][1] = op_br[1]
        elif op_bl_check==0:
            current_open_corners_filtered[2][0] = op_br[0]
            current_open_corners_filtered[2][1] = op_tl[1]
        elif op_br_check==0:
            current_open_corners_filtered[3][0] = op_bl[0]
            current_open_corners_filtered[3][1] = op_tr[1]
    elif np.sum(check) == 6:
        pos_out = np.where(check[0]==1)
        pos_out1 = op_corners[pos_out[0][0]]
        pos_out2 = op_corners[pos_out[0][1]]
        
        d1 = np.sum([np.linalg.norm(pos_out1-op_corners)])
        d2 = np.sum([np.linalg.norm(pos_out2-op_corners)])
        
        if d1>d2:
            out = pos_out[0][0]
        else:
            out = pos_out[0][1]
        
        if out==0:
            current_open_corners_filtered[0][0] = op_tr[0]
            current_open_corners_filtered[0][1] = op_bl[1]
        elif out==1:
            current_open_corners_filtered[1][0] = op_tl[0]
            current_open_corners_filtered[1][1] = op_br[1]
        elif out==2:
            current_open_corners_filtered[2][0] = op_br[0]
            current_open_corners_filtered[2][1] = op_tl[1]
        elif out==3:
            current_open_corners_filtered[3][0] = op_bl[0]
            current_open_corners_filtered[3][1] = op_tr[1]

    return current_open_corners_filtered

def corn_filt_4(face_open_area, face_open_corners, cte_fil4 = .2):
    """
        Filter4: Rid off small openings. If area of an opening is
        smaller than an area percentage of the biggest in the face (15% --adjust)
        Maybe compare with second biggest area (avoid compare agains doors)
        if there are al least two areas
    Args:
        face_open_area (array): openings' areas in px
        face_open_corners (array): opening corners for facade

    Returns:
        face_open_corners_ord (array): updated ordered opening corners getting rid off small openings
    """
    
    if len(face_open_area)>1:
        face_open_area = np.array(face_open_area)
        cp_face_open_area = np.copy(face_open_area)
        cp_face_open_area.sort()
        s_biggest_area = cp_face_open_area[-2]
        cte = cte_fil4 #!
        small_openings = np.where(face_open_area<cte*s_biggest_area) 
        for indx in sorted(small_openings[0], reverse=True):
            del face_open_corners[indx]
    
    #Organizing corners as is required in opening_projector.py
    face_open_corners = np.array(face_open_corners)
    sh = face_open_corners.shape
    face_open_corners = face_open_corners.reshape((sh[0]*sh[1],2))
    face_open_corners[:,[0, 1]] = face_open_corners[:,[1, 0]] 
    
    #Checking the correct order (tl-tr-bl-br). If not,re-order it in face_open_corners_ord
    face_open_corners_ord = np.zeros(face_open_corners.shape)
    for i in range(int(len(face_open_corners)/4)):
        temp = face_open_corners[4*i:4*(i+1)]
        x_sorted = np.sort(temp[:,0],axis=0)
        y_sorted = np.sort(temp[:,1],axis=0)
        for j, pt in enumerate(temp):
            check1 = np.where(pt[0]==x_sorted)
            check2 = np.where(pt[1]==y_sorted)
            if check1[0][0]<=1:
                if check2[0][0]<=1:
                    face_open_corners_ord[4*i] = pt
                else:
                    face_open_corners_ord[4*i+2] = pt
            else:
                if check2[0][0]<=1:
                    face_open_corners_ord[4*i+1] = pt
                else:
                    face_open_corners_ord[4*i+3] = pt
    
    return face_open_corners_ord

def corn_filt_4_5(face_open_corners_ord, facade):
    """_summary_

    Args:
        face_open_corners_ord (array): updated ordered opening corners getting rid off small openings
        facade (array): binary facade prediction

    Returns:
        face_open_corners_ord_fac (array): filtered openings deleting outsider of facade
        main_facade (array): binary mask with the facade area
    """
    
    #Segment facade and filter the points that lay outside this area
    
    #Extracting information of regions to select biggest area (in case of detecting more than 1 facade)    
    label_op = label(facade)
    regions_op = regionprops(label_op)
    areas = np.array([region.area for region in regions_op])
    max_area = np.max(areas)
    ind_max_area = np.where(areas==max_area)            
    reg_max_area = regions_op[ind_max_area[0][0]]
    
    #Array with the facade with maximum area
    main_facade = np.zeros_like(facade)
    coords = reg_max_area.coords
    main_facade[(coords[:,0], coords[:,1])] = True
    
    face_open_corners_ord_fac = np.copy(face_open_corners_ord)
    check = main_facade[(face_open_corners_ord_fac[:,1].astype('int')-1, face_open_corners_ord_fac[:,0].astype('int')-1)] #-1 as sometimes box in the contour           
    
    #if at least 2 corners lay inside the facade, do not delete points outside
    for i in range(int(len(check)/4)):
        if np.sum(check[4*i:4*(i+1)])>=2:
            check[4*i:4*(i+1)]=1
            
    face_open_corners_ord_fac = face_open_corners_ord_fac[np.where(check==1)]
        
    return face_open_corners_ord_fac, main_facade

def corn_filt_5(face_open_corners_ord, img, key, data_folder):
    """
    Filter5: Applying linear regression to align points vertically and horizontally

    Args:
        face_open_corners_ord (array): updated ordered opening corners getting rid off small openings
        img (array): facade image
        key (array): facade image name
        data_folder (atr): input data folder

    Returns:
        face_open_corners_ord_al (array): updated aligned opening corners after linear regression
    """
    
    cte = .2 #!
    edge_1 = face_open_corners_ord[1][0] - face_open_corners_ord[0][0]
    edge_2 = face_open_corners_ord[2][1] - face_open_corners_ord[0][1]
    
    threshold = cte*np.abs(np.min((edge_1,edge_2)))
    
    ###ALIGNING TO LINES
    #Vertical alignment
    vert_checker = np.zeros(face_open_corners_ord[:,0].shape) #Checker to identify points already aligned
    #if there is problems with not meeting the threshold, keep initial
    face_open_corners_ord_al = np.copy(face_open_corners_ord)
    skeleton = np.copy(img)
    for ii, pt in enumerate(face_open_corners_ord[:,0]):
        if vert_checker[ii] == 0:
            distances = np.abs(pt - face_open_corners_ord[:,0])
            meet_thr = np.where(distances<threshold)
            if ii%2==0: #if are in the left corners
                left_meet_thr = np.where(meet_thr[0]%2==0)
                meet_thr = meet_thr[0][left_meet_thr]
            else: #right
                right_meet_thr = np.where(meet_thr[0]%2!=0)
                meet_thr = meet_thr[0][right_meet_thr]
            vert_checker[meet_thr] = 1
            x_lr = face_open_corners_ord[:,0][meet_thr]
            y_lr = face_open_corners_ord[:,1][meet_thr]
            if len(x_lr)>2: 
                #To guarantee to do regretion (if not, put the same coordinate)
                #the coordinates x and y are swapped here because the regression produce vertical lines
                #that in some case give problem as a function cannot have 2 y coordinates for same x (approx problem)
                #it would rather to do the regression horizontally.
                face_open_corners_ord_al[:,1][meet_thr], face_open_corners_ord_al[:,0][meet_thr] = line_adjustor(y_lr,x_lr)
            else: 
                #If just two poits, they are from same openints. Put the same x coordinate
                face_open_corners_ord_al[:,0][meet_thr] = np.mean(face_open_corners_ord[:,0][meet_thr])
            pts = face_open_corners_ord_al[:,:2][meet_thr].astype(int)
            cv.polylines(skeleton, [pts], False, (0,0,255),10)
    #Horizontal alignment
    hori_checker = np.zeros(face_open_corners_ord[:,0].shape) #Checker to identify points already aligned
    for ii, pt in enumerate(face_open_corners_ord_al[:,1]):
        if hori_checker[ii] == 0:
            distances = np.abs(pt - face_open_corners_ord_al[:,1])
            meet_thr = np.where(distances<threshold)
            if ii%4==0 or ii%4==1: #top corners
                top_meet_thr1 = np.where(meet_thr[0]%4==0)
                top_meet_thr2 = np.where(meet_thr[0]%4==1)
                top_meet_thr = np.concatenate((top_meet_thr1,top_meet_thr2),axis=1)
                meet_thr = meet_thr[0][top_meet_thr]
            else: #bottom
                bottom_meet_thr1 = np.where(meet_thr[0]%4==2)
                bottom_meet_thr2 = np.where(meet_thr[0]%4==3)
                bottom_meet_thr = np.concatenate((bottom_meet_thr1,bottom_meet_thr2),axis=1)
                meet_thr = meet_thr[0][bottom_meet_thr]
            hori_checker[meet_thr] = 1
            x_lr = face_open_corners_ord_al[:,0][meet_thr][0] #using the horizontal aligned positions
            y_lr = face_open_corners_ord_al[:,1][meet_thr][0]
            if len(x_lr)>2:
                face_open_corners_ord_al[:,0][meet_thr], face_open_corners_ord_al[:,1][meet_thr] = line_adjustor(x_lr,y_lr)
            else: 
                #If just two poits, they are from same openints. Put the same x coordinate
                face_open_corners_ord_al[:,1][meet_thr] = np.mean(face_open_corners_ord_al[:,1][meet_thr])
                
            pts = face_open_corners_ord_al[:,:2][meet_thr][0].astype(int)
            cv.polylines(skeleton, [pts], False, (0,0,255),10)
    
    #saving skeleton 
    cv.imwrite('../results/' + data_folder + '/skeleton_' + key + '.png', skeleton)

    return face_open_corners_ord_al


def op_area(current_open_corners_filtered):
    """
    Computes opening areas using polygon vertices as
    |(x1y2−y1x2) + (x2y3−y2x3)..... + (xny1−ynx1)/2|
    for counter clock wise ordered nodes

    Args:
        current_open_corners_filtered (array): openings corners

    Returns:
        current_open_arra: openings areas
    """
    
    #finding area formed by the 4 points in the opening. It will help
    #in the filter 4
    op_tl = current_open_corners_filtered[0]
    op_tr = current_open_corners_filtered[1]
    op_br = current_open_corners_filtered[3]
    op_bl = current_open_corners_filtered[2]
    cons1 = (op_tl[1]*op_tr[0] + op_tr[1]*op_br[0] + op_br[1]*op_bl[0] + op_bl[1]*op_tl[0]) 
    cons2 = (op_tl[0]*op_tr[1] + op_tr[0]*op_br[1] + op_br[0]*op_bl[1] + op_bl[0]*op_tl[1]) 
    current_open_area = .5* (cons1 - cons2)

    return current_open_area

def op_detector_4p_1(opening_information, opening_labeled_corners, images_path, data_folder, two_views, cte_fil4 = .2):
    """_summary_
        Function to select just 4 points (4p) by opening according the ones nearest to the 
        boxes corners
    Args:
        opening_information (dict): information from deep learning opening detection
        opening_labeled_corners (dict): corners labeled according to opening that belong
        data_folder (str): folder name containing building data
        images_path (str): path to the input images
        two_views (bool, optional): if true computes detection for two facade views. Defaults to False.
    Returns:
        building_open_corners (dict): openings given as 4 corners that represent them
    """
   
    if two_views:
        im_fold="im/"
    else:
        im_fold="im1/"
    
    #Detect facade for filtering
    fac_prediction = fac_segmentation(data_folder, images_path+'im1/', out_return=True)    
    
    building_open_corners = {}
    progress = 0
    for key in opening_information:
        progress+=1
        print("4points process in {}%".format(progress*100/len(opening_information)))
        face_open_corners = list([])
        face_open_area = list([])
        build_box = 0 #if there is a box with building, this will be 1. To match with the oppening labels.
        building_corners = np.array(opening_labeled_corners[key])
        
        #Reading image
        img = cv.imread(images_path + im_fold + key + '.jpg')
        if not type(img)==np.ndarray:
            img = cv.imread(images_path + im_fold + key + '.JPG')
        if not type(img)==np.ndarray:
            img = cv.imread(images_path + im_fold + key + '.png')
        
        for i, box in enumerate(opening_information[key]['bboxes'][0]):
            if opening_information[key]['bboxes'][1][i]=='building':
                build_box = 1
            else:
                
                current_label = i - build_box
                meet_label = np.where(building_corners[:,1]==current_label)
                current_open_corners = building_corners[meet_label,0].T
                
                if len(current_open_corners)>=2:
                    #Filter1: The corner points selected are the ones closest to the box corners.
                    current_open_corners_filtered, box_corners = corn_filt_1_1(box, current_open_corners)
                      
                    #Filter2: if corners select a single point of box, change it.
                    current_open_corners_filtered = corn_filt_2(current_open_corners_filtered, box_corners)
                    
                    #Filter3: Rectifying outliers
                    current_open_corners_filtered = corn_filt_3(current_open_corners_filtered)
                    #Appending open_corners_filt
                    face_open_corners.append(current_open_corners_filtered)
                    
                    #Opening area
                    current_open_area = op_area(current_open_corners_filtered) 
                    face_open_area.append(current_open_area)
        
        #Filter4: Rid off small openings
        face_open_corners_ord = corn_filt_4(face_open_area, face_open_corners, cte_fil4 = cte_fil4)
        
        #Filter4.5: Rid off openings outside the main facade
        face_open_corners_ord_fac, main_facade = corn_filt_4_5(face_open_corners_ord, fac_prediction[key])
      
        #Filter 5: Aligning points to vertical and horizontal lines
        face_open_corners_ord_al = corn_filt_5(face_open_corners_ord_fac, img, key, data_folder)

        #Saving image with the filtered corners detected
        ccc=0
        for c in face_open_corners_ord_al:
            radius = int(img.shape[0]/100)
            cv.circle(img, (int(c[0]),int(c[1])), radius=radius, color=(0,0,255), thickness=-1)
            ccc+=1
        cv.imwrite('../results/' + data_folder + '/corn_filtered_' + key + '.png', img)        

        building_open_corners[key] = (face_open_corners_ord_al, main_facade)
        
    return building_open_corners


def op_detector_4p_2(opening_information, images_path, data_folder, bound_box, two_views, cte_fil4 = .2):
    """
        Function to select just 4 points (4p) by opening according the ones nearest to the 
        boxes corners. This function is used when the deep learning model is the unet
    Args:
        opening_information (dict): information from deep learning opening detection
        data_folder (str): folder name containing building data
        images_path (str): path to the input images
        bound_box (str): region or poly. Method to define bounding box
        two_views (bool, optional): if true computes detection for two facade views. Defaults to False.
    Returns:
        building_open_corners (dict): openings given as 4 corners that represent them
    """
    
    if two_views:
        im_fold="im/"
    else:
        im_fold="im1/"    
    
    #Detect facade for filtering
    fac_prediction = fac_segmentation(data_folder, images_path+im_fold, out_return=True)
    
    building_open_corners = {}
    progress = 0
    for key in opening_information:
        progress+=1
        print("4points process in {}%".format(progress*100/len(opening_information)))
        face_open_corners = list([])
        face_open_area = list([])
        building_corners = np.array(opening_information[key]['corner_centroids'])
        
        #Reading image
        img = cv.imread(images_path + im_fold + key + '.jpg')
        if not type(img)==np.ndarray:
            img = cv.imread(images_path + im_fold + key + '.JPG')
        if not type(img)==np.ndarray:
            img = cv.imread(images_path + im_fold + key + '.png')
        
        for i, box in enumerate(opening_information[key]['bboxes'][0]):
            if opening_information[key]['bboxes'][1][i]=='building':
                continue
            else:
                #Filter1:Initial opening corners based in distance between full corners and boxes
                current_open_corners_filtered = corn_filt_1_2(box, building_corners, bound_box=bound_box)
                
                #Filter3: Rectifying outliers
                #print("pro")
                current_open_corners_filtered = corn_filt_3(current_open_corners_filtered)

                #Appending open_corners_filt
                face_open_corners.append(current_open_corners_filtered)
                
                #Opening area
                current_open_area = op_area(current_open_corners_filtered) 
                face_open_area.append(current_open_area)
                
        #Filter4: Rid off small openings
        face_open_corners_ord = corn_filt_4(face_open_area, face_open_corners, cte_fil4 = cte_fil4)
                
        #Filter4.5: Rid off openings outside the main facade
        face_open_corners_ord_fac, main_facade = corn_filt_4_5(face_open_corners_ord, fac_prediction[key])
        #print("3---", len(face_open_corners_ord_fac))
        
        #Filter 5: Aligning points to vertical and horizontal lines
        face_open_corners_ord_al = corn_filt_5(face_open_corners_ord_fac, img, key, data_folder)
        
        #Saving image with the filtered corners detected
        ccc=0
        for c in face_open_corners_ord_al:
            radius = int(img.shape[0]/100)
            cv.circle(img, (int(c[0]),int(c[1])), radius=radius, color=(0,0,255), thickness=-1)
            ccc+=1
        cv.imwrite('../results/' + data_folder + '/corn_filtered_' + key + '.png', img)        
        
        building_open_corners[key] = (face_open_corners_ord_al, main_facade)
   
    return building_open_corners


def main_op_detector(data_folder, images_path, bound_box, op_det_nn, two_views=False, cte_fil4 = .2):
    """This function returns the openings detected by deep learning models as
    sets of four points for each opening.

    Args:
        data_folder (str): folder name containing building data
        images_path (str): path to the input images
        bound_box (str): method to compute bounding box regionor or poly
        op_det_nn (str): method to detect openings unet or ssd
        two_views (bool, optional): if true computes detection for two facade views. Defaults to False.

    Returns:
        opening_4points (dict): dictionary with opening information with four opening corners.
    """
    
    if op_det_nn == 'ssd':
        opening_information = op_detector_1(data_folder, images_path, two_views)
        opening_labeled_corners = op_detector_labeler(opening_information)
        opening_4points = op_detector_4p_1(opening_information, opening_labeled_corners, images_path, data_folder, two_views, cte_fil4 = cte_fil4)
    elif op_det_nn == 'unet':
        opening_information = op_detector_2(data_folder, images_path, bound_box, two_views)
        opening_4points = op_detector_4p_2(opening_information, images_path, data_folder, bound_box, two_views, cte_fil4 = cte_fil4)

    return opening_4points
