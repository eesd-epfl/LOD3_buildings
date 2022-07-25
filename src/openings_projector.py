#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:35:45 2020

This script uses camera data provided by meshroom
software located in the cameras.sfm file (StructureFromMotion folder)
See function added in CameraInit.py module (nodes) in meshroom.
It was implemented a script to save intrinsic and poses information
as dictionaries in json files.

This script contains the codes to generate LOD3 models.
The codes are based on "Generation LOD3 models from structure-from-motion and semantic segmentation" 
by Pantoja-Rosero et., al.
https://doi.org/10.1016/j.autcon.2022.104430

@author: pantoja
"""

from PIL import Image
import numpy as np
import pylab as pylab
import os
import matplotlib.pyplot as plt
import camera
import homography
import cv2
import camera
import sfm
from projection_tools import *

def track_kps(data_folder, imm1, imm2, im_n1, im_n2, kps_imm1=None):
    """

    #LKT to find other points
    #Lukas Kanade Tracker. Used to finde detected opening corners in im1
    over im2 using sequential frames

    Args:
        data_folder (str): input data folder name
        imm1 (array): image view1
        imm2 (array): image view2
        im_n1 (str): image name view1
        im_n2 (str): image name view2
        kps_imm1 (array, optional): points to be mapped from view1 to view2. Defaults to None.

    Returns:
        good_new (array): points mapped from view1 to view2 using LKT
    """
    
    # Create params for lucas kanade optical flow
    lk_params = dict( winSize  = (60,60), maxLevel = 10, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    #Finding kps to be traked (if not given)
    gray1 = cv2.cvtColor(imm1,cv2.COLOR_RGB2GRAY)
    shp1 = gray1.shape[:2]
    gray2 = cv2.cvtColor(imm2,cv2.COLOR_RGB2GRAY)
    shp2 = gray2.shape[:2]
    
    #if shapes are different, resize 2nd image to perform LKT. Later is needed to rescale points.
    if shp1!=shp2:
        gray2 = cv2.resize(gray2, (shp1[1],shp1[0]))
    
    if kps_imm1 is None:
        # Create params for ShiTomasi corner detection
        feature_params = dict( maxCorners = 100,
                              qualityLevel = 0.3,
                              minDistance = 7,
                              blockSize = 7 )
        p0 = cv2.goodFeaturesToTrack(gray1, mask = None, **feature_params)
    else:
        p0 = np.round(kps_imm1.reshape((len(kps_imm1),1,2))).astype('float32')
    
    # Create some random colors
    color = np.random.randint(0,255,(10000,3))
    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)
    # Select good points (if there is flow)
    good_new = p1[st==1]
    good_old = p0[st==1]
    
    #rescale if shapes are different
    if shp1!=shp2:
        scalex = shp2[1]/shp1[1]
        scaley = shp2[0]/shp1[0]
        good_new[:,0] = scalex * good_new[:,0]
        good_new[:,1] = scaley * good_new[:,1]    
    
    # Draw the tracks
    # Create a mask image for drawing purposes
    frame1 = np.copy(imm1)
    frame2 = np.copy(imm2)
    for i, (new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        print(frame1,(c,d),20,color[i].tolist(),-1)
        frame1 = cv2.circle(frame1,(int(c),int(d)),20,color[i].tolist(),-1)
        frame2 = cv2.circle(frame2,(int(a),int(b)),20,color[i].tolist(),-1)
    # Display the image with the flow lines
    #Save images with tracks
    cv2.imwrite('../results/'+data_folder+'/KLT_frame1_' + im_n1+ '.png', cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
    cv2.imwrite('../results/'+data_folder+'/KLT_frame2_' + im_n2+ '.png', cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
    
    plt.figure()
    plt.imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
    plt.figure()
    plt.imshow(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
    
    return good_new




def imgs_corners(im1, im2, imm1, imm2, i, keypoints_path, how2get_kp, opening_4points, data_folder, pp=None, save_op_corners=True):
    """

        Geting opening corners coordinates on images im1 and im2

    Args:
        im1 (list): list of input images view 1 for each facade
        im2 (list): list of input images view 2 for each facade
        imm1 (array): image for view one of facade
        imm2 (array): image for view two of facade
        i (int): facade id given by loop
        keypoints_path (str): path to keypoints
        how2get_kp (int): method to get opening corners to triangulate 0:from CNN detector+homography, 1: loading npy files, 2:clicking on two views, 3: v1 from npy+v2 with homography, 4:from CNN detector+homography
        opening_4points (dict): opening corners for each facade
        pp (int, optional): list with number of corners to be triangulated for each facade - used if how2get_kp==2
        data_folder (str): path to input data folder
        save_op_corners (bool, optional): _description_. Defaults to True.
    Returns:
        x1 (array): opening corners coordinates view1
        x2 (array): opening corners coordinates view2
        H (array): homography matrix to transform x1 to x2

    """
    v_H = 0
    #Depending of method selected on how2get_kp if findes the opening corner correspondences
    #for the 2 views
    if how2get_kp == 0:
        x1 = opening_4points[im1[i]][0]
        x1 = homography.make_homog(x1.T)
        mask_facade = opening_4points[im1[i]][1]
        H = fast_homography(imm1,imm2,im1[i], im2[i], data_folder, mask_facade=mask_facade, save_im_kps=True)
        x2 = np.dot(H,x1)
        x2[0,:] = x2[0,:]/x2[2,:]
        x2[1,:] = x2[1,:]/x2[2,:]
        x2[2,:] = x2[2,:]/x2[2,:]
        np.save(keypoints_path + "x1_{}".format(i), x1)
        np.save(keypoints_path + "x2_{}".format(i), x2)
        v_H=1
    elif how2get_kp == 1:
        #in case of have the corner points in a npy file
        x1 = np.load(keypoints_path + "x1_{}.npy".format(i))
        x2 = np.load(keypoints_path + "x2_{}.npy".format(i))
        H = None
    elif how2get_kp == 2:    
        #Selecting  correspondence points to be projected from image 1
        plt.figure()
        plt.imshow(imm1)
        print('Please click {} points'.format(pp[i]))
        x1 = np.array(pylab.ginput(pp[i],200))
        print('you clicked:',x1)
        plt.close()
           
        #Selecting  points to be projected from image 2
        plt.figure()
        plt.imshow(imm2)
        print('Please click {} points'.format(pp[i]))
        x2 = np.array(pylab.ginput(pp[i],200))
        print('you clicked:',x2)
        plt.close()
        
        # make homogeneous and normalize with inv(K)
        x1 = homography.make_homog(x1.T)
        x2 = homography.make_homog(x2.T)
        
        np.save(keypoints_path + "x1_{}".format(i), x1)
        np.save(keypoints_path + "x2_{}".format(i), x2)
        H = None
    
    elif how2get_kp ==3:
        x1 = np.load(keypoints_path + "x1_{}.npy".format(i))
        # make homogeneous and normalize with inv(K)
        x1 = homography.make_homog(x1.T)
        mask_facade = opening_4points[im1[i]][1]
        H = fast_homography(imm1,imm2,im1[i], im2[i], data_folder, mask_facade=mask_facade, save_im_kps=False)
        x2 = np.dot(H,x1)
        x2[0,:] = x2[0,:]/x2[2,:]
        x2[1,:] = x2[1,:]/x2[2,:]
        x2[2,:] = x2[2,:]/x2[2,:]
        v_H=1
        
    elif how2get_kp ==4:
        #LKT to find other points
        x1 = opening_4points[im1[i]][0]
        x2 = track_kps(data_folder, imm1,imm2,im1[i], im2[i], kps_imm1=x1)
        x1 = homography.make_homog(x1.T)
        x2 = homography.make_homog(x2.T)   
        np.save(keypoints_path + "x1_{}".format(i), x1)
        np.save(keypoints_path + "x2_{}".format(i), x2)
        v_H=0
        H = None
        
    
    
    if save_op_corners:
        #view1 
        radius = int(imm1.shape[0]/100)
        im_c1 = np.copy(imm1)
        im_c1 = cv2.cvtColor(im_c1, cv2.COLOR_RGB2BGR)
        for c in x1.T:
            cv2.circle(im_c1, (int(c[0]),int(c[1])), radius=radius, color=(0,0,255), thickness=-1)
        cv2.imwrite('../results/' + data_folder + '/opening_corners_' + im1[i] + '.png', im_c1)
        #view2
        radius = int(imm2.shape[0]/100)
        im_c2 = np.copy(imm2)
        im_c2 = cv2.cvtColor(im_c2, cv2.COLOR_RGB2BGR)
        for c in x2.T:
            cv2.circle(im_c2, (int(c[0]),int(c[1])), radius=radius, color=(0,0,255), thickness=-1)
        cv2.imwrite('../results/' + data_folder + '/opening_corners_' + im2[i] + '.png', im_c2)    
    
    return x1, x2, H, v_H

def proj_pt2pl(model,pt):
     '''
     Projectin pts to plane 

     Parameters
     ----------
     model : plane ransac model
     pt : point to be projected to plane

     Returns
     -------
     pt_p : pt projection on plane
     
     model -> x: [a b c]
           -> a + bx + cy + dz = 0  (here d=-1)
           n_p: normal to plane (b,c,d)
     '''
     pt1 = np.copy(pt)
     pt1[2] = model[0] + model[1]*pt[0] + model[2]*pt[1] #point on plane
     pt_pt1 = pt - pt1
     if np.linalg.norm(pt_pt1)==0:
         pt_p=np.copy(pt)
     else:
         n_1 = pt_pt1/np.linalg.norm(pt_pt1)
         n_p = np.array([model[1] ,model[2] ,-1])
         n_p = n_p/np.linalg.norm(n_p)
         cos_alpha = np.dot(n_1,n_p)
         pt_ptp = np.linalg.norm(pt_pt1)*cos_alpha
         pt_p = pt - pt_ptp*n_p
     
     return pt_p


def proj_op2plane(polyfit_path, X, dense):
    """

        Find a plane paralel to facade with a normal similar to mean of normals
        of all openings. Project corners to that plane. 
        Project 4 points of the opening to a single plane
        Calculate normals of all openings and find the closest to the 
        facade normal. Take the openings to a plane with the same normal.

    Args:
        polyfit_path (str): path to input folder with LOD2 model
        X (array): opening coorners in 3D
        dense (bool): if true, it loads LOD2 model obtained from dense pt cloud
    Returns:
        X (array): 3D opening coordinates projected to a plane
        n_pl (array): normal parameters of plane equation
        faces_normals: faces normals of LOD2 elements
        normal_op_plane: opening normal most similar to plane normal
    """
    
    #Normals of faces of the polygonal surface (polyfit)
    #Loading polyfit model vertices and faces
    vertices_p = list()
    faces_p = list()
    if dense:
        f = open(polyfit_path +"/polyfit_dense.obj", "r")
    else:
        f = open(polyfit_path +"/polyfit.obj", "r")
    for iii in f:
        l = iii.split()
        if l[0] == 'v':
            vertices_p.append([float(j) for j in l[1:]])
        else:
            faces_p.append([int(j) for j in l[1:]])
    f.close()

    #Getting normals of faces in polyfit model
    faces_normals = list([])
    for _, f in enumerate(faces_p):
        f_vert = list([])
        for j in f:
            f_vert.append(vertices_p[j-1])
        f_vert.append(vertices_p[f[0]-1])
        f_vert = np.array(f_vert)
        v1 = f_vert[1]-f_vert[0]
        v2 = f_vert[2]-f_vert[0]
        A,B,C = np.cross(v1,v2)
        #have to be normalized to make sense the distances
        faces_normals.append((np.array([A,B,C]))/(np.linalg.norm(np.array([A,B,C]))))
    faces_normals = np.array(faces_normals)    
    
    c_v = 0 #vertices counter. Helper to asociate vertices to a single opening
    op_normals = list([]) #list with the opening normals of the facade
    #Computing opening normal directions
    for j in range(int(len(X.T)/4)):
        a = np.copy(X[0:3,c_v])
        b = np.copy(X[0:3,c_v+1])
        c = np.copy(X[0:3,c_v+2])
        d = np.copy(X[0:3,c_v+3])
        #
        #to warranty same normal direction
        #a--->v1    b
        #|         /|
        #|      v4/ |v3
        #~v2     ~  ~
        # 
        #
        #c          d 
        #
        v1 = b-a
        v2 = c-a
        A1,B1,C1 = np.cross(v1,v2)
        v3 = d-b
        v4 = c-b
        A2,B2,C2 = np.cross(v3,v4)
        A = (A1+A2)/2
        B = (B1+B2)/2
        C = (C1+C2)/2
        c_v+=4
        op_normals.append((np.array([A,B,C]))/(np.linalg.norm(np.array([A,B,C]))))
       
    op_normals = np.array(op_normals)
    mean_op_normals = np.mean(op_normals, axis=0)
    
    #Look for the normal of the faces with minimum angle with the openings
    angle_normals = list([])
    for j in faces_normals:
        angle = np.arccos(np.dot(j,mean_op_normals))*180/np.pi
        if angle > 180:
            angle -= 180
        angle_normals.append(angle)
    angle_normals = np.array(angle_normals)
           
    #conditional to be invariant to the direction of the normal vector
    if np.min(np.abs(angle_normals-180)) > np.min(np.abs(angle_normals)):
        index_normal = np.argmin(np.abs(angle_normals))
    else:
        index_normal = np.argmin(np.abs(angle_normals-180))
        
    normal_op_plane = faces_normals[index_normal]   
    
    A = normal_op_plane[0]
    B = normal_op_plane[1]
    C = normal_op_plane[2]
    n_pl = [A,B,C] 
    
    #Using perpendicular projection
    #A(x-x0) + B(y-y0) + C(z-z0) = 0 => z = (Ax0/C + By0/C + z0) - Ax/C  - By/C 
    a = X.T[0,0:3] #create a plane that pass for this point. 
    m1 = A*a[0]/C + B*a[1]/C + a[2]
    m2 = -A/C
    m3 = -B/C
    model = np.array([m1,m2,m3])
    
    for ii, XX in enumerate(X.T):
        X[:3,ii] = proj_pt2pl(model,X[:3,ii])

    return X, n_pl, faces_normals, normal_op_plane


def open2local(X, faces_normals, normal_op_plane):
    """

        #Taking corners X to a local plane.
        #Finds a local plane to project X based in the direction of the openings edges.

    Args:
        X (array): 3D opening corners coordinates
        faces_normals (array): facade elements normals
        normal_op_plane (array): normal of opening similar to the face opening
    Returns:

        X_l (array): local coordinates of 3D opening corners
        T (array): transformation matrix to map 3D opening corners from global to local

    """
    
    dir_vect_h = np.zeros((int(len(X[0,:])/2), 3))
    for ee in range(int(len(X[0,:])/4)):
        ed_h1 = (X[:3, 4*ee + 1] - X[:3, 4*ee])/np.linalg.norm((X[:3, 4*ee + 1] - X[:3, 4*ee]))
        ed_h2 = (X[:3, 4*ee + 3] - X[:3, 4*ee + 2])/np.linalg.norm(X[:3, 4*ee + 3] - X[:3, 4*ee + 2])
        dir_vect_h[2*ee] = ed_h1
        dir_vect_h[2*ee+1] = ed_h2


    #Choosing building normal with similar direction to the edges
    mean_dir_h = (np.mean(dir_vect_h, axis=0))/(np.linalg.norm(np.mean(dir_vect_h, axis=0)))
    #Look for the dir_vect_h with minimum angle with building normals
    ang_dir_h = list([])
    for j in faces_normals:
        angle = np.arccos(np.dot(j/np.linalg.norm(j),mean_dir_h))*180/np.pi
        if angle > 180:
            angle -= 180
        ang_dir_h.append(angle)
    ang_dir_h = np.array(ang_dir_h)
           
    #conditional to be invariant to the direction of the normal vector
    if np.min(np.abs(ang_dir_h-180)) > np.min(np.abs(ang_dir_h)):
        ind_dir_h = np.argmin(np.abs(ang_dir_h))
    else:
        ind_dir_h = np.argmin(np.abs(ang_dir_h-180))
        
    normal_dir_h_plane = faces_normals[ind_dir_h]/ np.linalg.norm(faces_normals[ind_dir_h])
    normal_op_plane = normal_op_plane/np.linalg.norm(normal_op_plane)
    proj_norm_dir_h = normal_dir_h_plane - ((np.dot(normal_dir_h_plane,normal_op_plane))/((np.linalg.norm(normal_op_plane))**2))*normal_op_plane
    proj_norm_dir_h = proj_norm_dir_h/(np.linalg.norm(proj_norm_dir_h))
    
    A = np.copy(X[:3,0])
    B = A + proj_norm_dir_h
    N = np.copy(normal_op_plane)
    U = (B - A)/np.linalg.norm(B-A)
    V = np.cross(N,U)
    u = A + U
    v = A + V
    n = A + N
    #Solving the sistem for T:
    G = np.ones((4,4))
    G[:3,0] = np.copy(A)
    G[:3,1] = np.copy(u)
    G[:3,2] = np.copy(v)
    G[:3,3] = np.copy(n)
    
    L = np.array([[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,1,1,1]])
    
    T = np.dot(L,np.linalg.inv(G))
    
    #Finding local coordinates Xl with T matrix
    Xl = np.dot(T,X)
    Xl = np.round(Xl,6)

    return Xl, T


def op_aligning1(Xl, cte = .05):
    """
    Aligning the width and height of the openings (Aligment 1 --> to linear regression model).

    Args:
        Xl (array): local coordinates of opening corners
    Returns:
        Xl_al (array): aligned local coordinates of opening corners
    """
    print("THE CONSTANT IS ", cte)    
    #Threshold (depends on how the keypoints are organized)#!
    threshold = cte*np.abs(np.min(((Xl[0,0]-Xl[0,1]),(Xl[1,0]-Xl[1,2]))))
    
    #ALIGNING TO LINES
    #Vertical alignment
    vert_checker = np.zeros(Xl[0].shape) #Checker to identify points already aligned
    Xl_al = np.copy(Xl) #if there is problems with not meeting the threshold, keep initial
    for ii, pt in enumerate(Xl[0,:].T):
        if vert_checker[ii] == 0:
            distances = np.abs(pt - Xl[0])
            #distances = line_distances(Xl[0],Xl[1])
            meet_thr = np.where(distances<threshold)
            #print(meet_thr)
            if ii%2==0: #if are in the left corners
                left_meet_thr = np.where(meet_thr[0]%2==0)
                meet_thr = meet_thr[0][left_meet_thr]
            else: #right
                right_meet_thr = np.where(meet_thr[0]%2!=0)
                meet_thr = meet_thr[0][right_meet_thr]
            if np.sum(vert_checker[meet_thr])==0: #to avoid take again points already aligned    
                x_lr = Xl[0][meet_thr]
                y_lr = Xl[1][meet_thr]
                if len(x_lr)>2: #guarantee to do regretion 
                    Xl_al[0][meet_thr], Xl_al[1][meet_thr] = line_adjustor(x_lr,y_lr)
                else: #If just two poits, they are from same openints. Put the same x coordinate
                    Xl_al[0][meet_thr] = np.mean(Xl_al[0][meet_thr])
            vert_checker[meet_thr] = 1
    
    #Horizontal alignment
    hori_checker = np.zeros(Xl[1].shape) #Checker to identify points already aligned
    for ii, pt in enumerate(Xl_al[1,:]):
        if hori_checker[ii] == 0:
            distances = np.abs(pt - Xl_al[1])
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
            if np.sum(hori_checker[meet_thr])==0: #to avoid take again points already aligned
                x_lr = Xl_al[0][meet_thr][0]
                y_lr = Xl_al[1][meet_thr][0]
                if len(x_lr)>2:
                    Xl_al[0][meet_thr], Xl_al[1][meet_thr] = line_adjustor(x_lr,y_lr)
                else: #If just two poits, they are from same openints. Put the same x coordinate
                    Xl_al[1][meet_thr] = np.mean(Xl_al[1][meet_thr])
            hori_checker[meet_thr] = 1

    return Xl_al


def op_aligning2(Xl_al, cte = .35):
    """

    Aligning the width and height of the openings (Aligment 2 --> same width and height)

    Args:
        Xl_al (array): aligned local coordinates of opening corners (linear regression)
    Return:
        Xl_al2 (array): aligned local coordinates of opening corners (same width - height)
    """
    print("THE CONSTANT IS ", cte)    

    #Same width and height for openings#!    
    threshold = cte*np.abs(np.min(((Xl_al[0,0]-Xl_al[0,1]),(Xl_al[1,0]-Xl_al[1,2]))))     
       
    #Vertical alignment
    vert_checker = np.zeros(Xl_al[0].shape) #Checker to identify points already aligned
    Xl_al2 = np.copy(Xl_al) #if there is problems with not meeting the threshold, keep initial
    for ii, pt in enumerate(Xl_al[0,:]):
        if vert_checker[ii] == 0:
            distances = np.abs(pt - Xl_al[0])
            meet_thr = np.where(distances<threshold)
            if ii%2==0: #if are in the left corners
                left_meet_thr = np.where(meet_thr[0]%2==0)
                meet_thr = meet_thr[0][left_meet_thr]
            else: #right
                right_meet_thr = np.where(meet_thr[0]%2!=0)
                meet_thr = meet_thr[0][right_meet_thr]
            if np.sum(vert_checker[meet_thr])==0: #to avoid take again points already aligned
                mean_coordinate = np.mean(Xl_al[0][meet_thr])
                Xl_al2[0][meet_thr] = np.copy(mean_coordinate)
            vert_checker[meet_thr] = 1
    #Horizontal alignment
    hori_checker = np.zeros(Xl_al[0].shape) #Checker to identify points already aligned
    for ii, pt in enumerate(Xl_al[1,:]):
        if hori_checker[ii] == 0:
            distances = np.abs(pt - Xl_al[1])
            meet_thr = np.where(distances<threshold)
            if ii%4==0 or ii%4==1: #top corners
                top_meet_thr1 = np.where(meet_thr[0]%4==0)
                top_meet_thr2 = np.where(meet_thr[0]%4==1)
                top_meet_thr = np.concatenate((top_meet_thr1,top_meet_thr2), axis=1)
                meet_thr = meet_thr[0][top_meet_thr]
            else: #bottom
                bottom_meet_thr1 = np.where(meet_thr[0]%4==2)
                bottom_meet_thr2 = np.where(meet_thr[0]%4==3)
                bottom_meet_thr = np.concatenate((bottom_meet_thr1,bottom_meet_thr2), axis=1)
                meet_thr = meet_thr[0][bottom_meet_thr]
            if np.sum(hori_checker[meet_thr])==0: #to avoid take again points already aligned
                mean_coordinate = np.mean(Xl_al[1][meet_thr])
                Xl_al2[1][meet_thr] = np.copy(mean_coordinate)
            hori_checker[meet_thr] = 1
    

    return Xl_al2


def op_aligning3(Xl_al2, cte1 = .1, cte2 = .3):
    """

    Equalizing areas. Aligning cetroids. Calculating area of each opening. Increment or decrease
    edges to have same area.

    Args:
        Xl_al2 (array): aligned local coordinates of opening corners (same width - height)

    Returns:
        Xl_al3 (array): aligned local coordinates of opening corners (equal areas)
    """
    print("THE CONSTANTS ARE ", cte1, cte2)    
        
    Xl_al3 = np.copy(Xl_al2)
    centroids = [] 
    areas = []
    edges_h = []
    edges_v = []
    for j in range(int(Xl_al2.shape[1]/4)):
        xc = (Xl_al2.T[4*j,0] + Xl_al2.T[4*j+1,0])/2
        yc = (Xl_al2.T[4*j,1] + Xl_al2.T[4*j+2,1])/2
        centroids.append([xc,yc])
        edge_h = np.abs(Xl_al2.T[4*j,0] - Xl_al2.T[4*j+1,0])
        edge_v = np.abs(Xl_al2.T[4*j,1] - Xl_al2.T[4*j+2,1])
        edges_h.append(edge_h)
        edges_v.append(edge_v)
        areas.append(edge_h*edge_v)
    centroids = np.array(centroids)
    areas = np.array(areas)
    edges_h = np.array(edges_h)
    edges_v = np.array(edges_v)
    
    
    #Vertical centroids aligment#!
    threshold = cte1*np.abs(np.min(((Xl_al3[0,0]-Xl_al3[0,1]),(Xl_al3[1,0]-Xl_al3[1,2]))))     
    vert_checker = np.zeros(len(centroids)) #Checker to identify points already aligned
    centroids_al = np.copy(centroids) 
    for ii, pt in enumerate(centroids[:,0]):
        if vert_checker[ii] == 0:
            distances = np.abs(pt - centroids[:,0])
            meet_thr = np.where(distances<threshold)
            if np.sum(vert_checker[meet_thr])==0: #to avoid take again points already aligned
                mean_coordinate = np.mean(centroids[:,0][meet_thr])
                centroids_al[meet_thr,0] = np.copy(mean_coordinate)
            vert_checker[meet_thr] = 1
    #Horizontal centroids alignment
    hori_checker = np.zeros(len(centroids)) #Checker to identify points already aligned
    for ii, pt in enumerate(centroids[:,1]):
        if hori_checker[ii] == 0:
            distances = np.abs(pt - centroids[:,1])
            meet_thr = np.where(distances<threshold)
            if np.sum(hori_checker[meet_thr])==0: #to avoid take again points already aligned
                mean_coordinate = np.mean(centroids[:,1][meet_thr])
                centroids_al[meet_thr,1] = np.copy(mean_coordinate)
            hori_checker[meet_thr] = 1
    
    #Equalizing areas - to establish a threshold in the area diferences. #!
    threshold = cte2*np.min(areas)
    area_checker = np.zeros(len(areas))
    edges_h_e = np.copy(edges_h)
    edges_v_e = np.copy(edges_v)
    for ii, ar in enumerate(areas):
        if area_checker[ii] == 0:
            diferences = np.abs(ar - areas)
            meet_thr = np.where(diferences<threshold)
            if np.sum(area_checker[meet_thr])==0: #to avoid take again points already aligned
                mean_edge_h = np.mean(edges_h[meet_thr])
                mean_edge_v = np.mean(edges_v[meet_thr])
                edges_h_e[meet_thr] = np.copy(mean_edge_h)
                edges_v_e[meet_thr] = np.copy(mean_edge_v)
            area_checker[meet_thr] = 1
    
    #Generation new coordinates for openings with same area
    for j in range(int(Xl_al3.shape[1]/4)):
        #x coordinates
        Xl_al3[0,4*j]   = centroids_al[j][0] + edges_h_e[j]/2
        Xl_al3[0,4*j+1] = centroids_al[j][0] - edges_h_e[j]/2
        Xl_al3[0,4*j+2] = centroids_al[j][0] + edges_h_e[j]/2
        Xl_al3[0,4*j+3] = centroids_al[j][0] - edges_h_e[j]/2
        #y coordinates
        Xl_al3[1,4*j]   = centroids_al[j][1] + edges_v_e[j]/2
        Xl_al3[1,4*j+1] = centroids_al[j][1] + edges_v_e[j]/2
        Xl_al3[1,4*j+2] = centroids_al[j][1] - edges_v_e[j]/2
        Xl_al3[1,4*j+3] = centroids_al[j][1] - edges_v_e[j]/2
        
    #Testing final areas
    f_areas = []
    for j in range(int(Xl_al3.shape[1]/4)):
        edge_h = np.abs(Xl_al3.T[4*j,0] - Xl_al3.T[4*j+1,0])
        edge_v = np.abs(Xl_al3.T[4*j,1] - Xl_al3.T[4*j+2,1])
        f_areas.append(edge_h*edge_v)
    
    return Xl_al3

    
def op_projector(data_folder, images_path, keypoints_path, polyfit_path, how2get_kp,\
                 im1, im2, P, K, k_dist, opening_4points, pp=None, dense=False, ctes=[.05, .8, .1, .3]):
    """

        Creates 3D objects for the openings segmented with deep learning using the camera matrices
        correspondend to 2 view points of a group of building's facades

    Args:
        data_folder (str): path to input data folder
        images_path (str): path to input images
        keypoints_path (str): path to keypoints
        polyfit_path (str): path to LOD2 model produced by polyfit
        how2get_kp (int): method to get opening corners to triangulate 0:from CNN detector+homography, 1: loading npy files, 2:clicking on two views, 3: v1 from npy+v2 with homography, 4:from CNN detector+homography
        im1 (list): list of input images view 1 for each facade
        im2 (list): list of input images view 2 for each facade
        pp (int, optional): list with number of corners to be triangulated for each facade - used if how2get_kp==2
        P (dict): camera matrices for each camera pose
        K (dict): intrinsic matrices for each camera
        k_dist (dict): distorsion parameters for each camera
        opening_4points (dict): opening corners for each facade
        dense (bool, optional): if true, it loads the polyfit model from dense point cloud. Defaults to False.
        ctes (list): constants that define thresholds when refining and aligning openings -- the lower the less influence
    """
    
    
    #Loop through each par of pictures given by im1 and im2
    cc_vv = 1 #vertices counter. Helper to identify vertices in generated faces (all)
    cc_vv_c = 1 #vertices counter. Helper to identify vertices in generated cracks (all)
    for i in range(len(im1)): 
        #i=2
        print("Projection of openings in pics " + im1[i] + " and "\
              + im2[i])
        P1 = P[im1[i]]['P']
        P1_inid = P[im1[i]]['intrinsicId']
        P2 = P[im2[i]]['P']
        P2_inid = P[im2[i]]['intrinsicId']
        
        if os.path.isfile(images_path + "im1/" + im1[i] + '.jpg'):
            imm1 = np.array(Image.open(images_path + "im1/" + im1[i] + '.jpg'))
        elif os.path.isfile(images_path + "im1/" + im1[i] + '.JPG'):
            imm1 = np.array(Image.open(images_path + "im1/" + im1[i] + '.JPG'))
        else:
            imm1 = np.array(Image.open(images_path + "im1/" + im1[i] + '.png'))
        
        if os.path.isfile(images_path + "im2/" + im2[i] + '.jpg'):
            imm2 = np.array(Image.open(images_path + "im2/" + im2[i] + '.jpg'))
        elif os.path.isfile(images_path + "im2/" + im2[i] + '.JPG'):
            imm2 = np.array(Image.open(images_path + "im2/" + im2[i] + '.JPG'))
        else:
            imm2 = np.array(Image.open(images_path + "im2/" + im2[i] + '.png'))

        #Geting opening corners coordinates on images im1 and im2
        x1, x2, _, _ = imgs_corners(im1,im2,imm1,imm2,i, keypoints_path, how2get_kp, opening_4points, data_folder, pp=pp)
         
        #Correcting cordinates by lens distortion
        #x1_u = undistort_points(x1.T,K[P1_inid],k_dist[P1_inid]).T
        #x2_u = undistort_points(x2.T,K[P2_inid],k_dist[P2_inid]).T
        
        # triangulate inliers and remove points not in front of both cameras
        #X = sfm.triangulate(x1_u,x2_u,P1,P2) 
        X = sfm.triangulate(x1,x2,P1,P2) 
                   
        # project 3D points
        cam1 = camera.Camera(P1)
        cam2 = camera.Camera(P2)
        
        #Find a plane paralel to facade with a normal similar to mean of normals
        #of all openings. Project corners to that plane. 
        X, _, faces_normals, normal_op_plane = proj_op2plane(polyfit_path, X, dense)
        
        
        #CLEANING 2 - SAME WIDTH AND HEIGHT TO WINDOWS DOORS
        #Taking corners X to a local plane. Xl
        Xl, T = open2local(X, faces_normals, normal_op_plane)
        #Aligning the width and height of the openings (Aligment 1 --> to linear regression model).
        Xl_al = op_aligning1(Xl, cte = ctes[0])
       
        #CLEANING 2.1: aligning  each opening
        #Aligning the width and height of the openings (Aligment 2 --> same width and height)
        Xl_al2 = op_aligning2(Xl_al, cte = ctes[1])
        
        #Equalizing areas
        Xl_al3 = op_aligning3(Xl_al2, cte1 = ctes[2], cte2 = ctes[3])
        
        #Taking to global coordinates again
        X_al = np.dot(np.linalg.inv(T),Xl_al3) 
       
        #Check if directory exists, if not, create it
        check_dir = os.path.isdir('../results/' + data_folder)
        if not check_dir:
            os.makedirs('../results/' + data_folder) 
        
        #Writing an .obj file with information of the openings for each pics pair
        f = open('../results/' + data_folder + "/openings{}.obj".format(i), "w")
        
        for l in range(len(X_al.T)):
            f.write("v {} {} {}\n".format(X_al.T[l][0],X_al.T[l][1],X_al.T[l][2]))
        c_v = 1 #vertices counter. Helper to identify vertices in generated faces
        for j in range(int(len(X_al.T)/4)):
            f.write("f {} {} {}\n".format(c_v,c_v+1,c_v+2))
            f.write("f {} {} {}\n".format(c_v+1,c_v+2,c_v+3))
            c_v += 4
        f.close()
       
        #Writing an .obj file with information of the openings for all of them
        f = open('../results/' + data_folder + "/openings.obj".format(i), "a")
        for l in range(len(X_al.T)):
            f.write("v {} {} {}\n".format(X_al.T[l][0],X_al.T[l][1],X_al.T[l][2]))
       
        for j in range(int(len(X_al.T)/4)):
            f.write("f {} {} {}\n".format(cc_vv,cc_vv+1,cc_vv+2))
            f.write("f {} {} {}\n".format(cc_vv+1,cc_vv+2,cc_vv+3))
            cc_vv += 4
        f.close()
