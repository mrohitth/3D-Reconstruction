import numpy as np
import matplotlib.pyplot as plt

import os

from helper import visualize_keypoints, plot_3d_keypoint, connections_3d, colors
from q3_2_triangulate import triangulate

# Insert your package here


'''
Q6.1 Multi-View Reconstruction of keypoints.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx3 matrix with the 2D image coordinates and confidence per row
            C2, the 3x4 camera matrix
            pts2, the Nx3 matrix with the 2D image coordinates and confidence per row
            C3, the 3x4 camera matrix
            pts3, the Nx3 matrix with the 2D image coordinates and confidence per row
    Output: P, the Nx3 matrix with the corresponding 3D points for each keypoint per row
            err, the reprojection error.
'''
def MultiviewReconstruction(C1, pts1, C2, pts2, C3, pts3, Thres = 140):
    # Replace pass by your implementation
    pts1=pts1[pts1[:,2] > Thres]
    p1=pts1[:,:2]
    pts2=pts2[pts2[:,2] > Thres]
    p2=pts2[:,:2]
    pts3=pts3[pts3[:,2] > Thres]
    p3=pts3[:,:2]
    n, temp = p1.shape
    P = np.zeros((n,3))
    Phomo = np.zeros((n,4))
    for i in range(n):
        x1 = p1[i,0]
        y1 = p1[i,1]
        x2 = p2[i,0]
        y2 = p2[i,1]
        x3 = p3[i,0]
        y3 = p3[i,1]
        A1 = x1*C1[2,:] - C1[0,:]
        A2 = y1*C1[2,:] - C1[1,:]
        A3 = x2*C2[2,:] - C2[0,:]
        A4 = y2*C2[2,:] - C2[1,:]
        A5=  x3*C3[2,:] - C3[0,:]
        A6 = y3*C3[2,:] - C3[1,:]
        A = np.vstack((A1,A2,A3,A4,A5,A6))
        # print(A.shape)
        u, s, vh = np.linalg.svd(A)
        p = vh[-1, :]
        p = p/p[3]
        P[i, :] = p[0:3]
        Phomo[i, :] = p
        # print(p)
    p1_proj = np.matmul(C1,Phomo.T)
    lam1 = p1_proj[-1,:]
    p1_proj = p1_proj/lam1
    p2_proj = np.matmul(C2,Phomo.T)
    lam2 = p2_proj[-1,:]
    p2_proj = p2_proj/lam2
    err1 = np.sum((p1_proj[[0,1],:].T-p1)**2)
    err2 = np.sum((p2_proj[[0,1],:].T-p2)**2)
    err = err1 + err2
    # print(err)
    if(os.path.isfile('q6_1.npz')==False):
        np.savez('q6_1.npz',P=P)
    return P,err
    pass


'''
Q6.2 Plot Spatio-temporal (3D) keypoints
    :param car_points: np.array points * 3
'''
def plot_3d_keypoint_video(pts_3d_video):
    # Replace pass by your implementation
    pass


#Extra Credit
if __name__ == "__main__":
         
        
    pts_3d_video = []
    for loop in range(10):
        print(f"processing time frame - {loop}")

        data_path = os.path.join('../data/q6/','time'+str(loop)+'.npz')
        image1_path = os.path.join('../data/q6/','cam1_time'+str(loop)+'.jpg')
        image2_path = os.path.join('../data/q6/','cam2_time'+str(loop)+'.jpg')
        image3_path = os.path.join('../data/q6/','cam3_time'+str(loop)+'.jpg')

        im1 = plt.imread(image1_path)
        im2 = plt.imread(image2_path)
        im3 = plt.imread(image3_path)

        data = np.load(data_path)
        pts1 = data['pts1']
        pts2 = data['pts2']
        pts3 = data['pts3']

        K1 = data['K1']
        K2 = data['K2']
        K3 = data['K3']

        M1 = data['M1']
        M2 = data['M2']
        M3 = data['M3']

        #Note - Press 'Escape' key to exit img preview and loop further 
        # img = visualize_keypoints(im2, pts2)
        
        C1 = np.matmul(K1, M1)
        C2 = np.matmul(K2, M2)
        C3 = np.matmul(K3, M3)
    
        P,err=MultiviewReconstruction(C1, pts1, C2, pts2, C3, pts3)
        plot_3d_keypoint(P)
        # YOUR CODE HERE