from tkinter import X
import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous
from q2_1_eightpoint import eightpoint
from q2_2_sevenpoint import sevenpoint
from q3_2_triangulate import findM2


# Insert your package here
import scipy.optimize as optimize
from math import *
# Helper functions for this assignment. DO NOT MODIFY!!!
"""
Helper functions.

Written by Chen Kong, 2018.
Modified by Zhengyi (Zen) Luo, 2021
"""
def plot_3D_dual(P_before, P_after):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Blue: before; red: after")
    ax.scatter(P_before[:,0], P_before[:,1], P_before[:,2], c = 'blue')
    ax.scatter(P_after[:,0], P_after[:,1], P_after[:,2], c='red')
    while True:
        x, y = plt.ginput(1, mouse_stop=2)[0]
        plt.draw()


'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
            nIters, Number of iterations of the Ransac
            tol, tolerence for inliers
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers

    Hints:
    (1) You can use the calc_epi_error from q1 with threshold to calcualte inliers. Tune the threshold based on 
        the results/expected number of inliners. You can also define your own metric. 
    (2) Use the seven point alogrithm to estimate the fundamental matrix as done in q1
    (3) Choose the resulting F that has the most number of inliers
    (4) You can increase the nIters to bigger/smaller values
 
'''
def ransacF(pts1, pts2, M, nIters=1000, tol=10):
    # Replace pass by your implementation

    print("In ransac")

    N = pts1.shape[0]
    pts1_homo, pts2_homo = toHomogenous(pts1), toHomogenous(pts2)
    best_inlier = 0
    inlier_curr = np.zeros((pts1.shape[0],))
    # ----- TODO -----
    # YOUR CODE HERE
    
    choices = []
    ninlier = 0
    for i in range(nIters):
        print("Iterations :",i)
        try:
            choice = np.random.choice(range(pts1.shape[0]), 7)
            pts1_choice = pts1[choice, :]
            pts2_choice = pts2[choice, :]
            Fs = sevenpoint(pts1_choice, pts2_choice, M)
            for Fi in Fs:
                choices.append(choice)
                res = calc_epi_error(pts1_homo,pts2_homo, Fi)
                idx = np.where(res < tol)
                ninlier = np.array(idx).size
                if ninlier > best_inlier:
                    best_inlier = ninlier
                    F = Fi
                    idxmax=[]
                    idxmax.append(idx)
        except ValueError:
            print("Division by zero")
            
    inlier_curr[tuple(idxmax)] = 1
    inlier_curr = np.expand_dims(inlier_curr, axis = 1).astype(bool)

    return F, inlier_curr



'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
   
    theta = np.linalg.norm(r)
    if(theta == 0):
        return np.eye(3)
    u = r/theta
    #print(u.shape[0])
    u_cap = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    u = u.reshape((u.shape[0],1))
    ## print(u.shape)
    R = np.eye(3)*np.cos(theta) + (1 - np.cos(theta))*(u@u.T) + u_cap * np.sin(theta)
    return R


'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    
    A = (R - R.T)/2
    ro = np.array([A[2,1], A[0,2], A[1,0]])
    s = np.linalg.norm(ro)
    c = (R[0, 0]+R[1, 1]+R[2, 2]-1)/2
    
    if(s == 0 and c == 1):
        return np.zeros(3)
    elif(s == 0 and c == -1):
        v_ = R + np.eye(3)
        for i in range(3):
            if (np.count_nonzero(v_[:,i])) > 0:
                v = v_[:,i]
                print(v)
                break
        u = v/np.linalg.norm(v)
        r = u*np.pi
        
        if(np.linalg.norm(r) == np.pi and ((r[0,0] == 0 and r[1,0] == 0 and 
                r[2,0] < 0) or (r[0,0] == 0 and r[1,0] < 0) or (r[0,0] < 0))):
            r = -r
    else:
        theta = np.arctan2(s, c)
        u = ro/s
        r = u*theta
    return r


'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    residuals = None
    N = p1.shape[0]
    P = x[:-6].reshape((N,3))
    P = np.vstack((np.transpose(P), np.ones((1, N))))
    R2 = rodrigues(x[-6:-3].reshape((3,)))
    t2 = x[-3:].reshape((3,1))
    #print(R2.shape,t2.shape,R2,t2)
    M2 = np.hstack((R2, t2))
    
    C1 = K1 @ M1
    C2 = K2 @ M2
    
    p1_proj = C1 @ P
    p1_proj = p1_proj / p1_proj[2,:]
    p2_proj = C2 @ P
    p2_proj = p2_proj / p2_proj[2,:]
    p1_proj_coord = p1_proj[0:2,:].T
    p2_proj_coord = p2_proj[0:2,:].T
    residuals = np.concatenate([(p1-p1_proj_coord).reshape([-1]), (p2-p2_proj_coord).reshape([-1])])

    return residuals

    

'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
            o1, the starting objective function value with the initial input
            o2, the ending objective function value after bundle adjustment

    Hints:
    (1) Use the scipy.optimize.minimize function to minimize the objective function, rodriguesResidual. 
        You can try different (method='..') in scipy.optimize.minimize for best results. 
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    obj_start = obj_end = 0
    # ----- TODO -----
    # YOUR CODE HERE
    R2init, t2init = M2_init[:,0:3], M2_init[:,3]

    x0 = np.concatenate((P_init.flatten(), invRodrigues(R2init).flatten(), t2init.flatten()))

    print(P_init)
    def func(x): #K1, M1, p1, K2, p2, 
        return ((rodriguesResidual(K1, M1, p1, K2, p2, x))**2).sum()

    obj_start= func(x0)#**2).sum()
    x_upd = optimize.minimize(func, x0, method = 'CG' ).x  #leastsq

    
    
    obj_end= func(x_upd)#**2).sum()
    N = p1.shape[0]
    P = x_upd[:-6].reshape((N,3))
    R2 = rodrigues(x_upd[-6:-3].reshape((3,)))
    t2 = x_upd[-3:].reshape((3,1))
    M2 = np.hstack((R2, t2))

    return M2, P, obj_start, obj_end

    

if __name__ == "__main__":
              
    np.random.seed(1) #Added for testing, can be commented out

    some_corresp_noisy = np.load('../data/some_corresp_noisy.npz') # Loading correspondences
    intrinsics = np.load('../data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    noisy_pts1, noisy_pts2 = some_corresp_noisy['pts1'], some_corresp_noisy['pts2']
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')

    F, inliers = ransacF(noisy_pts1, noisy_pts2, M=np.max([*im1.shape, *im2.shape]))
    
    pts1_inliers = noisy_pts1[inliers.squeeze(), :]
    pts2_inliers = noisy_pts2[inliers.squeeze(), :]

    # F,_ = ransacF(pts1_inliers, pts2_inliers, M=np.max([*im1.shape, *im2.shape])) # this calculates the new F

    pts1_homogenous, pts2_homogenous = toHomogenous(noisy_pts1), toHomogenous(noisy_pts2)
    assert(F.shape == (3, 3))
    assert(F[2, 2] == 1)
    assert(np.linalg.matrix_rank(F) == 2)
    
    # Simple Tests to verify your implementation:
    from scipy.spatial.transform import Rotation as sRot
    rotVec = sRot.random()
    mat = rodrigues(rotVec.as_rotvec())

    assert(np.linalg.norm(rotVec.as_rotvec() - invRodrigues(mat)) < 1e-3)
    assert(np.linalg.norm(rotVec.as_matrix() - mat) < 1e-3)

    # YOUR CODE HERE
    
    M1, C1, M2_init,C2, P_init = findM2(F, pts1_inliers, pts2_inliers, intrinsics)
    # print(M2_init)
    M2, P_final, obj_start, obj_end = bundleAdjustment(K1, M1, pts1_inliers, K2, M2_init, pts2_inliers, P_init)
    print(f"Before {obj_start}, After {obj_end}")


    plot_3D_dual(P_init, P_final)


    