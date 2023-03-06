import numpy as np
import matplotlib.pyplot as plt
import os.path

from helper import camera2
from q2_1_eightpoint import eightpoint
from q3_1_essential_matrix import essentialMatrix

# Insert your package here


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.

    Hints:
    (1) For every input point, form A using the corresponding points from pts1 & pts2 and C1 & C2
    (2) Solve for the least square solution using np.linalg.svd
    (3) Calculate the reprojection error using the calculated 3D points and C1 & C2 (do not forget to convert from 
        homogeneous coordinates to non-homogeneous ones)
    (4) Keep track of the 3D points and projection error, and continue to next point 
    (5) You do not need to follow the exact procedure above. 
'''
def triangulate(C1, pts1, C2, pts2):

    N = pts1.shape[0]
    P = np.zeros((N, 3))

    for i in range(0, N):
            a1 = np.multiply(pts1[i, 0], C1[2, :]) - C1[0, :]
            a2 = np.multiply(pts1[i, 1], C1[2, :]) - C1[1, :]
            a3 = np.multiply(pts2[i, 0], C2[2, :]) - C2[0, :]
            a4 = np.multiply(pts2[i, 1], C2[2, :]) - C2[1, :]
            A = np.vstack((a1,a2,a3,a4))

            u,s,v = np.linalg.svd(A)
            f1 = v[3, :]
            f1 = f1/f1[3]
            P[i, :] = f1[:3]

    a = np.ones((1, N))
    p_temp = np.vstack((P.T, a))
    pts1_new = np.matmul(C1, p_temp)
    pts1_new = pts1_new / pts1_new[2, :]
    
    pts2_new = np.matmul(C2, p_temp)
    pts2_new = pts2_new / pts2_new[2, :]

    pts1_new = pts1_new[:2, :]
    pts2_new = pts2_new[:2, :]

    # print((pts1.T).shape, pts1_new.shape, (pts2.T).shape, pts2_new.shape)
    
    error = np.power(np.subtract(pts1.T, pts1_new),2) + np.power(np.subtract(pts2.T, pts2_new), 2)
    error = np.sum(error)
    # print (error)

    
    return P, error


'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''


def findM2(F, pts1, pts2, intrinsics, filename = 'q3_3.npz'):
    '''
    Q2.2: Function to find the camera2's projective matrix given correspondences
        Input:  F, the pre-computed fundamental matrix
                pts1, the Nx2 matrix with the 2D image coordinates per row
                pts2, the Nx2 matrix with the 2D image coordinates per row
                intrinsics, the intrinsics of the cameras, load from the .npz file
                filename, the filename to store results
        Output: [M2, C2, P] the computed M2 (3x4) camera projective matrix, C2 (3x4) K2 * M2, and the 3D points P (Nx3)
    
    ***
    Hints:
    (1) Loop through the 'M2s' and use triangulate to calculate the 3D points and projection error. Keep track 
        of the projection error through best_error and retain the best one. 
    (2) Remember to take a look at camera2 to see how to correctly reterive the M2 matrix from 'M2s'. 

    '''

    
    K1 = intrinsics['K1']
    K2 = intrinsics['K2']
    
    E = essentialMatrix(F, K1, K2)

    M1 = np.hstack(((np.eye(3)), np.zeros((3, 1))))
    M2s = camera2(E)
    row, col, num = np.shape(M2s)
    
    C1 = np.matmul(K1, M1)
    
    # print(num)
    for i in range(num):
        M2 = M2s[:, :, i]
        C2 = np.matmul(K2, M2)
        P, err = triangulate(C1, pts1, C2, pts2) 
        if (np.all(P[:,2] > 0)) :
            break
    #print("P is :",P)
    
    if(os.path.isfile('q3_3.npz')==False):
        np.savez('q3_3.npz', M2 = M2, C2 = C2, P = P)

    return M1, C1, M2, C2, P

   


if __name__ == "__main__":

    correspondence = np.load('../data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('../data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')

    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))

    
    M1, C1, M2, C2, P = findM2(F, pts1, pts2, intrinsics)
    print(f"M2: {M2}")
    print(f"C2 {C2}")
    if(os.path.isfile('q3_3.npz')==False):
        np.savez('q3_3.npz', M2 = M2, C2 = C2, P = P)

    # Simple Tests to verify your implementation:
    P_test, err = triangulate(C1, pts1, C2, pts2)
    print(f"Best Error {err}")
    assert(err < 500)