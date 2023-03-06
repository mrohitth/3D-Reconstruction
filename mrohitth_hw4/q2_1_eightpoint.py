import numpy as np
import matplotlib.pyplot as plt
import os.path

from helper import displayEpipolarF, calc_epi_error, toHomogenous, refineF

# Insert your package here



'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix

    HINTS:
    (1) Normalize the input pts1 and pts2 using the matrix T.
    (2) Setup the eight point algorithm's equation.
    (3) Solve for the least square solution using SVD. 
    (4) Use the function `_singularize` (provided) to enforce the singularity condition. 
    (5) Use the function `refineF` (provided) to refine the computed fundamental matrix. 
        (Remember to usethe normalized points instead of the original points)
    (6) Unscale the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    F = None            #fundamental matrix 
    N = pts1.shape[0]   # Extrating the number of points
    
    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)
    
    T = np.diag([1/M, 1/M, 1])
    
    pts1_norm = pts1_homogenous @ T
    pts2_norm = pts2_homogenous @ T
    A = []
    for i in range(pts2_norm.shape[0]):
        h1 =[pts1_norm[i, 0]*pts2_norm[i, 0], pts1_norm[i, 0]*pts2_norm[i, 1], pts1_norm[i, 0], 
            pts1_norm[i, 1]*pts2_norm[i, 0], pts1_norm[i, 1]*pts2_norm[i, 1], pts1_norm[i, 1], 
            pts2_norm[i, 0], pts2_norm[i, 1], 1]
        A.append(h1)
    A = np.array(A)
    u, s, vh = np.linalg.svd(A)
    

    F = vh[-1,:]
    #print(h.shape)

    F = F.reshape((3,3))
    F = F.T
    F = refineF(F,pts1_norm[:, :-1],pts2_norm[:, :-1] )
    F = np.transpose(T)@ F @ T
    
    F = F/F[2,2] #Finding the unique fundamental matrix by setting the scale to 1. 
    if(os.path.isfile('q2_1.npz')==False):
        np.savez('q2_1.npz',F = F, M = M)

    return F




if __name__ == "__main__":
        
    correspondence = np.load('../data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('../data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')

    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))
    print(F)
    N = pts1.shape[0]

    # Simple Tests to verify your implementation:
    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

    print("Error:", np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)))

    ## Important!! Uncomment this line to visualize, but before you submit, 
    displayEpipolarF(im1, im2, F)

    assert(F.shape == (3, 3))
    assert(F[2, 2] == 1)
    assert(np.linalg.matrix_rank(F) == 2)
    assert(np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1)