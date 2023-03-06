import numpy as np
import matplotlib.pyplot as plt


from helper import displayEpipolarF, calc_epi_error, toHomogenous, _singularize

# Insert your package here
import os.path

'''
Q2.2: Seven Point Algorithm for calculating the fundamental matrix
    Input:  pts1, 7x2 Matrix containing the corresponding points from image1
            pts2, 7x2 Matrix containing the corresponding points from image2
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated 3x3 fundamental matrixes.
    
    HINTS:
    (1) Normalize the input pts1 and pts2 scale paramter M.
    (2) Setup the seven point algorithm's equation.
    (3) Solve for the least square solution using SVD. 
    (4) Pick the last two colum vector of vT.T (the two null space solution f1 and f2)
    (5) Use the singularity constraint to solve for the cubic polynomial equation of  F = a*f1 + (1-a)*f2 that leads to 
        det(F) = 0. Sovling this polynomial will give you one or three real solutions of the fundamental matrix. 
        Use np.polynomial.polynomial.polyroots to solve for the roots
    (6) Unscale the fundamental matrixes and return as Farray
'''
def sevenpoint(pts1, pts2, M):

    Farray = []
    # ----- TODO -----
    # YOUR CODE HERE
    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

    T = np.diag([1/M, 1/M, 1])
    
    pts1_norm = pts1_homogenous @ T
    pts2_norm = pts2_homogenous @ T
    A = []
    for i in range(pts2_norm.shape[0]):
        h1 =[pts1_norm[i, 0]*pts2_norm[i, 0], pts1_norm[i, 0]*pts2_norm[i, 1], pts1_norm[i, 0], 
            pts1_norm[i, 1]*pts2_norm[i, 0], pts1_norm[i, 1]*pts2_norm[i, 1], 
            pts1_norm[i, 1], pts2_norm[i, 0], pts2_norm[i, 1], 1]
        A.append(h1)
    A = np.array(A)
    u, s, vh = np.linalg.svd(A)

    F1 = vh[-1,:]
    F2 = vh[-2,:]
    #prints(h.shape)
    
    F1 = F1.reshape((3,3))
    F2 = F2.reshape((3,3))
    
    c0 = np.linalg.det(F2)
    c2 = (np.linalg.det(2* F2 -F1) + np.linalg.det(F2))/2 - np.linalg.det(F2)
    c3 = (np.linalg.det(2*F1 -F2) - 2*c2 + c0 - 2* np.linalg.det(F1))/6
    c1 = np.linalg.det(F1) -c0 -c2 - c3
    
    alpha = np.polynomial.polynomial.polyroots((c0, c1, c2, c3))
    sol = [a.real for a in alpha if(a.imag == 0)]
  
    for i in range(len(sol)):
        a = sol[i]
        if a.imag == 0:
            r = a.real
            F = r*F1 + (1-r)*F2
            F = np.transpose(T)@ F @ T
            F = _singularize(F)
            Farray.append(F)
        else:
            continue

    Farray = np.stack(Farray, axis=-1)
    Farray /= Farray[2,2]
    
    if(os.path.isfile('q2_2.npz')==False):
        np.savez('q2_2.npz',F = Farray.T, M = M)
    
    return Farray.T



if __name__ == "__main__":
        
    correspondence = np.load('../data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('../data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')


    # ----- TODO -----
    # YOUR CODE HERE


    
    # Simple Tests to verify your implementation:
    # Test out the seven-point algorithm by randomly sampling 7 points and finding the best solution. 
    np.random.seed(1) #Added for testing, can be commented out
    
    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

    max_iter = 500
    pts1_homo = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_homo = np.hstack((pts2, np.ones((pts2.shape[0], 1))))

    ress = []
    F_res = []
    choices = []
    M=np.max([*im1.shape, *im2.shape])
    for i in range(max_iter):
        choice = np.random.choice(range(pts1.shape[0]), 7)
        pts1_choice = pts1[choice, :]
        pts2_choice = pts2[choice, :]
        Fs = sevenpoint(pts1_choice, pts2_choice, M)
        
        for F in Fs:
            choices.append(choice)
            res = calc_epi_error(pts1_homo,pts2_homo, F)
            F_res.append(F)
            ress.append(np.mean(res))
            
    index = np.argmin(np.abs(np.array(ress)))
    F = F_res[index]
    print(F)
    print("Error:", ress[index])
    displayEpipolarF(im1, im2, F)
    assert(F.shape == (3, 3))
    assert(F[2, 2] == 1)
    assert(np.linalg.matrix_rank(F) == 2)
    assert(np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1)