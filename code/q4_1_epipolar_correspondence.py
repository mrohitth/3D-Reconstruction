import numpy as np
import matplotlib.pyplot as plt
import os.path

from helper import _epipoles

from q2_1_eightpoint import eightpoint

# Insert your package here
# from scipy.ndimage import gaussian_filter

# Helper functions for this assignment. DO NOT MODIFY!!!
def epipolarMatchGUI(I1, I2, F):
    e1, e2 = _epipoles(F)

    sy, sx, _ = I2.shape
    P1, P2 = [], []

    f, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 9))
    ax1.imshow(I1)
    ax1.set_title('Select a point in this image')
    ax1.set_axis_off()
    ax2.imshow(I2)
    ax2.set_title('Verify that the corresponding point \n is on the epipolar line in this image')
    ax2.set_axis_off()

    while True:
        plt.sca(ax1)
        # x, y = plt.ginput(1, mouse_stop=2)[0]

        out = plt.ginput(1, timeout=3600, mouse_stop=2)

        if len(out) == 0:
            print(f"Closing GUI")
            break
        
        x, y = out[0]

        xc = int(x)
        yc = int(y)
        v = np.array([xc, yc, 1])
        l = F.dot(v)
        s = np.sqrt(l[0]**2+l[1]**2)

        if s==0:
            print('Zero line vector in displayEpipolar')

        l = l/s

        if l[0] != 0:
            ye = sy-1
            ys = 0
            xe = -(l[1] * ye + l[2])/l[0]
            xs = -(l[1] * ys + l[2])/l[0]
        else:
            xe = sx-1
            xs = 0
            ye = -(l[0] * xe + l[2])/l[1]
            ys = -(l[0] * xs + l[2])/l[1]

        # plt.plot(x,y, '*', 'MarkerSize', 6, 'LineWidth', 2);
        ax1.plot(x, y, '*', MarkerSize=6, linewidth=2)
        ax2.plot([xs, xe], [ys, ye], linewidth=2)

        # draw points
        x2, y2 = epipolarCorrespondence(I1, I2, F, xc, yc)
        ax2.plot(x2, y2, 'ro', MarkerSize=8, linewidth=2)


        p1 = [x, y]
        p2 = [x2, y2]
        # print(p1, p2)

        P1.append(p1)
        P2.append(p2)

        plt.draw()

    return P1, P2 




'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2
            
    Hints:
    (1) Given input [x1, x2], use the fundamental matrix to recover the corresponding epipolar line on image2
    (2) Search along this line to check nearby pixel intensity (you can define a search window) to 
        find the best matches
    (3) Use guassian weighting to weight the pixel simlairty

'''

def Gaussian(shape, sigma):
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    P1 = np.vstack((x1, y1, 1))
    e = np.matmul(F, P1)
    e = e/np.linalg.norm(e)
    a = e[0][0]
    b = e[1][0]
    c = e[2][0]

    step = 10
    sigma = 5
    min_dis = np.inf

    #filter code here
    x1 = int(round(x1))
    y1 = int(round(y1))
    x2 = 0
    y2 = 0
    patch1 = im1[y1-step:y1+step+1, x1-step:x1+step+1]
    kernel = Gaussian((2*step+1, 2*step+1), sigma)

    for i in range(y1-sigma*step, y1+sigma*step):
        x2_curr = (-b*i-c)/a
        # x2_temp = round(x2_curr)
        x2_curr = int(round(x2_curr))

        s_h = i-step
        e_h = i+step+1
        s_w = x2_curr-step
        e_w = x2_curr+step+1
        if s_w > 0 and e_w < im2.shape[1] and s_h > 0 and e_h < im2.shape[0]:
            patch2 = im2[s_h:e_h, s_w:e_w]

            weightedDist = []
            for l in range(0, patch2.shape[2]):
                dist = np.subtract(patch1[:, :, l], patch2[:, :, l])
                weightedDist.append(np.linalg.norm(np.matmul(kernel, dist)))
            error = sum(weightedDist)

            if error < min_dis:
                min_dis = error
                x2 = x2_curr
                y2 = i
    # print(f"Best Error {error}")
    return x2, y2
    



if __name__ == "__main__":

    correspondence = np.load('../data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('../data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')


    # ----- TODO -----
    # YOUR CODE HERE
    
    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))
    # epipolarMatchGUI(im1, im2, F)
    
    # Simple Tests to verify your implementation:
    x2, y2 = epipolarCorrespondence(im1, im2, F, 119, 217)
    assert(np.linalg.norm(np.array([x2, y2]) - np.array([118, 181])) < 10)
    
    P1, P2 = epipolarMatchGUI(im1, im2, F)

    if(os.path.isfile('q4_1.npz')==False):
        np.savez('q4_1.npz', F = F, P1 = P1, P2 = P2)