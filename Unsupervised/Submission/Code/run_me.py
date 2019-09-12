import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from math import log2
#import tqdm
#import copy
#import sys

def visualize(im1, im2, k):
	# displays two images
    im1 = im1.astype('uint8')
    im2 = im2.astype('uint8')
    f = plt.figure()
    f.add_subplot(1,2, 1)
    plt.imshow(im1)
    plt.axis('off')
    plt.title('Original')
    f.add_subplot(1,2, 2)
    plt.imshow(im2)
    plt.axis('off')
    plt.title('Cluster: '+str(k))
    plt.savefig('k_means_'+str(k)+'.jpg')
    plt.show()
    return None

def MSE(Im1, Im2):
	# computes error
	Diff_Im = Im2-Im1
	Diff_Im = np.power(Diff_Im, 2)
	Diff_Im = np.sum(Diff_Im, axis=2)
	Diff_Im = np.sqrt(Diff_Im)
	sum_diff = np.sum(np.sum(Diff_Im))
	avg_error = sum_diff / float(Im1.shape[0]*Im2.shape[1])
	return avg_error

### Start of my (@Tamas Palfi) added code functions.

#function to complete question 1(C) and 1(D)
def KMeans(orig_img):
    # first get dimensions of original image
    orig_dim_x = orig_img.shape[0]
    orig_dim_y = orig_img.shape[1]
    # first we need to fix the dimensionality of the original image to be flattened into two dimensions
    # get one dimension representing full image
    dim = orig_img.shape[0] * orig_img.shape[1]
    flatten_img = np.reshape(orig_img, (dim, 3))
    #compute the amount of bits in original for compression metrics
    orig_bits = dim * 24
    # set up a list of the k-values, list to store reconstruction errors, and list for compression rates
    k_vals = [2,5,10,25,50,100,200]
    reconst_err = []
    compress_rate = []
    #loop through the differing k values and run KMeans on each k value
    for k in k_vals:
        #create a copy of the image to be used for reconstruction
        new_img = np.zeros(flatten_img.shape)
        # loop through to change zeros to actual values
        for x in range(new_img.shape[0]):
            new_img[x] = flatten_img[x]
        # get the KMeans Model
        KMeans = MiniBatchKMeans(k) #create KMeans with specific k value
        #with the model created, fit it on the orginal image to get centroids
        centroids = KMeans.fit(new_img)
        clusters = centroids.cluster_centers_
        #predict which cluster each point goes to
        pred_arr = KMeans.predict(new_img)
        #with the predicted cluster values of each point, loop through to change the pixels to reflect that
        for pixel_index in range(new_img.shape[0]):
            new_img[pixel_index] = clusters[pred_arr[pixel_index]]
        #now we have our reconstructed image with k clusters - reshape to original dimensions
        final_img = np.reshape(new_img,(orig_dim_x,orig_dim_y,3))
        #print the side-by-side of each image
        visualize(orig_img, final_img, k)
        #get the reconstruction error and add to the list
        re_err = MSE(orig_img, final_img)
        reconst_err.append(re_err)
        # with clusters can get number of bits in new img for compression metrics
        #get number of pixels in new_imgs
        new_dim = orig_img.shape[0]* orig_img.shape[1]
        new_bits = 32 * 3 * k + new_dim * log2(k)
        #with number of pixel in both original img and compress img - compute compression rate
        comp_rate = (1 - (new_bits/orig_bits))* 100
        compress_rate.append(comp_rate)
    #with all the images done and metrics calculated, print
    print("Reconstruction Errors (MSE) in increasing order of k [2,5,10,25,50,100,200] respectively: ")
    print(reconst_err)
    print("Compression Rates in increasing order of k [2,5,10,25,50,100,200] respectively: ")
    print(compress_rate)


#############################################################################
# grab the image
original_image = np.array(Image.open('../../Data/singapore.jpg'))

### Added function calls for output

#only need one call to the KMeans function - solves both (C) and (D)
KMeans(original_image)