# ML-UnsupervisedLearning
This project was written with Python 3.6.

The main goal of this project was to look into the use of unsupervised methods for ML opposed to what is more commonly used such as supervised learning (regression,classification).  The method used for this project today is K-mean clustering unsupervised method.  The way in which K-means works is to assign centroids based on the value of k and nearby 'data points' will be assigned to the closest centroid, thus, forming a cluster.  

The **applciation** of K-means for this project will be to take an image of a skyline of a city and compressing the image to only a certain number of RGB combinations for each pixel effectively assigning each to a 'color cluster'.  The centroids represent an RGB color so a lower k would only have very few colors and a higher k would mor refelectively look like the original image.  The different values of k used were {2; 5; 10; 25; 50; 100; 200}.  Please check out the 'Report' PDF to see the results of this code which includes the reconstruction error and compression rate.

