/*
Name: Ebey Abraham
The following is my solution to the first task of the IP/CV internship evaluation
The code is genuine and has been written by me. 
I have explained my algorithm throughout the code through comments

The code was tested using OpenCV 3.2.0

Sources used for reference:
1.https://www.learnopencv.com/blob-detection-using-opencv-python-c/
*/

#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <iostream>

using namespace std;
using namespace cv;



int main(int argc, char *argv[])
{
    Mat img = imread(argv[1]);
    // Check if image is loaded or not
    if (!img.data)
        return -1;

    SimpleBlobDetector::Params BLOB;
    // parameter values for SimpleBlobDetector
    BLOB.thresholdStep = 5;
    BLOB.minThreshold = 10;
    BLOB.maxThreshold = 220;
    BLOB.minRepeatability = 2;
    BLOB.minDistBetweenBlobs = 1;
    BLOB.filterByColor = false;
    BLOB.filterByArea = true;
    BLOB.minArea = 500;
    BLOB.maxArea = 3000;
    BLOB.filterByCircularity = false;
    BLOB.filterByInertia = false;
    BLOB.minInertiaRatio =0.2;
    BLOB.filterByConvexity = false;
    /*
    The blobs have been detected based on their area, rest of the parameters have been set to false
    */
    
    
    vector<KeyPoint>  keyImg;
    
    Mat result=img.clone();
    
    Ptr<SimpleBlobDetector> sbd = SimpleBlobDetector::create(BLOB); 
    // Detect keypoints
    sbd->detect(img, keyImg, Mat());
    int i = 0;
    // iterate through keypoints and draw circles and mark centroids
    for (vector<KeyPoint>::iterator k = keyImg.begin(); k != keyImg.end(); ++k, ++i)
    {
    	Point center=k->pt;
     	int radius=k->size;
     	line(result,center-Point(5,5),center+Point(5,5),Scalar(0,255,0),3);
     	line(result,center-Point(5,-5),center+Point(5,-5),Scalar(0,255,0),3);
     	circle(result, center , radius, Scalar(0,0,255),2);
    }
    
    imshow("RESULT", result);
    //imshow("Original", img);
    waitKey(0);
    destroyWindow("RESULT");
    //destroyWindow("Original");
    return 0;
}