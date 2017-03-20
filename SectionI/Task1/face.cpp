#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main( int argc, const char** argv )
{
	Mat img = imread(argv[1]); // Read image data
    
	Mat image=img.clone();	// Create a clone of the original image
     
	if (img.empty()) // Check whether the image is loaded or not
   {
   	cout << "Image unable to load!" << endl;
      return -1;
   }
   // Store paths of cascade files
   String smile_cascade_path="haarcascade_smile.xml";
   String face_cascade_path="haarcascade_frontalface_alt.xml";
   String cat_cascade_path="haarcascade_frontalcatface.xml";
   String eye_cascade_path="haarcascade_eye_tree_eyeglasses.xml";
   // Declare objects of CascadeClassifier class
   CascadeClassifier face_cascade;	// For face detection
   CascadeClassifier smile_cascade;	// For smile detection
   CascadeClassifier cat_cascade;	
   CascadeClassifier eye_cascade;
   // Load the cascades and check whether they have been loaded
   if( !face_cascade.load( face_cascade_path ) )
   { 
     	cout<<"ERROR LOADING HAARCASCADE_FRONTALFACE!!";
    	return -1;
   }
   if( !smile_cascade.load( smile_cascade_path ) )
   { 
    	cout<<"ERROR LOADING HAARCASCADE_SMILE!!";
     	return -1;
   }
   if( !cat_cascade.load( cat_cascade_path ) )
   { 
    	cout<<"ERROR LOADING HAARCASCADE_FRONTAL-CAT-FACE!!";
     	return -1;
   }
   if( !eye_cascade.load( eye_cascade_path ) )
   { 
    	cout<<"ERROR LOADING HAARCASCADE_EYE!!";
     	return -1;
   }
     
     
   vector<Rect> faces;
   Mat gray;
   cvtColor(img,gray,CV_BGR2GRAY);

   // Detect faces in the image using haar-cascade
   face_cascade.detectMultiScale( gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(70, 70) );

   // Whiten out all the faces that have been detected
   // The left over image will be checked for faces that were not detected by haar-cascade-frontalface
   for(size_t i = 0; i < faces.size(); i++ )
   {
   	// Taking the top-left coordinates of i-th face that has been detected
   	int X=faces[i].x;
   	int Y=faces[i].y;
     	int W=faces[i].width;
     	int H=faces[i].height;
     	// Whitening out the i-th face
    	rectangle(gray,Point(X, Y-30),Point(X+W+10, Y+H+60),Scalar(255, 255, 255),-1);
	}
	
     	// Detect smiles in the left over image to find undetected faces
     	vector<Rect> smiles;
     	smile_cascade.detectMultiScale( gray, smiles, 1.3, 5, 0|CV_HAAR_SCALE_IMAGE, Size(40,40));
     
     	// Draw rectangles around the detected smiles
     	for(size_t i = 0; i < smiles.size(); i++ )
     	{
     		// Taking the top-left coordinates of i-th smile that has been detected
     		int X=smiles[i].x;
     		int Y=smiles[i].y;
     		int W=smiles[i].width;
     		int H=smiles[i].height;
     		//	Drawing rectangles around the detected smiles
     		// The area of the bounding rectangle has been increased to accomodate the faces that contain the detected smiles
    		rectangle(image,Point(X-20, Y-50),Point(X+W+20, Y+H+20),Scalar(0, 0, 255),2);
		}	
	
  		// Draw rectangles around all the faces that were detected using haar-cascade-frontalface
    	for(size_t i = 0; i < faces.size(); i++ )
     	{
     		int X=faces[i].x;
     		int Y=faces[i].y;
     		int W=faces[i].width;
     		int H=faces[i].height;
    		rectangle(image,Point(X, Y),Point(X+W, Y+H),Scalar(0, 0, 255),2);
		}
		
		vector<Rect> cat;
		cvtColor(img,gray,CV_BGR2GRAY);
		
		cat_cascade.detectMultiScale( gray, cat, 1.1, 1,0|CV_HAAR_SCALE_IMAGE,Size(80,80));
		
		for(size_t i = 0; i < cat.size(); i++ )
     	{
     		
     		int X=cat[i].x;
     		int Y=cat[i].y;
     		int W=cat[i].width;
     		int H=cat[i].height;
     		
    		rectangle(image,Point(X, Y),Point(X+W, Y+H),Scalar(255, 0, 0),3);
    		
    		Mat ROI=gray(cat[i]);
    		vector<Rect> eyes;
    		eye_cascade.detectMultiScale(ROI,eyes,1.1,2,0|CV_HAAR_SCALE_IMAGE,Size(30,30));
    		cout<<"BGR values of Ellen's eyes:\n";
    		for( size_t j = 0; j < eyes.size(); j++ )
     		{
       		int X=cat[i].x + eyes[j].x + eyes[j].width*0.5;
       		int Y=cat[i].y + eyes[j].y + eyes[j].height*0.5;
       		Point center(X,Y);
       		cout<<image.at<Vec3b>(Y,X)<<endl;
       		int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
       		circle( image, center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
     		}
		}	
      namedWindow("FACES",CV_WINDOW_AUTOSIZE);
     	imshow("FACES",image);	// Display the final image
     
     	waitKey(0); //	Wait indefinitely for a keypress
     	destroyWindow("FACES"); // Destroy all windows

     	return 0;
}