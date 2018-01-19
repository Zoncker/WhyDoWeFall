#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

void detectAndDisplay( Mat frame );

CascadeClassifier walmart_cascade, mobil_cascade, subway_cascade, atnt_cascade;

String window_name = "Capture - Logo detection";

int main( int argc, const char** argv )
{
    VideoCapture capture;
    Mat frame;

    //-- 1. Load the cascades
    if( !walmart_cascade.load( "cascades/walmart.xml" ) ){ printf("--(!)Error loading Walmart cascade\n"); return -1; };
    if( !mobil_cascade.load( "cascades/mobil.xml" ) ){ printf("--(!)Error loading Mobil cascade\n"); return -1; };
    if( !subway_cascade.load( "cascades/subway.xml" ) ){ printf("--(!)Error loading Subway cascade\n"); return -1; };
    if( !atnt_cascade.load( "cascades/att.xml" ) ){ printf("--(!)Error loading AT&T cascade\n"); return -1; };

    //-- 2. Read the video stream
    capture.open( 0 ); //
    if ( ! capture.isOpened() ) { printf("Error opening video capture\n"); return -1; }

    while ( capture.read(frame) )
    {
        if( frame.empty() )
        {
            printf(" --(!) No captured frame -- Break!");
            break;
        }
        //-- 3. Apply the classifier to the frame
        detectAndDisplay( frame );

        char c = (char)waitKey(10);
        if( c == 27 ) { break; } // escape
    }
    return 0;
}

void detectAndDisplay( Mat frame )
{
    vector<Rect> walmart;
    vector<Rect> mobil;
    vector<Rect> subway;
    vector<Rect> atnt;
    Mat frame_gray;

    cvtColor( frame, frame_gray, COLOR_RGB2GRAY ); //transforms into grayscale
    equalizeHist( frame_gray, frame_gray ); //Equalizes the histogram of a grayscale image.
//Finds different-sized objects in input image, which format is Mat CV_8U, "walmart" is a vector where each rectangle contains source obj, 1.1 is scale 
    walmart_cascade.detectMultiScale( frame_gray, walmart, 1.1, 20, 0|CASCADE_SCALE_IMAGE, Size(30, 30) ); 
    {
        Point center( walmart[i].x + walmart[i].width/2, walmart[i].y + walmart[i].height/2 );
        ellipse( frame, center, Size( walmart[i].width/2, walmart[i].height/2 ), 0, 0, 360, Scalar( 255, 242, 40 ), 4, 8, 0 );// bgr - cyan
    }

    mobil_cascade.detectMultiScale( frame_gray, mobil, 1.1, 20, 0 |CASCADE_SCALE_IMAGE, Size(30, 30) );

       for (size_t i = 0; i < mobil.size(); i++ )
        {
             Point mobil_center( mobil[i].x + mobil[i].width/2, mobil[i].y + mobil[i].height/2 );
             ellipse( frame, mobil_center, Size( mobil[i].width/2, mobil[i].height/2 ), 0, 0, 360, Scalar( 255, 26, 0 ), 4, 8, 0 ); //blue
        }

    subway_cascade.detectMultiScale( frame_gray, subway, 1.1, 20, 0 |CASCADE_SCALE_IMAGE, Size(30, 30) );

       for (size_t i = 0; i < subway.size(); i++ )
        {
             Point subway_center( subway[i].x + subway[i].width/2, subway[i].y + subway[i].height/2 );
             ellipse( frame, subway_center, Size( subway[i].width/2, subway[i].height/2 ), 0, 0, 360, Scalar( 0, 0, 255 ), 4, 8, 0 ); //red
        }

    atnt_cascade.detectMultiScale( frame_gray, atnt, 1.1, 20, 0 |CASCADE_SCALE_IMAGE, Size(30, 30) );

       for (size_t i = 0; i < atnt.size(); i++ )
        {
             Point atnt_center( atnt[i].x + atnt[i].width/2, atnt[i].y + atnt[i].height/2 );
             ellipse( frame, atnt_center, Size( atnt[i].width/2, atnt[i].height/2 ), 0, 0, 360, Scalar( 49, 63, 61 ), 4, 8, 0 ); //grey
        }

    imshow( window_name, frame );
}
