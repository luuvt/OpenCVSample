//
//  LaneDetector.hpp
//  OpenCVSample_iOS
//
//  Created by Luu Tran on 06/09/2022.
//  Copyright © 2022 test. All rights reserved.
//

#ifndef LaneDetector_hpp
#define LaneDetector_hpp

#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class LaneDetector {
    
    public:
    
    /*
     Returns image with lane overlay
     */
    Mat detect_lane(Mat image);
    
    private:
    
    /*
     Filters yellow and white colors on image
     */
    Mat filter_only_yellow_white(Mat image);
    
    /*
     Crops region where lane is most likely to be.
     Maintains image original size with the rest of the image blackened out.
     */
    Mat crop_region_of_interest(Mat image);
    
    /*
     Draws road lane on top image
     */
    Mat draw_lines(Mat image, vector<Vec4i> lines);
    
    /*
     Detects road lanes edges
     */
    Mat detect_edges(Mat image);
};
#endif /* LaneDetector_hpp */
