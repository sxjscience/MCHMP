//
//  ml_graphic_common.cpp
//  KSVD
//
//  Created by sxjscience on 13-9-28.
//  Copyright (c) 2013年 sxjscience. All rights reserved.
//

#include "ml_graphic_common.h"
#include "ml_openmp_common.h"
#include "ml_common.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "math.h"
void ML_GraphCommon::display_filter(Eigen::ArrayXXd filter, const std::string savepath, const std::string style ){
    if (style == "RGB") {
        //First Subtract mean value to set mid point to 0
        filter -= filter.mean();
        u_long cols = round(sqrt(filter.cols()));
        u_long channel_size = filter.rows()/3;
        u_long dim = sqrt(channel_size);
        u_long rows = ceil(filter.cols()/cols);
        Eigen::ArrayXXd R = filter.block(0, 0, channel_size, filter.cols());
        Eigen::ArrayXXd G = filter.block(channel_size, 0, channel_size, filter.cols());
        Eigen::ArrayXXd B = filter.block(channel_size*2, 0, channel_size, filter.cols());
        Eigen::ArrayXXd maxR = R.abs().colwise().maxCoeff();
        Eigen::ArrayXXd maxG = G.abs().colwise().maxCoeff();
        Eigen::ArrayXXd maxB = B.abs().colwise().maxCoeff();
        Eigen::ArrayXXd image_R = Eigen::ArrayXXd::Ones(dim*rows+rows-1, dim*cols+cols-1);
        Eigen::ArrayXXd image_G = Eigen::ArrayXXd::Ones(dim*rows+rows-1, dim*cols+cols-1);
        Eigen::ArrayXXd image_B = Eigen::ArrayXXd::Ones(dim*rows+rows-1, dim*cols+cols-1);
        for (int i = 0; i<channel_size; i++) {
            R.row(i) = R.row(i)/maxR;
            G.row(i) = G.row(i)/maxG;
            B.row(i) = B.row(i)/maxB;
        }
        for (int i=0; i<rows; i++) {
            for (int j=0; j<cols; j++) {
                if (i*cols+j+1>B.cols()) {
                    break;
                }
                Eigen::ArrayXXd temp = R.col(i*cols+j);
                temp.resize(dim,dim);
                image_R.block(i*(dim+1), j*(dim+1), dim, dim) = temp;
                temp = G.col(i*cols+j);
                temp.resize(dim,dim);
                image_G.block(i*(dim+1), j*(dim+1), dim, dim) = temp;
                temp = B.col(i*cols+j);
                temp.resize(dim,dim);
                image_B.block(i*(dim+1), j*(dim+1), dim, dim) = temp;
                
            }
        }
        image_R = (image_R+1)/2;
        image_G = (image_G+1)/2;
        image_B = (image_B+1)/2;
        //Begin to generate filter picture
        CvSize sz;
        sz.width = (int)image_R.cols();
        sz.height = (int)image_R.rows();
        IplImage* img = cvCreateImage(sz, IPL_DEPTH_64F, 3);
        for(int i =0;i<sz.height;i++){
            for (int j=0; j<sz.width; j++) {
                CvScalar s;
                s.val[0] = image_B(i,j);
                s.val[1] = image_G(i,j);
                s.val[2] = image_R(i,j);
                cvSet2D(img, i, j, s);
            }
        }
        cvNamedWindow("window");
        cvShowImage("window", img);
        cvWaitKey();
        cvDestroyWindow("window");
        cvReleaseImage(&img);
        
        
    }
}