//
//  ml_graphic.h
//  KSVD
//
//  Created by sxjscience on 13-9-17.
//  Copyright (c) 2013年 sxjscience. All rights reserved.
//

#ifndef __KSVD__ml_graphic__
#define __KSVD__ml_graphic__

#include <iostream>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <string>
#include <vector>
#include <map>
#include <set>
typedef unsigned long u_long;

class ML_Image{
public:
    Eigen::MatrixXd R;
    Eigen::MatrixXd G;
    Eigen::MatrixXd B;
    u_long origin_rows;
    u_long origin_cols;
    ML_Image(const IplImage* original_img,double min_size,std::string mode = "RGB");
    Eigen::MatrixXd get_patch_mat(int patch_size,int step_size = 1,std::string mode = "RGB");
};

class PatchGen{
private:
    std::string rootpath;
    u_long sample_image_num;
    u_long patch_num;
    std::vector<std::string> filename_vec;
    void _load_file_list(std::vector<std::string> &vec, const std::string &path);
    Eigen::MatrixXd _gen_patch_given_pic(std::string image_path,std::string mode = "RGB",double min_size = 100);
public:
    u_long patch_width;
    u_long patch_height;
    u_long patch_num_per_image;
    
    Eigen::MatrixXd patches;
    PatchGen();
    PatchGen(std::string rootpath,u_long patch_width = 8,u_long patch_height = 8,u_long patch_num_per_image = 200);
    void load_file_list(std::string rootpath);
    void gen_patch(u_long patch_width,u_long patch_height,u_long patch_num_per_image,std::string mode = "RGB",double min_size = 100);
    ~PatchGen();
    
    
    
};

class ImageSet{
private:
    std::string rootpath;
    void _load_file(std::string label);
public:
    std::map<std::string,u_long> m_label;
    std::vector<ML_Image> image_vec;
    std::vector<std::string> image_path_vec;
    std::vector<u_long> label_vec;
    double min_size;
    ImageSet(std::string rootpath = "");
    void load_file_list(std::string rootpath ="");
    void init_image_vec(double min_size,std::string mode = "RGB");
    ~ImageSet();
};




#endif /* defined(__KSVD__ml_graphic__) */
