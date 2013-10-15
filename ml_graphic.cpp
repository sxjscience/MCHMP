//
//  ml_graphic.cpp
//  KSVD
//
//  Created by sxjscience on 13-9-17.
//  Copyright (c) 2013年 sxjscience. All rights reserved.
//

#include "ml_graphic.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "ml_random.h"
#include "ml_openmp_common.h"
using namespace cv;

ML_Image::ML_Image(const IplImage* original_img,double min_size,std::string mode){
    CvSize sz;
    if (original_img->width<original_img->height) {
        sz.width = original_img->width*min_size/original_img->height;
        sz.height = min_size;
        
    }
    else{
        sz.width = min_size;
        sz.height = original_img->height*min_size/original_img->width;
    }
    IplImage *img = cvCreateImage(sz, original_img->depth, original_img->nChannels);
    cvResize(original_img, img);
    this->R = Eigen::MatrixXd::Zero(img->height,img->width);
    this->G = Eigen::MatrixXd::Zero(img->height, img->width);
    this->B = Eigen::MatrixXd::Zero(img->height, img->width);
    
    //#TODO Using OpenMP To Do this
    for (int i=0;i<img->height ; i++) {
        for (int j=0; j<img->width; j++) {
            CvScalar s = cvGet2D(img, i, j);
            R(i,j) = (double)s.val[2]/255;
            G(i,j) = (double)s.val[1]/255;
            B(i,j) = (double)s.val[0]/255;
        }
    }
    cvReleaseImage(&img);
}

Eigen::MatrixXd ML_Image::get_patch_mat(int patch_size, int step_size,std::string mode){
    this->origin_rows = (this->R.rows()-patch_size)/step_size+1;
    this->origin_cols = (this->R.cols()-patch_size)/step_size+1;
    if (mode == "RGB") {
        Eigen::MatrixXd R_patch = ML_OpenMP::matrix_patch_cols(this->R, patch_size, patch_size,step_size,step_size);
        Eigen::MatrixXd G_patch = ML_OpenMP::matrix_patch_cols(this->G, patch_size, patch_size,step_size,step_size);
        Eigen::MatrixXd B_patch = ML_OpenMP::matrix_patch_cols(this->B, patch_size, patch_size,step_size,step_size);
        Eigen::MatrixXd result(R_patch.rows()*3,R_patch.cols());
        result.topRows(R_patch.rows()) = R_patch;
        result.middleRows(R_patch.rows(), R_patch.rows()) = G_patch;
        result.bottomRows(R_patch.rows()) = B_patch;
        ML_OpenMP::matrix_remove_dc(result);
        return result;
    }
    else if(mode == "GRAY"){
        //0.299*R+0.587*G+0.114*B

        Eigen::MatrixXd result = ML_OpenMP::matrix_patch_cols(0.299*R+0.587*G+0.114*B, patch_size, patch_size,step_size,step_size);
        ML_OpenMP::matrix_remove_dc(result);
        return result;
    }
    
}


PatchGen::PatchGen(){
    this->rootpath = "";
    this->patch_width = 0;
    this->patch_height = 0;
    this->patch_num_per_image = 0;
}

PatchGen::PatchGen(std::string rootpath,u_long patch_width,u_long patch_height,u_long patch_num_per_image){
    this->rootpath = rootpath;
    this->patch_width = patch_width;
    this->patch_height = patch_height;
    this->patch_num_per_image = patch_num_per_image;
    this->load_file_list(rootpath);
    
}

void PatchGen::load_file_list(std::string rootpath){
    this->_load_file_list(this->filename_vec, rootpath);
    this->sample_image_num = this->filename_vec.size();
}

void PatchGen::_load_file_list(std::vector<std::string> &vec,const std::string &path){
    struct dirent* ent = NULL;
    DIR *pDir;
    pDir = opendir(path.c_str());
    if(pDir == NULL){
        return;
    }
    while (NULL != (ent = readdir(pDir))) {
        if (ent->d_type == 8) {
            //file
            std::string filename = ent->d_name;
            
            if (filename.substr(filename.find_last_of(".")+1) == "jpg") {
                std::cout<<path+"/"+filename<<std::endl;
                vec.push_back(path+"/"+filename);
            }
        }
        else{
            if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) {
                continue;
            }
            //directory
            this->_load_file_list(vec, path+"/"+ent->d_name);
        }
    }
}




Eigen::MatrixXd PatchGen::_gen_patch_given_pic(std::string image_path,std::string mode,double min_size){
    assert(this->patch_width>0 && this->patch_height>0 && this->patch_num_per_image>0);
    assert(this->patch_width<min_size && this->patch_height<min_size);
    u_long patch_size = this->patch_width * this->patch_height;
    ML_Random rng;
    /*
    First Step: Loading and resizing the original image
    Using cvLoadImage(...,1) to assure the image to load in RGB
     */
    IplImage *original_img = cvLoadImage(image_path.c_str(),1);
    if(original_img==NULL){
        std::cout<<"FATAL ERROR,\""<<image_path<<"\" DO NOT EXIST!!!(OR CANNOT BE OPENED)"<<std::endl;
        return Eigen::MatrixXd::Zero(0, 0);
    }
    CvSize sz;
    if (original_img->width<original_img->height) {
        sz.width = original_img->width*min_size/original_img->height;
        sz.height = min_size;

    }
    else{
        sz.width = min_size;
        sz.height = original_img->height*min_size/original_img->width;
    }
    IplImage *img = cvCreateImage(sz, original_img->depth, original_img->nChannels);
    cvResize(original_img, img);
    cvReleaseImage(&original_img);

    /*
    Second Step: Generating Random Patches according to parameters given by the user
     */
    if(mode == "RGB"){
        assert(img->nChannels==3);
        Eigen::MatrixXd result(patch_size*3,this->patch_num_per_image);
        
        //Generating Random Permutation
        //#TODO Using OpenMP
        std::vector<u_long> vec((img->width-this->patch_width+1)*(img->height-this->patch_height+1));
        for (int i=0; i<vec.size(); i++) {
            vec[i] = i;
        }
        assert(vec.size()>this->patch_num_per_image);
        rng.random_permutation_n(vec, this->patch_num_per_image);
        
        //Generating Patches
        //#TODO Using OpenMP
        for (int i=0; i<this->patch_num_per_image; i++) {
            u_long patch_index = vec[i];
            u_long patch_w = patch_index%(img->width-this->patch_width+1);
            u_long patch_h = patch_index/(img->width-this->patch_width+1);
            for (u_long w = patch_w; w<patch_w+this->patch_width; w++) {
                for (u_long h = patch_h; h<patch_h+this->patch_height; h++) {
                    //std::cout<<"Width:"<<img->width<<" Height:"<<img->height<<" patch_width:"<<this->patch_width<<" patch_height:"<<this->patch_height<<" w = "<<w<<" h = "<<h<<std::endl;
                    CvScalar s;
                    s = cvGet2D(img, (int)h, (int)w);
                    
                    //B:s.val[0] G:s.val[1] R:s.val[2]
                    for (int channel = 0; channel<3; channel++) {
                        u_long res_row_index = channel*patch_height*patch_width+(w-patch_w)*patch_height+(h-patch_h);
                        result(res_row_index,i) = s.val[2-channel]/(double)255;
                    }
                    
                }
            }
            
        }
        cvReleaseImage(&img);
        ML_OpenMP::matrix_remove_dc(result);
        return result;

    }
    else if(mode == "GRAY"){
        Eigen::MatrixXd result(patch_size,this->patch_num_per_image);
        
        //Generating Random Permutation
        //#TODO Using OpenMP
        std::vector<u_long> vec((img->width-this->patch_width+1)*(img->height-this->patch_height+1));
        for (int i=0; i<vec.size(); i++) {
            vec[i] = i;
        }
        assert(vec.size()>this->patch_num_per_image);
        rng.random_permutation_n(vec, this->patch_num_per_image);
        
        //Generating Patches
        //#TODO Using OpenMP
        for (int i=0; i<this->patch_num_per_image; i++) {
            u_long patch_index = vec[i];
            u_long patch_w = patch_index%(img->width-this->patch_width+1);
            u_long patch_h = patch_index/(img->width-this->patch_width+1);
            for (u_long w = patch_w; w<patch_w+this->patch_width; w++) {
                for (u_long h = patch_h; h<patch_h+this->patch_height; h++) {
                    //std::cout<<"Width:"<<img->width<<" Height:"<<img->height<<" patch_width:"<<this->patch_width<<" patch_height:"<<this->patch_height<<" w = "<<w<<" h = "<<h<<std::endl;
                    CvScalar s;
                    s = cvGet2D(img, (int)h, (int)w);
                    
                    //Gray = 0.299*R+0.587*G+0.114*B 
                    u_long res_row_index = (w-patch_w)*patch_height+(h-patch_h);
                    result(res_row_index,i) = (0.299*s.val[2]+0.587*s.val[1]+0.114*s.val[0])/(double)255;
                    
                }
            }
            
        }
        
        cvReleaseImage(&img);
        ML_OpenMP::matrix_remove_dc(result);
        return result;


    }
    else{
        Eigen::MatrixXd result(0,0);
        cvReleaseImage(&img);
        return result;
    }
    
}

void PatchGen::gen_patch(u_long patch_width,u_long patch_height,u_long patch_num_per_image,std::string mode,double min_size){
    this->patch_width = patch_width;
    this->patch_height = patch_height;
    this->patch_num_per_image = patch_num_per_image;
    u_long patch_size = this->patch_width*this->patch_height;
    this->sample_image_num = this->filename_vec.size();
    //#TODO Use OpenMP
    if(mode == "RGB"){
        this->patches = Eigen::MatrixXd::Zero(patch_size*3,this->patch_num_per_image*this->sample_image_num);
        for (u_long i = 0; i<this->sample_image_num; i++) {
            Eigen::MatrixXd mat = this->_gen_patch_given_pic(this->filename_vec[i],mode,min_size);
            std::cout<<"i:"<<i<<" row:"<<mat.rows()<<" col:"<<mat.cols()<<std::endl;
            this->patches.block(0, i*this->patch_num_per_image, patch_size*3, this->patch_num_per_image) = mat;
        }
        //Remove DC
        ML_OpenMP::matrix_remove_dc(this->patches,'c');
    }
    else if(mode == "GRAY"){
        this->patches = Eigen::MatrixXd::Zero(patch_size,this->patch_num_per_image*this->sample_image_num);
        for (u_long i = 0; i<this->sample_image_num; i++) {
            Eigen::MatrixXd mat = this->_gen_patch_given_pic(this->filename_vec[i],mode,min_size);
            std::cout<<"i:"<<i<<" row:"<<mat.rows()<<" col:"<<mat.cols()<<std::endl;
            this->patches.block(0, i*this->patch_num_per_image, patch_size, this->patch_num_per_image) = mat;
        }
        //Remove DC
        ML_OpenMP::matrix_remove_dc(this->patches,'c');
    }
    
}

PatchGen::~PatchGen(){
    
}

ImageSet::ImageSet(std::string rootpath){
    this->rootpath = rootpath;
}


void ImageSet::_load_file(std::string label){
    std::string path = this->rootpath+"/"+label;
    struct dirent* ent =NULL;
    DIR *pDir;
    pDir = opendir(path.c_str());
    assert(pDir!=NULL);
    while(NULL!=(ent =readdir(pDir))){
        if(ent->d_type==8){
            std::string filename = ent->d_name;
            this->image_path_vec.push_back(path+"/"+filename);
            this->label_vec.push_back(this->m_label[label]);
        }
    }
}

void ImageSet::load_file_list(std::string rootpath){
    if(rootpath != ""){
        this->rootpath = rootpath;
    }
    rootpath = this->rootpath;
    struct dirent* ent = NULL;
    DIR *pDir;
    pDir = opendir(rootpath.c_str());
    if(pDir == NULL){
        return;
    }
    while (NULL != (ent = readdir(pDir))) {
        if (ent->d_type == 8) {
            //file should be skipped
            continue;
            
        }
        else{
            if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) {
                continue;
            }
            //directory
            std::string label = ent->d_name;
            if(this->m_label.find(label) == this->m_label.end()){
                this->m_label[label] = this->m_label.size();
            }
            this->_load_file(label);
        }
    }
}

void ImageSet::init_image_vec(double min_size,std::string mode){
    this->min_size = min_size;
    for (int i=0;i<this->image_path_vec.size();i++){
        IplImage* img = cvLoadImage(image_path_vec[i].c_str(),1);
        assert(img!=NULL);
        ML_Image im(img,min_size,mode);
        cvReleaseImage(&img);
        std::cout<<im.R.rows()<<" "<<im.R.cols()<<std::endl;
        this->image_vec.push_back(im);
    }
}

ImageSet::~ImageSet(){

}
//void opencv_test(){
//    IplImage * pInpImg = 0;
//    pInpImg = cvLoadImage("/Users/sxjscience/Downloads/stereogram.jpg", CV_LOAD_IMAGE_UNCHANGED);
//    cvNamedWindow( "Display window", CV_WINDOW_AUTOSIZE );
//    cvShowImage("Display window",pInpImg);
//    waitKey(0);
//    // Remember to free image memory after using it!
//    cvReleaseImage(&pInpImg);
//    
//}

