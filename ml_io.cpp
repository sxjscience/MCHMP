//
//  ml_io.cpp
//  KSVD
//
//  Created by sxjscience on 13-10-1.
//  Copyright (c) 2013年 sxjscience. All rights reserved.
//

#include "ml_io.h"
#include <sys/stat.h>

void ML_IO::save_mat(const char* filename, const Eigen::MatrixXd & m){
    Eigen::MatrixXd::Index rows,cols;
    rows = m.rows();
    cols = m.cols();
    
    std::ofstream f(filename,std::ios::binary);
    f.write((char*)&rows,sizeof(m.rows()));
    f.write((char*)&cols,sizeof(m.cols()));
    f.write((char*)m.data(),sizeof(Eigen::MatrixXd::Scalar)*cols*rows);
    f.close();
}

Eigen::MatrixXd ML_IO::load_mat(const char* filename){
    Eigen::MatrixXd::Index rows,cols;
    std::ifstream f(filename,std::ios::binary);
    f.read((char*)&rows, sizeof(rows));
    f.read((char*)&cols, sizeof(cols));
    Eigen::MatrixXd result(rows,cols);
    f.read((char*)result.data(), sizeof(Eigen::MatrixXd::Scalar)*cols*rows);
    f.close();
    return result;
}

void ML_IO::save_mat_vector(const char* filename, const std::vector<Eigen::MatrixXd> &mat_vec){
    std::ofstream f(filename,std::ios::binary);
    u_long vec_size = mat_vec.size();
    f.write((char*)&vec_size,sizeof(vec_size));
    for(u_long i=0;i<vec_size;i++){
        Eigen::MatrixXd::Index rows,cols;
        rows = mat_vec[i].rows();
        cols = mat_vec[i].cols();
        f.write((char*)&rows,sizeof(mat_vec[i].rows()));
        f.write((char*)&cols,sizeof(mat_vec[i].cols()));
        f.write((char*)mat_vec[i].data(),sizeof(Eigen::MatrixXd::Scalar)*cols*rows);
    }
    f.close();

}

std::vector<Eigen::MatrixXd> ML_IO::load_mat_vector(const char* filename){
    std::ifstream f(filename,std::ios::binary);
    u_long vec_size;
    f.read((char*)&vec_size,sizeof(vec_size));
    std::vector<Eigen::MatrixXd> result(vec_size);
    for(u_long i=0;i<vec_size;i++){
        Eigen::MatrixXd::Index rows,cols;
        f.read((char*)&rows, sizeof(rows));
        f.read((char*)&cols, sizeof(cols));
        Eigen::MatrixXd temp(rows,cols);
        f.read((char*)temp.data(), sizeof(Eigen::MatrixXd::Scalar)*cols*rows);
        result[i] = temp;
    }
    f.close();
    return result;
}

bool ML_IO::file_exist(const std::string &filename){
    struct stat buffer;
    return (stat (filename.c_str(), &buffer) == 0);
}

void ML_IO::save_mat_to_txt(const char* filename, const Eigen::MatrixXd &m){
    std::ofstream f(filename,std::ios::binary);
    for(int i=0;i<m.rows();i++){
        for(int j=0;j<m.cols();j++){
            f<<m(i,j)<<" ";
        }
        f<<std::endl;
    }
    f.close();
}

void ML_IO::format_mat_to_liblinear(const char* filename, const Eigen::MatrixXd &feature_m, const Eigen::MatrixXd &label_m){
    std::ofstream f(filename,std::ios::binary);
    for (int i=0; i<feature_m.cols(); i++) {
        u_long label = (u_long)label_m(0,i);
        f<<label<<" ";
        for (int j=0; j<feature_m.rows(); j++) {
            f<<j+1<<":"<<feature_m(j,i)<<" ";
        }
        f<<std::endl;
    }
}
