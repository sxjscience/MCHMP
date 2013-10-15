//
//  ml_io.h
//  KSVD
//
//  Created by sxjscience on 13-10-1.
//  Copyright (c) 2013年 sxjscience. All rights reserved.
//

#ifndef __KSVD__ml_io__
#define __KSVD__ml_io__

#include <iostream>
#include <fstream>
#include "Eigen/Dense"
#include <string>
#include <vector>
typedef unsigned long u_long;

class ML_IO{
public:
    static void save_mat(const char* filename, const Eigen::MatrixXd & m);
    static Eigen::MatrixXd load_mat(const char* filename);
    static void save_mat_vector(const char* filename, const std::vector<Eigen::MatrixXd> &mat_vec);
    static std::vector<Eigen::MatrixXd> load_mat_vector(const char* filename);
    static bool file_exist(const std::string& filename);
    static void save_mat_to_txt(const char* filename, const Eigen::MatrixXd &m);
    static void format_mat_to_liblinear(const char*filename, const Eigen::MatrixXd &feature_m, const Eigen::MatrixXd &label_m);
    
};


#endif /* defined(__KSVD__ml_io__) */
