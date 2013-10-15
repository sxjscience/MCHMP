//
//  ml_openmp_common.h
//  KSVD
//
//  Created by sxjscience on 13-9-3.
//  Copyright (c) 2013年 sxjscience. All rights reserved.
//

#ifndef __KSVD__ml_openmp_common__
#define __KSVD__ml_openmp_common__

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
typedef unsigned long u_long;

//void quick_init_rand();

class ML_OpenMP{
public:
    static void matrix_normalization(Eigen::MatrixXd &mat,char operation = 'c');
    static void matrix_remove_dc(Eigen::MatrixXd &mat,char operation = 'c');

    static Eigen::MatrixXd matrix_with_selected_index(const Eigen::MatrixXd &mat,const std::vector<u_long> &row_index,const std::vector<u_long> &col_index);
    
    /*
     Function matrix_selected_index:
     Input:
     MatrixXd mat,
     Vector index,
     Operation c
     Output:
     Matrix With selected indexes in index
     
     */
    static Eigen::VectorXd vector_with_selected_index(const Eigen::VectorXd &vec,const std::vector<u_long> &index);
    static Eigen::MatrixXd matrix_with_selected_rows(const Eigen::MatrixXd &mat,const std::vector<u_long> &row_index);
    static Eigen::MatrixXd matrix_with_selected_rows(const Eigen::SparseMatrix<double> &mat,const std::vector<u_long> &row_index);

    static Eigen::MatrixXd matrix_with_selected_cols(const Eigen::MatrixXd &mat,const std::vector<u_long> &col_index);
    static Eigen::SparseMatrix<double> matrix_with_selected_cols(const Eigen::SparseMatrix<double> &mat,const std::vector<u_long> &col_index,const int T);
    
    static Eigen::MatrixXd matrix_patch_cols(const Eigen::MatrixXd &mat, const u_long &patch_width, const u_long &patch_height,const u_long &w_step=1, const u_long &h_step=1);
    static Eigen::MatrixXd matrix_with_selected_cols_max(const Eigen::MatrixXd &mat, const std::vector<u_long> &col_index);
    static Eigen::MatrixXd matrix_with_selected_rows_max(const Eigen::MatrixXd &mat, const std::vector<u_long> &row_index);
    
    /*
     Patch Mat: (?,origin_rows*origin_cols)
     Warning!!
    Columns of mat are small pathes of the patches of the original matrix. From Top to bottom then from left to right,that is
     1 4 7
     2 5 8
     3 6 9
     
     
     
     */
    
    static Eigen::MatrixXd matrix_max_pooling(const Eigen::MatrixXd &mat, const u_long pool_size, const u_long origin_rows,const u_long origin_cols);
    
    static Eigen::MatrixXd matrix_max_pooling(const Eigen::SparseMatrix<double> &mat, const u_long pool_size, const u_long origin_rows,const u_long origin_cols,const int T);
    
    /*
     
     Spatial Pyramid Levels could be [1,2,3], and there are 1*1 2*2 3*3 grids.Total:1*1+2*2+3*3 features
     */
    
    static Eigen::MatrixXd matrix_sp_max_pooling(const Eigen::MatrixXd &mat,const std::vector<u_long> spatial_pyramid_levels,const u_long origin_rows, const u_long origin_cols);
    
    static Eigen::MatrixXd matrix_patch_gen(const Eigen::MatrixXd &mat, const u_long patch_size, const u_long origin_rows, const u_long origin_cols);
    static Eigen::MatrixXd matrix_patch_sample(const Eigen::MatrixXd &mat, const u_long patch_size, const u_long origin_rows, const u_long origin_cols, const u_long sample_num);
    
        
};


#endif /* defined(__KSVD__openmp_common__) */
