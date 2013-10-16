//
//  ml_openmp_common.cpp
//  KSVD
//
//  Created by sxjscience on 13-9-3.
//  Copyright (c) 2013年 sxjscience. All rights reserved.
//

#include "ml_openmp_common.h"
#include "ml_random.h"
#include "time.h"
#include "math.h"

void ML_OpenMP::matrix_normalization(Eigen::MatrixXd &mat,char operation){
    if(operation=='r'){
#pragma omp parallel for
        for(u_long i=0;i<mat.rows();i++){
            mat.row(i).normalize();
        }
    }
    else if(operation=='c'){
#pragma omp parallel for
        for(u_long i=0;i<mat.cols();i++){
            mat.col(i).normalize();
        }
    }
}


void ML_OpenMP::matrix_remove_dc(Eigen::MatrixXd &mat,char operation){
    if(operation == 'r'){
        mat = mat.colwise() - mat.rowwise().mean();
    }
    else if(operation == 'c'){
        mat = mat.rowwise() - mat.colwise().mean();
    }
}


Eigen::MatrixXd ML_OpenMP::matrix_with_selected_index(const Eigen::MatrixXd &mat,const std::vector<u_long> &row_index,const std::vector<u_long> &col_index){
    u_long row_index_len = row_index.size();
    u_long col_index_len = col_index.size();
    Eigen::MatrixXd result(row_index_len,col_index_len);
#pragma omp parallel for
    for (u_long i =0; i<row_index_len;i++) {
#pragma omp parallel for
        for (u_long j=0; j<col_index_len; j++) {
            result(i,j) = mat(row_index[i],col_index[j]);
        }
    }
    return result;
}



Eigen::VectorXd ML_OpenMP::vector_with_selected_index(const Eigen::VectorXd &vec,const std::vector<u_long> &index){
    u_long index_len = index.size();
    Eigen::VectorXd result(index_len);
#pragma omp parallel for
    for(u_long i=0;i<index_len;i++){
        result(i) = vec(index[i]);
    }
    return result;
}


Eigen::MatrixXd ML_OpenMP::matrix_with_selected_rows(const Eigen::MatrixXd &mat,const std::vector<u_long> &row_index){
    u_long row_len = row_index.size();
    Eigen::MatrixXd result(row_len,mat.cols());
#pragma omp parallel for
    for (u_long i=0; i<row_len; i++) {
        result.row(i) = mat.row(row_index[i]);
    }
    return result;
}

Eigen::MatrixXd ML_OpenMP::matrix_with_selected_rows(const Eigen::SparseMatrix<double> &mat,const std::vector<u_long> &row_index){
    u_long row_len = row_index.size();
    Eigen::MatrixXd result(row_len,mat.cols());
#pragma omp parallel for
    for (u_long i=0; i<row_len; i++) {
        result.row(i) = mat.row(row_index[i]);
    }
    return result;
}



Eigen::MatrixXd ML_OpenMP::matrix_with_selected_cols(const Eigen::MatrixXd &mat,const std::vector<u_long> &col_index){
    clock_t start,finish;

    u_long col_len = col_index.size();
    Eigen::MatrixXd result(mat.rows(),col_len);
#pragma omp parallel for
    for (u_long i=0; i<col_len; i++) {
        result.col(i) = mat.col(col_index[i]);
    }

    return result;
}


Eigen::SparseMatrix<double> ML_OpenMP::matrix_with_selected_cols(const Eigen::SparseMatrix<double> &mat,const std::vector<u_long> &col_index,const int T){
    clock_t start,finish;
    
    u_long col_len = col_index.size();
    Eigen::SparseMatrix<double> result(mat.rows(),col_len);
    result.reserve(Eigen::VectorXi::Constant(result.cols(), T));
#pragma omp parallel for
    for (u_long i=0; i<col_len; i++) {
        result.col(i) = mat.col(col_index[i]);
    }
    
    return result;
}

Eigen::MatrixXd ML_OpenMP::matrix_patch_cols(const Eigen::MatrixXd &mat, const u_long &patch_width, const u_long &patch_height, const u_long &w_step, const u_long &h_step){
    u_long patch_rows = (mat.rows()-patch_height)/h_step + 1;
    u_long patch_cols = (mat.cols()-patch_width)/w_step + 1;
    Eigen::MatrixXd result(patch_width*patch_height,patch_rows*patch_cols);
#pragma omp parallel for
    for (int i=0; i+patch_height<=mat.rows(); i+=h_step){
#pragma omp parallel for
        for (int j=0;j+patch_width<=mat.cols();j+=w_step){
            u_long res_col_index = (j/w_step)*patch_rows+(i/h_step);
            for (int incre_r = 0; incre_r<patch_height; incre_r++) {
                for (int incre_c = 0; incre_c<patch_width; incre_c++) {
                    u_long res_row_index = incre_r*patch_width+incre_c;
                    result(res_row_index,res_col_index) = mat(i+incre_r,j+incre_c);
                }
            }
        }
    }
    return result;
}

Eigen::MatrixXd ML_OpenMP::matrix_with_selected_cols_max(const Eigen::MatrixXd &mat, const std::vector<u_long> &col_index){
    return ML_OpenMP::matrix_with_selected_cols(mat,col_index).rowwise().maxCoeff();
}

Eigen::MatrixXd ML_OpenMP::matrix_with_selected_rows_max(const Eigen::MatrixXd &mat, const std::vector<u_long> &row_index){
    return ML_OpenMP::matrix_with_selected_rows(mat,row_index).colwise().maxCoeff();
}
Eigen::MatrixXd ML_OpenMP::matrix_max_pooling(const Eigen::MatrixXd &mat, const u_long pool_size, const u_long origin_rows,const u_long origin_cols){
    u_long pooled_rows = origin_rows/pool_size;
    u_long pooled_cols = origin_cols/pool_size;
    Eigen::MatrixXd result(mat.rows(),pooled_rows*pooled_cols);
    
    for (int i = 0 ; i<pooled_rows; i++) {
        for (int j=0; j<pooled_cols; j++) {
            std::vector<u_long> pooled_indices(pool_size*pool_size);
#pragma omp parallel for
            for (int z =0; z<pooled_indices.size(); z++) {
                //Related patch: (i*pool_size+z%pool_size,j*pool_size+z/pool_size) ==> (j*pool_size+z/pool_size)*origin_rows+(i*pool_size+z%pool_size)
                pooled_indices[z] = (j*pool_size+z/pool_size)*origin_rows+(i*pool_size+z%pool_size);
            }
            result.col(j*pooled_rows+i) = ML_OpenMP::matrix_with_selected_cols_max(mat, pooled_indices);
        }
    }
    
    return result;
}

Eigen::MatrixXd ML_OpenMP::matrix_max_pooling(const Eigen::SparseMatrix<double> &mat, const u_long pool_size, const u_long origin_rows,const u_long origin_cols,const int T){
    u_long pooled_rows = origin_rows/pool_size;
    u_long pooled_cols = origin_cols/pool_size;
    Eigen::MatrixXd result(mat.rows(),pooled_rows*pooled_cols);
    for (int i = 0 ; i<pooled_rows; i++) {
        for (int j=0; j<pooled_cols; j++) {
            std::vector<u_long> pooled_indices(pool_size*pool_size);
#pragma omp parallel for
            for (int z =0; z<pooled_indices.size(); z++) {
                //Related patch: (i*pool_size+z%pool_size,j*pool_size+z/pool_size) ==> (j*pool_size+z/pool_size)*origin_rows+(i*pool_size+z%pool_size)
                pooled_indices[z] = (j*pool_size+z/pool_size)*origin_rows+(i*pool_size+z%pool_size);
            }
            result.col(j*pooled_rows+i) = Eigen::MatrixXd(ML_OpenMP::matrix_with_selected_cols(mat, pooled_indices,T)).rowwise().maxCoeff();
        }
    }
    
    return result;
}

Eigen::MatrixXd ML_OpenMP::matrix_sp_max_pooling(const Eigen::MatrixXd &mat,const std::vector<u_long> spatial_pyramid_levels,const u_long origin_rows, const u_long origin_cols){
    std::vector<u_long> feature_num(spatial_pyramid_levels.size()+1);
    feature_num[0] = 0;
    for (int i=1; i<feature_num.size(); i++) {
        feature_num[i] = spatial_pyramid_levels[i-1]*spatial_pyramid_levels[i-1] + feature_num[i-1];
    }
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(mat.rows()*feature_num[feature_num.size()-1],1);
    for (int t=0; t<spatial_pyramid_levels.size(); t++) {
        u_long level = spatial_pyramid_levels[t];
        for (int i=0; i<level; i++) {
            for (int j=0; j<level; j++) {
                u_long pyramid_rows = origin_rows/level;
                u_long pyramid_cols = origin_cols/level;
                std::vector<u_long> pyramid_indices(pyramid_rows*pyramid_cols);
                for (int w=0; w<pyramid_indices.size(); w++) {
                    //(pyramid_rows*i+w%pyramid_rows,pyramid_cols*j+w/pyramid_rows) ==> (pyramid_cols*j+w/pyramid_rows)*origin_rows+pyramid_rows*i+w%pyramid_rows
                    pyramid_indices[w] = (pyramid_cols*j+w/pyramid_rows)*origin_rows+pyramid_rows*i+w%pyramid_rows;
                }
                result.block(mat.rows()*(feature_num[t]+j*level+i), 0, mat.rows(), 1) = ML_OpenMP::matrix_with_selected_cols_max(mat, pyramid_indices);
            }
        }
    }
    
    //Perform contrast normalization at the end
    result.normalize();
    
    return result;
}

Eigen::MatrixXd ML_OpenMP::matrix_patch_gen(const Eigen::MatrixXd &mat, const u_long patch_size, const u_long origin_rows, const u_long origin_cols){
    u_long res_rows = mat.rows()*patch_size*patch_size;
    u_long res_cols = (origin_rows-patch_size+1)*(origin_cols-patch_size+1);
    u_long new_rows = (origin_rows-patch_size+1);
    u_long new_cols = (origin_cols-patch_size+1);
    Eigen::MatrixXd result(res_rows,res_cols);
#pragma omp parallel for
    for (int n=0; n<result.cols(); n++) {
        for (int i=0;i<patch_size;i++){
            for (int j=0;j<patch_size ; j++) {
                //(n%new_rows+i,n/new_rows+j) ===> (n/new_rows+j)*origin_rows+n%new_rows+i
                Eigen::MatrixXd single_col = mat.col((n/new_rows+j)*origin_rows+n%new_rows+i);
                double norm = sqrt(single_col.squaredNorm());
                double thres = 0.1;
                single_col = single_col/((norm>thres)?norm:thres);
                result.block((j*patch_size+i)*mat.rows(), n, mat.rows(), 1) = single_col;
            }
        }
    }
    return result;
}

Eigen::MatrixXd ML_OpenMP::matrix_patch_sample(const Eigen::MatrixXd &mat, const u_long patch_size, const u_long origin_rows, const u_long origin_cols, const u_long sample_num){
    std::cout<<"origin_rows"<<origin_rows<<std::endl;
    std::cout<<"origin_cols"<<origin_cols<<std::endl;
    ML_Random rng;
    u_long res_rows = mat.rows()*patch_size*patch_size;
    u_long res_cols = sample_num;
    u_long new_rows = (origin_rows-patch_size+1);
    u_long new_cols = (origin_cols-patch_size+1);
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(res_rows,res_cols);
    std::vector<u_long> indices(new_rows*new_cols);
#pragma omp parallel for
    for (int i=0; i<indices.size(); i++) {
        indices[i] = i;
    }
    rng.random_permutation_n(indices,sample_num);
#pragma omp parallel for
    for (int n=0; n<sample_num; n++) {
        u_long sel = indices[n];
        for (int i=0;i<patch_size;i++){
            for (int j=0;j<patch_size ; j++) {
                //(sel%new_rows+i,sel/new_rows+j) ===> (sel/new_rows+j)*origin_rows+sel%new_rows+i
                Eigen::MatrixXd single_col = mat.col((sel/new_rows+j)*origin_rows+sel%new_rows+i);
                double norm = sqrt(single_col.squaredNorm());
                double thres = 0.1;
                single_col = single_col/((norm>thres)?norm:thres);
                result.block((j*patch_size+i)*mat.rows(), n, mat.rows(), 1) = single_col;
            }
        }
    }
    std::cout<<"RUN!"<<std::endl;
    return result;
    
}


