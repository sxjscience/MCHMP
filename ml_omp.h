//
//  omp.h
//  KSVD
//
//  Created by sxjscience on 13-9-2.
//  Copyright (c) 2013年 sxjscience. All rights reserved.
//

#ifndef __KSVD__omp__
#define __KSVD__omp__

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
typedef unsigned long u_long;




Eigen::SparseVector<double> OMP_naive(Eigen::MatrixXd dictionary,Eigen::VectorXd signal,u_long T);

Eigen::SparseMatrix<double> OMP_cholesky(Eigen::MatrixXd dictionary,Eigen::MatrixXd signal_mat,u_long T,double episilon = 1E-6);

void OMP_batch(Eigen::SparseMatrix<double> &result, const Eigen::MatrixXd &dictionary,const Eigen::MatrixXd &signal_mat,u_long T,std::vector< std::vector<u_long> > &non_zero_index,const double episilon = 1E-6);

void OMP_batch_sign(Eigen::SparseMatrix<double> &result, const Eigen::MatrixXd &dictionary,const Eigen::MatrixXd &signal_mat,u_long T,std::vector< std::vector<u_long> > &non_zero_index,const double episilon = 1E-6);

void block_OMP_batch(Eigen::SparseMatrix<double> &result, const Eigen::MatrixXd &dictionary,const Eigen::MatrixXd &signal_mat, std::vector< std::vector<u_long> > &block , const u_long &T,std::vector< std::vector<u_long> > &non_zero_index,const double episilon = 1E-6);

void block_OMP_batch_sign(Eigen::SparseMatrix<double> &result, const Eigen::MatrixXd &dictionary,const Eigen::MatrixXd &signal_mat, std::vector< std::vector<u_long> > &block , const u_long &T,std::vector< std::vector<u_long> > &non_zero_index,const double episilon = 1E-6);

#endif /* defined(__KSVD__omp__) */
