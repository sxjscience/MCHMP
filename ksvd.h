//
//  ksvd.h
//  KSVD
//
//  Created by sxjscience on 13-9-5.
//  Copyright (c) 2013年 sxjscience. All rights reserved.
//

#ifndef __KSVD__ksvd__
#define __KSVD__ksvd__

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "ml_random.h"
#include <set>
typedef unsigned long u_long;

/*
 Normal KSVD algorithm
 
 */

void KSVD(Eigen::MatrixXd &dictionary, Eigen::SparseMatrix<double> &sparse_mat_result,const Eigen::MatrixXd &signal_mat, const u_long T, const u_long max_iter = 10, const double episilon = 1E-6);
/*
 
 Warning!!!
 Now in KSVD_opt_mutual_incoherence there is NO CLEAR DICT STEP
 */
void KSVD_opt_mutual_incoherence(Eigen::MatrixXd &dictionary, Eigen::SparseMatrix<double> &sparse_mat_result,const Eigen::MatrixXd &signal_mat, const u_long T, const u_long max_iter = 10, const double mu = 0.99, const double episilon = 1E-6, const u_long use_thresh = 4);


/*
 Warning!!!
 In pratice use KSVD_approx to achieve high speed
 */

void KSVD_approx(Eigen::MatrixXd &dictionary, Eigen::SparseMatrix<double> &sparse_mat_result,const Eigen::MatrixXd &signal_mat, const u_long T, const u_long max_iter = 10, const double mu = 0.99, const double episilon = 1E-6, const u_long use_thresh = 4);


/*
 MI_KSVD
 
 */
void KSVD_approx_mi(Eigen::MatrixXd &dictionary, Eigen::SparseMatrix<double> &sparse_mat_result,const Eigen::MatrixXd &signal_mat, const u_long T, const u_long max_iter = 10, const double mu = 0.99, const double episilon = 1E-6, const u_long use_thresh = 4);


/*
 MC_KSVD
 
 */
void KSVD_approx_mc(Eigen::MatrixXd &dictionary, Eigen::SparseMatrix<double> &sparse_mat_result,const Eigen::MatrixXd &signal_mat, const u_long T, const u_long max_iter = 10, const double mc_tradeoff = 0.1, const double mu = 0.99, const double episilon = 1E-6, const u_long use_thresh = 4);




/*
 Block KSVD
 
 */
void block_KSVD_approx(Eigen::MatrixXd &dictionary, std::vector<std::vector<u_long> > &block, Eigen::SparseMatrix<double> &sparse_mat_result,const Eigen::MatrixXd &signal_mat, const u_long T, const u_long max_iter = 10, const double mu = 0.99, const double episilon = 1E-6, const u_long use_thresh = 4);



void block_KSVD_approx_v2(Eigen::MatrixXd &dictionary, std::vector<std::vector<u_long> > &block, Eigen::SparseMatrix<double> &sparse_mat_result,const Eigen::MatrixXd &signal_mat, const u_long T, const u_long max_iter = 10, const double mu = 0.99, const double episilon = 1E-6, const u_long use_thresh = 4);



void IPR(Eigen::MatrixXd &dictionary, Eigen::SparseMatrix<double> &sparse_mat_result,const Eigen::MatrixXd &signal_mat,const double mu0, const u_long sparsity,const u_long iter = 5);




void clear_dict(Eigen::MatrixXd &dictionary,const Eigen::MatrixXd sparse_mat,const Eigen::MatrixXd signals,const std::vector<u_long>&unused_signals, const u_long &unused_signal_len, const std::vector<std::vector<u_long> > &non_zero_index, std::set<u_long>&replaced_atoms, const double mu, const u_long use_thresh = 4);



void handle_unused_dictionary_atom(const ML_Random &rng,Eigen::MatrixXd &dictionary, const u_long unused_atom_index, const Eigen::MatrixXd &signals,const Eigen::SparseMatrix<double> &sparse_mat,std::vector<u_long> &unused_signals, u_long &unused_signal_len, std::set<u_long> &replaced_atoms,const u_long max_selected_signal_num = 5000);

void handle_unused_dictionary_block(const ML_Random &rng,Eigen::MatrixXd &dictionary, const u_long unused_atom_index, const Eigen::MatrixXd &signals,const Eigen::SparseMatrix<double> &sparse_mat,std::vector<u_long> &unused_signals, u_long &unused_signal_len, std::set<u_long> &replaced_atoms,const u_long max_selected_signal_num = 5000);

double count_opt_function(Eigen::MatrixXd &d,Eigen::MatrixXd &g,Eigen::MatrixXd &error,Eigen::MatrixXd &dictionary,u_long col_num,double lambda,double uk,double wi,bool disp_opt_func);
/*
 
 Get the Mutual Coherence of the learnt dictionary
 sum(sum(<di,dj> i!=j))
 Warning!!!
 When Using this function ,D MUST BE column L2-Normalized!!!!
 */

double get_mutual_coherence(const Eigen::MatrixXd &dictionary);

double get_AMC(const Eigen::MatrixXd &dictionary);

double get_RMSE(const Eigen::MatrixXd &dictionary,const Eigen::MatrixXd &input,const u_long &sparsity);

double get_SNR(const Eigen::MatrixXd &dictionary,const Eigen::MatrixXd &X,const Eigen::MatrixXd &Y);

#endif /* defined(__KSVD__ksvd__) */
