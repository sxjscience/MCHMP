//
//  ksvd.cpp
//  KSVD
//
//  Created by sxjscience on 13-9-5.
//  Copyright (c) 2013年 sxjscience. All rights reserved.
//

#include "ksvd.h"
#include <Eigen/Sparse>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include <vector>
#include <set>
#include <sstream>
#include <fstream>
#include "omp.h"
#include "time.h"
#include "ml_openmp_common.h"
#include "ml_random.h"
#include "ml_common.h"




void clear_dict(Eigen::MatrixXd &dictionary,const Eigen::MatrixXd sparse_mat,const Eigen::MatrixXd signals,const std::vector<u_long>&unused_signals, const u_long &unused_signal_len, const std::vector<std::vector<u_long> > &non_zero_index, std::set<u_long>&replaced_atoms, const double mu, const u_long use_thresh){
    //TODO Count Error(Should be revised to openmp)
    std::vector<u_long> real_unused_signals(unused_signal_len);
    for (int i=0; i<unused_signal_len; i++) {
        real_unused_signals[i] = unused_signals[i];
    }
    
    Eigen::VectorXd error = ML_OpenMP::vector_with_selected_index((signals-dictionary*sparse_mat).colwise().squaredNorm(),real_unused_signals);
    for (u_long j=0;j<dictionary.cols();j++){
        Eigen::MatrixXd Gj = dictionary.transpose()*dictionary.col(j);
        Gj(j,0) = 0;
        if((Gj.rowwise().squaredNorm().maxCoeff()>mu*mu || non_zero_index[j].size()<use_thresh) && (replaced_atoms.find(j) == replaced_atoms.end())){
            u_long max_err_row,max_err_col;
            error.topRows(unused_signal_len).maxCoeff(&max_err_row,&max_err_col);
            u_long real_selected_signal_num = real_unused_signals[max_err_col];
            dictionary.col(j) = signals.col(real_selected_signal_num).normalized();
            ML_Common::vec_swap(real_unused_signals,max_err_col,unused_signal_len-1);
            error.row(max_err_col).swap(error.row(unused_signal_len-1));
            replaced_atoms.insert(j);
            
        }
    }
    
    
}

void handle_unused_dictionary_atom(const ML_Random &rng,Eigen::MatrixXd &dictionary, const u_long unused_atom_index, const Eigen::MatrixXd &signals,const Eigen::SparseMatrix<double> &sparse_mat,std::vector<u_long> &unused_signals, u_long &unused_signal_len, std::set<u_long> &replaced_atoms,const u_long max_selected_signal_num){
    
    /*
     Zero Step:
     Check
     */
    assert(unused_signal_len>0);
    
    /*
     First Step:
     Select several random indices from unused_signal.
     The selected num = min(max_selected_signal_num,unused_signal_len)
     After being selected, the indexes are swapped to the LOWER PART of the sequence.
    */
    
    u_long select_num = (max_selected_signal_num>unused_signal_len)?unused_signal_len:max_selected_signal_num;
    std::vector<u_long> selected_indices(select_num);
    for (int i=0; i<select_num; i++) {
        u_long rand_index = rng.random_u_long(0,unused_signal_len - i - 1);
        selected_indices[i] = unused_signals[rand_index];
        ML_Common::vec_swap(unused_signals, rand_index, unused_signal_len-i-1);
    }
    
    /*
     Second Step:
     Count the Error function of the randomly selected signals
     Get the signal index with the biggest error 
    */
    
    Eigen::MatrixXd error = ML_OpenMP::matrix_with_selected_cols(signals, selected_indices)-dictionary*ML_OpenMP::matrix_with_selected_cols(sparse_mat, selected_indices);
    u_long maxrow,maxcol;
    error.colwise().squaredNorm().maxCoeff(&maxrow,&maxcol);
    u_long choosen_index = selected_indices[maxcol];
    /*
     Third Step:
     Use the index choosed by the previous step to initialize the unused atom.
     */
    dictionary.col(unused_atom_index) = signals.col(choosen_index).normalized();
    ML_Common::vec_swap(unused_signals, choosen_index,unused_signal_len-1);
    unused_signal_len--;
    replaced_atoms.insert(unused_atom_index);
    return;
    
}


void handle_unused_dictionary_block(const ML_Random &rng,Eigen::MatrixXd &dictionary, const u_long unused_atom_index, const Eigen::MatrixXd &signals,const Eigen::SparseMatrix<double> &sparse_mat,std::vector<u_long> &unused_signals, u_long &unused_signal_len, std::set<u_long> &replaced_atoms,const u_long max_selected_signal_num){
    
    /*
     Zero Step:
     Check
     */
    assert(unused_signal_len>0);
    
    /*
     First Step:
     Select several random indices from unused_signal.
     The selected num = min(max_selected_signal_num,unused_signal_len)
     After being selected, the indexes are swapped to the LOWER PART of the sequence.
     */
    
    u_long select_num = (max_selected_signal_num>unused_signal_len)?unused_signal_len:max_selected_signal_num;
    std::vector<u_long> selected_indices(select_num);
    for (int i=0; i<select_num; i++) {
        u_long rand_index = rng.random_u_long(0,unused_signal_len - i - 1);
        selected_indices[i] = unused_signals[rand_index];
        ML_Common::vec_swap(unused_signals, rand_index, unused_signal_len-i-1);
    }
    
    /*
     Second Step:
     Count the Error function of the randomly selected signals
     Get the signal index with the biggest error
     */
    
    Eigen::MatrixXd error = ML_OpenMP::matrix_with_selected_cols(signals, selected_indices)-dictionary*ML_OpenMP::matrix_with_selected_cols(sparse_mat, selected_indices);
    u_long maxrow,maxcol;
    error.colwise().squaredNorm().maxCoeff(&maxrow,&maxcol);
    u_long choosen_index = selected_indices[maxcol];
    /*
     Third Step:
     Use the index choosed by the previous step to initialize the unused atom.
     */
    dictionary.col(unused_atom_index) = signals.col(choosen_index).normalized();
    ML_Common::vec_swap(unused_signals, choosen_index,unused_signal_len-1);
    unused_signal_len--;
    replaced_atoms.insert(unused_atom_index);
    return;
    
}


void KSVD(Eigen::MatrixXd &dictionary, Eigen::SparseMatrix<double> &sparse_mat_result,const Eigen::MatrixXd &signal_mat, const u_long T, const u_long max_iter, const double episilon){
    u_long col_num = dictionary.cols();
    u_long row_num = dictionary.rows();
    time_t start,finish;

    Eigen::VectorXd zero_col = Eigen::VectorXd::Zero(dictionary.rows());
    ML_Random rng;
    for(int iter = 0;iter<max_iter;iter++){
        //Check stopping criteria
        
        //Using Batch OMP to generate sparse codes
        std::vector<std::vector<u_long> > non_zero_index(dictionary.cols());
//        Eigen::SparseMatrix<double> sparse_mat((int)dictionary.cols(),(int)signal_mat.cols());
        Eigen::SparseMatrix<double> sparse_mat(sparse_mat_result.rows(),sparse_mat_result.cols());
        sparse_mat.reserve(Eigen::VectorXi::Constant(sparse_mat.cols(), T));
        start = clock();
        OMP_batch(sparse_mat, dictionary, signal_mat, T, non_zero_index);
        finish = clock();
        std::cout<<"Time For OMP_batch at iter:"<<iter<<" "<<(double)(finish-start)/CLOCKS_PER_SEC<<std::endl;
        
        std::cout<<"Iter "<<iter+1<<"/"<<max_iter<<":"<<std::endl;
        std::cout<<"    Current RMSE:"<<(signal_mat-dictionary*sparse_mat).squaredNorm()/(double)sparse_mat.cols()<<std::endl;
        for (int j=0;j<dictionary.cols();j++){
            std::vector<u_long> selected_index(non_zero_index[j]);
            if(selected_index.size()==0){
                continue;
            }
            dictionary.col(j) = zero_col;
            //Get Error Term
            Eigen::MatrixXd error = ML_OpenMP::matrix_with_selected_cols(signal_mat,selected_index) - dictionary*ML_OpenMP::matrix_with_selected_cols(sparse_mat,selected_index,T);
            //Compute SVD
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(error,Eigen::ComputeThinU | Eigen::ComputeThinV);
            Eigen::VectorXd singular_val = svd.singularValues();
            Eigen::VectorXd rowV= svd.matrixV().transpose().row(0);
            
            //Update column j of the dictionary
            dictionary.col(j) = svd.matrixU().col(0);
            //Update related sparse codes
            for (int i=0; i<selected_index.size(); i++) {
                
                sparse_mat.coeffRef(j,(int)selected_index[i]) = rowV(i)*singular_val(0);
            }
            std::cout<<"RMSE for "<<j<<":"<<(signal_mat-dictionary*sparse_mat).squaredNorm()/(double)sparse_mat.cols()<<std::endl;

        }
        //Using K steps of SVD to update the dictionary
        
    
    }
    std::vector<std::vector<u_long> > non_zero_index(dictionary.cols());
    OMP_batch(sparse_mat_result, dictionary, signal_mat, T, non_zero_index);

}

void KSVD_opt_mutual_incoherence(Eigen::MatrixXd &dictionary, Eigen::SparseMatrix<double> &sparse_mat_result,const Eigen::MatrixXd &signal_mat, const u_long T, const u_long max_iter, const double mu, const double episilon,const u_long use_thresh){
    std::ofstream fout;

    if(Param::statics_save_path!=""){
        fout.open(Param::statics_save_path.c_str());
    }
    u_long col_num = dictionary.cols();
    u_long row_num = dictionary.rows();
    double signalF = signal_mat.squaredNorm();

    time_t start,finish;
    Eigen::VectorXd zero_col = Eigen::VectorXd::Zero(dictionary.rows());
    ML_Random rng;
    u_long unused_signal_len = sparse_mat_result.cols();
    std::vector<u_long> unused_signals(unused_signal_len);
    for (int i=0; i<unused_signal_len; i++) {
        unused_signals[i] = i;
    }

    for(int iter = 0;iter<max_iter;iter++){
        //Check stopping criteria
        
        
        //Using Batch OMP to generate sparse codes
        std::vector<std::vector<u_long> > non_zero_index(dictionary.cols());
        //        Eigen::SparseMatrix<double> sparse_mat((int)dictionary.cols(),(int)signal_mat.cols());
        Eigen::SparseMatrix<double> sparse_mat(sparse_mat_result.rows(),sparse_mat_result.cols());
        sparse_mat.reserve(Eigen::VectorXi::Constant(sparse_mat.cols(), T));
        start = clock();
        OMP_batch(sparse_mat, dictionary, signal_mat, T, non_zero_index);
        finish = clock();
        std::cout<<"Time For OMP_batch at iter:"<<iter<<" "<<(double)(finish-start)/CLOCKS_PER_SEC<<std::endl;

        std::cout<<"Iter "<<iter+1<<"/"<<max_iter<<":"<<std::endl;
        double SNR = 20*log10(signalF/(signal_mat-dictionary*sparse_mat).squaredNorm());
        double MC = get_mutual_coherence(dictionary);
        double AMC = get_AMC(dictionary);
        std::cout<<"    Current SNR:"<<SNR<<std::endl;
        std::cout<<"    Current Mutual Coherence:"<<MC<<std::endl;
        std::cout<<"    Current AMC:"<<AMC<<std::endl;
        
        if(Param::statics_save_path!=""){
            fout<<SNR<<" "<<MC<<" "<<AMC<<std::endl;
        }

        //Initialize Some parameters
        std::set<u_long> replaced_atoms;
        unused_signal_len = sparse_mat.cols();
        
        for (int j=0;j<dictionary.cols();j++){
            std::vector<u_long> selected_index(non_zero_index[j]);
//            std::cout<<"Mutual Coherence:"<<get_mutual_coherence(dictionary)<<std::endl;
            if(selected_index.size()==0){
                /*
                 
                 If the dictionary atom has not been used,perform
                 */
                std::cout<<"UNUSED ATOM!!"<<std::endl;
                handle_unused_dictionary_atom(rng,dictionary,j,signal_mat,sparse_mat,unused_signals,unused_signal_len,replaced_atoms);
                
                continue;
            }
            
            dictionary.col(j) = zero_col;
            //Get Error Term
            Eigen::MatrixXd error = ML_OpenMP::matrix_with_selected_cols(signal_mat,selected_index) - dictionary*ML_OpenMP::matrix_with_selected_cols(sparse_mat,selected_index,T);
            //Compute SVD
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(error,Eigen::ComputeThinU | Eigen::ComputeThinV);
            Eigen::VectorXd singular_val = svd.singularValues();
            Eigen::VectorXd rowV= svd.matrixV().transpose().row(0);
            
            //Update column j of the dictionary
            dictionary.col(j) = svd.matrixU().col(0);
            //Update related sparse codes
            for (int i=0; i<selected_index.size(); i++) {
                
                sparse_mat.coeffRef(j,(int)selected_index[i]) = rowV(i)*singular_val(0);
            }
//            std::cout<<"RMSE for "<<j<<":"<<(signal_mat-dictionary*sparse_mat).squaredNorm()/(double)sparse_mat.cols()<<std::endl;
            
        }
        //Clean Dictionary Atoms
        //clear_dict(dictionary,sparse_mat,signal_mat,unused_signals,unused_signal_len,non_zero_index,replaced_atoms,0.99,use_thresh);
        //Using K steps of SVD to update the dictionary
        
        
    }
    sparse_mat_result.reserve(Eigen::VectorXi::Constant(sparse_mat_result.cols(), T));
    std::vector<std::vector<u_long> > non_zero_index(dictionary.cols());
    OMP_batch(sparse_mat_result, dictionary, signal_mat, T, non_zero_index);
    if(Param::statics_save_path!=""){
        fout.close();
    }

}

void KSVD_approx_mi(Eigen::MatrixXd &dictionary, Eigen::SparseMatrix<double> &sparse_mat_result,const Eigen::MatrixXd &signal_mat, const u_long T, const u_long max_iter, const double mu, const double episilon,const u_long use_thresh){
    std::ofstream fout;
    
    if(Param::statics_save_path!=""){
        fout.open(Param::statics_save_path.c_str());
    }

    u_long col_num = dictionary.cols();
    u_long row_num = dictionary.rows();
    double signalF = signal_mat.squaredNorm();

    time_t start,finish;
    Eigen::VectorXd zero_col = Eigen::VectorXd::Zero(dictionary.rows());
    ML_Random rng;
    u_long unused_signal_len = sparse_mat_result.cols();
    std::vector<u_long> unused_signals(unused_signal_len);
    for (int i=0; i<unused_signal_len; i++) {
        unused_signals[i] = i;
    }
    
    std::vector<u_long> dictionary_perm(dictionary.cols());
    for (int i=0; i<dictionary_perm.size(); i++) {
        dictionary_perm[i] = i;
    }
    
    
    for(int iter = 0;iter<max_iter;iter++){
        //First Randomly permutate the dictionary_perm array
        rng.random_permutation_n(dictionary_perm, dictionary_perm.size());
        std::cout<<"Permutation:"<<dictionary_perm[0]<<" "<<dictionary_perm[1]<<" "<<dictionary_perm[2]<<std::endl;
        //Using Batch OMP to generate sparse codes
        std::vector<std::vector<u_long> > non_zero_index(dictionary.cols());
        //        Eigen::SparseMatrix<double> sparse_mat((int)dictionary.cols(),(int)signal_mat.cols());
        Eigen::SparseMatrix<double> sparse_mat(sparse_mat_result.rows(),sparse_mat_result.cols());
        sparse_mat.reserve(Eigen::VectorXi::Constant(sparse_mat.cols(), T));
        start = clock();
        OMP_batch(sparse_mat, dictionary, signal_mat, T, non_zero_index);
        finish = clock();
        std::cout<<"KSVD approx_mi Time For OMP_batch at iter:"<<iter<<" "<<(double)(finish-start)/CLOCKS_PER_SEC<<std::endl;
        
        std::cout<<"Iter "<<iter+1<<"/"<<max_iter<<":"<<std::endl;
        double SNR = 20*log10(signalF/(signal_mat-dictionary*sparse_mat).squaredNorm());
        double MC = get_mutual_coherence(dictionary);
        double AMC = get_AMC(dictionary);
        std::cout<<"    Current SNR:"<<SNR<<std::endl;
        std::cout<<"    Current Mutual Coherence:"<<MC<<std::endl;
        std::cout<<"    Current AMC:"<<AMC<<std::endl;
        
        if(Param::statics_save_path!=""){
            fout<<SNR<<" "<<MC<<" "<<AMC<<std::endl;
        }
        //Initialize Some parameters
        std::set<u_long> replaced_atoms;
        unused_signal_len = sparse_mat.cols();
        
        for (int j=0;j<dictionary.cols();j++){
            
            std::vector<u_long> selected_index(non_zero_index[dictionary_perm[j]]);
            //std::cout<<"Mutual Coherence:"<<get_mutual_coherence(dictionary)<<std::endl;
            if(selected_index.size()==0){
                /*
                 
                 If the dictionary atom has not been used,perform
                 */
                std::cout<<"UNUSED ATOM!!"<<std::endl;
                handle_unused_dictionary_atom(rng,dictionary,dictionary_perm[j],signal_mat,sparse_mat,unused_signals,unused_signal_len,replaced_atoms);
                
                continue;
            }
            dictionary.col(dictionary_perm[j]) = zero_col;

            
            //Get Error Term

            
            Eigen::MatrixXd error = ML_OpenMP::matrix_with_selected_cols(signal_mat,selected_index) - dictionary*ML_OpenMP::matrix_with_selected_cols(sparse_mat,selected_index,T);
            
//            std::cout<<"error="<<error<<std::endl;

            //Compute Approximate g and d
            Eigen::MatrixXd g(selected_index.size(),1);
            for (int i=0; i<selected_index.size(); i++) {
                g(i,0) = sparse_mat.coeffRef(dictionary_perm[j],(int)selected_index[i]);
            }

            Eigen::MatrixXd d = error*g;

            d.normalize();

            u_long mi_max_iter =3;
            double trade_off = 0.05;
            Eigen::MatrixXd ddelta = zero_col;
            for (int t=0; t<mi_max_iter; t++) {
                ddelta = zero_col;
                Eigen::MatrixXd dcorr = d.transpose()*dictionary;
                for (u_long s = 0; s<dictionary.cols(); s++) {
                    if (s!=dictionary_perm[j]) {
                        ddelta += (dcorr(0,s)>0 ? 1 : -1)*dictionary.col(s);
                    }
                }
                ddelta = trade_off*ddelta/(dcorr.cols()-1);
                d =d - ddelta;
                d.normalize();

            }
            
            g = error.transpose()*d;
            
            
            //Update column j of the dictionary
            dictionary.col(dictionary_perm[j]) = d;
            //Update related sparse codes

            for (int i=0; i<selected_index.size(); i++) {
                
                sparse_mat.coeffRef(dictionary_perm[j],(int)selected_index[i]) = g(i,0);
            }


//            std::cout<<"RMSE for "<<j<<":"<<(signal_mat-dictionary*sparse_mat).squaredNorm()/(double)sparse_mat.cols()<<std::endl;

        }
        //Clean Dictionary Atoms
        //clear_dict(dictionary,sparse_mat,signal_mat,unused_signals,unused_signal_len,non_zero_index,replaced_atoms,0.99,use_thresh);
        //Using K steps of SVD to update the dictionary
        
        
    }
    sparse_mat_result.reserve(Eigen::VectorXi::Constant(sparse_mat_result.cols(), T));
    std::vector<std::vector<u_long> > non_zero_index(dictionary.cols());
    OMP_batch(sparse_mat_result, dictionary, signal_mat, T, non_zero_index);

    if(Param::statics_save_path!=""){
        fout.close();
    }
}

double count_opt_function(Eigen::MatrixXd &d,Eigen::MatrixXd &g,Eigen::MatrixXd &error,Eigen::MatrixXd &dictionary,u_long col_num,double lambda,double uk,double wi,bool disp_opt_func){
    double max_val = 0;
    Eigen::MatrixXd dcorr = d.transpose()*dictionary;
    for (u_long i=0;i<dictionary.cols();i++){
        if(i!=col_num && fabs(dcorr(0,i))>max_val){
            max_val = fabs(dcorr(0,i));
        }
    }
    double res = 0;
    if(disp_opt_func){
        res = g.squaredNorm()*d.squaredNorm()-2*((g.transpose()*error.transpose())*d).sum()+lambda*max_val+uk/2*(d.squaredNorm()-1)*(d.squaredNorm()-1)-wi*(d.squaredNorm()-1);
    }
    else{
        res = g.squaredNorm()*d.squaredNorm()-2*((g.transpose()*error.transpose())*d).sum()+lambda*max_val;
    }
    return res;
    
}


void KSVD_approx_mc(Eigen::MatrixXd &dictionary, Eigen::SparseMatrix<double> &sparse_mat_result,const Eigen::MatrixXd &signal_mat, const u_long T, const u_long max_iter,const double mc_tradeoff, const double mu, const double episilon, const u_long use_thresh){
    
    std::ofstream fout;
    Eigen::MatrixXd initD = dictionary;
    if(Param::statics_save_path!=""){
        fout.open(Param::statics_save_path.c_str());
    }
    
    u_long col_num = dictionary.cols();
    u_long row_num = dictionary.rows();
    double signalF = signal_mat.squaredNorm();
    time_t start,finish;
    Eigen::VectorXd zero_col = Eigen::VectorXd::Zero(dictionary.rows());
    ML_Random rng;
    u_long unused_signal_len = sparse_mat_result.cols();
    std::vector<u_long> unused_signals(unused_signal_len);
    for (int i=0; i<unused_signal_len; i++) {
        unused_signals[i] = i;
    }
    
    std::vector<u_long> dictionary_perm(dictionary.cols());
    std::vector<u_long> d_usage_map(dictionary.cols());

    for (int i=0; i<dictionary_perm.size(); i++) {
        dictionary_perm[i] = i;
        d_usage_map[i] = 0;
    }
    //Precompute DTD and other parameters
    Eigen::ArrayXXd DTD = (dictionary.transpose()*dictionary);
//    Eigen::VectorXd maxVal = Eigen::VectorXd::Zeros(dictionary.cols());
//    Eigen::VectorXd maxOtherVal(dictionary.cols()) = Eigen::VectorXd::Zeros(dictionary.cols());;
//    std::vector<u_long> maxValCol(dictionary.cols());
//    std::vector<u_long> maxOtherValCol1(dictionary.cols());
//    std::vector<u_long> maxOtherValCol2(dictionary.cols());
//    
//    for(int i=0;i<dictionary.cols();i++){
//        for (int j=i+1;j<dictionary.cols();j++){
//            if(DTD(i,j)>=maxVal(i)){
//                maxVal(i) = DTD(i,j);
//                maxValCol[i] = j;
//            }
//            if(DTD(i,j)>=maxVal(j)){
//                maxVal(j) = DTD(i,j);
//                maxValCol[j] = i;
//            }
//            for (int t = 0;t<dictionary.cols();t++){
//                if(i!=t && j!=t && DTD(i,j)>maxOtherVal(t)){
//                    maxOtherValCol1[t] = i;
//                    maxOtherValCol2[t] = j;
//                }
//            }
//        }
//    }
    
    double mutual_coherence = 0;
    double col1,col2;
    for (int i=0;i<dictionary.cols();i++){
        for (int j=i+1;j<dictionary.cols();j++){
            if(fabs(DTD(i,j))>mutual_coherence){
                mutual_coherence = fabs(DTD(i,j));
                col1 = i;
                col2 = j;
            }
        }
    }
    
    
    u_long j = 0;
    for(int iter = 0;iter<max_iter;iter++){
        //First Randomly permutate the dictionary_perm array
        for (int i=0; i<dictionary_perm.size(); i++) {
            dictionary_perm[i] = i;
            d_usage_map[i] = 0;
        }
        //Using Batch OMP to generate sparse codes
        std::vector<std::vector<u_long> > non_zero_index(dictionary.cols());
        //        Eigen::SparseMatrix<double> sparse_mat((int)dictionary.cols(),(int)signal_mat.cols());
        Eigen::SparseMatrix<double> sparse_mat(sparse_mat_result.rows(),sparse_mat_result.cols());
        sparse_mat.reserve(Eigen::VectorXi::Constant(sparse_mat.cols(), T));
        start = clock();
        OMP_batch(sparse_mat, dictionary, signal_mat, T, non_zero_index);
        finish = clock();
        std::cout<<"KSVD approx_mc Time For OMP_batch at iter:"<<iter<<" "<<(double)(finish-start)/CLOCKS_PER_SEC<<std::endl;
        
        std::cout<<"Iter "<<iter+1<<"/"<<max_iter<<":"<<std::endl;
        double SNR = 20*log10(signalF/(signal_mat-dictionary*sparse_mat).squaredNorm());
        double MC = get_mutual_coherence(dictionary);
        double AMC = get_AMC(dictionary);
        std::cout<<"    Current SNR:"<<SNR<<std::endl;
        std::cout<<"    Current Mutual Coherence:"<<MC<<std::endl;
        std::cout<<"    Current AMC:"<<AMC<<std::endl;
        
        if(Param::statics_save_path!=""){
            fout.open(Param::statics_save_path.c_str());
            fout<<SNR<<" "<<MC<<" "<<AMC<<std::endl;
        }

        //Initialize Some parameters
        std::set<u_long> replaced_atoms;
        unused_signal_len = sparse_mat.cols();
        for (int j=0;j<dictionary.cols();j++){
//            DTD = dictionary.transpose()*dictionary;
//            bool mutual_coherence_opt = 0;
//            for (int i=0;i<dictionary.cols();i++){
//                for (int t=i+1;t<dictionary.cols();t++){
//                    if(fabs(DTD(i,t))>mutual_coherence){
//                        mutual_coherence = fabs(DTD(i,t));
//                        col1 = i;
//                        col2 = t;
//                    }
//                }
//            }
//            std::cout<<col1<<" "<<col2<<" "<<mutual_coherence<<std::endl;
//            if(d_usage_map[col1]==1 && d_usage_map[col2] == 1){
//                u_long sel_col_index = rng.random_u_long(j, dictionary.cols()-1);
//                ML_Common::vec_swap(dictionary_perm,sel_col_index,j);
//                d_usage_map[dictionary_perm[j]] = 1;
//                mutual_coherence_opt = 0;
//            }
//            else if(d_usage_map[col1] == 0){
//                ML_Common::vec_swap(dictionary_perm,j,col1);
//                d_usage_map[dictionary_perm[j]] = 1;
//                mutual_coherence_opt = 1;
//            }
//            else if(d_usage_map[col2] == 0){
//                ML_Common::vec_swap(dictionary_perm,j,col2);
//                d_usage_map[dictionary_perm[j]] = 1;
//                mutual_coherence_opt = 1;
//
//            }
//            std::cout<<dictionary_perm[j]<<std::endl;
//            std::cout<<"    RMSE:"<<(signal_mat-dictionary*sparse_mat).squaredNorm()<<std::endl;
//            std::cout<<"    Mutual Coherence:"<<get_mutual_coherence(dictionary)<<std::endl;

            std::vector<u_long> selected_index(non_zero_index[dictionary_perm[j]]);
            //std::cout<<"Mutual Coherence:"<<get_mutual_coherence(dictionary)<<std::endl;
            if(selected_index.size()==0){
                /*
                 
                 If the dictionary atom has not been used,perform
                 */
                std::cout<<"UNUSED ATOM!!"<<std::endl;
                handle_unused_dictionary_atom(rng,dictionary,dictionary_perm[j],signal_mat,sparse_mat,unused_signals,unused_signal_len,replaced_atoms);
                
                continue;
            }
            Eigen::MatrixXd d = dictionary.col(dictionary_perm[j]);
            dictionary.col(dictionary_perm[j]) = zero_col;
            
            
            //Get Error Term
            
            
            Eigen::MatrixXd error = ML_OpenMP::matrix_with_selected_cols(signal_mat,selected_index) - dictionary*ML_OpenMP::matrix_with_selected_cols(sparse_mat,selected_index,T);
            
            
            
            //Compute Approximate g and d
            Eigen::MatrixXd g(selected_index.size(),1);
            for (int i=0; i<selected_index.size(); i++) {
                g(i,0) = sparse_mat.coeffRef(dictionary_perm[j],(int)selected_index[i]);
            }
//            Eigen::MatrixXd d;
//            Eigen::MatrixXd e;
//            double trade_off = 5 ;
//            for (int iter = 0; iter<1; iter++) {
//                e = error*g;
//                double max_val = 0;
//                double curr_val;
//                for(int t=0;t<dictionary.cols();t++){
//                    if(t!=dictionary_perm[j]){
//                        curr_val = (trade_off*dictionary.col(t)-2*e).squaredNorm();
//                        if(curr_val>max_val){
//                            max_val = curr_val;
//                            d = -(trade_off*dictionary.col(t)-2*e);
//                        }
//                        
//                        curr_val = (-trade_off*dictionary.col(t)-2*e).squaredNorm();
//                        if(curr_val>max_val){
//                            max_val = curr_val;
//                            d = -(trade_off*dictionary.col(t)-2*e);
//                        }
//                    }
//                }
//                d.normalize();
//                g = error.transpose()*d;
//            }
            if(1){
                u_long max_iter_coordinate = 1;
                u_long max_iter_subgrad = 5;
                u_long max_iter_lagrange = 6 ;
                double lambda = mc_tradeoff;
                
    //            std::cout<<j<<std::endl;
                for(u_long iter_coordinate = 0;iter_coordinate<max_iter_coordinate;iter_coordinate++){
                    d = error*g;
                    d.normalize();
                    /*Minimize: g^T*g*d^T*d - 2*g^T*error^T*d + lamda*max{|<d,di>|}
                     Constriant: d^T*d = 1
                     First Fix g and optimize d using augmented lagrange and subgradient method
                     So the function becomes:
                        lambda*max{|<d,di>|} + uk/2*(d^T*d-1)^2 - wi*(d^T*d-1)
                     */
                    
                    double episilon = 0.1;
                    double uk = 10;
                    double wi = 1;
//                    std::cout<<"   Iter"<<iter_coordinate<<"/"<<max_iter_coordinate<<" Current Lost For The True Function:"<<count_opt_function(d,g,error,dictionary,dictionary_perm[j],lambda,uk,wi,0)<<std::endl;
                    for(u_long iter_lagrange = 0;iter_lagrange<max_iter_lagrange;iter_lagrange++){
                        Eigen::MatrixXd bestd = d;
//                        double first_term = g.squaredNorm()*d.squarDedNorm()-2*((g.transpose()*error.transpose())*d).sum()+uk/2*(d.squaredNorm()-1)*(d.squaredNorm()-1)-wi*(d.squaredNorm()-1);
                        double bestLost = count_opt_function(bestd,g,error,dictionary,dictionary_perm[j],lambda,uk,wi,1);
    //                    std::cout<<"   Current Best Lost For Opt Function:"<<bestLost<<std::endl;
    //                    std::cout<<"   Current Lost For The True Function:"<<count_opt_function(bestd,g,error,dictionary,dictionary_perm[j],lambda,uk,wi,0)<<std::endl;
    //                    std::cout<<"   Current d.squaredNorm():"<<d.squaredNorm()<<std::endl;
                        double subgrad_step = 0.00001;
                        for(u_long iter_subgrad = 0;iter_subgrad<max_iter_subgrad;iter_subgrad++){
                            //Find Max Value
                            double max_val = 0;
                            Eigen::MatrixXd dcorr = d.transpose()*dictionary;
                            u_long sel_max_col = 0;
                            for(u_long s = 0;s<dictionary.cols();s++){
                                if(s!=dictionary_perm[j]&& fabs(dcorr(0,s))>max_val){
                                    max_val = fabs(dcorr(0,s));
                                    sel_max_col = s;
                                }
                            }

                            //Count Gradient
//                            double curr_first_term = g.squaredNorm()*d.squaredNorm()-2*((g.transpose()*error.transpose())*d).sum()+uk/2*(d.squaredNorm()-1)*(d.squaredNorm()-1)-wi*(d.squaredNorm()-1);
                            Eigen::MatrixXd grad;
//                            if(curr_first_term<=first_term){
                                grad = 2*g.squaredNorm()*d-2*error*g+lambda*(dcorr(0,sel_max_col)>0?1:-1)*dictionary.col(sel_max_col)+uk*(d.squaredNorm()-1)*2*d-wi*2*d;
//                            }
//                            else{
//                                grad = 2*g.squaredNorm()*d-2*error*g;
//                            }
                            d = d-subgrad_step*grad;
//                            curr_first_term = g.squaredNorm()*d.squaredNorm()-2*((g.transpose()*error.transpose())*d).sum()+uk/2*(d.squaredNorm()-1)*(d.squaredNorm()-1)-wi*(d.squaredNorm()-1);
                            double currLost = count_opt_function(d,g,error,dictionary,dictionary_perm[j],lambda,uk,wi,1);
    //                        std::cout<<"      Lost For iter<<"<<iter_subgrad+1<<"/"<<max_iter_subgrad<<":"<<currLost<<std::endl;
                            if(currLost<bestLost){
                                bestLost = currLost;
                                bestd = d;
                            }
//                            if(currLost-bestLost<0.1){
//                                break;
//                            }
                        }
                        d = bestd;
                        if((d.squaredNorm()-1)<episilon){
                            wi = wi-uk*(d.squaredNorm()-1);
                        }
                        else{
                            uk*=2;
                        }
                        
                    }
                    d.normalize();
                    g = error.transpose()*d;
                    
                    
                    
                }
                dictionary.col(dictionary_perm[j]) = d;
                 
            }
            else{
                d=error*g;
                d.normalize();
                g = error.transpose()*d;
                dictionary.col(dictionary_perm[j]) = d;

            }
            
            
            
//            d = error*g;
//            d.normalize();
//
//            for(int iter =0;iter<5;iter++){
//                d = error*g;
//                d.normalize();                
//                //Begin mc coherence Optimize Step
//                
//                u_long mc_max_iter =10;
//                double trade_off = 0.01;
//                Eigen::MatrixXd ddelta = zero_col;
//                
//                for (int t=0; t<mc_max_iter; t++) {
//                    double max_val = 0;
//                    u_long sel_col;
//                    Eigen::MatrixXd dcorr = (d.transpose()*dictionary);
//    //                std::cout<<dcorr.mean()<<std::endl;
//                    for(int s = 0;s<dictionary.cols();s++){
//                        if(s!=dictionary_perm[j] && fabs(dcorr(0,s))>max_val){
//                            sel_col = s;
//                            max_val = fabs(dcorr(0,s));
//                        }
//                    }
//                    d -= trade_off*(dcorr(0,sel_col)>0?1:-1)*dictionary.col(sel_col);
//                    d.normalize();
//                }
//
//                
//                g = error.transpose()*d;
//            }
//            dictionary.col(dictionary_perm[j]) = d;

            //Update column j of the dictionary
            
            
            
            
            //Update related sparse codes
            
            for (int i=0; i<selected_index.size(); i++) {
                
                sparse_mat.coeffRef(dictionary_perm[j],(int)selected_index[i]) = g(i,0);
            }
            
            
            //            std::cout<<"RMSE for "<<j<<":"<<(signal_mat-dictionary*sparse_mat).squaredNorm()/(double)sparse_mat.cols()<<std::endl;
            
        }
        
        //Clean Dictionary Atoms
        //clear_dict(dictionary,sparse_mat,signal_mat,unused_signals,unused_signal_len,non_zero_index,replaced_atoms,0.99,use_thresh);
        //Using K steps of SVD to update the dictionary
        
        
    }
    sparse_mat_result.reserve(Eigen::VectorXi::Constant(sparse_mat_result.cols(), T));
    std::vector<std::vector<u_long> > non_zero_index(dictionary.cols());
    OMP_batch(sparse_mat_result, dictionary, signal_mat, T, non_zero_index);

    if(Param::statics_save_path!=""){
        fout.close();
    }


    
}




void KSVD_approx(Eigen::MatrixXd &dictionary, Eigen::SparseMatrix<double> &sparse_mat_result,const Eigen::MatrixXd &signal_mat, const u_long T, const u_long max_iter, const double mu, const double episilon,const u_long use_thresh){
    u_long col_num = dictionary.cols();
    u_long row_num = dictionary.rows();
    time_t start,finish;
    double signalF = signal_mat.squaredNorm();
    Eigen::VectorXd zero_col = Eigen::VectorXd::Zero(dictionary.rows());
    ML_Random rng;
    u_long unused_signal_len = sparse_mat_result.cols();
    std::vector<u_long> unused_signals(unused_signal_len);
    for (int i=0; i<unused_signal_len; i++) {
        unused_signals[i] = i;
    }
    
    std::vector<u_long> dictionary_perm(dictionary.cols());
    for (int i=0; i<dictionary_perm.size(); i++) {
        dictionary_perm[i] = i;
    }
    
    
    for(int iter = 0;iter<max_iter;iter++){
        //First Randomly permutate the dictionary_perm array
        rng.random_permutation_n(dictionary_perm, dictionary_perm.size());
        std::cout<<"Permutation:"<<dictionary_perm[0]<<" "<<dictionary_perm[1]<<" "<<dictionary_perm[2]<<std::endl;
        //Using Batch OMP to generate sparse codes
        std::vector<std::vector<u_long> > non_zero_index(dictionary.cols());
        //        Eigen::SparseMatrix<double> sparse_mat((int)dictionary.cols(),(int)signal_mat.cols());
        Eigen::SparseMatrix<double> sparse_mat(sparse_mat_result.rows(),sparse_mat_result.cols());
        sparse_mat.reserve(Eigen::VectorXi::Constant(sparse_mat.cols(), T));
        start = clock();
        OMP_batch(sparse_mat, dictionary, signal_mat, T, non_zero_index);
        finish = clock();
        std::cout<<"KSVD Time For OMP_batch at iter:"<<iter<<" "<<(double)(finish-start)/CLOCKS_PER_SEC<<std::endl;
        
        std::cout<<"Iter "<<iter+1<<"/"<<max_iter<<":"<<std::endl;
        std::cout<<"    Current SNR:"<<20*log10(signalF/(signal_mat-dictionary*sparse_mat).squaredNorm())<<std::endl;
        std::cout<<"    Current Mutual Coherence:"<<get_mutual_coherence(dictionary)<<std::endl;
        std::cout<<"    Current AMC:"<<get_AMC(dictionary)<<std::endl;
        //Initialize Some parameters
        std::set<u_long> replaced_atoms;
        unused_signal_len = sparse_mat.cols();
        
        for (int j=0;j<dictionary.cols();j++){
            
            std::vector<u_long> selected_index(non_zero_index[dictionary_perm[j]]);
            //std::cout<<"Mutual Coherence:"<<get_mutual_coherence(dictionary)<<std::endl;
            if(selected_index.size()==0){
                /*
                 
                 If the dictionary atom has not been used,perform
                 */
                std::cout<<"UNUSED ATOM!!"<<std::endl;
                handle_unused_dictionary_atom(rng,dictionary,dictionary_perm[j],signal_mat,sparse_mat,unused_signals,unused_signal_len,replaced_atoms);
                
                continue;
            }
            dictionary.col(dictionary_perm[j]) = zero_col;
            
            
            //Get Error Term
            
            
            Eigen::MatrixXd error = ML_OpenMP::matrix_with_selected_cols(signal_mat,selected_index) - dictionary*ML_OpenMP::matrix_with_selected_cols(sparse_mat,selected_index,T);
            
            
            
            //Compute Approximate g and d
            Eigen::MatrixXd g(selected_index.size(),1);
            for (int i=0; i<selected_index.size(); i++) {
                g(i,0) = sparse_mat.coeffRef(dictionary_perm[j],(int)selected_index[i]);
            }
            
            Eigen::MatrixXd d = error*g;
            d.normalize();
            g = error.transpose()*d;
            
            
            //Update column j of the dictionary
            dictionary.col(dictionary_perm[j]) = d;
            //Update related sparse codes
            
            for (int i=0; i<selected_index.size(); i++) {
                
                sparse_mat.coeffRef(dictionary_perm[j],(int)selected_index[i]) = g(i,0);
            }
            
            
            //            std::cout<<"RMSE for "<<j<<":"<<(signal_mat-dictionary*sparse_mat).squaredNorm()/(double)sparse_mat.cols()<<std::endl;
            
        }
        //Clean Dictionary Atoms
        //clear_dict(dictionary,sparse_mat,signal_mat,unused_signals,unused_signal_len,non_zero_index,replaced_atoms,0.99,use_thresh);
        //Using K steps of SVD to update the dictionary
    
    }
    sparse_mat_result.reserve(Eigen::VectorXi::Constant(sparse_mat_result.cols(), T));
    std::vector<std::vector<u_long> > non_zero_index(dictionary.cols());
    OMP_batch(sparse_mat_result, dictionary, signal_mat, T, non_zero_index);

}


void block_KSVD_approx(Eigen::MatrixXd &dictionary, std::vector<std::vector<u_long> > &block, Eigen::SparseMatrix<double> &sparse_mat_result,const Eigen::MatrixXd &signal_mat, const u_long T, const u_long max_iter, const double mu, const double episilon, const u_long use_thresh){
    u_long col_num = dictionary.cols();
    u_long row_num = dictionary.rows();
    time_t start,finish;
    Eigen::VectorXd zero_col = Eigen::VectorXd::Zero(dictionary.rows());
    ML_Random rng;
    u_long unused_signal_len = sparse_mat_result.cols();
    std::vector<u_long> unused_signals(unused_signal_len);
    for (int i=0; i<unused_signal_len; i++) {
        unused_signals[i] = i;
    }
    
    std::vector<u_long> dictionary_perm(dictionary.cols());
    for (int i=0; i<dictionary_perm.size(); i++) {
        dictionary_perm[i] = i;
    }
    
    
    for(int iter = 0;iter<max_iter;iter++){
        //First Randomly permutate the dictionary_perm array
        rng.random_permutation_n(dictionary_perm, dictionary_perm.size());
        std::cout<<"Permutation:"<<dictionary_perm[0]<<" "<<dictionary_perm[1]<<" "<<dictionary_perm[2]<<std::endl;
        //Using Batch OMP to generate sparse codes
        std::vector<std::vector<u_long> > non_zero_index(dictionary.cols());
        //        Eigen::SparseMatrix<double> sparse_mat((int)dictionary.cols(),(int)signal_mat.cols());
        Eigen::SparseMatrix<double> sparse_mat(sparse_mat_result.rows(),sparse_mat_result.cols());
        sparse_mat.reserve(Eigen::VectorXi::Constant(sparse_mat.cols(), T*block[0].size()));
        start = clock();
        block_OMP_batch(sparse_mat, dictionary, signal_mat, block, T, non_zero_index);
        finish = clock();
        std::cout<<"Time For Block OMP batch at iter:"<<iter<<" "<<(double)(finish-start)/CLOCKS_PER_SEC<<std::endl;
        
        std::cout<<"Iter "<<iter+1<<"/"<<max_iter<<":"<<std::endl;
        std::cout<<"    Current RMSE:"<<(signal_mat-dictionary*sparse_mat).squaredNorm()/(double)sparse_mat.cols()<<std::endl;
        std::cout<<"    Current Mutual Coherence:"<<get_mutual_coherence(dictionary)<<std::endl;

        //Initialize Some parameters
        std::set<u_long> replaced_atoms;
        unused_signal_len = sparse_mat.cols();
        
        for (int j=0;j<dictionary.cols();j++){
            
            std::vector<u_long> selected_index(non_zero_index[dictionary_perm[j]]);
            
            //std::cout<<"Mutual Coherence:"<<get_mutual_coherence(dictionary)<<std::endl;
            if(selected_index.size()==0){
                /*
                 
                 If the dictionary atom has not been used,perform
                 */
                std::cout<<"UNUSED ATOM!!"<<std::endl;
                handle_unused_dictionary_atom(rng,dictionary,dictionary_perm[j],signal_mat,sparse_mat,unused_signals,unused_signal_len,replaced_atoms);
                
                continue;
            }
            dictionary.col(dictionary_perm[j]) = zero_col;
            
            
            //Get Error Term
            
            
            Eigen::MatrixXd error = ML_OpenMP::matrix_with_selected_cols(signal_mat,selected_index) - dictionary*ML_OpenMP::matrix_with_selected_cols(sparse_mat,selected_index,T);
            
            
            
            //Compute Approximate g and d
            Eigen::MatrixXd g(selected_index.size(),1);
            for (int i=0; i<selected_index.size(); i++) {
                g(i,0) = sparse_mat.coeffRef(dictionary_perm[j],(int)selected_index[i]);
            }
            
            Eigen::MatrixXd d = error*g;
            d.normalize();
            g = error.transpose()*d;
            
            
            //Update column j of the dictionary
            dictionary.col(dictionary_perm[j]) = d;
            //Update related sparse codes
            
            for (int i=0; i<selected_index.size(); i++) {
                
                sparse_mat.coeffRef(dictionary_perm[j],(int)selected_index[i]) = g(i,0);
            }
            
            
            //            std::cout<<"RMSE for "<<j<<":"<<(signal_mat-dictionary*sparse_mat).squaredNorm()/(double)sparse_mat.cols()<<std::endl;
            
        }
        //Clean Dictionary Atoms
        //clear_dict(dictionary,sparse_mat,signal_mat,unused_signals,unused_signal_len,non_zero_index,replaced_atoms,0.99,use_thresh);
        //Using K steps of SVD to update the dictionary
        
        
    }
    
}



void block_KSVD_approx_v2(Eigen::MatrixXd &dictionary, std::vector<std::vector<u_long> > &block, Eigen::SparseMatrix<double> &sparse_mat_result,const Eigen::MatrixXd &signal_mat, const u_long T, const u_long max_iter, const double mu, const double episilon, const u_long use_thresh){
    u_long col_num = dictionary.cols();
    u_long row_num = dictionary.rows();
    time_t start,finish;
    Eigen::VectorXd zero_col = Eigen::VectorXd::Zero(dictionary.rows());
    ML_Random rng;
    u_long unused_signal_len = sparse_mat_result.cols();
    std::vector<u_long> unused_signals(unused_signal_len);
    for (int i=0; i<unused_signal_len; i++) {
        unused_signals[i] = i;
    }
    
    std::vector<u_long> dictionary_block_perm(block.size());
    for (int i=0; i<dictionary_block_perm.size(); i++) {
        dictionary_block_perm[i] = i;
    }
    
    
    for(int iter = 0;iter<max_iter;iter++){
        //First Randomly permutate the dictionary_perm array
        rng.random_permutation_n(dictionary_block_perm, dictionary_block_perm.size());
        //Using Batch OMP to generate sparse codes
        std::vector<std::vector<u_long> > non_zero_index(dictionary.cols());
        //        Eigen::SparseMatrix<double> sparse_mat((int)dictionary.cols(),(int)signal_mat.cols());
        Eigen::SparseMatrix<double> sparse_mat(sparse_mat_result.rows(),sparse_mat_result.cols());
        sparse_mat.reserve(Eigen::VectorXi::Constant(sparse_mat.cols(), T*block[0].size()));
        start = clock();
        block_OMP_batch(sparse_mat, dictionary, signal_mat, block, T, non_zero_index);
        finish = clock();
        std::cout<<"Time For Block OMP batch at iter:"<<iter<<" "<<(double)(finish-start)/CLOCKS_PER_SEC<<std::endl;
        
        std::cout<<"Iter "<<iter+1<<"/"<<max_iter<<":"<<std::endl;
        std::cout<<"    Current RMSE:"<<(signal_mat-dictionary*sparse_mat).squaredNorm()/(double)sparse_mat.cols()<<std::endl;
        std::cout<<"    Current Mutual Coherence:"<<get_mutual_coherence(dictionary)<<std::endl;

        //Initialize Some parameters
        std::set<u_long> replaced_atoms;
        unused_signal_len = sparse_mat.cols();
        
        for (int j=0;j<block.size();j++){
            
            std::vector<u_long> selected_index(non_zero_index[block[dictionary_block_perm[j]][0]]);
            
            //std::cout<<"Mutual Coherence:"<<get_mutual_coherence(dictionary)<<std::endl;
            if(selected_index.size()==0){
                /*
                 
                 If the dictionary block has not been used,perform
                 */
                std::cout<<"UNUSED ATOM!!"<<std::endl;
//                handle_unused_dictionary_atom(rng,dictionary,dictionary_block_perm[j],signal_mat,sparse_mat,unused_signals,unused_signal_len,replaced_atoms);
                
                continue;
            }
            for (int i=0; i<block[dictionary_block_perm[j]].size(); i++) {
                dictionary.col(block[dictionary_block_perm[j]][i]) = zero_col;
            }
            
            
            //Get Error Term
            
            
            Eigen::MatrixXd error = ML_OpenMP::matrix_with_selected_cols(signal_mat,selected_index) - dictionary*ML_OpenMP::matrix_with_selected_cols(sparse_mat,selected_index,T);
            
            
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(error,Eigen::ComputeThinU | Eigen::ComputeThinV);
            Eigen::VectorXd singular_val = svd.singularValues();
            Eigen::MatrixXd rowV= svd.matrixV().transpose().topRows(block[dictionary_block_perm[j]].size());
            
            
            //Update dictionary
            
            for (int i=0; i<block[dictionary_block_perm[j]].size(); i++) {
                dictionary.col(block[dictionary_block_perm[j]][i]) = svd.matrixU().col(i);
            }
            
            
            //Update related sparse codes
            
            for (int i=0; i<selected_index.size(); i++) {
                for (int t=0; t<block[dictionary_block_perm[j]].size(); t++) {
                    sparse_mat.coeffRef(block[dictionary_block_perm[j]][t],(int)selected_index[i]) = rowV(t,i)*singular_val(t);
                }
            }
            
            
            
            
            //            std::cout<<"RMSE for "<<j<<":"<<(signal_mat-dictionary*sparse_mat).squaredNorm()/(double)sparse_mat.cols()<<std::endl;
            
        }
        //Clean Dictionary Atoms
        //clear_dict(dictionary,sparse_mat,signal_mat,unused_signals,unused_signal_len,non_zero_index,replaced_atoms,0.99,use_thresh);
        //Using K steps of SVD to update the dictionary
        
        
    }
    
}


void IPR(Eigen::MatrixXd &dictionary, Eigen::SparseMatrix<double> &sparse_mat_result,const Eigen::MatrixXd &signal_mat,const double mu0, const u_long sparsity,const u_long iter){
    double signalF = signal_mat.squaredNorm();
    for (u_long w=0; w<iter; w++) {
        std::cout<<"In IPR iter "<<w<<" :"<<std::endl;
        std::cout<<"Dictionary cols"<<dictionary.cols()<<" "<<dictionary.rows()<<std::endl;
        std::cout<<"   SNR:"<<20*log10(signalF/(signal_mat-dictionary*sparse_mat_result).squaredNorm())<<std::endl;
        std::cout<<"   Mutual Coherence:"<<get_mutual_coherence(dictionary)<<std::endl;
        std::cout<<"   AMC:"<<get_AMC(dictionary)<<std::endl;
        Eigen::MatrixXd DTD = dictionary.transpose()*dictionary;
        for (int i=0;i<dictionary.cols();i++){
            for (int j=0; j<dictionary.cols(); j++) {
                if(i==j){
                    DTD(i,j) = 1;
                }
                else{
                    if (DTD(i,j)>mu0) {
                        DTD(i,j) = mu0;
                    }
                    else if(DTD(i,j)<-mu0){
                        DTD(i,j) = -mu0;
                    }
                }
            }
        }
        std::cout<<"DTD:"<<DTD.rows()<<" "<<DTD.cols()<<std::endl;
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(DTD,Eigen::ComputeThinV|Eigen::ComputeThinU);
        Eigen::MatrixXd lambda = svd.singularValues();
        Eigen::MatrixXd V = svd.matrixV();
        Eigen::MatrixXd S(dictionary.rows(),dictionary.rows());
        for (int i=0; i<dictionary.rows(); i++) {
                S(i,i) = lambda(i,0);
        }
        
//        svd.compute(V*S*V.transpose(),Eigen::ComputeThinV|Eigen::ComputeThinU);
//        lambda = svd.singularValues();
//        for (int i=0; i<dictionary.rows(); i++) {
//            S(i,i) = sqrt(lambda(i,0));
//        }
//        V = svd.matrixV().leftCols(dictionary.rows());
        dictionary = S*V.transpose();
        std::cout<<(dictionary.transpose()*dictionary).diagonal()<<std::endl;
        Eigen::MatrixXd DS = dictionary*sparse_mat_result;
        Eigen::MatrixXd CC = signal_mat*DS.transpose();
        svd.compute(CC,Eigen::ComputeFullV|Eigen::ComputeFullU);
        dictionary = (svd.matrixV()*svd.matrixU().transpose())*dictionary;
    }
}

double get_mutual_coherence(const Eigen::MatrixXd &dictionary){
    Eigen::ArrayXXd  DTD = (dictionary.transpose()*dictionary).array();
    double result = 0;
    
    for(int i=0;i<dictionary.cols();i++){
        for (int j=i+1;j<dictionary.cols();j++){
            if(result<fabs(DTD(i,j))){
                result = fabs(DTD(i,j));
            }
        }
    }
    
    return result;
}

double get_AMC(const Eigen::MatrixXd &dictionary){
    Eigen::ArrayXXd DTD = (dictionary.transpose()*dictionary).array();
    double result = 0;
    for (int i=0;i<dictionary.cols();i++){
        for (int j=0;j<dictionary.cols();j++){
            if(i!=j){
                result+=fabs(DTD(i,j));
            }
        }
    }
    result/=dictionary.cols()*(dictionary.cols()-1);
    return result;
}

double get_RMSE(const Eigen::MatrixXd &dictionary, const Eigen::MatrixXd &input, const u_long &sparsity){
    std::vector<std::vector<u_long> > non_zero_index(dictionary.cols());
    Eigen::SparseMatrix<double> sparse_mat(dictionary.cols(),input.cols());
    sparse_mat.reserve(Eigen::VectorXi::Constant(sparse_mat.cols(), sparsity));
    OMP_batch(sparse_mat, dictionary, input, sparsity, non_zero_index);
    return (input-dictionary*sparse_mat).squaredNorm()/input.size();

}

double get_SNR(const Eigen::MatrixXd &dictionary,const Eigen::MatrixXd &X,const Eigen::MatrixXd &Y){
    return 20*log10(Y.squaredNorm()/(Y-dictionary*X).squaredNorm());
}