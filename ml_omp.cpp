//
//  omp.cpp
//  KSVD
//
//  Created by sxjscience on 13-9-2.
//  Copyright (c) 2013年 sxjscience. All rights reserved.
//

#include "ml_omp.h"
#include "ml_openmp_common.h"
#include <vector>
#include "math.h"
#include "time.h"
#include <Eigen/Cholesky>

Eigen::SparseVector<double> OMP_naive(Eigen::MatrixXd dictionary,Eigen::VectorXd signal,u_long T){
    u_long row_num = dictionary.rows();
    u_long col_num = dictionary.cols();
    Eigen::SparseVector<double> result((int)col_num);
    std::vector<u_long> selected_index;
    Eigen::VectorXd residual = signal;
    Eigen::MatrixXd dictionary_T = dictionary.transpose();
    Eigen::MatrixXd real_dict(row_num,T);
    Eigen::VectorXd multi(col_num);
    Eigen::VectorXd sparse_vec;
    while(selected_index.size()<T){
        
        //Count D'*r
        multi = (dictionary_T*residual);
        //Select max index for <Di,r>
        double max_val = 0;
        u_long sel_index = selected_index.size();
        for(u_long i = 0;i<col_num;i++){
            if(fabs(multi(i))>max_val){
                max_val = fabs(multi(i));
                sel_index = i;
            }
        }
        selected_index.push_back(sel_index);
        //Update Real Dictionary
        real_dict.col(selected_index.size()-1) = dictionary.col(sel_index);

        //Count Pseudo Inverse
        const Eigen::MatrixXd &D = real_dict.leftCols(selected_index.size());
        Eigen::MatrixXd pinv_D = (D.transpose()*D).inverse()*D.transpose();
        sparse_vec = pinv_D*signal;
        residual = signal - D*sparse_vec;
        //std::cout<<"Residual:"<<residual.norm()<<std::endl;
    }
    
    //Get Result
    for(int i=0;i<selected_index.size();i++){
        result.insert((int)selected_index[i]) = sparse_vec(i);
    }
    return result;
}

/*
 
 Warning!!!!When Running cholesky OMP, dictionary MUST BE NORMALIZED

*/
Eigen::SparseMatrix<double> OMP_cholesky(Eigen::MatrixXd dictionary,Eigen::MatrixXd signal_mat,u_long T,double episilon){
    u_long row_num = dictionary.rows();
    u_long col_num = dictionary.cols();
    u_long signal_num = signal_mat.cols();

    Eigen::MatrixXd dictionary_T = dictionary.transpose();
    Eigen::SparseMatrix<double> result((int)col_num,(int)signal_num);

    for (u_long v = 0; v<signal_num; v++) {
        const Eigen::VectorXd &signal = signal_mat.col(v);

        Eigen::SparseVector<double> ele_result((int)col_num);
        std::vector<u_long> selected_index;
        //r = x
        Eigen::VectorXd residual = signal;
        //Get D.T
        //Real Dictionary to Store the selected cols of dictionary
        Eigen::MatrixXd real_dict(row_num,T);
        //Get D.T*x
        Eigen::VectorXd alpha = dictionary_T*signal;
        Eigen::VectorXd real_alpha(T);
        Eigen::MatrixXd L = Eigen::MatrixXd::Zero(T,T);
        Eigen::VectorXd multi(col_num);
        Eigen::VectorXd sparse_vec;
        
        //Init L
        L(0,0) = 1;
        int n = 0;
        while(selected_index.size()<T){
            
            
            
            //Count D'*r
            multi = (dictionary_T*residual);
            
            

            

            //Select max index for <Di,r>
            
            
            double max_val = 0;
            u_long sel_index = selected_index.size();
            for(u_long i = 0;i<col_num;i++){
                if(fabs(multi(i))>max_val){
                    max_val = fabs(multi(i));
                    sel_index = i;
                }
            }
            
            
            //Update L(Cholesky Decomposition)
            if (n>0) {            
                const Eigen::MatrixXd &D_T = real_dict.leftCols(selected_index.size()).transpose();
                Eigen::VectorXd w = L.topLeftCorner(n, n).triangularView<Eigen::Lower>().solve(D_T*dictionary.col(sel_index));
                L.block(n,0,1,n) = w.transpose();
                L(n,n) = sqrt(1-w.squaredNorm());
            }
            
            
            

            //Push Selected Index
            selected_index.push_back(sel_index);
            
            //Update Real Dictionary And Real Alpha
            real_dict.col(selected_index.size()-1) = dictionary.col(sel_index);
            real_alpha(selected_index.size()-1) = alpha(sel_index);
            
            
            //Using Cholesky Method to sovle the equation:D.T*D*Y = D.T*X
            const Eigen::MatrixXd &D = real_dict.leftCols(selected_index.size());
            sparse_vec = real_alpha.topRows(selected_index.size());
            L.topLeftCorner(n+1, n+1).triangularView<Eigen::Lower>().solveInPlace(sparse_vec);
            L.topLeftCorner(n+1, n+1).transpose().triangularView<Eigen::Upper>().solveInPlace(sparse_vec);
            residual = signal - D*sparse_vec;
            

            n++;
        }
        
        //Get Result
        for(int i=0;i<selected_index.size();i++){
            ele_result.insert((int)selected_index[i]) = sparse_vec(i);
        }
        result.col((int)v) = ele_result;
    }
    return result;
    
}

/*
 
 Warning!!!!
 When Running cholesky OMP, dictionary MUST BE NORMALIZED
 
 When you are writing productional codes,please use OMP_batch!!
 
 Rows of non_zero_index should BE THE SAME AS cols of dictionary
 */
void OMP_batch(Eigen::SparseMatrix<double> &result, const Eigen::MatrixXd &dictionary,const Eigen::MatrixXd &signal_mat,u_long T,std::vector< std::vector<u_long> > &non_zero_index,const double episilon){
    assert(T<=dictionary.cols());
    assert(result.rows() == dictionary.cols());
    assert(result.cols() == signal_mat.cols());
    std::cout<<"Begin To Run OMP Batch"<<std::endl;
    time_t start,finish;
    u_long row_num = dictionary.rows();
    u_long col_num = dictionary.cols();
    assert(non_zero_index.size()==col_num);
    u_long signal_num = signal_mat.cols();
    //Preprocess
    Eigen::MatrixXd dictionary_T = dictionary.transpose();
    
    start = clock();
    Eigen::MatrixXd DTD = dictionary_T*dictionary;
    finish = clock();
    std::cout<<"Time for count DTD:"<<(finish-start)/(double)CLOCKS_PER_SEC<<std::endl;
    
    start = clock();
    Eigen::MatrixXd DTS = dictionary_T*signal_mat;
    finish = clock();
    std::cout<<"Time for count DTS:"<<(finish-start)/(double)CLOCKS_PER_SEC<<std::endl;

    
#pragma omp parallel for
    for (u_long v = 0; v<signal_num; v++) {
        
        const Eigen::VectorXd &signal = signal_mat.col(v);
        std::vector<u_long> selected_index;
        Eigen::VectorXd euclid_zero = DTS.col(v);
        Eigen::VectorXd euclid = euclid_zero;
        Eigen::MatrixXd L = Eigen::MatrixXd::Zero(T,T);
        
        Eigen::VectorXd sparse_vec;
        //Init L
        L(0,0) = 1;
        int n = 0;
        
        
        while(selected_index.size()<T){
            

            //Select max index for <Di,r>  <==> Select max index for alpha
            double max_val = 0;
            u_long sel_index = selected_index.size();
            for(u_long i = 0;i<col_num;i++){
                if(fabs(euclid(i))>max_val){
                    max_val = fabs(euclid(i));
                    sel_index = i;
                }
            }
            if(max_val<episilon){
                break;
            }
            

            

            
            //Update L(Lower Triangular Matrix of Cholesky Decomposition)
            if (n>0) {
                std::vector<u_long> col_index(1,sel_index);
                Eigen::VectorXd w = L.topLeftCorner(n, n).triangularView<Eigen::Lower>().solve(ML_OpenMP::matrix_with_selected_index(DTD,selected_index,col_index));
                L.block(n,0,1,n) = w.transpose();
                L(n,n) = sqrt(fabs(1-w.squaredNorm()));//IMPORTANT TO USE fabs BECAUSE w.squaredNorm() can be 1
            }
            
            //Push Selected Index
            selected_index.push_back(sel_index);
            
            
            //Using Cholesky Method to sovle the equation:
            //Solve Y: D.T*D*Y = D.T*X <==> Solve Y: G(I)*Y = Euclid(I)
            
            sparse_vec = ML_OpenMP::vector_with_selected_index(euclid_zero,selected_index);
            L.topLeftCorner(n+1, n+1).triangularView<Eigen::Lower>().solveInPlace(sparse_vec);
            L.topLeftCorner(n+1, n+1).transpose().triangularView<Eigen::Upper>().solveInPlace(sparse_vec);

            euclid = euclid_zero - ML_OpenMP::matrix_with_selected_cols(DTD,selected_index) * sparse_vec;
            n++;
            

        }
        
        
        //Get Result
        
        for(int i=0;i<selected_index.size();i++){
            //Update non_zero_index
            
//            start = clock();
            non_zero_index[selected_index[i]].push_back(v);
//            finish = clock();
//            std::cout<<"Vec Insertion:"<<(finish-start)/(double)CLOCKS_PER_SEC<<std::endl;

            //Get Sparse Result
//            start = clock();
            result.insert((int)selected_index[i], (int)v) = sparse_vec[i];
//            finish = clock();
//            std::cout<<"Sparse Insertion:"<<(finish-start)/(double)CLOCKS_PER_SEC<<std::endl;

            
        }
        

        
    }
    return;
    
}


/*
 
 Warning!!!!! When Storing Result,batch_sign seperate the positive and negative parts!!!!
 
 
 */

void OMP_batch_sign(Eigen::SparseMatrix<double> &result, const Eigen::MatrixXd &dictionary,const Eigen::MatrixXd &signal_mat,u_long T,std::vector< std::vector<u_long> > &non_zero_index,const double episilon){
    assert(T<dictionary.cols());
    assert(result.rows()/2 == dictionary.cols());
    assert(result.cols() == signal_mat.cols());
    std::cout<<"Begin To Run OMP Batch"<<std::endl;
    time_t start,finish;
    u_long row_num = dictionary.rows();
    u_long col_num = dictionary.cols();
    assert(non_zero_index.size()==col_num);
    u_long signal_num = signal_mat.cols();
    //Preprocess
    Eigen::MatrixXd dictionary_T = dictionary.transpose();
    
    start = clock();
    Eigen::MatrixXd DTD = dictionary_T*dictionary;
    finish = clock();
    std::cout<<"Time for count DTD:"<<(finish-start)/(double)CLOCKS_PER_SEC<<std::endl;
    
    start = clock();
    Eigen::MatrixXd DTS = dictionary_T*signal_mat;
    finish = clock();
    std::cout<<"Time for count DTS:"<<(finish-start)/(double)CLOCKS_PER_SEC<<std::endl;
    
    
#pragma omp parallel for
    for (u_long v = 0; v<signal_num; v++) {
        
        const Eigen::VectorXd &signal = signal_mat.col(v);
        std::vector<u_long> selected_index;
        Eigen::VectorXd euclid_zero = DTS.col(v);
        Eigen::VectorXd euclid = euclid_zero;
        Eigen::MatrixXd L = Eigen::MatrixXd::Zero(T,T);
        
        Eigen::VectorXd sparse_vec;
        //Init L
        L(0,0) = 1;
        int n = 0;
        
        
        while(selected_index.size()<T){
            
            
            //Select max index for <Di,r>  <==> Select max index for alpha
            double max_val = 0;
            u_long sel_index = selected_index.size();
            for(u_long i = 0;i<col_num;i++){
                if(fabs(euclid(i))>max_val){
                    max_val = fabs(euclid(i));
                    sel_index = i;
                }
            }
            if(max_val<episilon){
                break;
            }
            
            
            
            
            
            //Update L(Lower Triangular Matrix of Cholesky Decomposition)
            if (n>0) {
                std::vector<u_long> col_index(1,sel_index);
                Eigen::VectorXd w = L.topLeftCorner(n, n).triangularView<Eigen::Lower>().solve(ML_OpenMP::matrix_with_selected_index(DTD,selected_index,col_index));
                L.block(n,0,1,n) = w.transpose();
                L(n,n) = sqrt(fabs(1-w.squaredNorm()));//IMPORTANT TO USE fabs BECAUSE w.squaredNorm() can be 1
            }
            
            
            //Push Selected Index
            selected_index.push_back(sel_index);
            
            
            //Using Cholesky Method to sovle the equation:
            //Solve Y: D.T*D*Y = D.T*X <==> Solve Y: G(I)*Y = Euclid(I)
            
            sparse_vec = ML_OpenMP::vector_with_selected_index(euclid_zero,selected_index);
            L.topLeftCorner(n+1, n+1).triangularView<Eigen::Lower>().solveInPlace(sparse_vec);
            L.topLeftCorner(n+1, n+1).transpose().triangularView<Eigen::Upper>().solveInPlace(sparse_vec);
            
            euclid = euclid_zero - ML_OpenMP::matrix_with_selected_cols(DTD,selected_index) * sparse_vec;
            n++;
            
            
        }
        
        
        //Get Result
        
        for(int i=0;i<selected_index.size();i++){
            //Update non_zero_index
            
            //            start = clock();
            non_zero_index[selected_index[i]].push_back(v);
            //            finish = clock();
            //            std::cout<<"Vec Insertion:"<<(finish-start)/(double)CLOCKS_PER_SEC<<std::endl;
            
            //Get Sparse Result
            //            start = clock();
            if(sparse_vec[i]>=0){
                result.insert((int)selected_index[i], (int)v) = sparse_vec[i];
            }
            else{
                result.insert(result.rows()/2+(int)selected_index[i], (int)v) = -sparse_vec[i];
            }
            //            finish = clock();
            //            std::cout<<"Sparse Insertion:"<<(finish-start)/(double)CLOCKS_PER_SEC<<std::endl;
            
            
        }
        
        
        
    }
    return;
    
}

void block_OMP_batch(Eigen::SparseMatrix<double> &result, const Eigen::MatrixXd &dictionary,const Eigen::MatrixXd &signal_mat, std::vector< std::vector<u_long> > &block , const u_long &T,std::vector< std::vector<u_long> > &non_zero_index,const double episilon){
    assert(T<dictionary.cols());
    assert(result.rows() == dictionary.cols());
    assert(result.cols() == signal_mat.cols());
    std::cout<<"Begin To Run Block OMP Batch"<<std::endl;
    time_t start,finish;
    u_long row_num = dictionary.rows();
    u_long col_num = dictionary.cols();
    assert(non_zero_index.size()==col_num);
    u_long signal_num = signal_mat.cols();
    //Preprocess DTD DTS and cholesky matrix L for each block;
    Eigen::MatrixXd dictionary_T = dictionary.transpose();
    
    start = clock();
    Eigen::MatrixXd DTD = dictionary_T*dictionary;
    finish = clock();
    std::cout<<"Time for count DTD:"<<(finish-start)/(double)CLOCKS_PER_SEC<<std::endl;
    
    start = clock();
    Eigen::MatrixXd DTS = dictionary_T*signal_mat;
    finish = clock();
    std::cout<<"Time for count DTS:"<<(finish-start)/(double)CLOCKS_PER_SEC<<std::endl;
    
    std::vector<Eigen::MatrixXd> block_cholesky_L(block.size());
    for (int i=0; i<block.size(); i++) {
        Eigen::LLT<Eigen::MatrixXd> llt(ML_OpenMP::matrix_with_selected_index(DTD, block[i], block[i]));
        block_cholesky_L[i] = llt.matrixL();
    }
    
    
    
    
    for (u_long v = 0; v<signal_num; v++) {
        
        const Eigen::VectorXd &signal = signal_mat.col(v);
        std::vector<u_long> selected_block_index;
        std::vector<u_long> selected_index;
        Eigen::VectorXd euclid_zero = DTS.col(v);
        Eigen::VectorXd euclid = euclid_zero;
        Eigen::MatrixXd L;
        
        Eigen::VectorXd sparse_vec;

        int n = 0;
        

        while(selected_block_index.size()<T){
            
            //Select max block index for <Di,r>  <==> Select max index for euclid
            double max_val = 0;
            u_long sel_block_index = selected_block_index.size();
            for(u_long i = 0;i<block.size();i++){
                double block_val = 0;
                for (int j=0;j<block[i].size();j++){
                    block_val+=euclid[block[i][j]]*euclid[block[i][j]];
                }
                if(block_val>max_val){
                    max_val = block_val;
                    sel_block_index = i;
                }
            }
            if(max_val<episilon){
                break;
            }
            
            
            
            
            //Update L(Lower Triangular Matrix of Cholesky Decomposition)
            if (n>0) {
                Eigen::MatrixXd newL = Eigen::MatrixXd::Zero(L.rows()+block[sel_block_index].size(),L.cols()+block[sel_block_index].size());
                newL.topLeftCorner(L.rows(), L.cols()) = L;
                Eigen::MatrixXd WT = ML_OpenMP::matrix_with_selected_index(DTD, selected_index , block[sel_block_index]);
                L.triangularView<Eigen::Lower>().solveInPlace(WT);
                                
                newL.block(L.rows(), 0, WT.cols(), WT.rows()) = WT.transpose();
                Eigen::LLT<Eigen::MatrixXd> llt(ML_OpenMP::matrix_with_selected_index(DTD, block[sel_block_index], block[sel_block_index])-WT.transpose()*WT);
                Eigen::MatrixXd V = llt.matrixL();
                newL.block(L.rows(),L.cols(),V.rows(),V.cols()) = V;
                L = newL;
            }
            else{
                L = block_cholesky_L[sel_block_index];

            }

            
            //Push Selected Index
            selected_block_index.push_back(sel_block_index);
            for (int i=0; i<block[sel_block_index].size(); i++) {
                selected_index.push_back(block[sel_block_index][i]);
            }
            
            //Using Cholesky Method to sovle the equation:
            //Solve Y: D.T*D*Y = D.T*X <==> Solve Y: G(I)*Y = Euclid(I)
            
            sparse_vec = ML_OpenMP::vector_with_selected_index(euclid_zero,selected_index);

            L.triangularView<Eigen::Lower>().solveInPlace(sparse_vec);
            L.transpose().triangularView<Eigen::Upper>().solveInPlace(sparse_vec);
            
            euclid = euclid_zero - ML_OpenMP::matrix_with_selected_cols(DTD,selected_index) * sparse_vec;

            n++;
            
            
        }
        

        //Get Result

        for(int i=0;i<selected_index.size();i++){
            //Update non_zero_index
            
            //            start = clock();
            non_zero_index[selected_index[i]].push_back(v);
            //            finish = clock();
            //            std::cout<<"Vec Insertion:"<<(finish-start)/(double)CLOCKS_PER_SEC<<std::endl;
            
            //Get Sparse Result
            //            start = clock();
            result.insert((int)selected_index[i], (int)v) = sparse_vec[i];
            //            finish = clock();
            //            std::cout<<"Sparse Insertion:"<<(finish-start)/(double)CLOCKS_PER_SEC<<std::endl;
            
            
        }

        
        
    }
    return;

    
}

void block_OMP_batch_sign(Eigen::SparseMatrix<double> &result, const Eigen::MatrixXd &dictionary,const Eigen::MatrixXd &signal_mat, std::vector< std::vector<u_long> > &block , const u_long &T,std::vector< std::vector<u_long> > &non_zero_index,const double episilon){
    assert(T<=dictionary.cols());
    assert(result.rows()/2 == dictionary.cols());
    assert(result.cols() == signal_mat.cols());
    std::cout<<"Begin To Run Block OMP Batch"<<std::endl;
    time_t start,finish;
    u_long row_num = dictionary.rows();
    u_long col_num = dictionary.cols();
    assert(non_zero_index.size()==col_num);
    u_long signal_num = signal_mat.cols();
    //Preprocess DTD DTS and cholesky matrix L for each block;
    Eigen::MatrixXd dictionary_T = dictionary.transpose();
    
    start = clock();
    Eigen::MatrixXd DTD = dictionary_T*dictionary;
    finish = clock();
    std::cout<<"Time for count DTD:"<<(finish-start)/(double)CLOCKS_PER_SEC<<std::endl;
    
    start = clock();
    Eigen::MatrixXd DTS = dictionary_T*signal_mat;
    finish = clock();
    std::cout<<"Time for count DTS:"<<(finish-start)/(double)CLOCKS_PER_SEC<<std::endl;
    
    std::vector<Eigen::MatrixXd> block_cholesky_L(block.size());
    for (int i=0; i<block.size(); i++) {
        Eigen::LLT<Eigen::MatrixXd> llt(ML_OpenMP::matrix_with_selected_index(DTD, block[i], block[i]));
        block_cholesky_L[i] = llt.matrixL();
    }
    
    
    
    
    for (u_long v = 0; v<signal_num; v++) {
        
        const Eigen::VectorXd &signal = signal_mat.col(v);
        std::vector<u_long> selected_block_index;
        std::vector<u_long> selected_index;
        Eigen::VectorXd euclid_zero = DTS.col(v);
        Eigen::VectorXd euclid = euclid_zero;
        Eigen::MatrixXd L;
        
        Eigen::VectorXd sparse_vec;
        
        int n = 0;
        
        
        while(selected_block_index.size()<T){
            
            //Select max block index for <Di,r>  <==> Select max index for euclid
            double max_val = 0;
            u_long sel_block_index = selected_block_index.size();
            for(u_long i = 0;i<block.size();i++){
                double block_val = 0;
                for (int j=0;j<block[i].size();j++){
                    block_val+=euclid[block[i][j]]*euclid[block[i][j]];
                }
                if(block_val>max_val){
                    max_val = block_val;
                    sel_block_index = i;
                }
            }
            if(max_val<episilon){
                break;
            }
            
            
            
            
            //Update L(Lower Triangular Matrix of Cholesky Decomposition)
            if (n>0) {
                Eigen::MatrixXd newL = Eigen::MatrixXd::Zero(L.rows()+block[sel_block_index].size(),L.cols()+block[sel_block_index].size());
                newL.topLeftCorner(L.rows(), L.cols()) = L;
                Eigen::MatrixXd WT = ML_OpenMP::matrix_with_selected_index(DTD, selected_index , block[sel_block_index]);
                L.triangularView<Eigen::Lower>().solveInPlace(WT);
                
                newL.block(L.rows(), 0, WT.cols(), WT.rows()) = WT.transpose();
                Eigen::LLT<Eigen::MatrixXd> llt(ML_OpenMP::matrix_with_selected_index(DTD, block[sel_block_index], block[sel_block_index])-WT.transpose()*WT);
                Eigen::MatrixXd V = llt.matrixL();
                newL.block(L.rows(),L.cols(),V.rows(),V.cols()) = V;
                L = newL;
            }
            else{
                L = block_cholesky_L[sel_block_index];
                
            }
            
            
            //Push Selected Index
            selected_block_index.push_back(sel_block_index);
            for (int i=0; i<block[sel_block_index].size(); i++) {
                selected_index.push_back(block[sel_block_index][i]);
            }
            
            //Using Cholesky Method to sovle the equation:
            //Solve Y: D.T*D*Y = D.T*X <==> Solve Y: G(I)*Y = Euclid(I)
            
            sparse_vec = ML_OpenMP::vector_with_selected_index(euclid_zero,selected_index);
            
            L.triangularView<Eigen::Lower>().solveInPlace(sparse_vec);
            L.transpose().triangularView<Eigen::Upper>().solveInPlace(sparse_vec);
            
            euclid = euclid_zero - ML_OpenMP::matrix_with_selected_cols(DTD,selected_index) * sparse_vec;
            
            n++;
            
            
        }
        
        
        
        
        
        //Get Result
        
        for(int i=0;i<selected_index.size();i++){
            //Update non_zero_index
            
            //            start = clock();
            non_zero_index[selected_index[i]].push_back(v);
            //            finish = clock();
            //            std::cout<<"Vec Insertion:"<<(finish-start)/(double)CLOCKS_PER_SEC<<std::endl;
            
            //Get Sparse Result
            //            start = clock();
            if(sparse_vec[i]>=0){
                result.insert((int)selected_index[i], (int)v) = sparse_vec[i];
            }
            else{
                result.insert(result.rows()/2+(int)selected_index[i], (int)v) = -sparse_vec[i];
            }
            //            finish = clock();
            //            std::cout<<"Sparse Insertion:"<<(finish-start)/(double)CLOCKS_PER_SEC<<std::endl;
            
            
        }
                
    }
    return;
    
    
}

