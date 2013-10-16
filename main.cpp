//
//  main.cpp
//  KSVD
//
//  Created by sxjscience on 13-9-2.
//  Copyright (c) 2013年 sxjscience. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Cholesky>
#include "ml_omp.h"
#include "ksvd.h"
#include "stdlib.h"
#include "time.h"
#include "assert.h"
#include "ml_openmp_common.h"
#include "ml_graphic.h"
#include "ml_graphic_common.h"
#include "ml_io.h"
#include "ml_hmp.h"
#include "ml_helper.h"
#include "ml_common.h"


typedef unsigned long u_long;

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;


void test_omp(){
    Eigen::MatrixXd D(4,4);
    D<<1,2,3,4,
    2,3,4,5,
    3,4,5,6,
    5,6,1,2;
    ML_OpenMP::matrix_normalization(D,'c');
    
    Eigen::MatrixXd X(4,1);
    X<<1.0,
    4.2,
    0.9,
    0.8;
    u_long T = 4;
    Eigen::SparseMatrix<double> result(D.cols(),X.cols());
    vector<vector<u_long> > non_zero_index(D.cols());
    OMP_batch(result, D, X, T, non_zero_index);
    cout<<result<<endl;
    
}

void test_block_omp(){
    Eigen::MatrixXd D(4,4);
    D<<1,2,3,4,
    2,3,4,5,
    3,4,5,6,
    5,6,1,2;
    ML_OpenMP::matrix_normalization(D,'c');
    
    Eigen::MatrixXd X(4,1);
    X<<1.0,
    4.2,
    0.9,
    0.8;
    u_long T = 2;
    Eigen::SparseMatrix<double> result(D.cols(),X.cols());
    vector<vector<u_long> > block(2);
    for (int i=0; i<2; i++) {
        block[i].push_back(2*i);
        block[i].push_back(2*i+1);
    }
    vector<vector<u_long> > non_zero_index(D.cols());
    block_OMP_batch(result, D, X, block, T, non_zero_index);
    cout<<result<<endl;
    
}



void test_ksvd(){
    Eigen::MatrixXd D(4,4);
    D<<1,2,3,4,
    2,3,4,5,
    3,4,5,6,
    5,6,1,2;
    
    ML_OpenMP::matrix_normalization(D,'c');
    Eigen::MatrixXd X(4,6);
    X<<1,2,3,4,5,6,
    2,3,4,5,6,7,
    1.2,3,4,5.6,6.1,7.9,
    9,8,7,6,5,4;
    u_long T = 2;
    Eigen::SparseMatrix<double> sparse_mat(D.cols(),X.cols());
    u_long iter = 50;
    KSVD(D,sparse_mat,X,T,iter);
    cout<<"D="<<endl;
    cout<<D<<endl;
    cout<<"Sparse_mat="<<endl;
    cout<<sparse_mat<<endl;
    cout<<D*sparse_mat<<endl;
}

void test_ksvd_opt(){
    Eigen::MatrixXd D(4,4);
    D<<1,2,3,4,
    2,3,4,5,
    3,4,5,6,
    5,6,1,2;
    
    ML_OpenMP::matrix_normalization(D,'c');
    Eigen::MatrixXd X(4,6);
    X<<1,2,3,4,5,6,
    2,3,4,5,6,7,
    1.2,3,4,5.6,6.1,7.9,
    9,8,7,6,5,4;
    u_long T = 2;
    Eigen::SparseMatrix<double> sparse_mat(D.cols(),X.cols());
    u_long iter = 20;
    KSVD_opt_mutual_incoherence(D,sparse_mat,X,T,iter);
    cout<<"D="<<endl;
    cout<<D<<endl;
    cout<<"Sparse_mat="<<endl;
    cout<<sparse_mat<<endl;
    cout<<D*sparse_mat<<endl;
    
}

void compare_ksvd_opt_non_opt(){
    Eigen::MatrixXd D = Eigen::MatrixXd::Random(100,200);
    ML_OpenMP::matrix_normalization(D,'c');
    Eigen::MatrixXd Dopt = D;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(100,500);
    u_long T = 20;
    u_long iter = 10;
    Eigen::SparseMatrix<double> sparse_mat(D.cols(),X.cols());
    Eigen::SparseMatrix<double> sparse_mat_opt(D.cols(),X.cols());
    KSVD(D,sparse_mat,X,T,iter);
    KSVD_opt_mutual_incoherence(Dopt,sparse_mat_opt,X,T,iter,0.99,1E-6,5);
    cout<<"D="<<endl<<D<<endl;
    cout<<"Dopt="<<endl<<Dopt<<endl;
    
    cout<<"Mutual Coherence For D:"<<get_mutual_coherence(D)<<endl;
    cout<<"RMSE For D:"<<(X-D*sparse_mat).squaredNorm()<<endl;

    cout<<"Mutual Coherence For Dopt:"<<get_mutual_coherence(Dopt)<<endl;
    cout<<"RMSE For Dopt:"<<(X-Dopt*sparse_mat_opt).squaredNorm()<<endl;
}

void image_ksvd_test(){
    clock_t start,finish;
    string savepath = "/Users/sxjscience/Documents/Alibaba/Program/test.mlmat";
    PatchGen pg;
    start = clock();
    pg.load_file_list("/Users/sxjscience/Documents/Alibaba/Related_Paper/mhmp_cvpr/datasets/imagenetsmall/imagenetsmalltrain");
    pg.gen_patch(8,8,200,"RGB",300);
    finish = clock();
    cout<<"Time For Patch Generation:"<<(double)(finish-start)/CLOCKS_PER_SEC<<endl;
    start = clock();
    u_long dictsize = 400;
    u_long T = 4;
    u_long iter = 1;
    Eigen::MatrixXd &X = pg.patches;
    Eigen::MatrixXd D = Eigen::MatrixXd::Random(pg.patches.rows(),dictsize);
    ML_OpenMP::matrix_normalization(D,'c');
    Eigen::SparseMatrix<double> sparse_mat(D.cols(),X.cols());
    cout<<pg.patches.cols()<<endl;
    finish = clock();
    cout<<"Time For Parameter Initialization:"<<(double)(finish-start)/CLOCKS_PER_SEC<<endl;
    start = clock();
    
//    int block_size = 4;
//    vector<vector<u_long> > block(dictsize/block_size);
//    for (int i=0; i<block.size(); i++) {
//        for (int j=0; j<block_size; j++) {
//            block[i].push_back(i*block_size+j);
//        }
//    }
//    
//    block_KSVD_approx_v2(D,block,sparse_mat,X,T,iter);
    Eigen::MatrixXd Dmi = D;
    Eigen::MatrixXd Dmc = D;
  
    
    /* TEST FOR IPR*/
    KSVD_approx(D, sparse_mat, X, T,iter);
    IPR(D, sparse_mat, X, 0.7, T,10);
    
    /*END TEST*/
    
//    Dmc = D;
    
//    KSVD_approx(D,sparse_mat,X,T,iter);
//    KSVD_approx_mi(Dmi,sparse_mat,X,T,iter);
    Eigen::SparseMatrix<double> sparse_mat_mc(D.cols(),X.cols());

    
    KSVD_approx_mc(Dmc,sparse_mat_mc,X,T,iter,1000);

//    std::cout<<"KSVD RMSE:"<<get_RMSE(D,X,T)<<std::endl;
//    std::cout<<"KSVD Mutual Coherence:"<<get_mutual_coherence(D)<<std::endl;
//    std::cout<<"KSVD_MI RMSE:"<<get_RMSE(Dmi,X,T)<<std::endl;
//    std::cout<<"KSVD_MI Mutual Coherence:"<<get_mutual_coherence(Dmi)<<std::endl;
//    std::cout<<"KSVD_MI AMC:"<<get_AMC(Dmi)<<std::endl;
    std::cout<<"KSVD_MC RMSE:"<<get_RMSE(Dmc,X,T)<< std::endl;
    std::cout<<"KSVD_MC Mutual Coherence:"<<get_mutual_coherence(Dmc)<<std::endl;
    std::cout<<"KSVD_MC AMC:"<<get_AMC(Dmc)<<std::endl;

//    ML_IO::save_mat(savepath.c_str(), D);
//    ML_GraphCommon::display_filter(D);
    ML_GraphCommon::display_filter(Dmi);
    ML_GraphCommon::display_filter(Dmc);

    finish = clock();
    cout<<"Time For KSVD:"<<(double)(finish-start)/CLOCKS_PER_SEC<<endl;

}

void test_ImageSet(){
    clock_t start,finish;
    ImageSet trainset,testset;
    double min_size = 100.1;
    double patch_size = 8;
    trainset.load_file_list("/Users/sxjscience/Documents/Alibaba/Related_Paper/mhmp_cvpr/datasets/imagenetsmall/imagenetsmalltrain");
    testset.load_file_list("/Users/sxjscience/Documents/Alibaba/Related_Paper/mhmp_cvpr/datasets/imagenetsmall/imagenetsmalltest");
    trainset.init_image_vec(min_size);
    testset.init_image_vec(min_size);
    start = clock();
    vector<Eigen::MatrixXd> train_feature_set(trainset.image_vec.size());
    for(int i=0;i<trainset.image_vec.size();i++){
        ML_Image &im = trainset.image_vec[i];
        cout<<i<<endl;
        train_feature_set[i] = im.get_patch_mat(patch_size);
    }
    finish = clock();
    cout<<"Time cost for feature_set"<<(double)(finish-start)/CLOCKS_PER_SEC<<endl;
    std::string D_path = "/Users/sxjscience/Documents/Alibaba/Program/test.mlmat";
    Eigen::MatrixXd D = ML_IO::load_mat(D_path.c_str());
    
    
}

void simple_test(){
    Eigen::MatrixXd D(4,6);
    D<<1,2,3,4,5,6,
    2,3,4,5,3,4,
    3,4,5,6,1,2,
    5,6,1,2,3,9;
    D = ML_OpenMP::matrix_patch_cols(D, 2, 2);
    cout<<D<<endl;
    vector<u_long> spatial_pyramid_levels(3);
    spatial_pyramid_levels[0] = 1;
    spatial_pyramid_levels[1] = 2;
    spatial_pyramid_levels[2] = 3;
    D = ML_OpenMP::matrix_sp_max_pooling(D, spatial_pyramid_levels, 4-2+1, 6-2+1);
    cout<<D<<endl;
}

void hmp_test(){
    HMP_Input in;
    HMP_Output out;
    std::vector<HMP_Layer> hlayer_vec(1);
//    std::vector<HMP_Layer> hlayer_vec;
    
    
    in.train_set_path = "/Users/sxjscience/Documents/Alibaba/Related_Paper/mhmp_cvpr/datasets/imagenetsmall/imagenetsmalltrain";
    in.test_set_path = "/Users/sxjscience/Documents/Alibaba/Related_Paper/mhmp_cvpr/datasets/imagenetsmall/imagenetsmalltest";
    in.scaling_minsize = 200;
    in.patch_size = 8;
    in.patch_per_sample = 200;
    in.pool_size = 4;
    in.dict_size = 75;
    in.sparsity = 4;
    in.iter = 50;
    in.block_size = 1;
    in.mode = "GRAY";
    in.learning_mode = "mc";
    
    hlayer_vec[0].dict_size = 500;
    hlayer_vec[0].patch_size = 4;
    hlayer_vec[0].sample_num_per_patch = 200;
    hlayer_vec[0].pool_size = 1;
    hlayer_vec[0].sparsity = 2;
    hlayer_vec[0].block_size = 2;
    hlayer_vec[0].sparsity_omp = 10;
    hlayer_vec[0].iter = 25;
    hlayer_vec[0].learning_mode = "mc";
    
    
    
    vector<u_long> spatial_pyramid_levels(3);
    spatial_pyramid_levels[0] = 1;
    spatial_pyramid_levels[1] = 2;
    spatial_pyramid_levels[2] = 3;
    out.spatial_pyramid_levels = spatial_pyramid_levels;
    
    ML_HMP hmp(in,hlayer_vec,out);
    hmp.init();
    hmp.learn();
    hmp.gen_features();
}

void multi_block_hmp_test(){

    time_t start;
    time_t finish;
    Multi_Block_HMP_Input in;
    HMP_Output out;
    std::vector<Multi_Block_HMP_Layer> hlayer_vec;
    std::cout<<"**********Begin Multi Block Hmp Test!!!******"<<std::endl;

    in.train_set_path = "/Users/sxjscience/Documents/Alibaba/Related_Paper/mhmp_cvpr/datasets/imagenetsmall/imagenetsmalltrain";
    in.test_set_path = "/Users/sxjscience/Documents/Alibaba/Related_Paper/mhmp_cvpr/datasets/imagenetsmall/imagenetsmalltest";
    in.scaling_minsize = 100;
    in.patch_size = 8;
    in.patch_per_sample = 200;
    in.pool_size = 4;
    in.mode = "RGB";
    in.dict_size.push_back(500);
    in.dict_size.push_back(500);
    in.dict_size.push_back(500);
    in.block_size.push_back(1);
    in.block_size.push_back(2);
    in.block_size.push_back(4);
    in.sparsity.push_back(4);
    in.sparsity.push_back(2);
    in.sparsity.push_back(1);
    in.iter.push_back(30);
    in.iter.push_back(30);
    in.iter.push_back(30);
    
    out.spatial_pyramid_levels.push_back(1);
    out.spatial_pyramid_levels.push_back(2);
    out.spatial_pyramid_levels.push_back(3);
    
    Multi_Block_HMP mhmp(in,hlayer_vec,out);
    
    start = clock();
    mhmp.init();
    finish = clock();
    std::cout<<"**********Time Spent for initialization:"<<(double)(finish-start)/CLOCKS_PER_SEC<<"*********"<<std::endl;
    
    start = clock();
    mhmp.learn();
    
    mhmp.gen_features();
    finish = clock();

    std::cout<<"**********Time Spent for Learning and generating features for the :"<<(double)(finish-start)/CLOCKS_PER_SEC<<"*********"<<std::endl;

    
}

void svm_test(){
    std::string train_feature_path = "/Users/sxjscience/Documents/Alibaba/Program/train_feature.mlmat";
    std::string test_feature_path = "/Users/sxjscience/Documents/Alibaba/Program/test_feature.mlmat";
    std::string train_label_path = "/Users/sxjscience/Documents/Alibaba/Program/train_label.mlmat";
    std::string test_label_path = "/Users/sxjscience/Documents/Alibaba/Program/test_label.mlmat";
    Eigen::MatrixXd train_feature = ML_IO::load_mat(train_feature_path.c_str());
    Eigen::MatrixXd test_feature = ML_IO::load_mat(test_feature_path.c_str());
    Eigen::MatrixXd train_label = ML_IO::load_mat(train_label_path.c_str());
    Eigen::MatrixXd test_label = ML_IO::load_mat(test_label_path.c_str());
    
    ML_IO::save_mat_to_txt("/Users/sxjscience/Documents/Alibaba/Program/train_feature.txt",train_feature);
    ML_IO::save_mat_to_txt("/Users/sxjscience/Documents/Alibaba/Program/test_feature.txt",test_feature);
    ML_IO::save_mat_to_txt("/Users/sxjscience/Documents/Alibaba/Program/train_label.txt",train_label);
    ML_IO::save_mat_to_txt("/Users/sxjscience/Documents/Alibaba/Program/test_label.txt",test_label);
}

void draw_filter(std::string path){
    Eigen::MatrixXd D = ML_IO::load_mat(path.c_str());
    ML_GraphCommon::display_filter(D);
}

void multi_train(){
    Param::save_test_feature_path = "/Users/sxjscience/Documents/Alibaba/Program/test_feature.mlmat";
    Param::save_train_feature_path = "/Users/sxjscience/Documents/Alibaba/Program/train_feature.mlmat";
    Param::save_test_label_path = "/Users/sxjscience/Documents/Alibaba/Program/test_label.mlmat";
    Param::save_train_label_path = "/Users/sxjscience/Documents/Alibaba/Program/train_label.mlmat";
    
    
    
    multi_block_hmp_test();
    svm_test();
}


void train(std::string template_name,std::string stat_file_name=""){
    time_t start,finish;
    time_t start_time,finish_time;
    Param::statics_save_path = stat_file_name;
    ML_HMP hmp = ML_Helper::hmp_with_template(template_name);
    
    
    start_time = clock();
    start = clock();
    hmp.init();
    finish = clock();
    std::cout<<"**********Time Spent for initialization:"<<(double)(finish-start)/CLOCKS_PER_SEC<<"*********"<<std::endl;

    
    start = clock();
    hmp.learn();
    finish = clock();
    std::cout<<"**********Time Spent for Learning and generating features for the :"<<(double)(finish-start)/CLOCKS_PER_SEC<<"*********"<<std::endl;

    start = clock();
    hmp.gen_features();
    finish = clock();
    std::cout<<"**********Time Spent for Learning and generating features for the :"<<(double)(finish-start)/CLOCKS_PER_SEC<<"*********"<<std::endl;
    finish_time = clock();
    
    std::cout<<"**********TOTOAL TIME:"<<(double)(finish_time-start_time)/CLOCKS_PER_SEC<<"*********"<<std::endl;

    
    std::cout<<"Now save as liblinear template:"<<std::endl;
    Eigen::MatrixXd feature_train_m = ML_IO::load_mat(Param::save_train_feature_path.c_str());
    Eigen::MatrixXd feature_test_m = ML_IO::load_mat(Param::save_test_feature_path.c_str());
    Eigen::MatrixXd label_train_m = ML_IO::load_mat(Param::save_train_label_path.c_str());
    Eigen::MatrixXd label_test_m = ML_IO::load_mat(Param::save_test_label_path.c_str());

    ML_IO::format_mat_to_liblinear(Param::liblinear_train_save_path.c_str(),feature_train_m,label_train_m);
    ML_IO::format_mat_to_liblinear(Param::liblinear_test_save_path.c_str(),feature_test_m,label_test_m);
    std::cout<<"....Template saved..."<<std::endl;
}

void classify_test(std::string template_name){
    ifstream fin(template_name.c_str());
    string dirname;
    vector<Eigen::MatrixXd> feature_train_m;
    vector<Eigen::MatrixXd> feature_test_m;
    Eigen::MatrixXd label_train_m;
    Eigen::MatrixXd label_test_m;
    bool has_load = 0;
    u_long dir_num = 0;
    fin>>dir_num;
    u_long train_mat_row = 0;
    u_long test_mat_row = 0;
    u_long train_mat_col = 0;
    u_long test_mat_col = 0;
    for(int n = 0;n<dir_num;n++) {
        fin>>dir_num;
        string train_feature_path = dirname+"/train_feature.mlmat";
        string test_feature_path = dirname+"/test_feature.mlmat";
        string label_train_path = dirname+"/train_label.mlmat";
        string label_test_path = dirname+"/test_label.mlmat";
        if (has_load == 0) {
            label_train_m = ML_IO::load_mat(label_train_path.c_str());
            label_test_m = ML_IO::load_mat(label_test_path.c_str());
            has_load =1;
        }
        else{
            Eigen::MatrixXd label_train_temp = ML_IO::load_mat(label_train_path.c_str());
            Eigen::MatrixXd label_test_temp = ML_IO::load_mat(label_test_path.c_str());
            for (int i=0; i<label_train_temp.rows(); i++) {
                for (int j=0; j<label_train_temp.cols(); j++) {
                    assert(label_train_temp(i,j)==label_train_m(i,j));
                }
            }
            for (int i=0; i<label_test_temp.rows(); i++) {
                for (int j=0; j<label_test_temp.cols(); j++) {
                    assert(label_test_temp(i,j)==label_test_m(i,j));
                }
            }
            
        }
        Eigen::MatrixXd f_train_m = ML_IO::load_mat(train_feature_path.c_str());
        Eigen::MatrixXd f_test_m = ML_IO::load_mat(test_feature_path.c_str());
        feature_train_m.push_back(f_train_m);
        feature_test_m.push_back(f_test_m);
        train_mat_row += f_train_m.rows();
        test_mat_row +=f_test_m.rows();
        if (train_mat_col == 0) {
            train_mat_col = f_train_m.cols();
        }
        else{
            assert(train_mat_col == f_train_m.cols());
        }
        
        if (test_mat_col == 0) {
            test_mat_col = f_test_m.cols();
        }
        else{
            assert(test_mat_col == f_test_m.cols());
        }
    }
    Eigen::MatrixXd train_feature(train_mat_row,train_mat_col);
    Eigen::MatrixXd test_feature(test_mat_row,test_mat_col);
    u_long curr_train_row = 0;
    u_long curr_test_row = 0;
    for (int i=0;i<feature_train_m.size();i++){
        train_feature.middleRows(curr_train_row, feature_train_m[i].rows()) = feature_train_m[i];
        curr_train_row += feature_train_m[i].rows();
    }

    for (int i=0;i<feature_test_m.size();i++){
        test_feature.middleRows(curr_train_row, feature_test_m[i].rows()) = feature_test_m[i];
        curr_train_row += feature_test_m[i].rows();
    }
    string train_data_save_path;
    string test_data_save_path;
    
    fin>>train_data_save_path;
    fin>>test_data_save_path;
    ML_IO::format_mat_to_liblinear(train_data_save_path.c_str(),train_feature,label_train_m);
    ML_IO::format_mat_to_liblinear(test_data_save_path.c_str(),test_feature,label_test_m);
    
}

//void test_statistics(string trainfiledir,string testfiledir,string algo){
//    clock_t start,finish;
//    ofstream total_static_file("total_statics.stat");
//
//    
//}

int main(int argc,char* argv[])
{
    if (argc <= 1) {
//        image_ksvd_test();
        //multi_train();
        image_ksvd_test();
//        train("/Users/sxjscience/Documents/Alibaba/Program/template/A2_MC");
//        svm_test();
    }
    else if (argc>1){
        string command = std::string(argv[1]);
        if (command == "-d"){
            draw_filter(argv[2]);
        }
        else if(command == "-c"){
            classify_test(argv[2]);
        }
        else if(command == "-t"){
            if(argc>3){
                train(argv[2],argv[3]);
            }
            else{
                train(argv[2]);
            }
        }
        else if(command == "-s"){
//            test_statistics(argv[2]);
        }
    }
    
//    image_ksvd_test();
    return 0;
}

