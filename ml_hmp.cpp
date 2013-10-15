//
//  ml_hmp.cpp
//  KSVD
//
//  Created by sxjscience on 13-10-5.
//  Copyright (c) 2013年 sxjscience. All rights reserved.
//

#include "ml_hmp.h"
#include "ksvd.h"
#include "omp.h"
#include "assert.h"
#include "math.h"
#include "time.h"
void HMP_Input::init(){
    if (mode == "RGB") {
        this->dictionary = Eigen::MatrixXd::Random(3*patch_size*patch_size,this->dict_size);
    }
    else if(mode == "GRAY"){
        this->dictionary = Eigen::MatrixXd::Random(patch_size*patch_size,this->dict_size);
    }

    ML_OpenMP::matrix_normalization(this->dictionary,'c');
    
    if(block_size>1){
        block = ML_Common::get_block(dict_size, block_size);
    }
    
}

void HMP_Input::learn(const Eigen::MatrixXd &input){
    Eigen::SparseMatrix<double> sparse_mat(this->dictionary.cols(),input.cols());
    time_t start,finish;
    start = clock();
    if(block_size==1){
        if(learning_mode == "mc"){
            KSVD_approx_mc(this->dictionary,sparse_mat,input,this->sparsity,this->iter,tradeoff);
        }
        else if(learning_mode == "mi"){
            KSVD_approx_mi(this->dictionary, sparse_mat, input, sparsity,iter);
        }
        else if(learning_mode == "approx"){
            KSVD_approx(dictionary, sparse_mat, input, sparsity,iter);
        }
        else if(learning_mode == "exact"){
            KSVD_opt_mutual_incoherence(dictionary, sparse_mat,input,sparsity,iter);
        }
    }
    else{
        block_KSVD_approx(this->dictionary,block,sparse_mat,input,this->sparsity,this->iter);
    }
    finish = clock();
    std::cout<<"*****Time Spent For Dictionary Learning:"<<(double)(finish-start)/CLOCKS_PER_SEC<<"******"<<std::endl;
    
}

std::vector<Eigen::MatrixXd> HMP_Input::gen_output(ImageSet& im_set){
    std::vector<Eigen::MatrixXd> result(im_set.image_vec.size());
    std::vector<u_long> row_vec(im_set.image_vec.size());
    std::vector<u_long> col_vec(im_set.image_vec.size());
    for (int i=0;i<im_set.image_vec.size();i++){
        ML_Image &im = im_set.image_vec[i];
        Eigen::MatrixXd X = im.get_patch_mat(this->patch_size,1,this->mode);
        
        std::cout<<"Original Features For Image "<<i<<" processed"<<std::endl;
        Eigen::SparseMatrix<double> sparse_mat(this->dictionary.cols()*2,X.cols());
        if(this->block_size==1){
            sparse_mat.reserve(Eigen::VectorXi::Constant(sparse_mat.cols(), this->sparsity));
            std::vector<std::vector<u_long> > non_zero_index(this->dictionary.cols());
            OMP_batch_sign(sparse_mat, this->dictionary, X, this->sparsity, non_zero_index);
            result[i] = ML_OpenMP::matrix_max_pooling(sparse_mat,this->pool_size,im.origin_rows,im.origin_cols,this->sparsity);

        }
        else{
            sparse_mat.reserve(Eigen::VectorXi::Constant(sparse_mat.cols(), this->sparsity*this->block_size));
            std::vector<std::vector<u_long> > non_zero_index(this->dictionary.cols());
            block_OMP_batch_sign(sparse_mat, this->dictionary, X,block, this->sparsity, non_zero_index);
            result[i] = ML_OpenMP::matrix_max_pooling(sparse_mat,this->pool_size,im.origin_rows,im.origin_cols,this->sparsity*this->block_size);

        }
        row_vec[i] = im.origin_rows/this->pool_size;
        col_vec[i] = im.origin_cols/this->pool_size;
    }
    this->output_row_vec = row_vec;
    this->output_col_vec = col_vec;
    return result;
}


Eigen::MatrixXd HMP_Output::gen_features(const std::vector<Eigen::MatrixXd> &input, const std::vector<u_long> &row_vec, const std::vector<u_long> &col_vec){
    assert(input.size() == row_vec.size());
    assert(row_vec.size()==col_vec.size());
    u_long feature_rows = 0;
    for (int i=0;i<spatial_pyramid_levels.size();i++){
        feature_rows+=this->spatial_pyramid_levels[i]*this->spatial_pyramid_levels[i];
    }
    feature_rows*=input[0].rows();
    Eigen::MatrixXd result(feature_rows,input.size());
    
    for (int i=0;i<input.size();i++){
        std::cout<<"input"<<i<<" "<<input[i].rows()<<" "<<input[i].cols()<<std::endl;
        std::cout<<"result:"<<result.rows()<<std::endl;
        result.col(i) = ML_OpenMP::matrix_sp_max_pooling(input[i],this->spatial_pyramid_levels,row_vec[i],col_vec[i]);
    }
    return result;
}


void HMP_Layer::init(const std::vector<Eigen::MatrixXd> &input){
    this->dictionary = Eigen::MatrixXd::Random(input[0].rows()*patch_size*patch_size,this->dict_size);
    ML_OpenMP::matrix_normalization(this->dictionary,'c');
    if (block_size>1) {
        block = ML_Common::get_block(dict_size, block_size);
    }
}

Eigen::MatrixXd HMP_Layer::gen_patches(std::vector<Eigen::MatrixXd> &input,std::vector<u_long> &input_row_vec,std::vector<u_long> &input_col_vec){
    Eigen::MatrixXd result(input[0].rows()*this->patch_size*this->patch_size,input.size()*this->sample_num_per_patch);
    for (int i=0; i<input.size(); i++) {
        result.middleCols(i*this->sample_num_per_patch, this->sample_num_per_patch) = ML_OpenMP::matrix_patch_sample(input[i], this->patch_size, input_row_vec[i], input_col_vec[i], this->sample_num_per_patch);
        
    }
    return result;
}



void HMP_Layer::learn(const Eigen::MatrixXd &input){
    Eigen::SparseMatrix<double> sparse_mat(this->dict_size,input.cols());
    time_t start,finish;
    start = clock();

    if (block_size>1) {
        block_KSVD_approx(dictionary, block, sparse_mat, input, sparsity,iter);
    }
    else{
        if(learning_mode == "mc"){
            KSVD_approx_mc(this->dictionary,sparse_mat,input,this->sparsity,this->iter,tradeoff);
        }
        else if(learning_mode == "mi"){
            KSVD_approx_mi(this->dictionary, sparse_mat, input, sparsity,iter);
        }
        else if(learning_mode == "approx"){
            KSVD_approx(dictionary, sparse_mat, input, sparsity,iter);
        }
        else if(learning_mode == "exact"){
            KSVD_opt_mutual_incoherence(dictionary, sparse_mat,input,sparsity,iter);
        }
    }
    finish = clock();

    std::cout<<"*****Time Spent For Dictionary Learning:"<<(double)(finish-start)/CLOCKS_PER_SEC<<"******"<<std::endl;


}

std::vector<Eigen::MatrixXd> HMP_Layer::gen_output(const std::vector<Eigen::MatrixXd> &input,std::vector<u_long> &input_row_vec,std::vector<u_long> &input_col_vec){
    std::vector<Eigen::MatrixXd> result(input.size());
    std::vector<u_long> row_vec(input.size());
    std::vector<u_long> col_vec(input.size());
    for (int i=0; i<input.size(); i++) {
        assert(input[i].cols() == input_row_vec[i]*input_col_vec[i]);
        Eigen::MatrixXd X = ML_OpenMP::matrix_patch_gen(input[i], this->patch_size,input_row_vec[i], input_col_vec[i]);
        std::cout<<"Layer Features For "<<i<<" processed"<<std::endl;
        Eigen::SparseMatrix<double> sparse_mat(this->dictionary.cols()*2,X.cols());
        if(block_size==1){
            sparse_mat.reserve(Eigen::VectorXi::Constant(sparse_mat.cols(), sparsity_omp));
            std::vector<std::vector<u_long> > non_zero_index(this->dictionary.cols());
            OMP_batch_sign(sparse_mat, this->dictionary, X, this->sparsity_omp, non_zero_index);
            result[i] = ML_OpenMP::matrix_max_pooling(sparse_mat,this->pool_size,input_row_vec[i]-this->patch_size+1,input_col_vec[i]-this->patch_size+1,this->sparsity_omp);
        }
        else{
            sparse_mat.reserve(Eigen::VectorXi::Constant(sparse_mat.cols(), sparsity_omp*block_size));
            std::vector<std::vector<u_long> > non_zero_index(this->dictionary.cols());
            block_OMP_batch_sign(sparse_mat, dictionary, X, block, sparsity_omp, non_zero_index);
            result[i] = ML_OpenMP::matrix_max_pooling(sparse_mat,this->pool_size,input_row_vec[i]-this->patch_size+1,input_col_vec[i]-this->patch_size+1,this->sparsity_omp*block_size);
        }
        row_vec[i] = (input_row_vec[i]-this->patch_size+1)/this->pool_size;
        col_vec[i] = (input_col_vec[i]-this->patch_size+1)/this->pool_size;


    }
    this->output_row_vec = row_vec;
    this->output_col_vec = col_vec;
    return result;


}


ML_HMP::ML_HMP(HMP_Input &hin,std::vector<HMP_Layer> &hlayer,HMP_Output &hout){
    this->hmp_input = hin;
    this->hmp_output = hout;
    this->hmp_layer = hlayer;
     
}

void ML_HMP::init(){
    hmp_input.init();
    pg.load_file_list(hmp_input.train_set_path.c_str());
    pg.gen_patch(hmp_input.patch_size,hmp_input.patch_size,hmp_input.patch_per_sample,hmp_input.mode,hmp_input.scaling_minsize);
    trainset.load_file_list(hmp_input.train_set_path.c_str());
    testset.load_file_list(hmp_input.test_set_path.c_str());
    trainset.init_image_vec(hmp_input.scaling_minsize);
    testset.m_label = trainset.m_label;
    testset.init_image_vec(hmp_input.scaling_minsize);
    
    
    assert(trainset.m_label.size() == testset.m_label.size());
    std::map<std::string,u_long>::iterator it;
    for (it = trainset.m_label.begin(); it!=trainset.m_label.end();++it ) {
        assert(testset.m_label.find(it->first)!=testset.m_label.end());
    }
    
    std::string train_label_path = Param::save_train_label_path;
    std::string test_label_path = Param::save_test_label_path;
    Eigen::MatrixXd test_label(1,testset.label_vec.size());
    Eigen::MatrixXd train_label(1,trainset.label_vec.size());
    for (int i=0; i<trainset.label_vec.size(); i++) {
        train_label(0,i) = trainset.label_vec[i];
    }
    for (int i=0; i<testset.label_vec.size(); i++) {
        test_label(0,i) = testset.label_vec[i];
    }
    ML_IO::save_mat(train_label_path.c_str(), train_label);
    ML_IO::save_mat(test_label_path.c_str(), test_label);
}

void ML_HMP::learn(){
    std::string D_path = "D.mlmat";
    std::string train_feature_path = Param::save_train_feature_path;
    ML_OpenMP::matrix_remove_dc(pg.patches);
    hmp_input.learn(pg.patches);
    ML_IO::save_mat(D_path.c_str(),hmp_input.dictionary);
    std::vector<Eigen::MatrixXd> input = hmp_input.gen_output(this->trainset);
    std::vector<u_long> row_vec = hmp_input.output_row_vec;
    std::vector<u_long> col_vec = hmp_input.output_col_vec;
    
    for(int i=0;i<hmp_layer.size();i++){
        hmp_layer[i].init(input);
        hmp_layer[i].learn(hmp_layer[i].gen_patches(input,row_vec,col_vec));
        input = hmp_layer[i].gen_output(input,row_vec,col_vec);
        row_vec = hmp_layer[i].output_row_vec;
        col_vec = hmp_layer[i].output_col_vec;
        char save_D_path[500];
        sprintf(save_D_path,"D%d.mlmat",i+1);
        ML_IO::save_mat(save_D_path, hmp_layer[i].dictionary);
    }
    train_features = hmp_output.gen_features(input, row_vec, col_vec);
    ML_IO::save_mat(train_feature_path.c_str(),train_features);
}



void ML_HMP::gen_features(){
    std::string test_feature_path = Param::save_test_feature_path;
    std::vector<Eigen::MatrixXd> input = hmp_input.gen_output(this->testset);
    std::vector<u_long> row_vec = hmp_input.output_row_vec;
    std::vector<u_long> col_vec = hmp_input.output_col_vec;
    for(int i=0;i<hmp_layer.size();i++){
        input = hmp_layer[i].gen_output(input,row_vec,col_vec);
        row_vec = hmp_layer[i].output_row_vec;
        col_vec = hmp_layer[i].output_col_vec;
    }
    test_features = hmp_output.gen_features(input, row_vec, col_vec);
    ML_IO::save_mat(test_feature_path.c_str(),test_features);
    
}



void Multi_Block_HMP_Input::init(){
    if (mode == "RGB") {
        for (u_long i=0; i<block_size.size(); i++) {
            dictionary.push_back(Eigen::MatrixXd::Random(3*patch_size*patch_size,dict_size[i]));
            ML_OpenMP::matrix_normalization(dictionary[i],'c');
        }
    }
    else if(mode == "GRAY"){
        for (u_long i=0; i<block_size.size(); i++) {
            dictionary.push_back(Eigen::MatrixXd::Random(patch_size*patch_size,dict_size[i]));
            ML_OpenMP::matrix_normalization(dictionary[i],'c');
        }
        
    }
}

Eigen::MatrixXd Multi_Block_HMP_Layer::gen_patches(std::vector<Eigen::MatrixXd> &input,std::vector<u_long> &input_row_vec,std::vector<u_long> &input_col_vec){
    Eigen::MatrixXd result(input[0].rows()*this->patch_size*this->patch_size,input.size()*this->sample_num_per_patch);
    for (int i=0; i<input.size(); i++) {
        result.middleCols(i*this->sample_num_per_patch, this->sample_num_per_patch) = ML_OpenMP::matrix_patch_sample(input[i], this->patch_size, input_row_vec[i], input_col_vec[i], this->sample_num_per_patch);
        
    }
    return result;
}

void Multi_Block_HMP_Input::learn(const Eigen::MatrixXd &input){


    for (int i=0; i<dictionary.size(); i++) {
        Eigen::SparseMatrix<double> sparse_mat(dictionary[i].cols(),input.cols());
        if (block_size[i]>1) {
            std::vector<std::vector<u_long> > block = ML_Common::get_block(dict_size[i], block_size[i]);
            block_KSVD_approx(dictionary[i], block, sparse_mat, input, sparsity[i],iter[i]);
        }
        else{
            KSVD_approx(dictionary[i], sparse_mat, input, sparsity[i],iter[i]);
        }
    }

}

std::vector<Eigen::MatrixXd> Multi_Block_HMP_Input::gen_output(ImageSet& im_set){
    std::vector<Eigen::MatrixXd> result(im_set.image_vec.size());
    std::vector<u_long> row_vec(im_set.image_vec.size());
    std::vector<u_long> col_vec(im_set.image_vec.size());

    for (int i=0; i<im_set.image_vec.size(); i++) {
        ML_Image &im = im_set.image_vec[i];
        Eigen::MatrixXd X = im.get_patch_mat(patch_size,1,mode);
        row_vec[i] = im.origin_rows/this->pool_size;
        col_vec[i] = im.origin_cols/this->pool_size;
        std::cout<<"Original Features For Image "<<i<<" processed"<<std::endl;
        
        u_long res_col =0;
        for (int j=0; j<dictionary.size(); j++) {
            res_col += dictionary[j].cols()*2;
        }
        
        result[i] = Eigen::MatrixXd::Zero(res_col, row_vec[i]*col_vec[i]);
        u_long curr_row = 0;
        for (int j=0; j<dictionary.size();j++ ) {
            Eigen::SparseMatrix<double> sparse_mat(dictionary[j].cols()*2,X.cols());
            if(block_size[j]==1){
                sparse_mat.reserve(Eigen::VectorXi::Constant(sparse_mat.cols(), sparsity[j]));
                std::vector<std::vector<u_long> > non_zero_index(dictionary[j].cols());
                OMP_batch_sign(sparse_mat, dictionary[j], X, sparsity[j], non_zero_index);
            }
            else{
                std::vector<std::vector<u_long> > block = ML_Common::get_block(dict_size[j], block_size[j]);
                sparse_mat.reserve(Eigen::VectorXi::Constant(sparse_mat.cols(), sparsity[j]*block_size[j]));
                std::vector<std::vector<u_long> > non_zero_index(dictionary[j].cols());
                block_OMP_batch_sign(sparse_mat, dictionary[j], X,block, sparsity[j], non_zero_index);
            }
            result[i].middleRows(curr_row,dictionary[j].cols()*2) = ML_OpenMP::matrix_max_pooling(sparse_mat,pool_size,im.origin_rows,im.origin_cols,sparsity[j]);
            curr_row += dictionary[j].cols()*2;
        }
        //Normalize
        ML_OpenMP::matrix_normalization(result[i],'c');


    }
    output_row_vec = row_vec;
    output_col_vec = col_vec;
    return result;
}

void Multi_Block_HMP_Layer::init(const std::vector<Eigen::MatrixXd> &input){
    for (int i=0; i<multi_block_hmp_layer_vec.size(); i++) {
        multi_block_hmp_layer_vec[i].init(input);
    }
}


void Multi_Block_HMP_Layer::learn(const Eigen::MatrixXd &input){


    for (int i=0; i<multi_block_hmp_layer_vec.size(); i++) {
        multi_block_hmp_layer_vec[i].learn(input);
    }

}


std::vector<Eigen::MatrixXd> Multi_Block_HMP_Layer::gen_output(const std::vector<Eigen::MatrixXd> &input,std::vector<u_long> &input_row_vec,std::vector<u_long> &input_col_vec){
    std::vector<std::vector<Eigen::MatrixXd> > mat(multi_block_hmp_layer_vec.size());
    for (int i=0; i<multi_block_hmp_layer_vec.size(); i++) {
        mat[i] = multi_block_hmp_layer_vec[i].gen_output(input, input_row_vec, input_col_vec);
    }
    std::vector<Eigen::MatrixXd> result(input.size());
    for (int i=0; i<input.size(); i++) {
        u_long res_row = 0;
        u_long res_col = 0;
        for (int j=0; j<multi_block_hmp_layer_vec.size(); j++) {
            res_row += mat[j][i].rows();
            res_col = mat[j][i].cols();
        }
        Eigen::MatrixXd res_mat(res_row,res_col);
        u_long curr_row = 0;
        for (int j=0; j<multi_block_hmp_layer_vec.size(); j++) {
            res_mat.block(curr_row, 0, mat[j][i].rows(), mat[j][i].rows()) = mat[j][i];
        }
        ML_OpenMP::matrix_normalization(res_mat,'c');
        result[i] = res_mat;
    }
    output_col_vec = multi_block_hmp_layer_vec[0].output_col_vec;
    output_row_vec = multi_block_hmp_layer_vec[0].output_row_vec;

    return result;
}

Multi_Block_HMP::Multi_Block_HMP(Multi_Block_HMP_Input &hin,std::vector<Multi_Block_HMP_Layer> hlayer,HMP_Output &hout){
    this->multi_hmp_input = hin;
    this->multi_hmp_output = hout;
    this->multi_hmp_layer = hlayer;
}


void Multi_Block_HMP::init(){
    multi_hmp_input.init();
    pg.load_file_list(multi_hmp_input.train_set_path.c_str());
    pg.gen_patch(multi_hmp_input.patch_size,multi_hmp_input.patch_size,multi_hmp_input.patch_per_sample,multi_hmp_input.mode,multi_hmp_input.scaling_minsize);
    trainset.load_file_list(multi_hmp_input.train_set_path.c_str());
    testset.load_file_list(multi_hmp_input.test_set_path.c_str());
    trainset.init_image_vec(multi_hmp_input.scaling_minsize);
    testset.m_label = trainset.m_label;
    testset.init_image_vec(multi_hmp_input.scaling_minsize);
    
    std::string train_label_path = Param::save_train_label_path;
    std::string test_label_path = Param::save_test_label_path;
    Eigen::MatrixXd test_label(1,testset.label_vec.size());
    Eigen::MatrixXd train_label(1,trainset.label_vec.size());
    for (int i=0; i<trainset.label_vec.size(); i++) {
        train_label(0,i) = trainset.label_vec[i];
    }
    for (int i=0; i<testset.label_vec.size(); i++) {
        test_label(0,i) = testset.label_vec[i];
    }
    ML_IO::save_mat(train_label_path.c_str(), train_label);
    ML_IO::save_mat(test_label_path.c_str(), test_label);

}
void Multi_Block_HMP::learn(){
    time_t start,finish;
    std::string D_path = "D.mlmat";
    std::string train_feature_path = Param::save_train_feature_path;
    ML_OpenMP::matrix_remove_dc(pg.patches);
    start = clock();
    multi_hmp_input.learn(pg.patches);
    finish = clock();
    std::cout<<"********Total Time spent for learning dictionaries of the INPUT LAYER:"<<(double)(finish-start)/CLOCKS_PER_SEC<<"***"<<std::endl;
    
    std::vector<Eigen::MatrixXd> input = multi_hmp_input.gen_output(this->trainset);
    std::vector<u_long> row_vec = multi_hmp_input.output_row_vec;
    std::vector<u_long> col_vec = multi_hmp_input.output_col_vec;
    
    for(int i=0;i<multi_hmp_layer.size();i++){
        multi_hmp_layer[i].init(input);
        start = clock();
        multi_hmp_layer[i].learn(multi_hmp_layer[i].gen_patches(input,row_vec,col_vec));
        finish = clock();
        std::cout<<"********Time Spent for learning dictionaries of Middle Layer "<<i<<":"<<(double)(finish-start)/CLOCKS_PER_SEC<<"***"<<std::endl;
        input = multi_hmp_layer[i].gen_output(input,row_vec,col_vec);
        row_vec = multi_hmp_layer[i].output_row_vec;
        col_vec = multi_hmp_layer[i].output_col_vec;
        char save_D_path[500];
        for (int j=0; j<multi_hmp_layer[i].multi_block_hmp_layer_vec.size(); j++) {
            sprintf(save_D_path,"D%d_%d.mlmat",i+1,j+1);
            ML_IO::save_mat(save_D_path, multi_hmp_layer[i].multi_block_hmp_layer_vec[j].dictionary);
        }
    }
    train_features = multi_hmp_output.gen_features(input, row_vec, col_vec);
    ML_IO::save_mat(train_feature_path.c_str(),train_features);

}

void Multi_Block_HMP::gen_features(){
    std::string test_feature_path = Param::save_test_feature_path;
    std::vector<Eigen::MatrixXd> input = multi_hmp_input.gen_output(this->testset);
    std::vector<u_long> row_vec = multi_hmp_input.output_row_vec;
    std::vector<u_long> col_vec = multi_hmp_input.output_col_vec;
//    for(int i=0;i<hmp_layer.size();i++){
//        input = hmp_layer[i].gen_output(input,row_vec,col_vec);
//        row_vec = hmp_layer[i].output_row_vec;
//        col_vec = hmp_layer[i].output_col_vec;
//    }
    test_features = multi_hmp_output.gen_features(input, row_vec, col_vec);
    ML_IO::save_mat(test_feature_path.c_str(),test_features);

}


