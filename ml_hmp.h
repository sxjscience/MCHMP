//
//  ML_HMP.h
//  KSVD
//
//  Created by sxjscience on 13-10-5.
//  Copyright (c) 2013年 sxjscience. All rights reserved.
//

#ifndef __KSVD__ml_hmp__
#define __KSVD__ml_hmp__

#include <iostream>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "ml_common.h"
#include "ml_openmp_common.h"
#include "ml_graphic.h"
#include "ml_graphic_common.h"
#include "ml_io.h"
typedef unsigned long u_long;




class HMP_Input{
private:
    
public:
    std::string train_set_path;
    std::string test_set_path;
    Eigen::MatrixXd dictionary;
    std::vector<std::vector<u_long> > block;

    u_long scaling_minsize;
    u_long patch_size;
    u_long patch_per_sample;
    u_long pool_size;
    u_long dict_size;
    u_long sparsity;
    u_long iter;
    u_long block_size;
    std::string learning_mode;
    std::string mode;
    double tradeoff;
    
    
    std::vector<u_long> output_row_vec;
    std::vector<u_long> output_col_vec;
    
    void init();
    void learn(const Eigen::MatrixXd &input);
    std::vector<Eigen::MatrixXd> gen_output(ImageSet& im_set);
};


class HMP_Layer{
public:
    Eigen::MatrixXd dictionary;
    u_long pool_size;
    u_long sparsity;
    u_long sparsity_omp;
    u_long dict_size;
    u_long patch_size;
    u_long sample_num_per_patch;
    u_long iter;
    u_long block_size;
    std::string learning_mode;
    double tradeoff;

    
    std::vector<std::vector<u_long> > block;
    
    std::vector<u_long> output_row_vec;
    std::vector<u_long> output_col_vec;
    
    Eigen::MatrixXd gen_patches(std::vector<Eigen::MatrixXd> &input,std::vector<u_long> &input_row_vec,std::vector<u_long> &input_col_vec);
    
    void init(const std::vector<Eigen::MatrixXd> &input);
    void learn(const Eigen::MatrixXd &input);
    std::vector<Eigen::MatrixXd> gen_output(const std::vector<Eigen::MatrixXd> &input,std::vector<u_long> &input_row_vec,std::vector<u_long> &input_col_vec);
};

class HMP_Output{
public:
    
    std::vector<u_long> spatial_pyramid_levels;
    Eigen::MatrixXd gen_features(const std::vector<Eigen::MatrixXd> &input, const std::vector<u_long> &row_vec, const std::vector<u_long> &col_vec);
};

class ML_HMP{
private:
    HMP_Input hmp_input;
    std::vector<HMP_Layer> hmp_layer;
    HMP_Output hmp_output;
    PatchGen pg;
    ImageSet trainset;
    ImageSet testset;
    Eigen::MatrixXd train_features;
    Eigen::MatrixXd test_features;
    
public:
    ML_HMP(HMP_Input &hin,std::vector<HMP_Layer> &hlayer,HMP_Output &hout);
    void init();
    void learn();
    void gen_features();
};

class Multi_Block_HMP_Input{
public:
    std::string train_set_path;
    std::string test_set_path;
    u_long scaling_minsize;
    u_long patch_size;
    u_long pool_size;
    u_long patch_per_sample;
    std::string mode;
    
    std::vector<Eigen::MatrixXd> dictionary;
    
    
    std::vector<u_long> sparsity;
    std::vector<u_long> dict_size;
    std::vector<u_long> iter;
    std::vector<u_long> block_size;
    
    std::vector<u_long> output_row_vec;
    std::vector<u_long> output_col_vec;

    void init();
    void learn(const Eigen::MatrixXd &input);
    std::vector<Eigen::MatrixXd> gen_output(ImageSet& im_set);
    

};

class Multi_Block_HMP_Layer{
public:
    
    u_long patch_size;
    u_long pool_size;
    u_long sample_num_per_patch;
    
    std::vector<HMP_Layer> multi_block_hmp_layer_vec;
    std::vector<u_long> output_row_vec;
    std::vector<u_long> output_col_vec;
    
    Eigen::MatrixXd gen_patches(std::vector<Eigen::MatrixXd> &input,std::vector<u_long> &input_row_vec,std::vector<u_long> &input_col_vec);
    void init(const std::vector<Eigen::MatrixXd> &input);
    void learn(const Eigen::MatrixXd &input);
    std::vector<Eigen::MatrixXd> gen_output(const std::vector<Eigen::MatrixXd> &input,std::vector<u_long> &input_row_vec,std::vector<u_long> &input_col_vec);
    
};

class Multi_Block_HMP{
private:
    Multi_Block_HMP_Input multi_hmp_input;
    std::vector<Multi_Block_HMP_Layer> multi_hmp_layer;
    HMP_Output multi_hmp_output;
    PatchGen pg;
    ImageSet trainset;
    ImageSet testset;
    Eigen::MatrixXd train_features;
    Eigen::MatrixXd test_features;
public:
    Multi_Block_HMP(Multi_Block_HMP_Input &hin,std::vector<Multi_Block_HMP_Layer> hlayer,HMP_Output &hout);
    void init();
    void learn();
    void gen_features();

};


#endif /* defined(__KSVD__ML_HMP__) */
