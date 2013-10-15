//
//  ml_helper.cpp
//  KSVD
//
//  Created by sxjscience on 13-10-9.
//  Copyright (c) 2013年 sxjscience. All rights reserved.
//

#include "ml_helper.h"
#include <fstream>


ML_HMP ML_Helper::hmp_with_template(std::string filename){
    std::ifstream infile(filename.c_str(),std::ios::binary);
    std::string layer_type;
    std::string attri;
    HMP_Input in;
    HMP_Output out;
    std::vector<HMP_Layer> hlayer_vec;
    HMP_Layer middle;
    
    while(infile>>layer_type){
        if(layer_type == "train_feature_save_path"){
            infile>>Param::save_train_feature_path;
        }
        else if(layer_type == "test_feature_save_path"){
            infile>>Param::save_test_feature_path;
        }
        else if(layer_type == "train_label_save_path"){
            infile>>Param::save_train_label_path;
        }
        else if(layer_type == "test_label_save_path"){
            infile>>Param::save_test_label_path;
        }
        else if(layer_type == "liblinear_test_save_path"){
            infile>>Param::liblinear_test_save_path;
        }
        else if(layer_type == "liblinear_train_save_path"){
            infile>>Param::liblinear_train_save_path;
        }
        else if (layer_type == "in") {
            infile>>attri;
            
            
            if (attri == "train_set_path") {
                infile>>in.train_set_path;
            }
            else if(attri=="test_set_path"){
                infile>>in.test_set_path;
            }
            else if(attri=="scaling_minsize"){
                infile>>in.scaling_minsize;
            }
            else if(attri == "patch_size"){
                infile>>in.patch_size;
            }
            else if(attri == "patch_per_sample"){
                infile>>in.patch_per_sample;
            }
            else if(attri == "pool_size"){
                infile>>in.pool_size;
            }
            else if(attri == "dict_size"){
                infile>>in.dict_size;
            }
            else if(attri == "sparsity"){
                infile>>in.sparsity;
            }
            else if(attri == "iter"){
                infile>>in.iter;
            }
            else if(attri == "block_size"){
                infile>>in.block_size;
            }
            else if(attri == "mode"){
                infile>>in.mode;
            }
            else if(attri == "learning_mode"){
                infile>>in.learning_mode;
            }
            else if(attri == "tradeoff"){
                infile>>in.tradeoff;
            }
        }
        else if(layer_type == "h_layer"){
            int layer_num;
            infile>>layer_num;
            if (layer_num>=hlayer_vec.size()) {
                for (int i=0; i<layer_num-hlayer_vec.size()+1; i++) {
                    hlayer_vec.push_back(middle);
                }
            }
            infile>>attri;
            if (attri == "dict_size") {
                infile>>hlayer_vec[layer_num].dict_size;
            }
            else if (attri == "patch_size"){
                infile>>hlayer_vec[layer_num].patch_size;
            }
            else if (attri == "sample_num_per_patch"){
                infile>>hlayer_vec[layer_num].sample_num_per_patch;
            }
            else if (attri == "pool_size"){
                infile>>hlayer_vec[layer_num].pool_size;
            }
            else if (attri == "sparsity"){
                infile>>hlayer_vec[layer_num].sparsity;
            }
            else if (attri == "sparsity_omp"){
                infile>>hlayer_vec[layer_num].sparsity_omp;
            }
            else if (attri == "block_size"){
                infile>>hlayer_vec[layer_num].block_size;
            }
            else if (attri == "iter"){
                infile>>hlayer_vec[layer_num].iter;
            }
            else if (attri == "learning_mode"){
                infile>>hlayer_vec[layer_num].learning_mode;
            }
            else if (attri == "tradeoff"){
                infile>>hlayer_vec[layer_num].tradeoff;
            }
            
        }
        else if(layer_type == "out"){
            infile>>attri;
            if (attri=="spatial_pyramid_levels") {
                u_long level_num;
                infile>>level_num;
                for (int i=0; i<level_num; i++) {
                    u_long level;
                    infile>>level;
                    out.spatial_pyramid_levels.push_back(level);
                }
                
            }
        }
    }
    return ML_HMP(in,hlayer_vec,out);
}