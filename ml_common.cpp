//
//  ml_common.cpp
//  KSVD
//
//  Created by sxjscience on 13-9-22.
//  Copyright (c) 2013年 sxjscience. All rights reserved.
//

#include "ml_common.h"


std::string Param::save_test_feature_path ="";
std::string Param::save_train_feature_path ="";
std::string Param::save_test_label_path ="";
std::string Param::save_train_label_path ="";
std::string Param::liblinear_train_save_path ="";
std::string Param::liblinear_test_save_path ="";
std::string Param::statics_save_path = "";

void ML_Common::vec_swap(std::vector<u_long> &vec,const u_long &i,const u_long &j){
    u_long temp = vec[i];
    vec[i] = vec[j];
    vec[j] = temp;
}

std::vector<std::vector<u_long> > ML_Common::get_block(u_long dict_size,u_long block_size){
    std::vector<std::vector<u_long> > block;
    for(int i=0;i<dict_size/block_size;i++){
        std::vector<u_long> subblock_index;
        for(int j=0;j<block_size;j++){
            subblock_index.push_back(i*block_size+j);
        }
        block.push_back(subblock_index);
    }
    return block;
}