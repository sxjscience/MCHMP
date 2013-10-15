//
//  ml_common.h
//  KSVD
//
//  Created by sxjscience on 13-9-22.
//  Copyright (c) 2013年 sxjscience. All rights reserved.
//

#ifndef __KSVD__ml_common__
#define __KSVD__ml_common__

#include <iostream>
#include <vector>
#include <string>
typedef unsigned long u_long;

class ML_Common{
public:
    static void vec_swap(std::vector<u_long> &vec,const u_long &i,const u_long &j);
    static std::vector<std::vector<u_long> > get_block(u_long dict_size,u_long block_size);
};

class Param{
public:
    static std::string save_train_feature_path;
    static std::string save_test_feature_path;
    static std::string save_train_label_path;
    static std::string save_test_label_path;
    static std::string liblinear_train_save_path;
    static std::string liblinear_test_save_path;
    static std::string statics_save_path;
    
};


#endif /* defined(__KSVD__ml_common__) */
