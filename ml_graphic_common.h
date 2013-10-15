//
//  ml_graphic_common.h
//  KSVD
//
//  Created by sxjscience on 13-9-28.
//  Copyright (c) 2013年 sxjscience. All rights reserved.
//

#ifndef __KSVD__ml_graphic_common__
#define __KSVD__ml_graphic_common__

#include <iostream>
#include <string>
#include <Eigen/Dense>
typedef unsigned long u_long;


class ML_GraphCommon {
public:
    /*
     Function: display_filter
     Input: A filter matrix which can be RGB or gray
     */
    static void display_filter(Eigen::ArrayXXd filter, const std::string savepath = "", const std::string style = "RGB");
    
};


#endif /* defined(__KSVD__ml_graphic_common__) */
