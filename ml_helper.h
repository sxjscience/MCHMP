//
//  ml_helper.h
//  KSVD
//
//  Created by sxjscience on 13-10-9.
//  Copyright (c) 2013年 sxjscience. All rights reserved.
//

#ifndef __KSVD__ml_helper__
#define __KSVD__ml_helper__

#include <iostream>
#include <string>
#include "ml_hmp.h"
typedef unsigned long u_long;


class ML_Helper{
public:
    static ML_HMP hmp_with_template(std::string filename);
    
};


#endif /* defined(__KSVD__ml_helper__) */
