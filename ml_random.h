//
//  ml_random.h
//  KSVD
//
//  Created by sxjscience on 13-9-3.
//  Copyright (c) 2013年 sxjscience. All rights reserved.
//

#ifndef __KSVD__ml_random__
#define __KSVD__ml_random__

#include <iostream>
#include <gsl/gsl_rng.h>
#include <vector>
typedef unsigned long u_long;
class ML_Random{
private:
    const gsl_rng_type *T = gsl_rng_default;
    gsl_rng * r;
public:
    ML_Random();
    ~ML_Random();
    
    /*
     Name: 
        random_u_long
     Usage: 
        Generate a random number within the range ==> [low_bound,upper_bound]
     */
    u_long random_u_long (u_long lower_bound,u_long upper_bound)const;
    
    /*
     Name:
        random_permutation_n
     Usage:
        Randomly select n numbers from the vector and put these selected numbers to the beginning par† of the vector.
     */
    
    void random_permutation_n(std::vector<u_long> &vec, u_long n);
};




#endif /* defined(__KSVD__ml_random__) */
