//
//  ml_random.cpp
//  KSVD
//
//  Created by sxjscience on 13-9-3.
//  Copyright (c) 2013年 sxjscience. All rights reserved.
//

#include "ml_random.h"
#include <time.h>
#include <vector>
#include "ml_common.h"
#include "assert.h"


ML_Random::ML_Random(){
    this->r = gsl_rng_alloc (T);
    gsl_rng_set(this->r, time(NULL));
}
ML_Random::~ML_Random(){
    gsl_rng_free (this->r);
}


u_long ML_Random::random_u_long(u_long lower_bound,u_long upper_bound)const{
    assert(upper_bound>=lower_bound);
    if(upper_bound == lower_bound){
        return upper_bound;
    }
    return lower_bound+gsl_rng_uniform_int(this->r, upper_bound-lower_bound);


}

void ML_Random::random_permutation_n(std::vector<u_long> &vec, u_long n){
    assert(n<=vec.size());
    for (u_long i = 0; i<n; i++) {
        u_long rand_index = this->random_u_long(i, vec.size()-1);
        ML_Common::vec_swap(vec, rand_index, i);
    }
}