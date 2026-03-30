#pragma once

#include <cstdint>
#include "seal/util/uintarithsmallmod.h"

void transform_to_rev_cuda(
    uint64_t* values, 
    int log_n, 
    const seal::util::MultiplyUIntModOperand* roots, 
    uint64_t modulus,
    const seal::util::MultiplyUIntModOperand* scalar
);

void transform_from_rev_cuda(
    uint64_t* values, 
    int log_n, 
    const seal::util::MultiplyUIntModOperand* roots, 
    uint64_t modulus,
    const seal::util::MultiplyUIntModOperand* scalar
);
