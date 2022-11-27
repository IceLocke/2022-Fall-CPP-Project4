#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "lib/OpenBLAS/include/cblas.h"
#include "matrix.h"


void randMatrix(struct Matrix *m) {
    srand(clock());
    size_t i, len = m->col * m->row;
    for (i = 0; i < len; i++) {
        m->pData[i] = rand() % 1000;
    }
}

void test(size_t size) {
    struct Matrix *a = createMatrix(size, size);
    struct Matrix *b = createMatrix(size, size);
    struct Matrix *c = createMatrix(size, size);
    struct Matrix *d = createMatrix(size, size);

    clock_t start, end, inteval;

    printf("Test start, matrix size: %lu\n", size);
    printf("Start generate matrix with random elements\n");
    randMatrix(a); randMatrix(b);
    printf("Generated\n");
    copyMatrix(c, a);

    printf("Start calculating\n");

    printf("Plain method start\n");
    start = clock();
    multiplyMatrix(a, b);
    end = clock();
    printf("Plain method: %ldms\n", end - start);

    printf("OpenBLAS SGEMM start\n");
    start = clock();
    copyMatrix(a, c);
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                a->row, b->col, a->col, 1,
                a->pData, a->row, 
                b->pData, b->row, 
                1, d->pData, a->row);
    end = clock();
    printf("OpenBLAS SGEMM: %ldms\n", end - start);

    printf("Reordered for start\n");
    start = clock();
    multiplyMatrix_Reorderd(a, b);
    end = clock();
    printf("Reordered for: %ldms\n", end - start);

    printf("Packing + SIMD method start\n");
    copyMatrix(a, c);
    start = clock();
    multiplyMatrix_Packing_SIMD(a, b);
    end = clock();
    printf("Packing + SIMD method: %ldms\n", end - start);

    printf("Packing + SIMD + OpenMP method start\n");
    copyMatrix(a, c);
    start = clock();
    multiplyMatrix_Packing_SIMD_OMP(a, b);
    end = clock();
    printf("Packing + SIMD + OpenMP method: %ldms\n", end - start);

    printf("Packing + SIMD + OpenMP + Paralleling method start\n");
    copyMatrix(a, c);
    start = clock();
    multiplyMatrix_Packing_SIMD_OMP_Paralleld(a, b);
    end = clock();
    printf("Packing + SIMD + OpenMP + Paralleling method: %ldms\n", end - start);

    printf("Advanced packing + SIMD + OpenMP + Paralleling method start\n");
    copyMatrix(a, c);
    start = clock();
    multiplyMatrix_AdvancedPacking_SIMD_OMP_Paralleld(a, b);
    end = clock();
    printf("Advanced packing + SIMD + OpenMP + Paralleling method: %ldms\n", end - start);

    printf("Advanced packing + SIMD + OpenMP + Paralleling + Blocking method start\n");
    copyMatrix(a, c);
    start = clock();
    multiplyMatrix_Packing_SIMD_OMP_Blocking(a, b);
    end = clock();
    printf("Advanced packing + SIMD + OpenMP + Paralleling + Blocking: %ldms\n", end - start);

    deleteMatrix(a);
    deleteMatrix(b);
    deleteMatrix(c);
    deleteMatrix(d);
}

int main() {

    // test(16);
    // test(128);
    // test(1024); 
    // test(2048);
    test(4096);
    // test(8192);

    // out of mem
    // test(65536);
}