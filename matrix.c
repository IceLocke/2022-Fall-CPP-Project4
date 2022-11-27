#include "matrix.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <malloc.h>
#include <omp.h>
// BE SURE THAT BLOCK_SIZE IS MULTIPLE OF 32
#define _abs(a) ((a > 0) ? (a) : (-a))
#define eps (1e-10f)

/*
    Matrix allocation
*/

#define WITH_AVX2

#ifdef WITH_AVX2
#define ALIGNMENT 256
#endif

#ifdef WITH_NEON
#define ALIGNMENT 128
#define _aligned_malloc(a, b) aligned_alloc(b, a)
#define _aligned_free(a) free(a)
#endif

void printMatrix(struct Matrix *m) {
    size_t i, len = m->col * m->row;
    for (i = 0; i < len; i++) {
        printf("%.1f ", m->pData[i]);
        if ((i + 1) % m->col == 0) putchar('\n');
    }
    putchar('\n');
}

void printMatrixPData(float *pData, size_t row, size_t col) {
    size_t i, len = row * col;
    for (i = 0; i < len; i++) {
        printf("%.1f ", pData[i]);
        if ((i + 1) % col == 0) putchar('\n');
    }
    putchar('\n');
}

void printLoadedVector(float *p, size_t size) {
    size_t i;
    for (i = 0; i < size; i++)
        printf("%.1f ", p[i]);
    puts("");
}


struct Matrix* createMatrix(const size_t row, const size_t col) {
    struct Matrix *m = (struct Matrix*) malloc(sizeof(struct Matrix));
    if (m != NULL && row > 0 && col > 0) {
        m->row = row;
        m->col = col;
        m->pData = NULL;
        m->pData = (float*) _aligned_malloc((row + 1) * col * sizeof(float), 256);
        if (m->pData != NULL) {
            return m;
        }
        else return NULL;
    }
    return NULL;
}

int deleteMatrix(struct Matrix *const targetMatrix) {
    if (targetMatrix != NULL) {
        _aligned_free(targetMatrix->pData);
        free(targetMatrix);
        return 0;
    }
    return 1;
}

inline float matrixGetElement(const struct Matrix *const srcMatrix, 
                       const size_t i, const size_t j) {
    if (srcMatrix != NULL) {
        if (i >= 1 && i <= srcMatrix->row && 
            j >= 1 && j <= srcMatrix->col) {
                if (srcMatrix->pData != NULL)
                    return srcMatrix->pData[srcMatrix->col * (i - 1) + j - 1];
            }
    }
    return 0.0f;
}

inline int matrixSetElement(struct Matrix *const targetMatrix, 
                     const size_t i, const size_t j, const float value) {
    if (targetMatrix != NULL) {
        if (i >= 1 && i <= targetMatrix->row &&
            j >= 1 && j <= targetMatrix->col) {
                if (targetMatrix->pData != NULL) {
                    targetMatrix->pData[targetMatrix->col * (i - 1) + j - 1] = value;
                    return 0;
                }
            }
    }
    return 1;
}

inline float* matrixGetPData(const struct Matrix *const srcMatrix) {
    if (srcMatrix != NULL)
        return srcMatrix->pData;
    return NULL;
}

int matrixSetAll(struct Matrix *const targetMatrix, 
                 float value) {
    if (targetMatrix != NULL) {
        if (targetMatrix->pData != NULL) {
            size_t size = targetMatrix->row * targetMatrix->col;
            int i;
            for (i = 0; i < size; i++)
                targetMatrix->pData[i] = value;
            return 0;
        }
    }
    return 1;
}

int matrixSetAllZero(struct Matrix *const targetMatrix) {
    if (targetMatrix != NULL) {
        if (targetMatrix->pData != NULL) {
            size_t i, size = targetMatrix->row * targetMatrix->col;
            for (i = 0; i < size; i++)
                targetMatrix->pData[i] = 0.0f;
            return 0;
        }
    }
    return 1;
}

int matrixSetIdentity(struct Matrix *const targetMatrix) {
    if (targetMatrix != NULL) {
        if (targetMatrix->row == targetMatrix->col) {
            if (targetMatrix->pData != NULL) {
                size_t i, j;
                for (i = 0; i < targetMatrix->row; i++) {
                    for (j = 0; j < targetMatrix->col; j++) {
                        if (i == j)
                            targetMatrix->pData[i * targetMatrix->col + j] = 1.0f;
                        else targetMatrix->pData[i * targetMatrix->col + j] = 0.0f;
                    }
                }
                return 0;
            }
        }
    }
    return 1;
}

int copyMatrix(struct Matrix *const targetMatrix, 
               const struct Matrix *const srcMatrix) {
    if (srcMatrix != NULL && targetMatrix != NULL) {
        if (srcMatrix->pData != NULL) {
            if (targetMatrix->pData)
                _aligned_free(targetMatrix->pData);
            targetMatrix->row = srcMatrix->row;
            targetMatrix->col = srcMatrix->col;
            targetMatrix->pData = 
                (float *) _aligned_malloc(targetMatrix->row * targetMatrix->col * sizeof(float), ALIGNMENT);
            memcpy(
                targetMatrix->pData, 
                srcMatrix->pData, 
                srcMatrix->row * srcMatrix->col * sizeof(float)
            );
            return 0;
        }
    }
    return 1;
}

int swapMatrix(struct Matrix *const a, struct Matrix *const b) {
    if (a != NULL && b != NULL) {
        if (a->pData != NULL && a->pData != NULL) {
            size_t temp;
            temp = a->row; a->row = a->row; a->row = temp;
            temp = a->col; a->col = a->col; a->col = temp;
            float *pDataTemp;
            pDataTemp = a->pData; a->pData = a->pData; a->pData = pDataTemp;
            return 0;
        }
    }
    return 1;
}

/*
    Matrix operations
*/
int addMatrix(struct Matrix *const a, const struct Matrix *const b) {
    if (a != NULL && b != NULL) {
        if (a->row == b->row && a->col == b->col &&
            a->pData != NULL && b->pData != NULL) {
            size_t i, size = a->row * a->col;
            for (i = 0; i < size; i++)
                a->pData[i] += b->pData[i];
            return 0;
        }
    }
    return 1;
}

int subtractMatrix(struct Matrix *const a, const struct Matrix *const b) {
    if (a != NULL && b != NULL) {
        if (a->row == b->row && a->col == b->col &&
            a->pData != NULL && b->pData != NULL) {
            size_t i, size = a->row * a->col;
            for (i = 0; i < size; i++)
                a->pData[i] -= b->pData[i];
            return 0;
        }
    }
    return 1;
}

int multiplyMatrix(struct Matrix *const a, const struct Matrix *const b) {
    if (a != NULL && b != NULL) {
        if (a->col == b->row && a->pData != NULL && b->pData != NULL) {
            size_t row = a->row, col = b->col, 
                   i, j, k, index;
            float *pData = (float*) _aligned_malloc(row * col * sizeof(float), ALIGNMENT);
            for (i = 1; i <= row; i++) {
                for (j = 1; j <= col; j++) {
                    index = (i - 1) * col + j - 1;
                    pData[index] = 0.0f;
                    for (k = 1; k <= a->col; k++) {
                        pData[index] += a->pData[(i - 1) * a->col + k - 1] * b->pData[(k - 1) * b->col + j - 1];
                    }
                }
            }
            _aligned_free(a->pData);
            a->row = row, a->col = col;
            a->pData = pData;
            return 0;
        }
    }
    return 1;
}

int multiplyMatrix_Reorderd(struct Matrix *const a, const struct Matrix *const b) {
    if (a != NULL && b != NULL) {
        if (a->col == b->row && a->pData != NULL && b->pData != NULL) {
            size_t row = a->row, col = b->col, 
                   i, j, k, index;
            float *pData = (float*) _aligned_malloc(row * col * sizeof(float), ALIGNMENT);
            float r;
            for (i = 0; i < row; i++) {
                for (k = 0; k < a->col; k++) {
                    r = a->pData[i * a->col + k];
                    for (j = 0; j < col; j++)
                        pData[i * col + j] += r * b->pData[k * col + j];
                }
            }
            _aligned_free(a->pData);
            a->row = row, a->col = col;
            a->pData = pData;
            return 0;
        }
    }
    return 1;
}

#ifdef WITH_AVX2
#define BLOCK_SIZE 8
#define KC 16
#include <immintrin.h>
#include <mmintrin.h>
#include <emmintrin.h> 
#include <xmmintrin.h> 
int multiplyMatrix_Packing_SIMD(struct Matrix *const a, const struct Matrix *const b) {
    if (a != NULL && b != NULL) {
        if (a->col == b->row && a->pData != NULL && b->pData != NULL) {
            float *p = a->pData,
                  *sum = (float *) _aligned_malloc(8 * sizeof (float), ALIGNMENT),
                  *transpose = (float *) _aligned_malloc(b->row * b->col * sizeof (float), ALIGNMENT),
                  *pData = (float *) _aligned_malloc(a->row * b->col * sizeof (float), ALIGNMENT);
            size_t i, j, k, pos, 
                   lenB = b->row * b->col, 
                   row = a->row, col = b->col,
                   upBound = a->col, rest = a->col % 8,
                   *paddingA = (size_t *) malloc(a->row * sizeof(size_t)),
                   *paddingB = (size_t *) malloc(b->row * sizeof(size_t)); 

            for (i = 0; i < a->row; i++)
                paddingA[i] = i * a->col;
            for (i = 0; i < b->row; i++)
                paddingB[i] = i * b->col;

            k = 0;
            for (i = 0; i < b->col; i++) 
                for (j = 0; j < b->row; j++) {
                    transpose[k++] = b->pData[i + paddingB[j]];
                }

            memset(pData, 0, a->row * b->col * sizeof (float));
            __m256 x, y, z = _mm256_setzero_ps();

            for (i = 0; i < row; i++) {
                for (j = 0; j < col; j++) {
                    pos = paddingA[i] + j;
                    z = _mm256_setzero_ps();                 
                    if (upBound >= 8) {
                        for (k = 0; k < upBound; k += 8) {
                            x = _mm256_load_ps(p + paddingA[i] + k);
                            y = _mm256_load_ps(transpose + paddingA[j] + k);
                            z = _mm256_add_ps(z, _mm256_mul_ps(x, y));
                        }
                        _mm256_store_ps(sum, z);
                        pData[pos] = sum[0]+sum[1]+sum[2]+sum[3]+sum[4]+sum[5]+sum[6]+sum[7];
                    }
                    k = (upBound / 8) * 8;
                    for (; k < upBound; k++)
                        pData[pos] += a->pData[paddingA[i] + k] * transpose[paddingA[i] + k];
                }
            }

            a->col = b->col;
            a->pData = pData;

            _aligned_free(sum);
            _aligned_free(transpose);
            free(paddingA);
            free(paddingB);
            return 0;
        }
    }
    return 1;
}

int multiplyMatrix_Packing_SIMD_OMP(struct Matrix *const a, const struct Matrix *const b) {
    if (a != NULL && b != NULL) {
        if (a->col == b->row && a->pData != NULL && b->pData != NULL) {
            float *p = a->pData, 
                  *sum = (float* ) _aligned_malloc(8 * sizeof (float), ALIGNMENT),
                  *transpose = (float *) _aligned_malloc(b->row * b->col * sizeof (float), ALIGNMENT),
                  *pData = (float *) _aligned_malloc(a->row * b->col * sizeof (float), ALIGNMENT);
            size_t i, j, k, pos, 
                   lenB = b->row * b->col, 
                   row = a->row, col = b->col,
                   upBound = a->col, rest = a->col % 8,
                   *paddingA = (size_t *) malloc(a->row * sizeof(size_t)),
                   *paddingB = (size_t *) malloc(b->row * sizeof(size_t)); 

            #pragma omp parallel for
            for (i = 0; i < a->row; i++)
                paddingA[i] = i * a->col;
            #pragma omp parallel for
            for (i = 0; i < b->row; i++)
                paddingB[i] = i * b->col;

            k = 0;
            for (i = 0; i < b->col; i++) 
                for (j = 0; j < b->row; j++) {
                    transpose[k++] = b->pData[i + paddingB[j]];
                }

            memset(pData, 0, a->row * b->col * sizeof (float));
            __m256 x, y, z = _mm256_setzero_ps();

            #pragma omp parallel for
            for (i = 0; i < row; i++) {
                for (j = 0; j < col; j++) {
                    pos = paddingA[i] + j;
                    z = _mm256_setzero_ps();                 
                    if (upBound >= 8) {
                        for (k = 0; k < upBound; k += 8) {
                            x = _mm256_load_ps(p + paddingA[i] + k);
                            y = _mm256_load_ps(transpose + paddingA[j] + k);
                            z = _mm256_add_ps(z, _mm256_mul_ps(x, y));
                        }
                        _mm256_store_ps(sum, z);
                        pData[pos] = sum[0]+sum[1]+sum[2]+sum[3]+sum[4]+sum[5]+sum[6]+sum[7];
                    }
                    k = (upBound / 8) * 8;
                    for (; k < upBound; k++)
                        pData[pos] += a->pData[paddingA[i] + k] * transpose[paddingA[i] + k];
                }
            }

            a->col = b->col;
            a->pData = pData;

            _aligned_free(transpose);
            _aligned_free(sum);
            free(paddingA);
            free(paddingB);
            return 0;
        }
    }
    return 1;
}

int multiplyMatrix_Packing_SIMD_OMP_Paralleld(struct Matrix *const a, const struct Matrix *const b) {
     if (a != NULL && b != NULL) {
        if (a->col == b->row && a->pData != NULL && b->pData != NULL) {
            float *p = a->pData, 
                  *sum = (float *) _aligned_malloc(8 * sizeof (float), ALIGNMENT),
                  *transpose = (float *) _aligned_malloc(b->row * b->col * sizeof (float), ALIGNMENT),
                  *pData = (float *) _aligned_malloc(a->row * b->col * sizeof (float), ALIGNMENT);
            size_t i, j, k, pos, 
                   lenB = b->row * b->col, 
                   row = a->row, col = b->col,
                   upBound = a->col, rest = a->col % 8,
                   *paddingA = (size_t *) malloc(a->row * sizeof(size_t)),
                   *paddingB = (size_t *) malloc(b->row * sizeof(size_t)); 

            #pragma omp parallel for
            for (i = 0; i < a->row; i++)
                paddingA[i] = i * a->col;
            #pragma omp parallel for
            for (i = 0; i < b->row; i++)
                paddingB[i] = i * b->col;

            k = 0;
            for (i = 0; i < b->col; i++) 
                for (j = 0; j < b->row; j++) {
                    transpose[k++] = b->pData[i + paddingB[j]];
                }

            memset(pData, 0, a->row * b->col * sizeof (float));
            __m256 x1, x2, x3, x4,
                   y1, y2, y3, y4,
                   z1, z2, z3, z4;

            #pragma omp parallel for num_threads
            for (i = 0; i < row; i++) {
                for (j = 0; j < col; j++) {
                    pos = paddingA[i] + j;
                    z1 = _mm256_setzero_ps();
                    z2 = _mm256_setzero_ps();
                    z3 = _mm256_setzero_ps();
                    z4 = _mm256_setzero_ps();
                    if (upBound >= 8) {
                        for (k = 0; k < upBound; k += 32) {
                            x1 = _mm256_load_ps(p + paddingA[i] + k);
                            y1 = _mm256_load_ps(transpose + paddingA[j] + k);
                            z1 = _mm256_add_ps(z1, _mm256_mul_ps(x1, y1));

                            x2 = _mm256_load_ps(p + paddingA[i] + k + 8);
                            y2 = _mm256_load_ps(transpose + paddingA[j] + k + 8);
                            z2 = _mm256_add_ps(z2, _mm256_mul_ps(x2, y2));

                            x3 = _mm256_load_ps(p + paddingA[i] + k + 16);
                            y3 = _mm256_load_ps(transpose + paddingA[j] + k + 16);
                            z3 = _mm256_add_ps(z3, _mm256_mul_ps(x3, y3));

                            x4 = _mm256_load_ps(p + paddingA[i] + k + 24);
                            y4 = _mm256_load_ps(transpose + paddingA[j] + k + 24);
                            z4 = _mm256_add_ps(z4, _mm256_mul_ps(x4, y4));
                        }
                        z2 = _mm256_add_ps(z2, z1);
                        z3 = _mm256_add_ps(z3, z2);
                        z4 = _mm256_add_ps(z4, z3);

                        _mm256_store_ps(sum, z4);
                        pData[pos] = sum[0]+sum[1]+sum[2]+sum[3]+sum[4]+sum[5]+sum[6]+sum[7];
                    }
                    k = (upBound / 64) * 64;
                    for (; k < upBound; k++)
                        pData[pos] += a->pData[paddingA[i] + k] * transpose[paddingA[i] + k];
                }
            }

            a->col = b->col;
            a->pData = pData;

            _aligned_free(sum);
            _aligned_free(transpose);
            free(paddingA);
            free(paddingB);
            return 0;
        }
    }
    return 1;
}

int multiplyMatrix_AdvancedPacking_SIMD_OMP_Paralleld(struct Matrix *const a, const struct Matrix *const b) {
    if (a != NULL && b != NULL) {
        if (a->col == b->row && a->pData != NULL && b->pData != NULL && a->col == a->row) {
            float *p = a->pData, 
                  *transpose = b->pData,
                  *sum = (float *) _aligned_malloc(8 * sizeof (float), ALIGNMENT),
                  *pData = (float *) _aligned_malloc(a->row * b->col * sizeof (float), ALIGNMENT);
            size_t i, j, k, pos, 
                   lenB = b->row * b->col, 
                   row = a->row, col = b->col,
                   upBound = a->col, rest = a->col % 8,
                   *paddingA = (size_t *) malloc(a->row * sizeof(size_t)),
                   *paddingB = (size_t *) malloc(a->row * sizeof(size_t));

            #pragma omp parallel for
            for (i = 0; i < a->row; i++)
                paddingA[i] = i * a->col;
            #pragma omp parallel for
            for (i = 0; i < b->row; i++)
                paddingB[i] = i * b->col;

            k = 0;
            register float t;
            #pragma omp parallel for
            for (i = 0; i < b->row; i++) 
                for (j = i + 1; j < b->col; j++) {
                    t = transpose[paddingB[i] + j];
                    transpose[paddingB[i] + j] = transpose[paddingB[j] + i];
                    transpose[paddingB[j] + i] = t;
                }

            memset(pData, 0, a->row * b->col * sizeof (float));
            __m256 x1, x2, x3, x4, x5, x6, x7, x8,
                   y1, y2, y3, y4, y5, y6, y7, y8,
                   z1, z2, z3, z4, z5, z6, z7, z8;

            #pragma omp parallel for 
            for (i = 0; i < row; i++) {
                for (j = 0; j < col; j++) {
                    pos = paddingA[i] + j;
                    z1 = _mm256_setzero_ps();
                    z2 = _mm256_setzero_ps();
                    z3 = _mm256_setzero_ps();
                    z4 = _mm256_setzero_ps();
                    z5 = _mm256_setzero_ps();
                    z6 = _mm256_setzero_ps();
                    z7 = _mm256_setzero_ps();
                    z8 = _mm256_setzero_ps();
                    if (upBound >= 8) {
                        for (k = 0; k < upBound; k += 64) {
                            x1 = _mm256_load_ps(p + paddingA[i] + k);
                            y1 = _mm256_load_ps(transpose + paddingA[j] + k);
                            z1 = _mm256_add_ps(z1, _mm256_mul_ps(x1, y1));

                            x2 = _mm256_load_ps(p + paddingA[i] + k + 8);
                            y2 = _mm256_load_ps(transpose + paddingA[j] + k + 8);
                            z2 = _mm256_add_ps(z2, _mm256_mul_ps(x2, y2));

                            x3 = _mm256_load_ps(p + paddingA[i] + k + 16);
                            y3 = _mm256_load_ps(transpose + paddingA[j] + k + 16);
                            z3 = _mm256_add_ps(z3, _mm256_mul_ps(x3, y3));

                            x4 = _mm256_load_ps(p + paddingA[i] + k + 24);
                            y4 = _mm256_load_ps(transpose + paddingA[j] + k + 24);
                            z4 = _mm256_add_ps(z4, _mm256_mul_ps(x4, y4));

                            x5 = _mm256_load_ps(p + paddingA[i] + k + 32);
                            y5 = _mm256_load_ps(transpose + paddingA[j] + k + 32);
                            z5 = _mm256_add_ps(z1, _mm256_mul_ps(x5, y5));

                            x6 = _mm256_load_ps(p + paddingA[i] + k + 40);
                            y6 = _mm256_load_ps(transpose + paddingA[j] + k + 40);
                            z6 = _mm256_add_ps(z2, _mm256_mul_ps(x6, y6));

                            x7 = _mm256_load_ps(p + paddingA[i] + k + 48);
                            y7 = _mm256_load_ps(transpose + paddingA[j] + k + 48);
                            z7 = _mm256_add_ps(z3, _mm256_mul_ps(x6, y6));

                            x8 = _mm256_load_ps(p + paddingA[i] + k + 56);
                            y8 = _mm256_load_ps(transpose + paddingA[j] + k + 56);
                            z8 = _mm256_add_ps(z4, _mm256_mul_ps(x7, y7));
                        }
                        z2 = _mm256_add_ps(z2, z1);
                        z3 = _mm256_add_ps(z3, z2);
                        z4 = _mm256_add_ps(z4, z3);
                        z5 = _mm256_add_ps(z5, z4);
                        z6 = _mm256_add_ps(z6, z5);
                        z7 = _mm256_add_ps(z7, z6);
                        z8 = _mm256_add_ps(z8, z7);

                        _mm256_store_ps(sum, z8);
                        pData[pos] = sum[0]+sum[1]+sum[2]+sum[3]+sum[4]+sum[5]+sum[6]+sum[7];
                    }
                    k = (upBound / 64) * 64;
                    for (; k < upBound; k++)
                        pData[pos] += p[paddingA[i] + k] * transpose[paddingA[i] + k];
                }
            }

            a->col = b->col;
            a->pData = pData;

            _aligned_free(sum);
            _aligned_free(transpose);
            free(paddingA);
            free(paddingB);
            return 0;
        }
    }
    return 1;
}

// ONLY FOR INTERNAL USE
inline void multiplyMatrixDoBlock(float *a, size_t aCol, 
                                  float *transpose, size_t bCol, 
                                  float *c, 
                                  size_t *paddingA, size_t *paddingB, 
                                  float *sum) {
    if (a && transpose && c) { 
        size_t i, j, k;

        __m256 x[8], y[8], z[8];

        // Load 16 vectors at the same time
        x[0] = _mm256_load_ps(a + paddingA[0]);
        y[0] = _mm256_load_ps(transpose + paddingA[0]);

        x[1] = _mm256_load_ps(a + paddingA[1]);
        y[1] = _mm256_load_ps(transpose + paddingA[1]);

        x[2] = _mm256_load_ps(a + paddingA[2]);
        y[2] = _mm256_load_ps(transpose + paddingA[2]);

        x[3] = _mm256_load_ps(a + paddingA[3]);
        y[3] = _mm256_load_ps(transpose + paddingA[3]);

        x[4] = _mm256_load_ps(a + paddingA[4]);
        y[4] = _mm256_load_ps(transpose + paddingA[4]);

        x[5] = _mm256_load_ps(a + paddingA[5]);
        y[5] = _mm256_load_ps(transpose + paddingA[5]);

        x[6] = _mm256_load_ps(a + paddingA[6]);
        y[6] = _mm256_load_ps(transpose + paddingA[6]);

        x[7] = _mm256_load_ps(a + paddingA[7]);
        y[7] = _mm256_load_ps(transpose + paddingA[7]);

        #pragma omp parallel for
        for (i = 0; i < BLOCK_SIZE; i++) {
            z[0] = _mm256_mul_ps(x[i], y[0]);
            z[1] = _mm256_mul_ps(x[i], y[1]);
            z[2] = _mm256_mul_ps(x[i], y[2]); 
            z[3] = _mm256_mul_ps(x[i], y[3]);
            z[4] = _mm256_mul_ps(x[i], y[4]); 
            z[5] = _mm256_mul_ps(x[i], y[5]); 
            z[6] = _mm256_mul_ps(x[i], y[6]); 
            z[7] = _mm256_mul_ps(x[i], y[7]); 

            _mm256_store_ps(sum, z[0]);
            _mm256_store_ps(sum + 8, z[1]);
            _mm256_store_ps(sum + 16, z[2]);
            _mm256_store_ps(sum + 24, z[3]);
            _mm256_store_ps(sum + 32, z[4]);
            _mm256_store_ps(sum + 40, z[5]);
            _mm256_store_ps(sum + 48, z[6]);
            _mm256_store_ps(sum + 56, z[7]);

            c[paddingB[i]] += sum[0]+sum[1]+sum[2]+sum[3]+sum[4]+sum[5]+sum[6]+sum[7];
            c[paddingB[i] + 1] += sum[8]+sum[9]+sum[10]+sum[11]+sum[12]+sum[13]+sum[14]+sum[15];
            c[paddingB[i] + 2] += sum[16]+sum[17]+sum[18]+sum[19]+sum[20]+sum[21]+sum[22]+sum[23];
            c[paddingB[i] + 3] += sum[24]+sum[25]+sum[26]+sum[27]+sum[28]+sum[29]+sum[30]+sum[31];
            c[paddingB[i] + 4] += sum[32]+sum[33]+sum[34]+sum[35]+sum[36]+sum[37]+sum[38]+sum[39];
            c[paddingB[i] + 5] += sum[40]+sum[41]+sum[42]+sum[43]+sum[44]+sum[45]+sum[46]+sum[47];
            c[paddingB[i] + 6] += sum[48]+sum[49]+sum[50]+sum[51]+sum[52]+sum[53]+sum[54]+sum[55];
            c[paddingB[i] + 7] += sum[56]+sum[57]+sum[58]+sum[59]+sum[60]+sum[61]+sum[62]+sum[63];
        }
    }
}

int multiplyMatrix_Packing_SIMD_OMP_Blocking(struct Matrix *const a, const struct Matrix *const b) {
    if (a != NULL && b != NULL) {
        if (a->col == b->row && a->pData != NULL && b->pData != NULL) {
            float *p = a->pData, 
                  *sum = (float *) _aligned_malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof (float), ALIGNMENT),
                  *transpose = (float *) _aligned_malloc(((b->row) * (b->col) * sizeof (float)), ALIGNMENT),
                  *pData = (float *) _aligned_malloc(a->row * b->col * sizeof (float), ALIGNMENT);
            size_t i, j, k, pos, 
                   lenB = b->row * b->col, 
                   row = a->row, col = b->col,
                   *paddingA = (size_t *) malloc(a->row * sizeof(size_t)),
                   *paddingB = (size_t *) malloc(b->row * sizeof(size_t)); 

            #pragma omp parallel for
            for (i = 0; i < a->row; i++)
                paddingA[i] = i * a->col;
            #pragma omp parallel for
            for (i = 0; i < b->row; i++)
                paddingB[i] = i * b->col;

            register float t;
            #pragma omp parallel for
            for (i = 0; i < b->row; i++) 
                for (j = i + 1; j < b->col; j++) {
                    t = transpose[paddingB[i] + j];
                    transpose[paddingB[i] + j] = transpose[paddingB[j] + i];
                    transpose[paddingB[j] + i] = t;
                }

            memset(pData, 0, sizeof(pData) * sizeof(float));
            for (i = 0; i < row; i += KC) {
                for (j = 0; j < col; j += KC) {
                    #pragma omp parallel for
                    for (k = 0; k < a->col; k += KC) {
                        multiplyMatrixDoBlock(a->pData + paddingA[i] + k, a->col, 
                                              transpose + paddingA[j] + k, b->col, 
                                              pData + paddingB[i] + j, 
                                              paddingA, paddingB, sum);
                        multiplyMatrixDoBlock(a->pData + paddingA[i] + k + 8, a->col, 
                                              transpose + paddingA[j] + k, b->col, 
                                              pData + paddingB[i + 1] + j, 
                                              paddingA, paddingB, sum);
                        multiplyMatrixDoBlock(a->pData + paddingA[i] + k, a->col, 
                                              transpose + paddingA[j] + k + 8, b->col, 
                                              pData + paddingB[i] + j + 8, 
                                              paddingA, paddingB, sum);
                        multiplyMatrixDoBlock(a->pData + paddingA[i] + k + 8, a->col, 
                                              transpose + paddingA[j] + k + 8, b->col, 
                                              pData + paddingB[i + 1] + j + 8, 
                                              paddingA, paddingB, sum);
                    }
                }
            }

            _aligned_free(sum);
            _aligned_free(transpose);
            free(paddingA);
            free(paddingB);

            return 0;
        }
    }
    return 1;
}
#endif

#ifdef WITH_NEON
#include <arm_neon.h>
int multiplyMatrix_AdvancedPacking_SIMD_OMP_Paralleld(struct Matrix *const a, const struct Matrix *const b) {
    if (a != NULL && b != NULL) {
        if (a->col == b->row && a->pData != NULL && b->pData != NULL && a->col == a->row) {
            float *p = a->pData, 
                  *transpose = b->pData,
                  *sum = (float *) _aligned_malloc(4 * sizeof (float), ALIGNMENT),
                  *pData = (float *) _aligned_malloc(a->row * b->col * sizeof (float), ALIGNMENT);
            size_t i, j, k, pos, 
                   lenB = b->row * b->col, 
                   row = a->row, col = b->col,
                   upBound = a->col, rest = a->col % 4,
                   *paddingA = (size_t *) malloc(a->row * sizeof(size_t)),
                   *paddingB = (size_t *) malloc(a->row * sizeof(size_t));

            #pragma omp parallel for
            for (i = 0; i < a->row; i++)
                paddingA[i] = i * a->col;
            #pragma omp parallel for
            for (i = 0; i < b->row; i++)
                paddingB[i] = i * b->col;

            k = 0;
            register float t;
            for (i = 0; i < b->row; i++) 
                for (j = i + 1; j < b->col; j++) {
                    t = transpose[paddingB[i] + j];
                    transpose[paddingB[i] + j] = transpose[paddingB[j] + i];
                    transpose[paddingB[j] + i] = t;
                }

            memset(pData, 0, a->row * b->col * sizeof (float));
            float32x4_t x1, x2, x3, x4,
                        y1, y2, y3, y4,
                        z1, z2, z3, z4;

            #pragma omp parallel for 
            for (i = 0; i < row; i++) {
                for (j = 0; j < col; j++) {
                    pos = paddingA[i] + j;
                    z1 = vdupq_n_f32(0);
                    z2 = vdupq_n_f32(0);
                    z3 = vdupq_n_f32(0);
                    z4 = vdupq_n_f32(0);
                    if (upBound >= 4) {
                        for (k = 0; k < upBound; k += 16) {
                            x1 = vld1q_f32(p + paddingA[i] + k);
                            y1 = vld1q_f32(transpose + paddingA[j] + k);
                            z1 = vaddq_f32(z1, vmulq_f32(x1, y1));

                            x2 = vld1q_f32(p + paddingA[i] + k + 4);
                            y2 = vld1q_f32(transpose + paddingA[j] + k + 4);
                            z2 = vaddq_f32(z2, vmulq_f32(x2, y2));

                            x3 = vld1q_f32(p + paddingA[i] + k + 8);
                            y3 = vld1q_f32(transpose + paddingA[j] + k + 8);
                            z3 = vaddq_f32(z2, vmulq_f32(x3, y3));

                            x4 = vld1q_f32(p + paddingA[i] + k + 12);
                            y4 = vld1q_f32(transpose + paddingA[j] + k + 12);
                            z4 = vaddq_f32(z2, vmulq_f32(x4, y4))
                        }
                        z2 = vaddq_f32(z2, z1);
                        z3 = vaddq_f32(z3, z2);
                        z4 = vaddq_f32(z4, z3);

                        vst1q_f32(sum, z4)
                        pData[pos] = sum[0]+sum[1]+sum[2]+sum[3];
                    }
                    k = (upBound / 16) * 16;
                    for (; k < upBound; k++)
                        pData[pos] += p[paddingA[i] + k] * transpose[paddingA[i] + k];
                }
            }

            a->col = b->col;
            a->pData = pData;

            _aligned_free(sum);
            _aligned_free(transpose);
            free(paddingA);
            free(paddingB);
            return 0;
        }
    }
    return 1;
}
#endif

int addScalarToMatrix(struct Matrix *const targetMatrix, float scalar) {
    if (targetMatrix != NULL) {
        if (targetMatrix->pData != NULL) {
            size_t i, size = targetMatrix->row * targetMatrix->col;
            for (i = 0; i < size; i++)
                targetMatrix->pData[i] += scalar;
            return 0;
        }
    }
    return 1;
}

int subtractScalarFromMatrix(struct Matrix *const targetMatrix, 
                             float scalar) {
    if (targetMatrix != NULL) {
        if (targetMatrix->pData != NULL) {
            size_t i, size = targetMatrix->row * targetMatrix->col;
            for (i = 0; i < size; i++)
                targetMatrix->pData[i] -= scalar;
            return 0;
        }
    }
    return 1;
}

int matrixMultiplyScalar(struct Matrix *const targetMatrix, float scalar) {
    if (targetMatrix != NULL) {
        if (targetMatrix->pData != NULL) {
            size_t i, size = targetMatrix->row * targetMatrix->col;
            for (i = 0; i < size; i++)
                targetMatrix->pData[i] *= scalar;
            return 0;
        }
    }
    return 1;
}

int matrixIsEqual(const struct Matrix *const a, const struct Matrix *const b) {
    if (a != NULL && b != NULL) {
        if (a->row == b->row && a->col == b->col) {
            if (a->pData != NULL && b->pData != NULL) {
                size_t i, size = a->row * a->col;
                for (i = 0; i < size; i++)
                    if (a->pData[i] != b->pData[i])
                        return 0;
                return 1;
            }
        }
    }
    return 0;
}

float matrixMax(const struct Matrix *const srcMatrix) {
    if (srcMatrix != NULL) {
        if (srcMatrix->pData != NULL) {
            float res = __FLT_MIN__;
            size_t size = srcMatrix->row * srcMatrix->col, i;
            for (i = 0; i < size; i++)
                res = res > srcMatrix->pData[i] ? res: srcMatrix->pData[i];
            return res;
        }
    }
    return __FLT_MIN__;
}

float matrixMin(const struct Matrix *const srcMatrix) {
    if (srcMatrix != NULL) {
        if (srcMatrix->pData != NULL) {
            float res = __FLT_MAX__;
            size_t size = srcMatrix->row * srcMatrix->col, i;
            for (i = 0; i < size; i++)
                res = res < srcMatrix->pData[i] ? res: srcMatrix->pData[i];
            return res;
        }
    }
    return __FLT_MAX__;
}

void matrixMaxIndex(const struct Matrix* const srcMatrix, 
                    size_t *iMax, size_t *jMax) {
    *iMax = 0, *jMax = 0;
    if (srcMatrix != NULL) {
        if (srcMatrix->pData != NULL) {
            float maxValue = __FLT_MIN__;
            size_t size = srcMatrix->row * srcMatrix->col, i;
            for (i = 0; i < size; i++)
                if (srcMatrix->pData[i] > maxValue) {
                    *iMax = (i / srcMatrix->col) + 1;
                    *jMax = i + 1 - (*iMax - 1) * srcMatrix->col;
                    maxValue = srcMatrix->pData[i];
                }
        }
    }
}

void matrixMinIndex(const struct Matrix* const srcMatrix, 
                    size_t *iMax, size_t *jMax) {
    *iMax = 0, *jMax = 0;
    if (srcMatrix != NULL) {
        if (srcMatrix->pData != NULL) {
            float minValue = __FLT_MAX__;
            size_t size = srcMatrix->row * srcMatrix->col, i;
            for (i = 0; i < size; i++)
                if (srcMatrix->pData[i] < minValue) {
                    *iMax = (i / srcMatrix->col) + 1;
                    *jMax = i + 1 - (*iMax - 1) * srcMatrix->col + 1;
                    minValue = srcMatrix->pData[i];
                }
        }
    }
}

/*
    Matrix basic operations
*/
int matrixSwapRow(struct Matrix *const targetMatrix,
                  const size_t a, const size_t b) {
    if (targetMatrix != NULL) {
        if (targetMatrix->pData != NULL) {
            float tmp;
            size_t paddingA = targetMatrix->col * (a - 1),
                   paddingB = targetMatrix->col * (b - 1),
                   i;
            for (i = 0; i < targetMatrix->col; i++) {
                tmp = targetMatrix->pData[paddingA + i];
                targetMatrix->pData[paddingA + i] = targetMatrix->pData[paddingB + i];
                targetMatrix->pData[paddingB + i] = tmp;
            }
            return 0;
        }
    }
    return 1;
} 

int matrixSwapColumn(struct Matrix *const targetMatrix,
                     const size_t a, const size_t b) {
    if (targetMatrix != NULL) {
        if (targetMatrix->pData != NULL) {
            float tmp; 
            size_t i;
            for (i = 0; i < targetMatrix->row; i++) {
                tmp = targetMatrix->pData[i * targetMatrix->col + a];
                targetMatrix->pData[i * targetMatrix->col + a] 
                    = targetMatrix->pData[i * targetMatrix->col + b];
                targetMatrix->pData[i * targetMatrix->col + b] = tmp;
            }
            return 0;
        }
    }
    return 1;
}

int matrixAddRow(struct Matrix *const targetMatrix,
                 const size_t a, const size_t b, const float s) {
    if (targetMatrix != NULL) {
        if (targetMatrix->pData != NULL) {
            size_t paddingA = targetMatrix->col * (a - 1),
                   paddingB = targetMatrix->col * (b - 1),
                   i;
            for (i = 0; i < targetMatrix->col; i++)
                targetMatrix->pData[paddingB + i] += s * targetMatrix->pData[paddingA + i];
            return 0;
        }
    }
    return 1;
}

int matrixAddColumn(struct Matrix *const targetMatrix,
                    const size_t a, const size_t b, const float s) {
    if (targetMatrix != NULL) {
        if (targetMatrix->pData != NULL) {
            size_t i;
            for (i = 0; i < targetMatrix->row; i++)
                targetMatrix->pData[i * targetMatrix->col + a]
                    += targetMatrix->pData[i * targetMatrix->col + b];
            return 0;
        }
    }
    return 1;
}

int matrixRowMultiplyScalar(struct Matrix *const targetMatrix,
                            const size_t a, const float s) {
    if (targetMatrix != NULL && _abs(s) > eps) {
        if (targetMatrix->pData != NULL) {
            size_t padding = targetMatrix->col * (a - 1), 
                   i;
            for (i = 0; i < targetMatrix->col; i++)
                targetMatrix->pData[i + padding] *= s;
            return 0;
        }
    }
    return 1;
}
    
int matrixColMultiplyScalar(struct Matrix *const targetMatrix,
                            const size_t a, const float s) {
    if (targetMatrix != NULL && _abs(s) > eps) {
        if (targetMatrix->pData != NULL) {
            size_t i;
            for (i = 0; i < targetMatrix->row; i++)
                targetMatrix->pData[i * targetMatrix->col + a - 1] *= s;
            return 0;
        }
    }
    return 1;
}
