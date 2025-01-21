#include <assert.h>
#include <immintrin.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>
#include <fstream>

using namespace std;
using namespace std::chrono;

const size_t matrix_size = 16 * 4;

void mul_matrix(double* A, size_t cA, size_t rA,
    const double* B, size_t cB, size_t rB,
    const double* C, size_t cC, size_t rC)
{   
    // Можно переписать через if
    assert(cB == rC && cA == cC && rA == rB);
    assert((cA & 0x3f) == 0); // 0x3f = 63, проверяем что cA делится на 64 без остатка (кратно 64)

    for (size_t i = 0; i < cA; i++)
    {
        for (size_t j = 0; j < rA; j++)
        {
            A[i * rA + j] = 0;
            for (size_t k = 0; k < cB; k++)
            {
                A[i * rA + j] += B[k * rB + j] * C[i * rC + k];
            }
        }
    }
}



void mul_matrix_avx512(double* A, const double* B, const double* C,
    size_t cA, size_t rA,
    size_t cB, size_t rB,
    size_t cC, size_t rC)
{
    assert(cB == rC && cA == cC && rA == rB);
    assert((cA & 0x3f) == 0);

    for (size_t i = 0; i < rB / 8; i++)
    {
        for (size_t j = 0; j < cC; j++)
        {
            __m512d sum = _mm512_setzero_pd();
            for (size_t k = 0; k < rC; k++)
            {
                __m512d bCol = _mm512_loadu_pd(B + rB * k + i * 8);
                __m512d broadcasted = _mm512_set1_pd(C[j * rC + k]);
                sum = _mm512_fmadd_pd(bCol, broadcasted, sum);
            }

            _mm512_storeu_pd(A + j * rA + i * 8, sum);
        }
    }
}

void mul_matrix_avx2(double* A, const double* B, const double* C,
    size_t cA, size_t rA,
    size_t cB, size_t rB,
    size_t cC, size_t rC)
{
    assert(cB == rC && cA == cC && rA == rB);
    assert((cA & 0x3f) == 0); 

    for (size_t i = 0; i < rB / 4; i++)
    {
        for (size_t j = 0; j < cC; j++)
        {
            __m256d sum = _mm256_setzero_pd();
            for (size_t k = 0; k < rC; k++)
            {
                __m256d bCol = _mm256_loadu_pd(B + rB * k + i * 4);
                __m256d broadcasted = _mm256_set1_pd(C[j * rC + k]);
                sum = _mm256_fmadd_pd(bCol, broadcasted, sum);
            }

            _mm256_storeu_pd(A + j * rA + i * 4, sum);
        }
    }
}


int main(int argc, char** argv)
{
    ofstream outfile("outfile.csv");
    if (!outfile.is_open()) {
        std::cout << "Error! file is not open.\n";
        return -1;
    }


    vector<double> A(matrix_size * matrix_size),
        B(matrix_size * matrix_size, 0), 
        C(matrix_size * matrix_size, 0),
        D(matrix_size * matrix_size);

    auto t1 = steady_clock::now();
    mul_matrix(A.data(), matrix_size, matrix_size,
        B.data(), matrix_size, matrix_size,
        C.data(), matrix_size, matrix_size);
    auto t2 = steady_clock::now();
    double duration1 = duration_cast<microseconds>(t2 - t1).count();
    cout << "Matrix multiplication (in loop)\n";
    cout << "Duration = " << duration1 << " micro. Acceleration = 1\n\n";
    outfile << "Mode,Duration,Acceleration\n" << "Default," << duration1 << ",1\n";


    t1 = steady_clock::now();
    mul_matrix_avx2(D.data(), B.data(), C.data(), matrix_size, matrix_size, matrix_size, matrix_size, matrix_size, matrix_size);
    t2 = steady_clock::now();
    double duration2 = duration_cast<microseconds>(t2 - t1).count();
    cout << "Matrix multiplication (AVX2)\n";
    cout << "Duration = " << duration_cast<microseconds>(t2 - t1).count() << " micro. Acceleration = " << duration1 / duration2 << "\n\n";
    outfile << "AVX2," << duration2 << "," << duration1 / duration2;


    if (!memcmp(static_cast<void*>(A.data()),
        static_cast<void*>(D.data()),
        matrix_size * matrix_size * sizeof(double)))
    {
        cout << "The matrices are equal.";
    }

    /*for (size_t c = 0; c < matrix_size; c++)
    {
        for (size_t r = 0; r < matrix_size; r++)
        {
            if ((r != c) != A[c * matrix_size + r])
            {
                cout << "Not equal.\n\n";
                return 0;
            }
        }
    }*/

    return 0;
}