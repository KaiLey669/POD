#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>
#include <immintrin.h>
#include <fstream>

using namespace std;
using namespace std::chrono;

#define columns 2048 * 2
#define rows 2048 * 2

void add_matrix(double* A, const double* B, const double* C, size_t colsc, size_t rowsc) {
	for (size_t i = 0; i < colsc * rowsc; i++) {
		A[i] = B[i] + C[i];
	}
}

void add_matrix_avx(double* A, const double* B, const double* C, size_t colsc, size_t rowsc) {
	for (size_t i = 0; i < rowsc * colsc / 4; i++) {
		__m256d b = _mm256_loadu_pd(&(B[i * 4]));
		__m256d c = _mm256_loadu_pd(&(C[i * 4]));
		__m256d a = _mm256_add_pd(b, c);

		_mm256_storeu_pd(&(A[i * 4]), a);
	}
}

void add_matrix_avx512(const double* A, const double* B, double* C, size_t colsc, size_t rowsc)
{
	for (size_t i = 0; i < rowsc / 8; i++)
	{
		for (size_t j = 0; j < colsc; j++)
		{
			__m512d a = _mm512_loadu_pd(&(A[i + j * rowsc]));
			__m512d b = _mm512_loadu_pd(&(B[i + j * rowsc]));
			__m512d c = _mm512_add_pd(a, b);

			_mm512_storeu_pd(&(C[i + j * rowsc]), c);
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


	vector<double> B(columns * rows, 1), C(columns * rows, -1), A(columns * rows);

	auto t1 = steady_clock::now();
	add_matrix(A.data(), B.data(), C.data(), columns, rows);
	auto t2 = steady_clock::now();

	double duration1 = duration_cast<microseconds>(t2 - t1).count();
	cout << "Matrix addition (in loop)\n";
	cout << "Duration = " << duration1 << " micro. Acceleration = 1\n\n";
	outfile << "Mode,Duration,Acceleration\n" << "Default," << duration1 << ",1\n";


	t1 = steady_clock::now();
	add_matrix_avx(A.data(), B.data(), C.data(), columns, rows);
	t2 = steady_clock::now();

	double duration2 = duration_cast<microseconds>(t2 - t1).count();
	cout << "Matrix addition (AVX2)\n";
	cout << "Duration = " << duration_cast<microseconds>(t2 - t1).count() << " micro. Acceleration = " << duration1 / duration2;
	outfile << "AVX2," << duration2 << "," << duration1 / duration2;


	return 0;
}