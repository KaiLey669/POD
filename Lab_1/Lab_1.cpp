#include <iostream>
#include <chrono>
#include <omp.h>
#include <thread>
#include <fstream>

using namespace std;

const double N = 100000000;


double func(double x) {
    return x * x;
}

double integrate(double a, double b)
{
    double sum = 0;
    double dx = (b - a)/N;

    for (int i = 0; i < N; i++) {
        sum += func(a + i * dx);
    }

    return dx * sum;
}

double integrate_omp(double a, double b)
{
    double sum = 0;
    double dx = (b - a)/N;

    #pragma omp parallel
    {
        unsigned t = omp_get_thread_num();
        unsigned T = omp_get_num_threads();
        double local_sum = 0;

        for (size_t i = t; i < N; i += T){
            local_sum += func(a + i * dx);
        }

        #pragma omp critical
        {
            sum += local_sum;
        }
    }

    return dx * sum;
}


int main()
{
    ofstream outfile("outfile.csv");
    if (!outfile.is_open()) {
        std::cout << "Error! file is not open.\n";
        return -1;
    }

    double a = -1;
    double b = 1;

    double t1 = omp_get_wtime();
    double result = integrate(a, b);
    double t2 = omp_get_wtime() - t1;

    cout << "threads = 1, S = " << result << ", duration = " << t2 << "s, acceleration = 1\n";
    outfile << "t,duration,acceleration\n1," << t2 << ",1\n";

    double duration1 = t2;


    unsigned int threads_num = thread::hardware_concurrency();
    cout << "\nThreads = " << threads_num << "\n";

    for (size_t i = 2; i <= threads_num; i++) {
        omp_set_num_threads(i);
        t1 = omp_get_wtime();
        result = integrate_omp(a, b);
        t2 = omp_get_wtime() - t1;

        cout << "threads = " << i << ", S = " << result << ", duration = " << t2 << "s, acceleration = " << duration1 / t2 << "\n";
        outfile << i << "," << t2 << "," << (duration1 / t2);
        if (i < threads_num) {
            outfile << "\n";
        }
    }

    outfile.close();
    
    return 0;
}
