#include <iostream>
#include <numbers>
#include <complex>
#include <bit>
#include <vector>
#include <iomanip>
#include <omp.h>
#include <fstream>


void shuffle(const std::complex<double>* inp, std::complex<double>* out, std::size_t n)
{
    for (std::size_t i = 0; i < n / 2; i++)
    {
        out[i] = inp[i * 2];
        out[i + n / 2] = inp[i * 2 + 1];
    }
}

void fft(const std::complex<double>* inp, std::complex<double>* out, std::size_t n, int inverse)
{
    if (n == 1) {
        *out = *inp;
        return;
    }

    std::complex<double>* tmp_out = new std::complex<double>[n];
    shuffle(inp, tmp_out, n);

    fft(tmp_out, out, n / 2, inverse);
    fft(tmp_out + n / 2, out + n / 2, n / 2, inverse);
    delete[] tmp_out;

    for (std::size_t i = 0; i < n / 2; i++) {
        auto w = std::polar(1.0, -2 * std::numbers::pi_v<double> *i * inverse / n);
        auto r1 = out[i];
        auto r2 = out[i + n / 2];
        out[i] = r1 + w * r2;
        out[i + n / 2] = r1 - w * r2;
    }
}

int main()
{
    std::ofstream outfile("outfile.csv");
    if (!outfile.is_open()) {
        std::cout << "Error! file is not open.\n";
        return -1;
    }

    constexpr std::size_t n = 1 << 4;
    std::vector<std::complex<double>> original(n), spectre(n), restored(n);

    for (std::size_t i = 0; i < n / 2; i++) {
        original[i] = i;
        original[n - 1 - i] = i;
    }


    /*for (std::size_t i = 0; i < n; i++)
    {
        original[i] = 5;
    }*/

    outfile << "mode,T,a,b\n";
    std::cout << "original:\n";
    for (std::size_t i = 0; i < n; i++) {
        std::cout << "[" << i << "] " << original[i] << "\n";
        outfile << "original," << i << "," << original[i] << "\n";
    }
    outfile << "\n";

    outfile << "mode,T,a,b\n";
    fft(original.data(), spectre.data(), n, 1);
    std::cout << "\n\nDFT(original):\n";
    for (std::size_t i = 0; i < n; i++) {
        std::cout << "[" << i << "] " << std::fixed << spectre[i] << "\n";
        outfile << "DFT," << i << "," << spectre[i] << "\n";
    }
    outfile << "\n";

    outfile << "mode,T,a,b\n";
    fft(spectre.data(), restored.data(), n, -1);
    std::cout << "\n\nIDFT(DFT):\n";
    for (std::size_t i = 0; i < n; i++) {
        std::cout << "[" << i << "] " << std::fixed << restored[i] / static_cast<std::complex<double>>(n) << "\n";
        outfile << "IDFT," << i << "," << restored[i] / static_cast<std::complex<double>>(n) << "\n";
    }

    outfile.close();
    return 0;
}