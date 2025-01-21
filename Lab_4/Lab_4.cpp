#include "vector_mod.h"
#include "test.h"
#include "performance.h"
#include <iostream>
#include <iomanip>
#include "num_threads.h"
#include <fstream>


int main(int argc, char** argv)
{
	std::ofstream outfile("outfile.csv");
	if (!outfile.is_open()) {
		std::cout << "Error! file is not open.\n";
		return -1;
	}

	std::cout << "==Correctness tests. ";
	//set_num_threads(3);
	for (std::size_t iTest = 1; iTest < test_data_count; ++iTest)
	{
		if (test_data[iTest].result != vector_mod(test_data[iTest].dividend, test_data[iTest].dividend_size, test_data[iTest].divisor))
		{
			std::cout << "FAILURE==\n";
			return -1;
		}
	}
	std::cout << "ok.==\n";

	std::cout << "==Performance tests. ";
	auto measurements = run_experiments();
	std::cout << "Done==\n";
	std::cout << std::setfill(' ') << std::setw(2) << "T:" << " |" << std::setw(3 + 2 * sizeof(IntegerWord)) << "Value:" << " | " <<
		std::setw(14) << "Duration, ms:" << " | Acceleration:\n";

	outfile << "T,Duration,Acceleration\n";

	for (std::size_t T = 1; T <= measurements.size(); ++T)
	{
		auto duration = measurements[T - 1].time.count();
		auto acceleration = (static_cast<double>(measurements[0].time.count()) / measurements[T - 1].time.count());
		std::cout << std::setw(2) << T << " | 0x" << std::setw(2 * sizeof(IntegerWord)) << std::setfill('0') << std::hex << measurements[T - 1].result;
		std::cout << " | " << std::setfill(' ') << std::setw(14) << std::dec << measurements[T - 1].time.count();
		std::cout << " | " << (static_cast<double>(measurements[0].time.count()) / measurements[T - 1].time.count()) << "\n";
		outfile << T << "," << duration << "," << acceleration << "\n";
	}

	outfile.close();
	return 0;
}