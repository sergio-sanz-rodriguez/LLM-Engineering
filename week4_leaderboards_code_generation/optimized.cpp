
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace std;

double calculate(long long iterations, int param1, int param2) {
    double result = 1.0;
    for (long long i = 1; i <= iterations; ++i) {
        double j = (double)i * param1 - param2;
        result -= (1.0 / j);
        j = (double)i * param1 + param2;
        result += (1.0 / j);
    }
    return result;
}

int main() {
    auto start_time = chrono::high_resolution_clock::now();
    double result = calculate(100000000, 4, 1) * 4;
    auto end_time = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
    double seconds = duration.count() / 1000000.0;

    cout << fixed << setprecision(12) << "Result: " << result << endl;
    cout << fixed << setprecision(6) << "Execution Time: " << seconds << " seconds" << endl;

    return 0;
}
