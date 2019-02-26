// This file is a part of BAT.jl, licensed under the MIT License (MIT).

// Build using
//
//     g++ -O2 external_mvnormal_density.cxx -o external_mvnormal_density


#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

#include <inttypes.h>
#include <arpa/inet.h>


double log_likelihood(const double *params, size_t n_params) {
    const double *x = params;
    assert(n_params == 2);

    const double sigma[][2] = {
        {1.0, 1.5},
        {1.5, 4.0}
    };

    const double log_sqrt_det_2pi_sigma = 2.1176849603770567;

    const double inv_sigma[][2] = {
        { 2.2857142857142856, -0.8571428571428572},
        {-0.8571428571428571,  0.5714285714285714}
    };

    double s = 0;
    for (size_t i = 0; i < 2; ++i) for (size_t j = 0; j < 2; ++j) s += x[i] * inv_sigma[i][j] * x[j];

    return -s / 2 - log_sqrt_det_2pi_sigma;
}


int run_log_likelihood_service(std::istream &input, std::ostream &output) {
    using namespace std;

    const int32_t bat_msg_type = 0x42415430; // "BAT0"

    const uint32_t typehash_GetLogDensityValueDMsg = 0x1164e043;
    const uint32_t typehash_LogDensityValueDMsg = 0xc0c6d511;

    std::vector<char> buffer;

    while (input.good()) {
        int32_t msgtype = 0;
        input.read((char*)&msgtype, sizeof(msgtype));
        if (!input.good()) return 0;
        assert(msgtype == bat_msg_type);
        int32_t req_len = 0;
        input.read((char*)&req_len, sizeof(req_len));
        assert(input.good());
        // cerr << "DEBUG: Reading request with length " << req_len << endl;
        buffer.resize(req_len);
        input.read((char*)&buffer[0], buffer.size());
        char* input_data = (char*)&buffer[0];

        size_t header_len = sizeof(uint32_t) + sizeof(int32_t) + sizeof(int32_t) + sizeof(int32_t);
        assert(req_len >= header_len);

        uint32_t request_tp = *((uint32_t*)input_data); input_data += sizeof(uint32_t);
        assert(request_tp == typehash_GetLogDensityValueDMsg);
        int32_t request_id = *((int32_t*)input_data); input_data += sizeof(int32_t);
        int32_t density_id = *((int32_t*)input_data); input_data += sizeof(int32_t);
        int32_t n_params = *((int32_t*)input_data); input_data += sizeof(int32_t);
        assert(input_data - (char*)&buffer[0] == header_len);
        assert(n_params >= 0);
        assert(header_len = header_len + n_params * sizeof(double));
        double* params = (double*)input_data;

        // cerr << "DEBUG: params ="; for (size_t i = 0; i < n_params; ++i) cerr << " " << params[i]; cerr << endl;
        double ll_value = log_likelihood(params, n_params);
        // cerr << "DEBUG: log(likelihood) value = " << ll_value << endl;

        int32_t resplen = sizeof(uint32_t) + sizeof(int32_t) + sizeof(int32_t) + sizeof(double);
        buffer.resize(resplen);
        char* output_data = &buffer[0];
        *((uint32_t*)output_data) = typehash_LogDensityValueDMsg; output_data += sizeof(uint32_t);
        *((int32_t*)output_data) = request_id; output_data += sizeof(int32_t);
        *((int32_t*)output_data) = density_id; output_data += sizeof(int32_t);
        *((double*)output_data) = ll_value; output_data += sizeof(double);

        // cerr << "DEBUG: Writing response with length " << resplen << endl;
        output.write((char*)&bat_msg_type, sizeof(msgtype));
        output.write((char*)&resplen, sizeof(resplen));
        output.write((char*)&buffer[0], buffer.size());
    }

    return 0;
}


int main() {
    double test_x[2] = {1.23, 2.34};

    assert(fabs(log_likelihood(test_x, 2) - -2.9441421032342) < 1E-7);

    run_log_likelihood_service(std::cin, std::cout);

    return 0;
}
