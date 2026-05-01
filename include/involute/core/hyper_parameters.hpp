#pragma once
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace involute::core {
    struct HyperParameters {
        double beta = 1.0;
        std::vector<double> lambda{1.0};
        std::vector<double> delta{1.0};

        double lambda_at(size_t i) const {
            if (i >= lambda.size()) {
                throw std::out_of_range("HyperParameters::lambda index out of range");
            }
            return lambda[i];
        }

        double delta_at(size_t i) const {
            if (i >= delta.size()) {
                throw std::out_of_range("HyperParameters::delta index out of range");
            }
            return delta[i];
        }

        double &lambda_at(size_t i) {
            if (i >= lambda.size()) {
                throw std::out_of_range("HyperParameters::lambda index out of range");
            }
            return lambda[i];
        }

        double &delta_at(size_t i) {
            if (i >= delta.size()) {
                throw std::out_of_range("HyperParameters::delta index out of range");
            }
            return delta[i];
        }
    };
} // namespace involute::core
