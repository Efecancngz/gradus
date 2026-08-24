#pragma once

#include <functional>
#include <vector>

namespace gradus::testutil {

// Central-difference numerical gradient of `f` at `x`. Used to verify
// analytic backward() implementations against a ground truth that doesn't
// depend on the analytic code being correct.
inline std::vector<double> numerical_gradient(
    const std::function<double(const std::vector<double>&)>& f,
    std::vector<double> x,
    double epsilon = 1e-5) {
    std::vector<double> grad(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        double original = x[i];

        x[i] = original + epsilon;
        double f_plus = f(x);

        x[i] = original - epsilon;
        double f_minus = f(x);

        x[i] = original;
        grad[i] = (f_plus - f_minus) / (2.0 * epsilon);
    }
    return grad;
}

}  // namespace gradus::testutil
