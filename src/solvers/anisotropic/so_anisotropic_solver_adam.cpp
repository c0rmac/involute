#include "involute/solvers/anisotropic/so_anisotropic_solver_adam.hpp"
#include "involute/core/math.hpp"
#include <cmath>
#include <algorithm>
#include <vector>

namespace involute::solvers {
    using namespace involute::core;

    // ==============================================================================
    // Static helpers
    // ==============================================================================

    double SOAnisotropicSolverADAM::diameter_so(int d) {
        return M_PI * std::sqrt(static_cast<double>(d * (d - 1)) / 2.0);
    }

    // ==============================================================================
    // Constructor
    // ==============================================================================

    SOAnisotropicSolverADAM::SOAnisotropicSolverADAM(SOAnisotropicSolverADAMConfig cfg)
        : BaseSolver(SolverConfig{
            .N = cfg.N,
            .d = cfg.d,
            .params = HyperParameters{
                .beta   = 1.0,
                .lambda = {cfg.lambda},
                .delta  = {cfg.delta <= 0.0 ? diameter_so(cfg.d) : cfg.delta}
            },
            .dtype             = cfg.dtype,
            .convergence       = cfg.convergence,
            .parameter_adapter = nullptr,
            .debug             = cfg.debug
        })
        , adam_cfg_{cfg.beta1, cfg.beta2, cfg.epsilon}
    {}

    // ==============================================================================
    // SO(d) SVD projection
    // ==============================================================================

    Tensor SOAnisotropicSolverADAM::enforce_so_d(const Tensor &U, const Tensor &Vt) {
        int N = U.shape()[0];
        int d = U.shape()[1];

        Tensor p_tensor = math::matmul(U, Vt);
        Tensor dets = math::det(p_tensor);

        Tensor zero    = Tensor(0.0f, U.dtype());
        Tensor one     = Tensor(1.0f, U.dtype());
        Tensor neg_one = Tensor(-1.0f, U.dtype());

        Tensor clean_dets = math::where(math::greater(dets, zero), one, neg_one);
        Tensor dets_exp = math::reshape(clean_dets, {N, 1, 1});

        std::vector<float> mask_data(d * d, 0.0f);
        mask_data[d * d - 1] = 1.0f;
        Tensor mask = math::array(mask_data, {d, d}, U.dtype());

        Tensor I = math::eye(d, U.dtype());
        Tensor D = math::add(I, math::multiply(mask, math::subtract(dets_exp, one)));

        return math::matmul(math::matmul(U, D), Vt);
    }

    // ==============================================================================
    // BaseSolver interface (Vector-Aware Overrides)
    // ==============================================================================

    std::vector<Tensor> SOAnisotropicSolverADAM::initialize_particles() {
        const int N = config_.N;
        const int d = config_.d;
        const int D = d * (d - 1) / 2;

        t_ = 0;
        best_energy_ = std::numeric_limits<double>::infinity();

        Tensor Z = math::random_normal({N, d, d}, config_.dtype);
        auto [U, S, Vt] = math::svd(Z);
        Tensor p_so3 = enforce_so_d(U, Vt);
        math::eval(p_so3);

        var_cpu_.assign(N, std::vector<double>(D * D, 0.0));

        sampler::anisotropic::SOdAnisotropicHeterogeneousGaussianSampler::Config hcfg;
        hcfg.N               = N;
        hcfg.d               = d;
        hcfg.dtype           = config_.dtype;
        hcfg.num_threads     = 8;
        hcfg.gamma_frob_rtol = 0.01;
        hcfg.num_chains      = 1;
        hcfg.burn_in         = 2000;
        hcfg.leapfrog_steps  = 5;
        hcfg.init_epsilon    = 1e-4;
        hcfg.target_accept   = 0.65;

        hetero_sampler_ = std::make_unique<
            sampler::anisotropic::SOdAnisotropicHeterogeneousGaussianSampler>(hcfg);

        return {p_so3};
    }

    std::vector<Tensor> SOAnisotropicSolverADAM::compute_consensus(
        const std::vector<Tensor> &particles, const Tensor &weights)
    {
        const Tensor &p = particles[0];

        Tensor weighted      = math::multiply(p, math::expand_dims(weights, {1, 2}));
        Tensor M_amb         = math::sum(weighted, {0});
        Tensor M_amb_batched = math::expand_dims(M_amb, {0});

        auto [U_b, S_b, Vt_b] = math::svd(M_amb_batched);
        Tensor Q_b = enforce_so_d(U_b, Vt_b);

        return {math::sum(Q_b, {0})};
    }

    void SOAnisotropicSolverADAM::on_consensus_evaluated(
        const std::vector<Tensor> &consensus, double energy)
    {
        if (energy < best_energy_) {
            best_energy_    = energy;
            best_consensus_ = consensus;
        }
    }

    std::vector<Tensor> SOAnisotropicSolverADAM::step(
        const std::vector<Tensor> &particles,
        const std::vector<Tensor> &consensus,
        HyperParameters params)
    {
        t_++;
        const int N = config_.N;
        const int d = config_.d;
        const int D = d * (d - 1) / 2;

        const std::vector<Tensor> &centre_vec = (best_energy_ < std::numeric_limits<double>::infinity())
                                                ? best_consensus_ : consensus;
        const Tensor &centre = centre_vec[0];
        const Tensor &p_curr = particles[0];

        const double lambda   = params.lambda_at(0);
        const double delta_sq = params.delta_at(0) * params.delta_at(0);
        const double bias2    = 1.0 - std::pow(adam_cfg_.beta2, t_);

        Tensor centre_T = math::transpose(math::expand_dims(centre, {0}), {0, 2, 1});
        Tensor R        = math::matmul(centre_T, p_curr);
        Tensor v_t      = math::matrix_log(R);

        math::eval(v_t);
        std::vector<float> v_flat = math::to_float_vector(v_t);

        std::vector<std::vector<double>> gammas_lie(N, std::vector<double>(D * D, 0.0));

        for (int i = 0; i < N; i++) {
            int a = 0;
            for (int j = 0; j < d; j++) {
                for (int k = j + 1; k < d; k++, a++) {
                    const float v_jk = v_flat[i * d * d + j * d + k];

                    var_cpu_[i][a] = adam_cfg_.beta2 * var_cpu_[i][a]
                                   + (1.0 - adam_cfg_.beta2) * v_jk * v_jk;

                    const double v_hat   = var_cpu_[i][a] / bias2;
                    const double gamma_a = lambda / (delta_sq * std::sqrt(v_hat) + adam_cfg_.epsilon);

                    gammas_lie[i][a * D + a] = gamma_a;
                }
            }
        }

        hetero_sampler_->update_gammas(gammas_lie, 500);
        hetero_sampler_->set_m_hat(centre);

        return {hetero_sampler_->sample()};
    }

    bool SOAnisotropicSolverADAM::check_manifold_constraint(
        const std::vector<Tensor> &p_vec) const
    {
        const Tensor &p = p_vec[0];
        Tensor p_T  = math::transpose(p, {0, 2, 1});
        Tensor Pt_P = math::matmul(p_T, p);
        Tensor I    = math::eye(config_.d, config_.dtype);
        Tensor diff = math::subtract(Pt_P, I);

        double total = math::to_double(math::sum(math::square(diff)));
        double mse   = total / (config_.N * config_.d * config_.d);

        return (mse <= 1e-5);
    }

} // namespace involute::solvers
