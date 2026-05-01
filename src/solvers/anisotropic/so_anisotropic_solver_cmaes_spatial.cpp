#include "involute/solvers/anisotropic/so_anisotropic_solver_cmaes_spatial.hpp"
#include "involute/core/math.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

namespace involute::solvers {
using namespace involute::core;

// ==============================================================================
// Constructor
// ==============================================================================

SOAnisotropicSolverCMAESSpatial::SOAnisotropicSolverCMAESSpatial(
    SOAnisotropicSolverCMAESConfig cfg)
    : BaseSolver(SolverConfig{
        .N = cfg.N, .d = cfg.d,
        .params = HyperParameters{.beta = 1.0, .lambda = {1.0}, .delta = {1.0}},
        .dtype  = cfg.dtype,
        .convergence       = cfg.convergence,
        .parameter_adapter = nullptr,
        .debug             = cfg.debug
    })
    , cmaes_cfg_{cfg.sigma0, cfg.mu, cfg.c_c, cfg.c_sigma, cfg.c_1,
                 cfg.c_mu, cfg.d_sigma, cfg.burn_in, cfg.warm_start, cfg.gamma_rtol}
    , D_dof_(cfg.d * (cfg.d - 1) / 2)
    , lambda_(cfg.N)
    , mu_(0)
{}

// ==============================================================================
// CMA-ES parameter initialisation
// ==============================================================================

void SOAnisotropicSolverCMAESSpatial::init_cmaes_params() {
    const double D_dof = static_cast<double>(D_dof_);

    mu_ = (cmaes_cfg_.mu > 0 && cmaes_cfg_.mu < lambda_) ? cmaes_cfg_.mu
                                                          : lambda_ / 2;
    w_.resize(mu_);
    double w_sum = 0.0;
    for (int i = 0; i < mu_; i++) {
        w_[i]  = std::log((lambda_ + 1.0) / 2.0) - std::log(i + 1.0);
        w_sum += w_[i];
    }
    for (auto& wi : w_) wi /= w_sum;

    double w2_sum = 0.0;
    for (auto wi : w_) w2_sum += wi * wi;
    mu_eff_ = 1.0 / w2_sum;

    c_sigma_ = (cmaes_cfg_.c_sigma > 0.0) ? cmaes_cfg_.c_sigma
             : (mu_eff_ + 2.0) / (D_dof + mu_eff_ + 5.0);
    c_c_     = (cmaes_cfg_.c_c > 0.0) ? cmaes_cfg_.c_c
             : (4.0 + mu_eff_ / D_dof) / (D_dof + 4.0 + 2.0 * mu_eff_ / D_dof);
    c_1_     = (cmaes_cfg_.c_1 > 0.0) ? cmaes_cfg_.c_1
             : 2.0 / ((D_dof + 1.3) * (D_dof + 1.3) + mu_eff_);
    const double c_mu_raw = 2.0 * (mu_eff_ - 2.0 + 1.0 / mu_eff_) /
                            ((D_dof + 2.0) * (D_dof + 2.0) + mu_eff_);
    c_mu_    = (cmaes_cfg_.c_mu > 0.0) ? cmaes_cfg_.c_mu
             : std::min(1.0 - c_1_, c_mu_raw);
    d_sigma_ = (cmaes_cfg_.d_sigma > 0.0) ? cmaes_cfg_.d_sigma
             : 1.0 + c_sigma_ + 2.0 * std::sqrt((mu_eff_ - 1.0) / (D_dof + 1.0));
    chi_D_   = std::sqrt(D_dof) * (1.0 - 1.0 / (4.0 * D_dof) + 1.0 / (21.0 * D_dof * D_dof));
}

double SOAnisotropicSolverCMAESSpatial::intrinsic_norm(const Tensor& V) const {
    using namespace math;
    Tensor trace_tensor = sum(square(V));
    eval(trace_tensor);
    return std::sqrt(0.5 * std::max(0.0, to_double(trace_tensor)));
}

Tensor SOAnisotropicSolverCMAESSpatial::transport_matrix(
    const Tensor& M, const Tensor& R) const
{
    using namespace math;
    Tensor RT   = transpose(R, {1, 0});
    Tensor RMRT = matmul(matmul(R, M), RT);
    eval(RMRT);
    return RMRT;
}

// =============================================================================
// Main solve loop — Algorithm 4.9 (spatial-precision variant)
// =============================================================================

CBOResult SOAnisotropicSolverCMAESSpatial::solve(ObjectiveFunction* obj) {
    obj_single_  = obj;
    obj_product_ = nullptr;
    using namespace math;

    const int   d      = config_.d;
    const int   lambda = config_.N;
    const DType dtype  = config_.dtype;

    const bool do_log = std::find(config_.debug.begin(), config_.debug.end(),
                                   Debugger::Log) != config_.debug.end();
    const bool do_history = std::find(config_.debug.begin(), config_.debug.end(),
                                       Debugger::History) != config_.debug.end();

    lambda_ = lambda;
    init_cmaes_params();
    sigma_ = cmaes_cfg_.sigma0;

    Sigma_   = eye(d, dtype);
    p_c_     = full({d, d}, 0.0f, dtype);
    p_sigma_ = full({d, d}, 0.0f, dtype);

    {
        Tensor Z  = random_normal({d, d}, dtype);
        Tensor Zb = expand_dims(Z, {0});
        auto [U, S, Vt] = svd(Zb);
        Tensor UV = matmul(U, Vt);
        Tensor dets = det(UV);
        eval(dets);
        float dv = to_float_vector(dets)[0];
        Tensor m = squeeze(UV, {0});
        if (dv < 0.0f) {
            std::vector<float> buf = to_float_vector(m);
            for (int r = 0; r < d; r++) buf[r * d + (d - 1)] *= -1.0f;
            m_hat_ = { array(buf, {d, d}, dtype) };
        } else {
            m_hat_ = { m };
        }
        eval(m_hat_[0]);
    }

    sampler::anisotropic::SOdAnisotropicSpatialGaussianSampler::Config scfg;
    scfg.num_samples              = lambda;
    scfg.dtype                    = dtype;
    scfg.angle_cfg.d              = d;
    scfg.angle_cfg.burn_in        = cmaes_cfg_.burn_in;
    scfg.angle_cfg.leapfrog_steps = 5;
    scfg.angle_cfg.num_threads    = 1;

    std::vector<double> Gamma_init(d * d, 0.0);
    const double inv_s2 = 1.0 / (sigma_ * sigma_);
    for (int i = 0; i < d; i++) Gamma_init[i * d + i] = inv_s2;

    aniso_sampler_ = std::make_unique<
        sampler::anisotropic::SOdAnisotropicSpatialGaussianSampler>(
            m_hat_[0], d, Gamma_init, scfg);

    best_energy_ = std::numeric_limits<double>::infinity();
    best_m_hat_  = m_hat_;

    CBOResult res;
    std::vector<StepRecord> history;
    double prev_energy = std::numeric_limits<double>::infinity();
    std::vector<Tensor> prev_m_hat = m_hat_;
    std::vector<Tensor> last_particles = { full({lambda, d, d}, 0.0f, dtype) };

    int final_k = 0;
    for (int k = 0; ; k++) {
        final_k = k;

        auto [eigvals, eigvecs] = eigh(Sigma_);
        eigvals = clamp(eigvals, 1e-10, 1e12);
        Tensor inv_ev    = divide(full({d}, 1.0f, dtype), eigvals);
        Tensor Vt_e      = transpose(eigvecs, {1, 0});
        Tensor inv_Sigma = matmul(matmul(eigvecs, diag_embed(inv_ev)), Vt_e);
        Tensor Gamma_k   = multiply(inv_Sigma,
            Tensor(static_cast<float>(1.0 / (sigma_ * sigma_)), dtype));
        eval(Gamma_k);

        aniso_sampler_->update_gamma(to_double_vector(Gamma_k));
        aniso_sampler_->set_m_hat(m_hat_[0]);

        std::vector<Tensor> particles = { aniso_sampler_->sample() };
        eval(particles[0]);
        last_particles = particles;

        Tensor mhat_T = transpose(m_hat_[0], {1, 0});
        std::vector<Tensor> V_all(lambda);
        for (int i = 0; i < lambda; i++) {
            Tensor Xi    = squeeze(slice(particles[0], i, i + 1, 0), {0});
            Tensor Ri    = matmul(mhat_T, Xi);
            Tensor logR  = squeeze(matrix_log(expand_dims(Ri, {0})), {0});
            Tensor logRT = transpose(logR, {1, 0});
            V_all[i] = multiply(subtract(logR, logRT), Tensor(0.5f, dtype));
            eval(V_all[i]);
        }

        Tensor costs = evaluate_objective(particles);
        eval(costs);
        std::vector<float> cost_f = to_float_vector(costs);

        std::vector<int> order(lambda);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](int a, int b){ return cost_f[a] < cost_f[b]; });

        double current_energy = math::to_double(evaluate_objective(m_hat_));
        on_consensus_evaluated(m_hat_, current_energy);

        if (do_log) {
            std::cout << "\r[SOAnisotropicSolverCMAESSpatial] k=" << k
                      << " E=" << current_energy
                      << " best=" << best_energy_ << " σ=" << sigma_ << std::flush;
        }
        if (do_history) history.push_back({k, current_energy, 1.0, 1.0, sigma_});

        SolverState state;
        state.step              = k;
        state.current_energy    = current_energy;
        state.prev_energy       = prev_energy;
        state.current_consensus = &m_hat_;
        state.prev_consensus    = &prev_m_hat;
        state.d                 = d;
        state.N                 = lambda;

        if (current_energy < 0.01) break;
        if (config_.convergence && config_.convergence->check(state)) break;

        prev_energy = current_energy;
        prev_m_hat  = m_hat_;

        Tensor M_k = full({d, d}, 0.0f, dtype);
        for (int i = 0; i < mu_; i++)
            M_k = add(M_k, multiply(V_all[order[i]],
                      Tensor(static_cast<float>(w_[i]), dtype)));
        eval(M_k);

        Tensor expM = squeeze(matrix_exp(expand_dims(M_k, {0})), {0});
        m_hat_[0] = matmul(m_hat_[0], expM);
        eval(m_hat_[0]);

        Tensor R_k = squeeze(matrix_exp(expand_dims(
            multiply(M_k, Tensor(-0.5f, dtype)), {0})), {0});
        eval(R_k);

        std::vector<Tensor> V_elite(mu_);
        for (int i = 0; i < mu_; i++) V_elite[i] = transport_matrix(V_all[order[i]], R_k);

        p_c_ = add(
            multiply(transport_matrix(p_c_, R_k),
                     Tensor(1.0f - static_cast<float>(c_c_), dtype)),
            multiply(multiply(M_k, Tensor(1.0f / static_cast<float>(sigma_), dtype)),
                     Tensor(static_cast<float>(std::sqrt(c_c_ * (2.0 - c_c_) * mu_eff_)), dtype)));

        p_sigma_ = add(
            multiply(transport_matrix(p_sigma_, R_k),
                     Tensor(1.0f - static_cast<float>(c_sigma_), dtype)),
            multiply(multiply(M_k, Tensor(1.0f / static_cast<float>(sigma_), dtype)),
                     Tensor(static_cast<float>(std::sqrt(c_sigma_ * (2.0 - c_sigma_) * mu_eff_)), dtype)));
        eval(p_sigma_);

        sigma_ *= std::exp((c_sigma_ / d_sigma_) * (intrinsic_norm(p_sigma_) / chi_D_ - 1.0));
        sigma_  = std::clamp(sigma_, 1e-12, 1e3);

        Tensor rank1  = multiply(matmul(p_c_, p_c_), Tensor(-1.0f, dtype));
        Tensor rankMu = full({d, d}, 0.0f, dtype);
        for (int i = 0; i < mu_; i++) {
            Tensor Vi_s = multiply(V_elite[i],
                Tensor(1.0f / static_cast<float>(sigma_), dtype));
            rankMu = add(rankMu,
                multiply(multiply(matmul(Vi_s, Vi_s), Tensor(-1.0f, dtype)),
                         Tensor(static_cast<float>(w_[i]), dtype)));
        }

        Sigma_ = add(
            multiply(transport_matrix(Sigma_, R_k),
                     Tensor(1.0f - static_cast<float>(c_1_ - c_mu_), dtype)),
            add(multiply(rank1,  Tensor(static_cast<float>(c_1_),  dtype)),
                multiply(rankMu, Tensor(static_cast<float>(c_mu_), dtype))));
        Sigma_ = multiply(add(Sigma_, transpose(Sigma_, {1, 0})), Tensor(0.5f, dtype));
        eval(Sigma_);
    }

    res.converged       = true;
    res.final_consensus = best_m_hat_;
    res.final_particles = last_particles;
    res.min_energy      = best_energy_;
    res.iterations_run  = final_k + 1;
    res.history         = std::move(history);
    return res;
}

// =============================================================================
// BaseSolver Virtual Overrides
// =============================================================================

std::vector<Tensor> SOAnisotropicSolverCMAESSpatial::initialize_particles() {
    return { math::full({config_.N, config_.d, config_.d}, 0.0f, config_.dtype) };
}

std::vector<Tensor> SOAnisotropicSolverCMAESSpatial::compute_consensus(
    const std::vector<Tensor>&, const Tensor&) {
    return m_hat_;
}

std::vector<Tensor> SOAnisotropicSolverCMAESSpatial::step(
    const std::vector<Tensor>&, const std::vector<Tensor>&,
    HyperParameters) {
    return {};
}

void SOAnisotropicSolverCMAESSpatial::on_consensus_evaluated(
    const std::vector<Tensor>& consensus, double energy)
{
    if (energy < best_energy_) {
        best_energy_ = energy;
        best_m_hat_  = consensus;
    }
}

bool SOAnisotropicSolverCMAESSpatial::check_manifold_constraint(
    const std::vector<Tensor>& p_vec) const
{
    const Tensor& p = p_vec[0];
    Tensor pT  = math::transpose(p, {0, 2, 1});
    Tensor PtP = math::matmul(pT, p);
    Tensor diff = math::subtract(PtP, math::eye(config_.d, config_.dtype));
    double mse  = math::to_double(math::sum(math::square(diff)))
                  / (config_.N * config_.d * config_.d);
    return mse <= 1e-5;
}

} // namespace involute::solvers
