#include "involute/solvers/anisotropic/so_anisotropic_solver_cmaes.hpp"
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

SOAnisotropicSolverCMAES::SOAnisotropicSolverCMAES(SOAnisotropicSolverCMAESConfig cfg)
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

void SOAnisotropicSolverCMAES::init_cmaes_params() {
    const double D = static_cast<double>(D_dof_);

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
             : (mu_eff_ + 2.0) / (D + mu_eff_ + 5.0);
    c_c_     = (cmaes_cfg_.c_c > 0.0) ? cmaes_cfg_.c_c
             : (4.0 + mu_eff_ / D) / (D + 4.0 + 2.0 * mu_eff_ / D);
    c_1_     = (cmaes_cfg_.c_1 > 0.0) ? cmaes_cfg_.c_1
             : 2.0 / ((D + 1.3) * (D + 1.3) + mu_eff_);
    const double c_mu_raw = 2.0 * (mu_eff_ - 2.0 + 1.0 / mu_eff_) /
                            ((D + 2.0) * (D + 2.0) + mu_eff_);
    c_mu_    = (cmaes_cfg_.c_mu > 0.0) ? cmaes_cfg_.c_mu
             : std::min(1.0 - c_1_, c_mu_raw);
    d_sigma_ = (cmaes_cfg_.d_sigma > 0.0) ? cmaes_cfg_.d_sigma
             : 1.0 + c_sigma_ + 2.0 * std::sqrt((mu_eff_ - 1.0) / (D + 1.0));
    chi_D_   = std::sqrt(D) * (1.0 - 1.0 / (4.0 * D) + 1.0 / (21.0 * D * D));
}

// =============================================================================
// Geometry helpers
// =============================================================================

std::vector<double> SOAnisotropicSolverCMAES::skew_to_vec(const Tensor& V) const {
    const int d = config_.d;
    std::vector<float> V_flat = math::to_float_vector(V);
    std::vector<double> v(D_dof_);
    int idx = 0;
    for (int j = 0; j < d - 1; j++)
        for (int k = j + 1; k < d; k++)
            v[idx++] = static_cast<double>(V_flat[j * d + k]);
    return v;
}

Tensor SOAnisotropicSolverCMAES::vec_to_skew(const std::vector<double>& v, DType dtype) const {
    const int d = config_.d;
    std::vector<float> V_flat(d * d, 0.0f);
    int idx = 0;
    for (int j = 0; j < d - 1; j++)
        for (int k = j + 1; k < d; k++) {
            float val = static_cast<float>(v[idx++]);
            V_flat[j * d + k] =  val;
            V_flat[k * d + j] = -val;
        }
    return math::array(V_flat, {d, d}, dtype);
}

Tensor SOAnisotropicSolverCMAES::compute_lie_transport(const Tensor& R) const {
    using namespace math;
    const int d = config_.d;
    const int D = D_dof_;
    const DType dtype = config_.dtype;

    std::vector<int> js(D), ks(D);
    int idx = 0;
    for (int j = 0; j < d - 1; j++)
        for (int k = j + 1; k < d; k++) { js[idx] = j; ks[idx] = k; ++idx; }

    std::vector<float> R_flat = to_float_vector(R);
    std::vector<float> RP(static_cast<size_t>(D) * D);
    for (int b = 0; b < D; b++) {
        int jb = js[b], kb = ks[b];
        for (int a = 0; a < D; a++) {
            int ja = js[a], ka = ks[a];
            float r_jb_ja = R_flat[jb * d + ja];
            float r_kb_ka = R_flat[kb * d + ka];
            float r_jb_ka = R_flat[jb * d + ka];
            float r_kb_ja = R_flat[kb * d + ja];
            RP[b * D + a] = r_jb_ja * r_kb_ka - r_jb_ka * r_kb_ja;
        }
    }
    return array(RP, {D, D}, dtype);
}

double SOAnisotropicSolverCMAES::vec_norm(const Tensor& v) const {
    using namespace math;
    Tensor sq = sum(square(v));
    eval(sq);
    return std::sqrt(std::max(0.0, to_double(sq)));
}

// =============================================================================
// Main solve loop — Algorithm 4.10
// =============================================================================

CBOResult SOAnisotropicSolverCMAES::solve(ObjectiveFunction* obj) {
    obj_single_  = obj;
    obj_product_ = nullptr;
    using namespace math;

    const int    d      = config_.d;
    const int    lambda = config_.N;
    const int    D      = D_dof_;
    const DType  dtype  = config_.dtype;

    const bool do_log = std::find(config_.debug.begin(), config_.debug.end(),
                                   Debugger::Log) != config_.debug.end();
    const bool do_history = std::find(config_.debug.begin(), config_.debug.end(),
                                       Debugger::History) != config_.debug.end();

    lambda_ = lambda;
    init_cmaes_params();
    sigma_ = cmaes_cfg_.sigma0;

    C_       = eye(D, dtype);
    p_c_     = full({D}, 0.0f, dtype);
    p_sigma_ = full({D}, 0.0f, dtype);

    {
        Tensor Z  = random_normal({d, d}, dtype);
        Tensor Zb = expand_dims(Z, {0});
        auto [U, S, Vt] = svd(Zb);
        Tensor UV   = matmul(U, Vt);
        Tensor dets = det(UV);
        eval(dets);
        float dv  = to_float_vector(dets)[0];
        Tensor m  = squeeze(UV, {0});
        if (dv < 0.0f) {
            std::vector<float> buf = to_float_vector(m);
            for (int r = 0; r < d; r++) buf[r * d + (d - 1)] *= -1.0f;
            m_hat_ = { array(buf, {d, d}, dtype) };
        } else {
            m_hat_ = { m };
        }
        eval(m_hat_[0]);
    }

    sampler::anisotropic::SOdAnisotropicGaussianSampler::Config scfg;
    scfg.num_samples              = lambda;
    scfg.dtype                    = dtype;
    scfg.angle_cfg.d              = d;
    scfg.angle_cfg.burn_in        = cmaes_cfg_.burn_in;
    scfg.angle_cfg.leapfrog_steps = 5;
    scfg.angle_cfg.num_threads    = 1;

    const double inv_s2 = 1.0 / (sigma_ * sigma_);
    std::vector<double> Gamma_init(static_cast<size_t>(D) * D, 0.0);
    for (int i = 0; i < D; i++) Gamma_init[i * D + i] = inv_s2;

    aniso_sampler_ = std::make_unique<
        sampler::anisotropic::SOdAnisotropicGaussianSampler>(
            m_hat_[0], d, Gamma_init, scfg);

    best_energy_ = std::numeric_limits<double>::infinity();
    best_m_hat_  = m_hat_;

    CBOResult res;
    std::vector<StepRecord> history;
    double prev_energy  = std::numeric_limits<double>::infinity();
    std::vector<Tensor> prev_m_hat  = m_hat_;
    std::vector<Tensor> last_particles = { full({lambda, d, d}, 0.0f, dtype) };

    int final_k = 0;
    for (int k = 0; ; k++) {
        final_k = k;

        auto [ev_C, evec_C] = eigh(C_);
        ev_C = clamp(ev_C, 1e-10, 1e12);
        Tensor inv_ev_C  = divide(full({D}, 1.0f, dtype), ev_C);
        Tensor evec_C_T  = transpose(evec_C, {1, 0});
        Tensor C_inv     = matmul(matmul(evec_C, diag_embed(inv_ev_C)), evec_C_T);
        Tensor Gamma_k   = multiply(C_inv, Tensor(static_cast<float>(1.0 / (sigma_ * sigma_)), dtype));
        eval(Gamma_k);

        aniso_sampler_->update_gamma(to_double_vector(Gamma_k));
        aniso_sampler_->set_m_hat(m_hat_[0]);

        Tensor particles_component = aniso_sampler_->sample();
        eval(particles_component);
        std::vector<Tensor> particles = { particles_component };
        last_particles = particles;

        Tensor costs = evaluate_objective(particles);
        eval(costs);
        std::vector<float> cost_f = to_float_vector(costs);

        std::vector<int> order(lambda);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](int a, int b) { return cost_f[a] < cost_f[b]; });

        double current_energy = math::to_double(evaluate_objective(m_hat_));
        on_consensus_evaluated(m_hat_, current_energy);

        if (current_energy < best_energy_) {
            best_energy_ = current_energy;
            best_m_hat_  = m_hat_;
        }

        if (do_log) {
            std::cout << "\r[SOAnisotropicSolverCMAES] k=" << k
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

        Tensor mhat_T = transpose(m_hat_[0], {1, 0});
        Tensor M_k = full({d, d}, 0.0f, dtype);

        std::vector<Tensor> V_all(lambda);
        for (int i = 0; i < lambda; i++) {
            Tensor Xi   = squeeze(slice(particles[0], i, i + 1, 0), {0});
            Tensor Ri   = matmul(mhat_T, Xi);
            Tensor logR = squeeze(matrix_log(expand_dims(Ri, {0})), {0});
            Tensor logRT = transpose(logR, {1, 0});
            V_all[i] = multiply(subtract(logR, logRT), Tensor(0.5f, dtype));
            eval(V_all[i]);
        }

        for (int i = 0; i < mu_; i++)
            M_k = add(M_k, multiply(V_all[order[i]], Tensor(static_cast<float>(w_[i]), dtype)));
        eval(M_k);

        Tensor expM = squeeze(matrix_exp(expand_dims(M_k, {0})), {0});
        m_hat_[0] = matmul(m_hat_[0], expM);
        eval(m_hat_[0]);

        Tensor R_k = squeeze(matrix_exp(expand_dims(
            multiply(M_k, Tensor(-0.5f, dtype)), {0})), {0});
        eval(R_k);

        Tensor R_P   = compute_lie_transport(R_k);
        Tensor R_P_T = transpose(R_P, {1, 0});

        Tensor C_t       = matmul(matmul(R_P, C_), R_P_T);
        Tensor p_c_t     = squeeze(matmul(R_P, expand_dims(p_c_, {1})), {1});
        Tensor p_sigma_t = squeeze(matmul(R_P, expand_dims(p_sigma_, {1})), {1});
        eval(C_t); eval(p_c_t); eval(p_sigma_t);

        std::vector<double> delta_m_vec = skew_to_vec(M_k);
        std::vector<float>  delta_m_f(D);
        for (int i = 0; i < D; i++) delta_m_f[i] = static_cast<float>(delta_m_vec[i] / sigma_);
        Tensor delta_m = array(delta_m_f, {D}, dtype);

        auto [ev_Ct, evec_Ct] = eigh(C_t);
        ev_Ct = clamp(ev_Ct, 1e-10, 1e12);
        Tensor C_t_invsqrt = matmul(matmul(evec_Ct,
            diag_embed(divide(full({D}, 1.0f, dtype), sqrt(ev_Ct)))),
            transpose(evec_Ct, {1, 0}));
        Tensor Cinvsqrt_dm = squeeze(matmul(C_t_invsqrt, expand_dims(delta_m, {1})), {1});

        float f_sigma = static_cast<float>(std::sqrt(c_sigma_ * (2.0 - c_sigma_) * mu_eff_));
        p_sigma_ = add(multiply(p_sigma_t, Tensor(1.0f - static_cast<float>(c_sigma_), dtype)),
                       multiply(Cinvsqrt_dm, Tensor(f_sigma, dtype)));

        double ps_norm = vec_norm(p_sigma_);
        sigma_ *= std::exp((c_sigma_ / d_sigma_) * (ps_norm / chi_D_ - 1.0));
        sigma_  = std::clamp(sigma_, 1e-12, 100.0);

        double ps_norm_norm = ps_norm / std::sqrt(1.0 - std::pow(1.0 - c_sigma_, 2.0 * (k + 1)));
        float h_sigma = (ps_norm_norm < (1.4 + 2.0 / (D + 1.0)) * chi_D_) ? 1.0f : 0.0f;
        float f_c = static_cast<float>(std::sqrt(c_c_ * (2.0 - c_c_) * mu_eff_));
        p_c_ = add(multiply(p_c_t, Tensor(1.0f - static_cast<float>(c_c_), dtype)),
                   multiply(delta_m, Tensor(h_sigma * f_c, dtype)));

        Tensor rank1   = matmul(expand_dims(p_c_, {1}), expand_dims(p_c_, {0}));
        Tensor rankMu  = full({D, D}, 0.0f, dtype);
        for (int i = 0; i < mu_; i++) {
            Tensor V_t = matmul(matmul(R_k, V_all[order[i]]), transpose(R_k, {1, 0}));
            std::vector<double> vi_v = skew_to_vec(V_t);
            std::vector<float> vi_f(D);
            for (int j = 0; j < D; j++) vi_f[j] = static_cast<float>(vi_v[j] / sigma_);
            Tensor v = array(vi_f, {D}, dtype);
            rankMu = add(rankMu, multiply(
                matmul(expand_dims(v, {1}), expand_dims(v, {0})),
                Tensor(static_cast<float>(w_[i]), dtype)));
        }

        C_ = add(multiply(C_t, Tensor(1.0f - static_cast<float>(c_1_ + c_mu_), dtype)),
                 add(multiply(rank1, Tensor(static_cast<float>(c_1_), dtype)),
                     multiply(rankMu, Tensor(static_cast<float>(c_mu_), dtype))));
        C_ = multiply(add(C_, transpose(C_, {1, 0})), Tensor(0.5f, dtype));
        eval(C_);
    }

    if (do_log)
        std::cout << "\n[SOAnisotropicSolverCMAES] Finished. best=" << best_energy_ << "\n";

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

std::vector<Tensor> SOAnisotropicSolverCMAES::initialize_particles() {
    return { math::full({config_.N, config_.d, config_.d}, 0.0f, config_.dtype) };
}

std::vector<Tensor> SOAnisotropicSolverCMAES::compute_consensus(
    const std::vector<Tensor>&, const Tensor&) {
    return m_hat_;
}

std::vector<Tensor> SOAnisotropicSolverCMAES::step(
    const std::vector<Tensor>&, const std::vector<Tensor>&,
    HyperParameters) {
    return {};
}

bool SOAnisotropicSolverCMAES::check_manifold_constraint(
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
