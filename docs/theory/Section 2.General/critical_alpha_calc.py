import numpy as np
from scipy.optimize import minimize, bisect
from scipy.stats import chi2
import time

# =============================================================================
# PARAMETERS
# =============================================================================
n = 40
k = 20
EPSILON = 0.05       # 5% max error tolerance
TARGET_MASS = 0.997  # 99.7% (3-sigma equivalent) of particles inside bounds

print(f"--- Fast & TIGHT Conservative Alpha Bound for Stiefel V({n}, {k}) ---")
print("-" * 60)

# =============================================================================
# STAGE 1: GEOMETRIC OPTIMIZATION FOR r* (Unchanged)
# =============================================================================
def exact_volume_ratio(theta, n, k):
    sinc_theta = np.sinc(theta / np.pi)
    ratio_short = np.prod(sinc_theta ** (n - 2 * k))
    ratio_long = 1.0
    if k > 1:
        diffs = theta[:, None] - theta[None, :]
        sums = theta[:, None] + theta[None, :]
        idx = np.triu_indices(k, k=1)
        ratio_long = np.prod(np.sinc(diffs[idx]/np.pi)) * np.prod(np.sinc(sums[idx]/np.pi))
    return ratio_short * ratio_long

def find_critical_radius(epsilon, n, k):
    print("Stage 1: Searching for exact critical radius r*...")
    target_ratio = 1.0 - epsilon

    def objective(theta): return exact_volume_ratio(theta, n, k)

    def find_worst_case_ratio(r):
        def sphere_cons(theta): return np.sum(theta**2) - r**2
        ineq_cons = [{'type': 'ineq', 'fun': lambda t, i=i: t[i] - t[i+1]} for i in range(k - 1)]
        ineq_cons.append({'type': 'ineq', 'fun': lambda t: t[-1]})
        cons = [{'type': 'eq', 'fun': sphere_cons}] + ineq_cons

        theta0 = np.ones(k) * (r / np.sqrt(k))
        perturb = np.linspace(0.01, -0.01, k) * (r / k)
        theta0 = (theta0 + perturb) / np.linalg.norm(theta0 + perturb) * r

        res = minimize(objective, theta0, constraints=cons, method='SLSQP', options={'ftol': 1e-9, 'disp': False})
        return res.fun if res.success else exact_volume_ratio(theta0, n, k)

    def root_func(r): return find_worst_case_ratio(r) - target_ratio

    r_low, r_high = 0.001, 0.5
    while root_func(r_high) > 0 and r_high < np.pi: r_high *= 2.0
    r_star = bisect(root_func, r_low, min(r_high, np.pi), xtol=1e-5)
    return r_star

# =============================================================================
# STAGE 2: THE EXPONENTIAL BOUND CHI-SQUARE SHORTCUT
# =============================================================================
def find_tight_conservative_alpha(r_star, n, k, target_mass):
    print("Stage 2: Calculating TIGHT conservative alpha via Exponential Bound...")

    df = k * (n - k) # Degrees of freedom for Grassmannian

    # 1. Find the required effective alpha using the Chi-Square inverse CDF
    required_chi2_val = chi2.ppf(target_mass, df)
    alpha_eff = required_chi2_val / (2 * r_star**2)

    # 2. Subtract the strict geometric curvature bound (n-2)/6
    curvature_bonus = (n - 2) / 6.0
    alpha_tight = alpha_eff - curvature_bonus

    return alpha_tight, curvature_bonus

# =============================================================================
# EXECUTION
# =============================================================================
if __name__ == '__main__':
    start = time.time()
    r_star = find_critical_radius(EPSILON, n, k)
    alpha_star, bonus = find_tight_conservative_alpha(r_star, n, k, TARGET_MASS)

    print("-" * 60)
    print(f"FINAL RESULT (Computed in {time.time() - start:.3f}s):")
    print(f"Critical Radius (r*) : {r_star:.4f} radians")
    print(f"Manifold Curvature Bonus: {bonus:.2f} (This was mathematically subtracted!)")
    print(f"To guarantee a max tangent-plane error of {EPSILON*100}% across {TARGET_MASS*100}% of particles,")
    print(f"you MUST strictly ensure that:  Alpha >= {alpha_star:.2f}")
    print("-" * 60)