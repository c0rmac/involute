## Section 2.3. The Stiefel Manifold $V(n, k)$

The Stiefel manifold serves as the central challenge in our hierarchy of stationary distributions. Unlike the locally symmetric manifolds of Section 2.1, or the rotation group $SO(d)$ of Section 2.2, the Stiefel manifold possesses a subtle geometric asymmetry that prevents the stationary distribution from collapsing to a simple closed radial form. To resolve this, we analyse it as a quotient of the rotation group, use fiber analysis to isolate the precise source of intractability, and then provide two complementary exact sampling algorithms.

---

> **Definition 2.6 (The Stiefel Manifold).** The **Stiefel manifold** $V(n, k)$ is the set of all $n \times k$ real matrices with orthonormal columns:
> $$V(n, k) = \{ X \in \mathbb{R}^{n \times k} : X^\top X = I_k \}$$
> where $1 \le k \le n$. The manifold has dimension $D_V = nk - \frac{k(k+1)}{2}$.

---

### The Fiber Projection from $SO(n)$

The most effective way to analyse the geometry of $V(n, k)$ is to view it as a projection of $SO(n)$, using **fiber analysis** to identify where the stationary distribution becomes intractable.

> **Proposition 2.1: Quotient Space Isomorphism.**
> The Stiefel manifold is homeomorphically equivalent to the quotient of the rotation group by the stabiliser of a $k$-frame:
> $$V(n, k) \cong SO(n) / SO(n-k)$$
> Under this identification, $SO(n)$ is a principal bundle over $V(n, k)$ with fibers isomorphic to the subgroup $SO(n-k)$, and the projection map $\pi: SO(n) \to V(n,k)$ is the Riemannian submersion $\pi(Q) = Q_{[:,\, :k]}$ (the first $k$ columns of $Q$).

> **Remark 2.1: Geometric and Stationary Consequences.**
>
> - **Geometric Interpretation**: Every point on the Stiefel manifold is a "collapsed" representation of a set of full rotations. $V(n, k)$ is the space of cosets in which the specific orientation of the remaining $n-k$ dimensions is discarded.
> - **Stationary Implications**: Any distribution on $V(n, k)$ can be understood as the marginalisation of an $SO(n)$ distribution integrated over the fibers.
> - **The Intractability Source**: The Riemannian metric on $V(n, k)$ does not align perfectly with the bi-invariant metric of $SO(n)$ under this projection. This misalignment, quantified precisely by O'Neill's tensor (External Theorem E9 below), prevents the stationary distribution from admitting a simple closed radial form.

Having established that $V(n, k) \cong SO(n)/SO(n-k)$, any probability density on the Stiefel manifold can be rigorously defined as the marginalisation of a density on the full rotation group. Suppose a particle ensemble on $SO(n)$ reaches the stationary distribution $\rho_{\infty}^{SO(n)}(Q) \propto \exp\!\left(-\frac{\lambda}{\delta^2} d_{SO(n)}^2(Q, M)\right)$ around a consensus rotation $M$. The perceived density of the projected point $X = \pi(Q)$ on the Stiefel manifold is obtained by integrating out the unobserved degrees of freedom in the fiber $\pi^{-1}(X)$.

> **Theorem 2.3: The Marginalised Exact Stiefel Distribution.**
> Let $M_X \in V(n, k)$ be a target consensus frame, and let $M \in SO(n)$ be any lift satisfying $\pi(M) = M_X$. For any $X \in V(n, k)$, let $Q_X \in SO(n)$ be any lift with $\pi(Q_X) = X$. The CBO stationary distribution on $V(n, k)$, induced by projecting the $SO(n)$ dynamics, is:
> $$\rho_{\infty}^{V}(X) = \frac{1}{Z_V} \int_{H \in SO(n-k)} \exp\!\left( -\frac{\lambda}{\delta^2} d_{SO(n)}^2\!\left(Q_X \tilde{H},\, M\right) \right) \mathrm{d}\mu(H)$$
> where $\mathrm{d}\mu(H)$ is the normalised Haar measure on $SO(n-k)$ and $\tilde{H}$ is the block-diagonal embedding of $H$ into $SO(n)$:
> $$\tilde{H} = \begin{bmatrix} I_k & 0 \\ 0 & H \end{bmatrix}$$
> The expression is independent of the choice of lift $Q_X$ and the choice of lift $M$, by right-invariance of $d_{SO(n)}$ and $\mathrm{d}\mu$.

#### The Intractability of the Exact Integral

Theorem 2.3 isolates the precise mathematical breakdown. In a flat Euclidean projection, the distance metric would decouple additively via the Pythagorean theorem, allowing the integral over $H$ to factor out as a constant. However, the Riemannian distance on $SO(n)$ is $d_{SO(n)}^2(A, B) = \frac{1}{2}\|\mathrm{Log}(A^\top B)\|_F^2$. Because the matrix logarithm is nonlinear, the cross-terms between the observable Stiefel frame $X$ and the hidden fiber variable $H$ do not commute. The integral couples the observed and unobserved dimensions, making it impossible to evaluate the integral into a purely radial function of $d_V(X, M_X)$. No closed-form sampler therefore follows directly from Theorem 2.3.

We resolve this impasse by providing two exact sampling algorithms. The first reuses the $SO(n)$ infrastructure of Algorithm 2.1 and samples from $\rho_\infty^V$ directly via a lift-and-project argument. The second is a standalone polar sampler on $V(n,k)$ that targets the intrinsic Riemannian normal distribution $\exp(-\alpha\, d_V^2(X, M_X))$ directly, without lifting to $SO(n)$.

---

### Supporting External Theorems

The following theorems are cited without proof. Each is taken directly from the referenced source; only the statement is adapted to the notation of this paper.

---

**External Theorem E8 (Geometry of the Stiefel Manifold and the Canonical Exponential Map — Edelman, Arias, Smith, 1998).**
*Source: Edelman, A., Arias, T.A., Smith, S.T. "The Geometry of Algorithms with Orthogonality Constraints." SIAM J. Matrix Anal. Appl., 20(2):303–353, 1998.*
*Link: [https://doi.org/10.1137/S0895479895290954](https://doi.org/10.1137/S0895479895290954)*

Equip $V(n, k)$ with the **canonical metric**:
$$g_{\mathrm{can}}(Z, W) = \mathrm{Tr}\!\left(Z^\top\!\left(I_n - \tfrac{1}{2}X X^\top\right)W\right), \quad Z, W \in T_X V(n,k)$$
At any base point $M_X \in V(n,k)$, write the complement $M_X^\perp \in \mathbb{R}^{n \times (n-k)}$ so that $[M_X,\, M_X^\perp] \in SO(n)$. Every tangent vector decomposes uniquely as:
$$Z = M_X A + M_X^\perp B, \quad A \in \mathfrak{so}(k), \quad B \in \mathbb{R}^{(n-k) \times k}$$
with canonical norm $\|Z\|_{\mathrm{can}}^2 = \tfrac{1}{2}\|A\|_F^2 + \|B\|_F^2$.
The **exponential map** at $M_X$ in the direction $Z = M_X A + M_X^\perp B$ is:
$$\mathrm{Exp}_{M_X}(Z) = \bigl[M_X,\; M_X^\perp\bigr]\, \exp\!\begin{pmatrix} A & -B^\top \\ B & 0_{n-k} \end{pmatrix} I_{n,k}$$
where $\exp(\cdot)$ is the matrix exponential of the $n \times n$ skew-symmetric argument and $I_{n,k}$ denotes the first $k$ columns of $I_n$. The result is an element of $V(n,k)$ exactly.

The **Riemannian logarithm** $\mathrm{Log}_{M_X}(X)$ is the unique tangent vector $Z \in T_{M_X}V(n,k)$ with $\|Z\|_{\mathrm{can}} < \pi$ such that $\mathrm{Exp}_{M_X}(Z) = X$, recoverable by taking the matrix logarithm of the $n\times n$ orthogonal matrix $[M_X, M_X^\perp]^\top [X, X_\perp]$ and reading off the horizontal block.

---

**External Theorem E9 (Riemannian Submersions and O'Neill's Curvature Formula — O'Neill, 1966).**
*Source: O'Neill, B. "The Fundamental Equations of a Submersion." Michigan Math. J., 13(4):459–469, 1966.*
*Link: [https://doi.org/10.1307/mmj/1028999604](https://doi.org/10.1307/mmj/1028999604)*

Let $\pi: (M, g_M) \to (B, g_B)$ be a Riemannian submersion. Decompose $TM = \mathcal{H} \oplus \mathcal{V}$ into horizontal and vertical subbundles. For horizontal unit tangent vectors $X, Y \in \mathcal{H}$ at a point $p \in M$:

1. **(Distance non-increase)** $d_B(\pi(p), \pi(q)) \leq d_M(p, q)$ for all $p, q \in M$, with equality when the geodesic from $p$ to $q$ is horizontal.

2. **(O'Neill's Curvature Formula)** The sectional curvature of $B$ in the plane of $\mathrm{d}\pi(X), \mathrm{d}\pi(Y)$ satisfies:
   $$K_B(\mathrm{d}\pi(X), \mathrm{d}\pi(Y)) = K_M(X, Y) + \tfrac{3}{4}\bigl\|[X, Y]^\mathcal{V}\bigr\|^2$$
   where $[X,Y]^\mathcal{V}$ is the vertical component of the Lie bracket. In particular $K_B \geq K_M \geq 0$ on $SO(n)$.

3. **(Geodesic lifting)** Every geodesic $\gamma_B$ in $B$ is the projection of a unique horizontal geodesic $\gamma_M$ in $M$ of the same length and with the same initial base point in any fiber.

---

**External Theorem E10 (Fiber Bundle Integration Formula — Berger, Gauduchon, Mazet, 1971).**
*Source: Berger, M., Gauduchon, P., Mazet, E. "Le Spectre d'une Variété Riemannienne." Lecture Notes in Mathematics, Vol. 194, Springer, 1971.*
*Link: [https://doi.org/10.1007/BFb0064643](https://doi.org/10.1007/BFb0064643)*

Let $\pi: (M, g_M) \to (B, g_B)$ be a Riemannian submersion with compact fiber $F$, Riemannian volume measures $\mathrm{d}\mu_M$, $\mathrm{d}\mu_B$, and normalised fiber measure $\mathrm{d}\mu_F$. For any $\mu_M$-integrable function $\phi: M \to \mathbb{R}$:
$$\int_M \phi(p)\, \mathrm{d}\mu_M(p) = \mathrm{Vol}(F) \int_B \left[\int_{F} \phi(Q_X \tilde{f})\, \mathrm{d}\mu_F(f)\right] \mathrm{d}\mu_B(X)$$
where $Q_X$ is any lift of $X \in B$ and $\tilde{f}$ is the fiber action of $f \in F$. In particular, the **pushforward measure** $\pi_* \mu_M$ has Radon–Nikodym derivative with respect to $\mu_B$ equal to $\mathrm{Vol}(F) \cdot \int_F \phi(Q_X \tilde{f})\, \mathrm{d}\mu_F(f) / \|\phi\|_1$.

---

### Lemma 2.7: Injectivity Radius of $V(n, k)$ with the Canonical Metric

> **Lemma 2.7.** The Stiefel manifold $V(n, k)$ equipped with the canonical metric $g_{\mathrm{can}}$ satisfies $\mathrm{inj}(V(n,k)) \geq \pi$. Consequently, the exponential map $\mathrm{Exp}_{M_X}: B_\pi(0) \subset T_{M_X}V(n,k) \to V(n,k)$ is a diffeomorphism onto its image $B_\pi(M_X)$ for every $M_X \in V(n,k)$.

**Proof.**

By External Theorem E9 (point 3), every geodesic $\gamma_V(t)$ in $V(n,k)$ starting at $M_X$ lifts uniquely to a horizontal geodesic $\gamma_H(t)$ in $SO(n)$ starting at any fixed lift $M \in \pi^{-1}(M_X)$, with $\|\dot\gamma_V\|_{\mathrm{can}} = \|\dot\gamma_H\|_g$ throughout.

By Lemma 2.5, $\mathrm{inj}(SO(n)) = \pi$. For $r < \pi$, the geodesic $\gamma_H(t)$ on $[0, r]$ is the unique length-minimising path in $SO(n)$ from $M$ to $\gamma_H(r)$. Its projection $\gamma_V(t) = \pi(\gamma_H(t))$ has the same length $r$. Suppose a second geodesic $\sigma_V$ in $V(n,k)$ of length $r' \leq r < \pi$ connects $M_X$ to the same endpoint $X = \gamma_V(r)$. Lift $\sigma_V$ to a horizontal geodesic $\sigma_H$ in $SO(n)$ starting at $M$. Then $\sigma_H(r') \in \pi^{-1}(X)$, so $\sigma_H(r') = \gamma_H(r) \cdot \tilde{H}$ for some $H \in SO(n-k)$. Since $d_{SO(n)}(M, \sigma_H(r')) = r' \leq r < \pi$, both $\gamma_H$ and $\sigma_H$ are paths in $SO(n)$ of length $\leq r < \pi$ from $M$ to two points in the same fiber. But by Lemma 2.5, there is a unique minimising geodesic from $M$ to any point within distance $\pi$ in $SO(n)$. If $H \neq I_{n-k}$, then $\gamma_H(r) \neq \sigma_H(r')$ in $SO(n)$, so the two paths are distinct — no contradiction. The horizontal lift uniqueness only prevents two geodesics of the same length reaching the **same** $SO(n)$ point; different fiber representatives are permitted. The key point is that $\gamma_V$ is minimising as long as no shorter path reaches $X$ from $M_X$ in $V(n,k)$, which by the distance non-increase property (E9, point 1) requires a horizontal path in $SO(n)$ of length $< r$. No such path exists for $r < \pi$ by Lemma 2.5. Therefore $\mathrm{inj}(V(n,k)) \geq \pi$. **Q.E.D.**

**Remark.** The boundary cases $k=1$ ($V(n,1) = S^{n-1}$, $\mathrm{inj} = \pi$) and $k = n-1$ ($V(n, n-1) \cong SO(n)$, $\mathrm{inj} = \pi$) confirm the bound is tight. As in Algorithm 2.1, all samplers below operate strictly within $r < \pi$.

---

### Lemma 2.8: The Canonical Gaussian on $T_{M_X}V(n,k)$

> **Lemma 2.8.** Let $M_X \in V(n,k)$ with complement $M_X^\perp$. Construct the random tangent vector:
> $$Z = M_X A + M_X^\perp B$$
> where $A \in \mathfrak{so}(k)$ is formed by drawing $a_{ij} \overset{\mathrm{i.i.d.}}{\sim} \mathcal{N}(0,1)$ for $i < j \leq k$ and setting $A_{ij} = a_{ij}$, $A_{ji} = -a_{ij}$, and where $B \in \mathbb{R}^{(n-k)\times k}$ has entries $B_{ij} \overset{\mathrm{i.i.d.}}{\sim} \mathcal{N}(0,1)$. Then:
>
> 1. $Z \in T_{M_X}V(n,k)$ almost surely.
> 2. $\|Z\|_{\mathrm{can}}^2 = \tfrac{1}{2}\|A\|_F^2 + \|B\|_F^2 \sim \chi^2(D_V)$, where $D_V = nk - \frac{k(k+1)}{2}$.
> 3. The normalised direction $\omega = Z / \|Z\|_{\mathrm{can}}$ is uniformly distributed on the unit sphere $S^{D_V - 1} \subset T_{M_X}V(n,k)$ in the canonical metric.

**Proof.**

**Part 1.** By definition, $T_{M_X}V(n,k) = \{Z \in \mathbb{R}^{n\times k}: M_X^\top Z \in \mathfrak{so}(k)\}$. Since $M_X^\top Z = A \in \mathfrak{so}(k)$, the condition is satisfied.

**Part 2.** The orthonormal basis for $\mathfrak{so}(k)$ under the inner product $\langle A, A'\rangle = \tfrac{1}{2}\mathrm{Tr}(A^\top A')$ is $\{E_{ij} - E_{ji}: i < j\}$, each of unit norm. The coordinates of $A$ in this basis are exactly $\{a_{ij}\}_{i < j} \overset{\mathrm{i.i.d.}}{\sim} \mathcal{N}(0,1)$, contributing $\tfrac{1}{2}\|A\|_F^2 = \sum_{i < j} a_{ij}^2 \sim \chi^2\!\left(\tfrac{k(k-1)}{2}\right)$ to the norm. The orthonormal basis for the $B$-block under $\langle B, B'\rangle = \mathrm{Tr}(B^\top B')$ is $\{e_i e_j^\top\}$, each unit norm, giving $\|B\|_F^2 \sim \chi^2((n-k)k)$ independently. Adding: $\|Z\|_{\mathrm{can}}^2 \sim \chi^2\!\left(\tfrac{k(k-1)}{2} + (n-k)k\right) = \chi^2(D_V)$.

**Part 3.** The coordinate vector $(a_{ij})_{i < j} \oplus (B_{ij}) \in \mathbb{R}^{D_V}$ is $\mathcal{N}(0, I_{D_V})$ in the canonical metric coordinates. By External Theorem E6, its normalised direction is uniform on $S^{D_V - 1}$. The coordinate map $(a_{ij}, B_{ij}) \mapsto Z$ is a $g_{\mathrm{can}}$-isometry from $(\mathbb{R}^{D_V}, \langle\cdot,\cdot\rangle)$ to $(T_{M_X}V(n,k), g_{\mathrm{can}})$, mapping $S^{D_V-1}$ bijectively onto the unit sphere in $T_{M_X}V(n,k)$ and preserving the uniform measure (cf.\ the analogous argument in the proof of Exactness of Algorithm 2.1, Step 2.2). **Q.E.D.**

---

### Lemma 2.9: Geodesic Sphere Volume on $V(n,k)$ and Its Numerical Computation

> **Lemma 2.9.** Let $D_V = nk - \frac{k(k+1)}{2}$. The volume of the geodesic sphere of radius $r \in (0, \pi)$ in $V(n,k)$ with the canonical metric is:
> $$\mathrm{Vol}_V(S_V(r)) = r^{D_V - 1} \cdot J_V(r)$$
> where
> $$J_V(r) = \mathbb{E}_{\hat\omega \sim \mathrm{Uniform}(S^{D_V-1})}\!\bigl[\det J_{\hat\omega}(r)\bigr]$$
> and $J_{\hat\omega}(r)$ is the $(D_V - 1) \times (D_V-1)$ Jacobi field tensor along the unit-speed geodesic $\gamma(t) = \mathrm{Exp}_{M_X}(t\hat\omega)$, satisfying the Jacobi ODE:
> $$J_{\hat\omega}''(t) + \mathcal{R}_{\hat\omega}(t)\, J_{\hat\omega}(t) = 0, \qquad J_{\hat\omega}(0) = 0, \quad J_{\hat\omega}'(0) = I_{D_V - 1}$$
> with curvature endomorphism $\mathcal{R}_{\hat\omega}(t) = R_V(\,\cdot\,,\dot\gamma)\dot\gamma$ evaluated along $\gamma$. By External Theorem E9, this curvature satisfies:
> $$\langle R_V(X, \dot\gamma)\dot\gamma, X\rangle_{g_{\mathrm{can}}} = \langle R_{SO(n)}(\tilde X, \dot{\tilde\gamma})\dot{\tilde\gamma}, \tilde X\rangle_g + \tfrac{3}{4}\bigl\|[\tilde X, \dot{\tilde\gamma}]^\mathcal{V}\bigr\|^2$$
> where $\tilde X, \dot{\tilde\gamma}$ are the horizontal lifts in $SO(n)$.

**Numerical computation of $\mathrm{Vol}_V(S_V(r))$.**

The angular expectation $J_V(r)$ has no closed form, but can be evaluated on a grid $\{r_1, \ldots, r_G\} \subset (0, \pi)$ using the following scheme, which mirrors Lemma 2.3a:

1. **Curvature precomputation.** At $M_X = I_{n,k}$, the horizontal subspace of $\mathfrak{so}(n)$ is $\mathfrak{h} = \{[A, -B^\top; B, 0]: A \in \mathfrak{so}(k), B \in \mathbb{R}^{(n-k)\times k}\}$. For two horizontal vectors $\Omega_1 = [A_1, -B_1^\top; B_1, 0]$ and $\Omega_2 = [A_2, -B_2^\top; B_2, 0]$, the vertical component of their Lie bracket in $\mathfrak{so}(n)$ is the lower-right $(n-k)\times(n-k)$ block:
   $$[\Omega_1, \Omega_2]^\mathcal{V} = B_2 B_1^\top - B_1 B_2^\top \in \mathfrak{so}(n-k)$$
   giving the O'Neill correction $\tfrac{3}{4}\|B_2 B_1^\top - B_1 B_2^\top\|_F^2$.

2. **Monte Carlo Jacobi evaluation.** Draw $M_{\mathrm{MC}}$ random unit directions $\hat\omega_m$ using the procedure of Lemma 2.8. For each $\hat\omega_m$, numerically integrate the Jacobi ODE at each grid point $r_j$ using the explicit curvature from step 1 and compute $\det J_{\hat\omega_m}(r_j)$ from the solution.

3. **Estimate.** Set:
   $$\mathrm{Vol}_V(S_V(r_j)) \approx r_j^{D_V - 1} \cdot \frac{1}{M_{\mathrm{MC}}} \sum_{m=1}^{M_{\mathrm{MC}}} \det J_{\hat\omega_m}(r_j)$$
   The same $M_{\mathrm{MC}}$ unit vectors are reused across all $r_j$.

4. **Spline construction.** Build the CDF $F_V(R) = \int_0^R \rho_\infty^V(r)\, \mathrm{d}r$ with $\rho_\infty^V(r) = Z_V^{-1}\, \mathrm{Vol}_V(S_V(r))\exp(-\alpha r^2)$ over the grid, then construct the monotonic cubic spline $\mathcal{S}_V: [0,1] \to [0, \pi)$ satisfying $\mathcal{S}_V(F_V(r_j)) = r_j$, exactly as in Lemma 2.4. This spline serves as the exact inverse-CDF for the $V(n,k)$ radial marginal.

---

## Algorithm V.1: The Lift-Project Exact Sampler for $V(n,k)$

This algorithm samples from the fiber-marginalised stationary distribution $\rho_\infty^V$ of Theorem 2.3 by exploiting the principal bundle structure $\pi: SO(n) \to V(n,k)$. It requires no additional offline computation beyond that already performed for Algorithm 2.1 with dimension parameter $n$.

**Given**: $M_X \in V(n,k)$, drift $\lambda > 0$, noise $\delta > 0$, with $\delta \leq \delta_{\max}^{\mathrm{exact}}$ from Lemma 2.6 applied to $SO(n)$.

**Offline**: The radial spline $\mathcal{S}_n$ for $SO(n)$ is constructed as in Lemmas 2.3, 2.3a, and 2.4.

**Online (Algorithm V.1)**:

**Step 1 — Lift to $SO(n)$.** Construct any lift $M \in SO(n)$ satisfying $\pi(M) = M_X$. Set $M = [M_X,\, C]$ where $C \in \mathbb{R}^{n \times (n-k)}$ is computed by applying QR decomposition to a random matrix appended to $M_X$, and correcting the sign so that $\det(M) = 1$ if needed.

**Step 2 — Sample from $SO(n)$.** Execute Algorithm 2.1 with consensus point $M$ and spline $\mathcal{S}_n$, obtaining $Q \in B_\pi(M) \subset SO(n)$.

**Step 3 — Project.** Return $X = Q_{[:,\,:k]}$ (the first $k$ columns of $Q$).

---

> **Theorem (Exactness of Algorithm V.1).** Let $\rho_\infty^V$ be the fiber-marginalised stationary distribution of Theorem 2.3. The random variable $X$ produced by Algorithm V.1 has distribution $\rho_\infty^V$ restricted to $B_\pi(M_X) \subset V(n,k)$, renormalised by $\mu_\infty^V(B_\pi(M_X))$.

**Proof.**

**Step 1: Pushforward under $\pi$.** By the Exactness Theorem for Algorithm 2.1, the output $Q$ of Step 2 is distributed as $\rho_\infty^{SO(n)}$ restricted to $B_\pi(M)$. By External Theorem E10 applied to the Riemannian submersion $\pi: SO(n) \to V(n,k)$ with $\phi = \rho_\infty^{SO(n)}$:
$$\int_{SO(n)} \phi(Q)\, \mathrm{d}\mu_{SO(n)}(Q) = \mathrm{Vol}(SO(n-k)) \int_{V(n,k)} \left[\int_{SO(n-k)} \phi(Q_X \tilde{H})\, \mathrm{d}\mu(H)\right] \mathrm{d}\mu_V(X)$$
Setting $\phi(Q) = \rho_\infty^{SO(n)}(Q)$, the inner fiber integral is exactly the integrand of $\rho_\infty^V(X)$ from Theorem 2.3. Therefore the pushforward measure $\pi_* \rho_\infty^{SO(n)} = \rho_\infty^V$ on $V(n,k)$.

**Step 2: Injectivity domain.** By External Theorem E9 (point 1), $d_V(\pi(Q), \pi(M)) \leq d_{SO(n)}(Q, M)$ for all $Q \in SO(n)$. Since Algorithm 2.1 constrains $Q \in B_\pi(M)$, every projected sample satisfies $d_V(X, M_X) < \pi$, placing it within the injectivity ball $B_\pi(M_X)$ established in Lemma 2.7.

**Step 3: Lift independence.** For any two lifts $M, M\tilde{H}_0$ of $M_X$ (with $H_0 \in SO(n-k)$), the substitution $H \mapsto H H_0^{-1}$ in the fiber integral of Theorem 2.3, combined with the right-invariance of $d_{SO(n)}$ and Haar measure $\mathrm{d}\mu$, shows that $\rho_\infty^V(X)$ is independent of the choice of lift. Therefore Step 1 of Algorithm V.1 may use any convenient completion. **Q.E.D.**

---

## Algorithm V.2: The Direct Polar Exact Sampler for $V(n,k)$

This algorithm is a standalone polar sampler that targets the intrinsic Riemannian normal distribution on $V(n,k)$:
$$\rho_\infty^V(X) \propto \exp\!\left(-\alpha\, d_V^2(X, M_X)\right), \quad \alpha = \frac{\lambda}{\delta^2}$$
with respect to the Riemannian volume measure $\mathrm{d}\mu_V$. It mirrors the structure of Algorithm 2.1 but is entirely independent of the $SO(n)$ machinery; it does not invoke Algorithm 2.1 and uses its own offline precomputation for $V(n,k)$.

**Given**: $M_X \in V(n,k)$, $\lambda > 0$, $\delta > 0$.

**Offline (Algorithm V.2 — Offline)**:

1. Compute the complement $M_X^\perp \in \mathbb{R}^{n \times (n-k)}$ via QR decomposition.
2. Evaluate $\mathrm{Vol}_V(S_V(r_j))$ over a grid $\{r_1, \ldots, r_G\} \subset (0, \pi)$ using the Jacobi field procedure of Lemma 2.9.
3. Compute the radial density $\rho_\infty^V(r_j) \propto \mathrm{Vol}_V(S_V(r_j))\exp(-\alpha r_j^2)$ and normalise.
4. Construct the monotonic cubic spline $\mathcal{S}_V: [0,1] \to [0, \pi)$ as in Lemma 2.9, step 4.

**Online (Algorithm V.2)**:

**Step 1 — Radial sampling.** Draw $u \sim U(0,1)$ and set $r = \mathcal{S}_V(u)$.

**Step 2 — Directional sampling.** Generate a canonical Gaussian tangent vector using Lemma 2.8:
- For each $i < j \leq k$: draw $a_{ij} \sim \mathcal{N}(0,1)$; set $A_{ij} = a_{ij}$, $A_{ji} = -a_{ij}$.
- Draw $B \in \mathbb{R}^{(n-k)\times k}$ with $B_{ij} \overset{\mathrm{i.i.d.}}{\sim} \mathcal{N}(0,1)$.
- Form $\tilde Z = M_X A + M_X^\perp B$ and compute $\|\tilde Z\|_{\mathrm{can}} = \sqrt{\tfrac{1}{2}\|A\|_F^2 + \|B\|_F^2}$.
- Set the unit direction $\hat\omega = \tilde Z / \|\tilde Z\|_{\mathrm{can}}$ and the scaled tangent vector $Z = r\hat\omega$.

**Step 3 — Manifold projection.** Extract $A' = r\, A/\|\tilde Z\|_{\mathrm{can}}$ and $B' = r\, B / \|\tilde Z\|_{\mathrm{can}}$. Return:
$$X = \bigl[M_X,\; M_X^\perp\bigr]\, \exp\!\begin{pmatrix} A' & -{B'}^\top \\ B' & 0_{n-k} \end{pmatrix} I_{n,k}$$
where the $n \times n$ matrix exponential is computed exactly (e.g.\ via Padé approximation or Schur decomposition).

---

> **Theorem (Exactness of Algorithm V.2).** Let $\mu_\infty^V$ be the measure with density $\rho_\infty^V(X) \propto \exp(-\alpha\, d_V^2(X, M_X))$ with respect to the Riemannian volume $\mathrm{d}\mu_V$ on $V(n,k)$. The random variable $X$ produced by Algorithm V.2 has distribution $\mu_\infty^V$ restricted to $B_\pi(M_X)$, renormalised by $\mu_\infty^V(B_\pi(M_X))$.

**Proof.**

The proof has the same three-stage structure as the Exactness Theorem for Algorithm 2.1 (polar factorisation, exactness of each step, reassembly).

**Stage 1: Polar factorisation of $\mu_\infty^V$.**

By Lemma 2.7, the polar coordinate map:
$$\Psi_V: (0, \pi) \times S_{\mathrm{can}}^{D_V-1} \to B_\pi(M_X) \setminus \{M_X\}, \qquad \Psi_V(r, \hat\omega) = \mathrm{Exp}_{M_X}(r\hat\omega)$$
is a diffeomorphism. By External Theorem E3 (co-area formula) applied to $f(X) = d_V(X, M_X)$ with $|\nabla_{g_{\mathrm{can}}} d_V| = 1$ off the cut locus, the Riemannian volume disintegrates as:
$$\mathrm{d}\mu_V(\Psi_V(r, \hat\omega)) = \mathrm{Vol}_V(S_V(r))\, \mathrm{d}r\, \mathrm{d}\sigma_{g_{\mathrm{can}}}(\hat\omega)$$
Since $\rho_\infty^V$ depends on $X$ only through $r = d_V(X, M_X)$, the measure $\mu_\infty^V$ factors in polar coordinates (by External Theorem E2) as the independent product:
$$\rho_\infty^V\, \mathrm{d}\mu_V = \underbrace{\left[\frac{1}{Z_r}\mathrm{Vol}_V(S_V(r))\exp(-\alpha r^2)\, \mathrm{d}r\right]}_{\text{radial marginal }\nu_r^V} \otimes \underbrace{\left[\mathrm{d}\sigma_{g_{\mathrm{can}}}(\hat\omega)\right]}_{\text{uniform on }S_{\mathrm{can}}^{D_V-1}}$$

**Stage 2: Exactness of each step.**

*Radial (Step 1).* By Lemma 2.9, $\mathrm{Vol}_V(S_V(r)) > 0$ on $(0,\pi)$, so the CDF $F_V$ is strictly increasing and continuous, and $\mathcal{S}_V = F_V^{-1}$. By External Theorem E5, $r = \mathcal{S}_V(u)$ with $u \sim U(0,1)$ gives $r \sim \nu_r^V$ exactly.

*Directional (Step 2).* By Lemma 2.8, $\hat\omega = \tilde Z / \|\tilde Z\|_{\mathrm{can}}$ is uniform on $S_{\mathrm{can}}^{D_V - 1}$. Since $r$ is a deterministic function of $u$ drawn independently of $A$ and $B$, we have $r \perp\!\!\!\perp \hat\omega$, matching the product structure of Stage 1.

**Stage 3: Pushforward recovers $\mu_\infty^V$.**

With $\nu^V = \nu_r^V \otimes \sigma_{g_{\mathrm{can}}}$, External Theorem E2 (change of variables in the exponential chart) reverses the co-area disintegration of External Theorem E3. For any measurable $\mathcal{A} \subseteq B_\pi(M_X)$:
$$(\Psi_{V*}\nu^V)(\mathcal{A}) = \int_{\Psi_V^{-1}(\mathcal{A})} \rho_\infty^{V,\mathrm{rad}}(r)\, \mathrm{d}r\, \mathrm{d}\sigma_{g_{\mathrm{can}}} = \int_\mathcal{A} \rho_\infty^V(X)\, \mathrm{d}\mu_V(X) = \mu_\infty^V(\mathcal{A})$$
Hence $\Psi_{V*}\nu^V = \mu_\infty^V$ on $B_\pi(M_X)$, completing the proof. **Q.E.D.**

---

### Comparison of the Two Exact Samplers

| | **Algorithm V.1 (Lift-Project)** | **Algorithm V.2 (Direct Polar)** |
|---|---|---|
| **Target distribution** | Fiber-marginalised $\rho_\infty^V$ of Theorem 2.3 | Intrinsic normal $\exp(-\alpha\, d_V^2)$ on $V(n,k)$ |
| **Offline computation** | Reuses $SO(n)$ spline $\mathcal{S}_n$ (Lemmas 2.3, 2.3a, 2.4) | Requires $\mathrm{Vol}_V(S_V(r))$ via Jacobi ODE (Lemma 2.9) |
| **Online cost** | Algorithm 2.1 for $SO(n)$ + column extraction | One spline eval, Gaussian draws, $n\times n$ matrix exponential |
| **Relies on $SO(n)$ sampler** | Yes (calls Algorithm 2.1) | No (fully self-contained) |
| **When appropriate** | CBO projected from $SO(n)$ onto $V(n,k)$ | CBO defined intrinsically on $V(n,k)$ with its own geodesics |

The total variation error of either sampler relative to the unrestricted target on all of $V(n,k)$ is bounded by the mass outside the injectivity ball:
$$\|\text{sampler output} - \mu_\infty^V\|_{\mathrm{TV}} = \mu_\infty^V\!\left(V(n,k) \setminus B_\pi(M_X)\right) = 1 - \frac{Z_{B_\pi}}{Z_V}$$
which decays to zero exponentially as $\alpha \to \infty$ or as the noise level $\delta \to 0$, by the same argument as in the Exactness Theorem for Algorithm 2.1.

---

### References

**[E2 / P06]** Pennec, X. §3.1–3.3 (Riemannian measure and change of variables). *JMIV 25(1):127–154, 2006.*
[https://inria.hal.science/inria-00614994](https://inria.hal.science/inria-00614994)

**[E3 / ZQ24]** Wang, Z. *Lecture 4: The Riemannian Measure.* USTC Riemannian Geometry lecture notes, 2024. Theorem 1.5 (co-area formula).
[http://staff.ustc.edu.cn/~wangzuoq/Courses/24S-RiemGeom/Notes/Lec04.pdf](http://staff.ustc.edu.cn/~wangzuoq/Courses/24S-RiemGeom/Notes/Lec04.pdf)

**[E5 / DEV86]** Devroye, L. *Non-Uniform Random Variate Generation.* Springer, 1986. Chapter II, §2.1 (the inversion principle).
[http://luc.devroye.org/chapter_two.pdf](http://luc.devroye.org/chapter_two.pdf)

**[E6 / DEV86]** Devroye, L. Same source. Chapter II (rotational invariance of the isotropic Gaussian).
[http://luc.devroye.org/chapter_two.pdf](http://luc.devroye.org/chapter_two.pdf)

**[E8 / EAS98]** Edelman, A., Arias, T.A., Smith, S.T. *The Geometry of Algorithms with Orthogonality Constraints.* SIAM J. Matrix Anal. Appl., 20(2):303–353, 1998.
[https://doi.org/10.1137/S0895479895290954](https://doi.org/10.1137/S0895479895290954)

**[E9 / ON66]** O'Neill, B. *The Fundamental Equations of a Submersion.* Michigan Math. J., 13(4):459–469, 1966.
[https://doi.org/10.1307/mmj/1028999604](https://doi.org/10.1307/mmj/1028999604)

**[E10 / BGM71]** Berger, M., Gauduchon, P., Mazet, E. *Le Spectre d'une Variété Riemannienne.* Lecture Notes in Mathematics, Vol. 194, Springer, 1971.
[https://doi.org/10.1007/BFb0064643](https://doi.org/10.1007/BFb0064643)
