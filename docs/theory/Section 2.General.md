## Section 2: Spectral Sampling and the Geometry of Compact Homogeneous Spaces

### Subsection 2.1: Introduction to Lie Groups and Lie Algebras

To establish a rigorous framework for optimization on curved manifolds, we utilize the theory of Lie groups and their associated Lie algebras. This approach allows for the linearization of a manifold's local geometry, facilitating the derivation of stochastic updates—such as those used in Consensus-Based Optimization (CBO)—within a flat, vector-space tangent domain.

---

#### **Definition 2.1: The Lie Group ($G$)**
A **Lie group** $G$ is a $D$-dimensional smooth manifold equipped with a group structure such that the group multiplication $\mu: G \times G \to G$, defined by $\mu(g, h) = gh$, and the inversion $\iota: G \to G$, defined by $\iota(g) = g^{-1}$, are $C^\infty$ smooth maps. This dual nature allows $G$ to be studied both as a geometric object and an algebraic symmetry group.

> **Example 2.1 (The Special Orthogonal Group):** The rotation group **$SO(n)$** is a compact Lie group defined as the set of all $n \times n$ real orthogonal matrices with unit determinant:
> $$SO(n) = \{ R \in \mathbb{R}^{n \times n} \mid R^\top R = I, \det(R) = 1 \}$$

#### **Definition 2.2: Left-Invariant Vector Fields and the Lie Algebra ($\mathfrak{g}$)**
For any $g \in G$, let $L_g: G \to G$ be the left translation map $L_g(h) = gh$. A vector field $X$ on $G$ is **left-invariant** if for all $g, h \in G$, the differential satisfies $(dL_g)_h(X_h) = X_{gh}$. The **Lie algebra** $\mathfrak{g}$ of $G$ is the real vector space consisting of all left-invariant vector fields, which is canonically isomorphic to the tangent space at the identity element, $T_e G$.

> **Example 2.2 ($\mathfrak{so}(n)$):** The Lie algebra of **$SO(n)$**, denoted **$\mathfrak{so}(n)$**, consists of the space of $n \times n$ **skew-symmetric matrices**:
> $$\mathfrak{so}(n) = \{ \Omega \in \mathbb{R}^{n \times n} \mid \Omega^\top = -\Omega \}$$
> For any $\Omega \in \mathfrak{so}(n)$, the trace is zero ($\text{Tr}(\Omega) = 0$), which represents the infinitesimal condition for preserving the determinant during a rotation.

#### **Definition 2.3: The Lie Bracket and Algebraic Structure**
The Lie algebra $\mathfrak{g}$ is equipped with a bilinear, alternating map $[\cdot, \cdot]: \mathfrak{g} \times \mathfrak{g} \to \mathfrak{g}$, known as the **Lie bracket**, which satisfies the **Jacobi identity**:
$$[X, [Y, Z]] + [Y, [Z, X]] + [Z, [X, Y]] = 0 \quad \forall X, Y, Z \in \mathfrak{g}$$

> **Example 2.3 (Matrix Commutator):** For matrix Lie algebras, the bracket is defined as the standard commutator:
> $$[A, B] = AB - BA$$
> If $[A, B] = 0$, the elements $A$ and $B$ are said to commute, meaning the flow of their associated vector fields is locally interchangeable.

#### **Definition 2.4: The Exponential Map ($\exp$)**
The **exponential map** $\exp: \mathfrak{g} \to G$ is the unique smooth map such that for each $X \in \mathfrak{g}$, the curve $\gamma(t) = \exp(tX)$ is the one-parameter subgroup of $G$ with initial velocity $\gamma'(0) = X$. This map provides the fundamental link between the linear algebra and the curved manifold, effectively "wrapping" the tangent space around the group.

> **Example 2.4 (Matrix Exponential):** For matrix groups, the map is computed via the power series:
> $$\exp(A) = \sum_{k=0}^{\infty} \frac{A^k}{k!}$$
> In CBO, this map is used to update particle positions while ensuring they remain strictly on the manifold $\mathcal{M}$.

#### **Definition 2.5: The Riemannian Logarithm and Injectivity Radius**
Let $x \in \mathcal{M}$. The **Riemannian exponential map** $\text{Exp}_x: T_x\mathcal{M} \to \mathcal{M}$ maps a tangent vector $v$ to the point $\gamma(1)$, where $\gamma$ is the unique geodesic starting at $x$ with initial velocity $v$.

The **injectivity radius** $\text{inj}(\mathcal{M})$ is the largest radius $r > 0$ such that $\text{Exp}_x$ is a diffeomorphism when restricted to the open ball $B_r(0) \subset T_x\mathcal{M}$. Within this ball, the inverse map is well-defined and is called the **Riemannian Logarithm**, denoted $\text{Log}_x: \text{Exp}_x(B_r(0)) \to T_x\mathcal{M}$. For optimization purposes, the Logarithm provides the unique shortest-path tangent vector pointing from $x$ to a target point $y$, such that $\|\text{Log}_x(y)\|_g = d_g(x, y)$.

#### **Definition 2.6: The Adjoint Representations ($Ad$ and $ad$)**
The **Adjoint representations** characterize the internal symmetries of the Lie group and its algebra by describing how they act upon themselves:
1. **The Group Adjoint ($Ad_g$):** For $g \in G$, $Ad_g: \mathfrak{g} \to \mathfrak{g}$ is the differential of the conjugation map $C_g(h) = ghg^{-1}$ at the identity. In matrix form:
   $$Ad_g(X) = gXg^{-1}$$
2. **The Algebra Adjoint ($ad_X$):** The differential of $Ad$ at the identity is $ad_X: \mathfrak{g} \to \mathfrak{g}$, which is defined directly by the Lie bracket:
   $$ad_X(Y) = [X, Y]$$

#### **Definition 2.7: Bi-invariant Riemannian Metrics**
A Riemannian metric $g$ on $G$ is **bi-invariant** if it is invariant under both left and right translations, ensuring that the intrinsic geometry (distances and angles) is preserved regardless of the frame of reference. For a compact Lie group, such a metric is uniquely determined by an $Ad$-invariant inner product on the algebra $\mathfrak{g}$.

> **Example 2.7 (The Scaled Frobenius Inner Product):** On the rotation group $SO(n)$, we typically utilize the bi-invariant metric defined by the trace operator:
> $$g(U, V) = \frac{1}{2} \text{Tr}(U^\top V)$$
> This metric allows the Riemannian distance $d_g(X, Y)$ to be calculated intrinsically as the norm of the matrix logarithm of the relative rotation.

---

### Subsection 2.2: Foundations of Homogeneous Quotient Spaces

Building upon the Lie-theoretic preliminaries, we define the structural decomposition of the manifold $\mathcal{M}$ as a quotient of group actions. This algebraic partitioning allows for the rigorous isolation of the ensemble's "shape" (which remains invariant under the stabilizer) from its "orientation" (which is governed by the fiber).

#### **Definition 2.8: The Homogeneous Quotient Space**
Let $G$ be a compact, connected Lie group and $H \subset G$ be a closed subgroup. The state space $\mathcal{M}$ is a **homogeneous space** defined as the space of left cosets:
$$\mathcal{M} = G/H = \{ gH : g \in G \}$$
The manifold $\mathcal{M}$ is a smooth manifold of dimension $\dim(G) - \dim(H)$. The canonical projection $\pi: G \to G/H$ given by $\pi(g) = gH$ is a smooth submersion, defining $G$ as a principal $H$-bundle over the base manifold $\mathcal{M}$.

> **Example 2.8 (The Stiefel Manifold and the Dimensionality Assumption):** The **Stiefel manifold** is identified as the homogeneous space $V(n, k) \cong SO(n) / SO(n-k)$. In this geometrical context, $G = SO(n)$ acts transitively on the space of $k$-frames in $\mathbb{R}^n$, and the subgroup $H = SO(n-k)$ serves as the stabilizer of a fixed reference frame.
>
> **Crucial Assumption:** Throughout this framework, when dealing with the Stiefel manifold, we explicitly assume $n \ge 2k$ (the "tall and skinny" frame). If $n < 2k$, the geometric rank of the space drops to $n-k$, and the formulas for the principal angles and the Dyson drift break down. Optimization on such "fat" frames is mathematically equivalent to optimizing their $(n-k)$-dimensional orthogonal complements, which effectively restores the safe $n \ge 2k$ condition.

#### **Definition 2.9: The Reductive Decomposition**

For a compact Lie group $G$ and a closed subgroup $H$, the Lie algebra $\mathfrak{g}$ admits a canonical **reductive decomposition**. This is the direct sum of vector spaces:
$$\mathfrak{g} = \mathfrak{h} \oplus \mathfrak{m}$$
where $\mathfrak{h}$ is the Lie algebra of the stabilizer $H$, and $\mathfrak{m}$ is its orthogonal complement with respect to the bi-invariant metric $g$ defined in **Definition 2.7**.

The $Ad(G)$-invariance of the metric $g$ ensures that $\mathfrak{m}$ is **$Ad(H)$-invariant**, satisfying $Ad(h)\mathfrak{m} \subset \mathfrak{m}$ for all $h \in H$. This algebraic structure identifies $\mathfrak{m}$ as the specific subspace of "non-stabilizing" generators that are canonically isomorphic to the tangent space of the manifold at the origin: $T_{[e]}(G/H) \cong \mathfrak{m}$.

> **Example 2.9 (Reductive Structure of the Stiefel Manifold):** Consider the Stiefel manifold $V(n, k) \cong SO(n)/SO(n-k)$. The Lie algebra $\mathfrak{g} = \mathfrak{so}(n)$ consists of $n \times n$ skew-symmetric matrices. We partition an arbitrary matrix $\Omega \in \mathfrak{so}(n)$ into blocks:
> $$\Omega = \begin{bmatrix} A & -B^\top \\ B & C \end{bmatrix}$$
> where $A \in \mathbb{R}^{k \times k}$ and $C \in \mathbb{R}^{(n-k) \times (n-k)}$ are skew-symmetric, and $B \in \mathbb{R}^{(n-k) \times k}$.
>
> 1. **The Stabilizer Algebra ($\mathfrak{h}$)**: Corresponds to rotations that act only on the $n-k$ dimensions orthogonal to the frame. These take the form:
     >    $$\mathfrak{h} = \left\{ \begin{bmatrix} \mathbf{0}_k & \mathbf{0} \\ \mathbf{0} & C \end{bmatrix} : C \in \mathfrak{so}(n-k) \right\}$$
> 2. **The Reductive Complement ($\mathfrak{m}$)**: Corresponds to the degrees of freedom that actually define the $k$-frame's position. These are the matrices where $C = 0$:
     >    $$\mathfrak{m} = \left\{ \begin{bmatrix} A & -B^\top \\ B & \mathbf{0}_{n-k} \end{bmatrix} : A \in \mathfrak{so}(k), B \in \mathbb{R}^{(n-k) \times k} \right\}$$
>
> This decomposition is the "engine" of our theory: it allows us to project the full $n(n-1)/2$ dimensions of $SO(n)$ noise down to the $nk - k(k+1)/2$ dimensions of the Stiefel manifold by simply "zeroing out" the $C$ block during the SDE update.

#### **Definition 2.10: The Adjoint Action and Orbit Stratification on Reductive Spaces**
Let $\mathcal{M} = G/H$ be a compact reductive homogeneous space with the Lie algebra decomposition $\mathfrak{g} = \mathfrak{h} \oplus \mathfrak{m}$, satisfying the reductive geometric condition $[\mathfrak{h}, \mathfrak{m}] \subset \mathfrak{m}$.

The stabilizer subgroup $H$ acts continuously on the tangent space $\mathfrak{m}$ via the restricted Adjoint representation, $Ad(H)$. For strict symmetric spaces, this action possesses sufficient degrees of freedom to conjugate any tangent vector into a flat maximal abelian subspace $\mathfrak{a}$ (the Cartan Decomposition). However, for general reductive homogeneous spaces (such as the Stiefel manifold), the commutator $[\mathfrak{m}, \mathfrak{m}]$ is not contained strictly within $\mathfrak{h}$. Consequently, $H$ is topologically insufficient to fully diagonalize $\mathfrak{m}$.

Therefore, instead of collapsing to a flat subspace, the continuous Adjoint action foliates the tangent space into a set of equivalence classes known as Adjoint orbits:
$$\mathcal{O}_V = \{ Ad_h(V) \mid h \in H \} \quad \text{for any } V \in \mathfrak{m}$$
The invariant geometric properties of the tangent space are thus parameterized not by a simple abelian chamber, but by the fully stratified quotient space (the orbit space) $\mathcal{S}_{shape} = \mathfrak{m} / Ad(H)$. This space mathematically retains the irreducible internal degrees of freedom that the stabilizer is too small to eliminate.

#### **Definition 2.11: The Root System and the Fundamental Weyl Chamber**

Let $\mathfrak{g}$ be a compact Lie algebra and $\mathfrak{a} \subset \mathfrak{m}$ be the Maximal Abelian Subspace as established in **Definition 2.10**. We define the **root system** $\Phi \subset \mathfrak{a}^*$ (where $\mathfrak{a}^*$ is the dual space of $\mathfrak{a}$) as the finite set of all non-zero linear functionals $\alpha$ such that the simultaneous eigenspace, known as the **root space** $\mathfrak{g}_\alpha$, is non-trivial:
$$\mathfrak{g}_\alpha = \{ X \in \mathfrak{g}_{\mathbb{C}} \mid [H, X] = i\alpha(H)X \quad \forall H \in \mathfrak{a} \}$$
The set $\Phi$ characterizes the directions of non-commutativity within the algebra. A subset $\Phi^+ \subset \Phi$ is designated as the **positive root system** if it satisfies the following strict conditions:
1.  **Completeness**: For every root $\alpha \in \Phi$, exactly one of the pair $\{ \alpha, -\alpha \}$ is contained in $\Phi^+$.
2.  **Closure**: For any $\alpha, \beta \in \Phi^+$, if their sum $\alpha + \beta$ is a root, then $\alpha + \beta \in \Phi^+$.

The **fundamental Weyl chamber** $\mathcal{W}$ is the open, convex cone in $\mathfrak{a}$ defined as the intersection of open half-spaces where all positive roots evaluate to strictly positive values:
$$\mathcal{W} = \{ H \in \mathfrak{a} \mid \alpha(H) > 0 \quad \forall \alpha \in \Phi^+ \}$$
The closure $\overline{\mathcal{W}}$ serves as the fundamental domain for the spectral map $\Phi$. The boundaries of the chamber, $\partial \mathcal{W}$, correspond to the kernels of the roots ($\ker \alpha$), representing the singular set where the dimensionality of the orientation fibers changes.

> **Example 2.11 (The Grassmannian Root System and the Affine Fundamental Alcove):**
> While the Stiefel manifold $V(n, k) \cong SO(n)/SO(n-k)$ is a reductive homogeneous space, it is not a strict symmetric space and therefore lacks a classical restricted root system. However, the Stiefel manifold is geometrically structured as a principal $SO(k)$-bundle over the Grassmannian manifold $Gr(n, k) \cong SO(n) / S(O(k) \times O(n-k))$, which *is* a rigorous symmetric space.
>
> Because the $SO(k)$ fibers (representing the unobserved internal rotational degrees of freedom) are metrically flat with respect to the principal angles $\theta$, the $\theta$-dependent Riemannian volume distortion of the Stiefel Adjoint orbits is strictly and entirely inherited from the symmetric base space, the Grassmannian.
>
> Therefore, the active geometric distortion is governed perfectly by the restricted root system of the Grassmannian. For $n > 2k$, these restricted roots form a single, cohesive classical root system of **Type $B_k$**. The standard positive restricted roots $\alpha \in \Phi^+$ partition into long and short roots:
>
> 1. **Long Roots (The $D_k$ Subsystem / Internal Frame Interactions):** These roots take the explicit form $\alpha(\theta) = \theta_i \pm \theta_j$ for $1 \le i < j \le k$. Their zero-sets enforce the strict internal ordering of the active principal angles ($\theta_i > \theta_j$). Each of these long roots possesses a geometric multiplicity of $m_\alpha = 1$.
> 2. **Short Roots (Boundary Frame Interactions):** These roots take the explicit form $\alpha(\theta) = \theta_i$ for $1 \le i \le k$. Their zero-sets establish the absolute lower bound of the linear Weyl chamber ($\theta_k \ge 0$). The geometric multiplicity of these short roots is exactly $m_\alpha = n - 2k$.
>
> While the linear roots $\Phi^+$ define the unbounded Weyl chamber, the geometry of a compact symmetric space is strictly confined by its Riemannian cut locus. This maximal geodesical extension is bounded by the **affine roots** of the system. For the Grassmannian, the bounding affine root evaluates to zero precisely at $\frac{\pi}{2}$, dictating the point where geodesics globally intersect and the orientation fibers topologically collapse.
>
> Together, the zero-sets of the standard linear roots and the bounding affine root define the strict topological boundaries of the fundamental domain (the affine Weyl alcove) for both the Grassmannian and the Stiefel shape spaces: $\frac{\pi}{2} \ge \theta_1 > \theta_2 > \dots > \theta_k \ge 0$.

#### **Definition 2.12: The Generalized Spectral Map and the Orbit Space**

To accommodate arbitrary compact homogeneous spaces $\mathcal{M} = G/H$ (including those lacking a strict symmetric space structure), we must generalize the geometric concept of "shape." For a general tangent vector $V \in \mathfrak{m}$, the stabilizer group $H$ may not possess sufficient degrees of freedom to perfectly diagonalize $V$ into a flat $k$-dimensional abelian subspace $\mathfrak{a}$.

Instead, the invariant geometric signature of a displacement is captured by its equivalence class under the Adjoint action of the stabilizer. We define the **Shape Space** (or Orbit Space) as the quotient:
$$\mathcal{S}_{shape} = \mathfrak{m} / Ad(H)$$

Let $s \in \mathcal{S}_{shape}$ denote a generalized shape coordinate. The **Generalized Spectral Map** $\Phi: \mathcal{M} \to \mathcal{S}_{shape}$ is the non-bijective operator that extracts this invariant geometric signature relative to the frozen consensus point $\widehat{M}_k$. For any $X \in \mathcal{M}$, the map $\Phi(X)$ yields the unique shape parameter $s$ such that the Riemannian logarithm satisfies the Adjoint orbit relation:
$$\text{Log}_{\widehat{M}_k}(X) = Ad_h(V(s)) \quad \text{for some } h \in H$$
where $V(s) \in \mathfrak{m}$ is the canonical Lie-algebraic embedding of the shape $s$.

This identity ensures that all degrees of freedom that *can* be rotated away by the stabilizer $H$ are marginalized into the orientation fiber, while $s$ strictly retains the irreducible internal geometry (the principal angles and any coupled internal frame rotations) necessary to accurately reconstruct the distance and trajectory.

> **Example 2.12 (The Stiefel Orbit Space and Dimensionality):** > For the Stiefel manifold $V(n, k) \cong SO(n)/SO(n-k)$, under the assumption $n \ge 2k$, the tangent space $\mathfrak{m}$ at the consensus point has dimension $nk - \frac{k(k+1)}{2}$. The stabilizer $H = SO(n-k)$ acts strictly on the unobserved $n-k$ dimensions.
>
> Because $H$ only acts on the left side of the off-diagonal tangent blocks, it cannot fully diagonalize the system. The shape parameter $s$ must strictly decompose into three distinct components to capture the full $k^2$ dimensions of the invariant orbit: $s = (\Omega_{int}, \theta, V_{right})$, where:
> 1. $\Omega_{int} \in \mathfrak{so}(k)$ represents the internal rotational degrees of freedom of the $k$-frame (Dimension: $\frac{k(k-1)}{2}$).
> 2. $\theta \in \mathbb{R}^k$ represents the principal angles tilting the frame (Dimension: $k$).
> 3. $V_{right} \in SO(k)$ represents the un-cancellable right-action rotational twist of the principal angles (Dimension: $\frac{k(k-1)}{2}$).
>
> To properly map these active dimensions into the full $(n-k) \times k$ off-diagonal tangent block, we define $\Sigma(\theta) \in \mathbb{R}^{(n-k) \times k}$ as the rectangular block matrix containing $\text{diag}(\theta)$ stacked on top of an $(n-2k) \times k$ matrix of strictly zero entries:
> $$\Sigma(\theta) = \begin{bmatrix} \text{diag}(\theta) \\ \mathbf{0}_{(n-2k) \times k} \end{bmatrix}$$
>
> The Canonical Embedding $V(s) \in \mathfrak{m}$ therefore takes the precise block structure:
> $$V(s) = \begin{bmatrix} \Omega_{int} & -V_{right} \Sigma(\theta)^\top \\ \Sigma(\theta) V_{right}^\top & \mathbf{0}_{n-k} \end{bmatrix}$$
>
> The Generalized Spectral Map $\Phi(X) = (\Omega_{int}, \theta, V_{right})$ perfectly extracts the $k^2$ degrees of freedom of the Adjoint orbit, entirely avoiding the mathematical bottleneck of dimensional collapse while rigorously satisfying the ambient matrix dimensions of the homogeneous space.

#### **Lemma 2.1: Metric Isometry and Regular Drift Projection on the Orbit Space**
Let $\mathcal{M} = G/H$ be a compact homogeneous space equipped with a $G$-invariant metric $g$. Let $\mathcal{S}_{shape} = \mathfrak{m} / Ad(H)$ be the stratified shape space, containing the open, dense principal stratum of regular shapes $\mathcal{S}_{reg}$. Let $\Phi(X) = s$ be the generalized spectral map.
1.  **Global Metric Isometry:** The squared Riemannian distance from any point $X$ to the consensus point $\widehat{M}_k$ is strictly equal to the metric norm of its canonical shape embedding $V(s) \in \mathfrak{m}$, which defines the intrinsic distance on the orbit space:
    $$d_g^2(X, \widehat{M}_k) = \|V(s)\|_g^2 = d_{\mathcal{S}}^2(s, s_0)$$
    where $s_0$ is the origin of the shape space.
2.  **Regular Drift Projection:** Let $\mathcal{M}_{reg} = \Phi^{-1}(\mathcal{S}_{reg})$ be the regular domain of the manifold. Restricted strictly to this principal stratum, the spectral map $\Phi: \mathcal{M}_{reg} \to \mathcal{S}_{reg}$ acts as a smooth Riemannian submersion. The geometric projection of the intrinsic CBO drift vector $\lambda \text{Log}_{X_t}(\widehat{M}_k)$ onto the regular shape space via the differential $\mathrm{d}\Phi$ acts strictly as an intrinsic Riemannian gradient descent toward the origin $s_0$:
    $$\mathrm{d}\Phi \left( \lambda \text{Log}_{X_t}(\widehat{M}_k) \right) = -\lambda \text{grad}_{g_{\mathcal{S}}} \Psi(s_t)$$
    where $\Psi(s) = \frac{1}{2} d_{\mathcal{S}}^2(s, s_0)$ is the intrinsic half-squared distance potential on the orbit space.

**Proof.**

*Part 1: Metric Isometry.*
By the definition of the Riemannian logarithm (Definition 2.5), the squared intrinsic distance is the squared norm of the tangent vector: $d_g^2(X, \widehat{M}_k) = \|\text{Log}_{\widehat{M}_k}(X)\|_g^2$.
By the definition of the Generalized Spectral Map (Definition 2.12), the logarithm decomposes into an Adjoint orbit: $\text{Log}_{\widehat{M}_k}(X) = Ad_h(V(s))$ for some $h \in H$.
Because the metric $g$ is uniquely determined by an $Ad$-invariant inner product on the Lie algebra $\mathfrak{g}$ (Definition 2.7), the Adjoint action of the stabilizer constitutes a strict isometry. Therefore, the metric norm is invariant under the conjugation by $h$:
$$d_g^2(X, \widehat{M}_k) = \|Ad_h(V(s))\|_g^2 = \|V(s)\|_g^2 = d_{\mathcal{S}}^2(s, s_0) \qquad \square_1$$

*Part 2: Regular Drift Projection.*
We must compute the pushforward of the continuous CBO drift vector $\lambda \text{Log}_{X_t}(\widehat{M}_k)$ onto the shape space. By the Principal Orbit Type Theorem, $\mathcal{S}_{shape}$ is a stratified space. We restrict our differential analysis to the open, dense principal stratum $\mathcal{S}_{reg}$, where the isotropy subgroups are conjugate and the orbit dimensions are constant, rendering $\mathcal{S}_{reg}$ a true smooth manifold and $\Phi: \mathcal{M}_{reg} \to \mathcal{S}_{reg}$ a strict Riemannian submersion.

By Part 1, we established the exact metric isometry $d_g(X, \widehat{M}_k) = d_{\mathcal{S}}(s, s_0)$. Therefore, the full manifold potential function $V(X) = \frac{\lambda}{2} d_g^2(X, \widehat{M}_k)$ is the pullback of the regular shape space potential $\Psi(s) = \frac{\lambda}{2} d_{\mathcal{S}}^2(s, s_0)$ via the spectral map: $V(X) = \Psi(\Phi(X))$.

Because $\Phi$ acts as a Riemannian submersion from $\mathcal{M}_{reg}$ to the regular orbit space $(\mathcal{S}_{reg}, g_{\mathcal{S}})$, the differential $\mathrm{d}\Phi$ maps the horizontal gradient of a pulled-back function exactly to the gradient of the base function.
Applying the differential operator yields:
$$\mathrm{d}\Phi \left( \lambda \text{Log}_{X_t}(\widehat{M}_k) \right) = \mathrm{d}\Phi \left( -\lambda \text{grad}_g V(X_t) \right) = -\lambda \text{grad}_{g_{\mathcal{S}}} \Psi(s_t)$$
This coordinate-free projection rigorously preserves the geometry of all shape components within the principal stratum. The measure-zero singular boundaries are handled independently via topological reflection. $\square_2$

---

#### **Lemma 2.2: Uniqueness, Stratification, and Class Function Identity of the Orbit Space**
Let $X \in \mathcal{M}$ be a point situated on a compact homogeneous space $\mathcal{M} = G/H$ equipped with a $G$-invariant metric $g$. Assume $X$ is constrained strictly within the injectivity radius $r = \text{inj}(\mathcal{M})$ of the consensus point $\widehat{M}_k$ (Definition 2.5). Let $\Phi: \mathcal{M} \to \mathcal{S}_{shape}$ be the generalized spectral map onto the orbit space $\mathcal{S}_{shape} = \mathfrak{m} / Ad(H)$ (Definition 2.12).

1.  **Orbit Uniqueness:** The generalized shape parameter $s = \Phi(X) \in \mathcal{S}_{shape}$ is strictly and uniquely defined for all $X$ within the injectivity ball.
2.  **Boundary Singularities and Stratification:** The shape space $\mathcal{S}_{shape}$ is not a simple flat simplex but a stratified topological space. On the boundaries between strata, the mapping exhibits geometric singularities where the dimension of the orientation fibers strictly decreases, indicating a structural collapse in the frame's degrees of freedom.
3.  **Class Function Identity:** The Riemannian distance functional $d_g(X, \widehat{M}_k)$ acts as a class function, depending exclusively on the invariant shape coordinate:
    $$d_g(X, \widehat{M}_k) = \|V(s)\|_g$$

**Proof.**

*Part 1: Orbit Uniqueness via Equivalence Partitioning.*

Let $B_r(0) \subset \mathfrak{m}$ be the open ball of radius $r = \text{inj}(\mathcal{M})$ centered at the origin of the tangent space $T_{\widehat{M}_k}\mathcal{M} \cong \mathfrak{m}$. By the definition of the injectivity radius (Definition 2.5), the Riemannian exponential map $\text{Exp}_{\widehat{M}_k}: B_r(0) \to \mathcal{M}$ is a global diffeomorphism onto its image.

Therefore, for any $X \in \text{Exp}_{\widehat{M}_k}(B_r(0))$, there exists a unique tangent vector $V = \text{Log}_{\widehat{M}_k}(X) \in \mathfrak{m}$.

The stabilizer subgroup $H$ acts continuously on the vector space $\mathfrak{m}$ via the Adjoint representation $Ad(H)$. By the fundamental properties of topological group actions, the set of all Adjoint orbits $\mathcal{O}_V = \{ Ad_h(V) \mid h \in H \}$ forms a strict and exhaustive equivalence partition of $\mathfrak{m}$.

Crucially, this geometric partitioning holds regardless of the dimension or continuity of the isotropy subgroup $H_V = \{ h \in H \mid Ad_h(V) = V \}$. Even if $H_V$ is a continuous, positive-dimensional Lie group (such as $SO(n-2k)$ for the Stiefel manifold), the orbit $\mathcal{O}_V$ remains a singular, rigorously defined equivalence class. Because the vector $V$ is unique, it belongs to exactly one Adjoint orbit. Since the shape space $\mathcal{S}_{shape} = \mathfrak{m} / Ad(H)$ is defined precisely as the quotient space of these orbits, the shape parameter $s$ indexing the orbit $\mathcal{O}_V$ is uniquely and unambiguously determined for all $X$ within the injectivity ball. $\square_1$

*Part 2: Boundary Singularities via the Principal Orbit Type Theorem.*

We must rigorously define the "boundary" of the shape space without relying on the algebraic roots of a symmetric space.

For any tangent vector $V \in \mathfrak{m}$, define its isotropy subgroup within the stabilizer as $H_V = \{ h \in H \mid Ad_h(V) = V \}$. The dimension of the orbit $\mathcal{O}_V$ is given by the quotient dimension: $\dim(\mathcal{O}_V) = \dim(H) - \dim(H_V)$.

By the **Principal Orbit Type Theorem** for smooth actions of compact Lie groups, the quotient space $\mathcal{S}_{shape} = \mathfrak{m} / Ad(H)$ is a stratified space. There exists an open, dense subset called the *principal stratum*, consisting of "regular" elements where the conjugacy class of $H_V$ is minimal, and thus the orbit dimension $\dim(\mathcal{O}_V)$ is maximal.

The structural "boundaries" or singular sets of $\mathcal{S}_{shape}$ correspond exactly to the lower-dimensional strata. As a sequence of regular shapes $s_n$ converges to a boundary shape $s_{\partial}$, the isotropy subgroup $H_V$ must undergo a discontinuous jump in dimension (e.g., from a discrete finite group to a continuous subgroup like $SO(2)$).

Because $\dim(H_V)$ strictly increases at the boundary, the dimension of the Adjoint orbit $\dim(\mathcal{O}_V)$ strictly decreases. Geometrically, this means the mapping from the orientation group $H$ to the tangent space $\mathfrak{m}$ becomes degenerate; multiple distinct choices of orientation frames $h \in H$ collapse and evaluate to the exact same tangent vector. This proves the existence of cut-locus-style geometric singularities at the boundaries of the orbit space strata. $\square_2$

*Part 3: Class Function Identity via Metric $G$-Invariance.*

We wish to prove that the intrinsic distance depends only on the shape orbit. By Part 1, within the injectivity radius, the Riemannian distance is the metric norm of the unique tangent vector:
$$d_g(X, \widehat{M}_k) = \|\text{Log}_{\widehat{M}_k}(X)\|_g$$

By the definition of the Generalized Spectral Map (Definition 2.12), the logarithm decomposes into a canonical shape embedding conjugated by some orientation frame: $\text{Log}_{\widehat{M}_k}(X) = Ad_h(V(s))$ for some $h \in H$. Substituting this yields:
$$d_g(X, \widehat{M}_k) = \|Ad_h(V(s))\|_g$$

By Definition 2.7, the bi-invariant Riemannian metric $g$ is uniquely determined by an $Ad$-invariant inner product on the Lie algebra. Therefore, the Adjoint action of $H$ acts by strict isometries on $\mathfrak{m}$, meaning it preserves the metric norm:
$$\|Ad_h(V(s))\|_g = \|V(s)\|_g$$

Because the quantity $\|V(s)\|_g$ depends strictly on the invariant shape parameter $s = \Phi(X)$ and is completely independent of the orientation frame $h \in H$, the distance functional is exactly constant on every $Ad(H)$-orbit. By definition, a function constant on the orbits of a group action is a class function. $\square_3$

---

#### **Definition 2.13: The Metric Volume Twist**
On the homogeneous space $G/H$, the Riemannian volume element $d\mu_g$ does not uniformly factor into an independent product of angular and fiber measures. The **geometric twist** quantifies this local metric distortion, which arises directly from the non-commutativity of the reductive complement $\mathfrak{m}$ with the stabilizer $\mathfrak{h}$.

> **Example 2.13 (The Stiefel Metric Twist):** In the specific case of $V(n, k)$, the geometric twist manifests mathematically as a $(\sin \theta_i)^{n-2k}$ multiplicative term within the volume density. This correctly accounts for the volumetric size of the $SO(n-k)$ fibers relative to the $n-2k$ purely unobserved dimensions that remain after accounting for both the frame's position and its internal rotational degrees of freedom.

#### **Theorem 2.1: The Generalized Integration Formula for Homogeneous Spaces**
Let $(\mathcal{M}, g)$ be a compact homogeneous space $\mathcal{M} = G/H$ equipped with a $G$-invariant metric, and let $\mathcal{S}_{shape} = \mathfrak{m} / Ad(H)$ be its corresponding shape orbit space (Definition 2.12).

If $f: \mathcal{M} \to \mathbb{R}$ is an integrable class function—meaning $f(X)$ depends exclusively on the invariant geometric shape $s = \Phi(X)$ and is constant across the orientation fibers—then the integral of $f$ over the manifold within the injectivity radius decomposes exactly into an integral over the shape space:
$$\int_{\mathcal{M}} f(X) \mathrm{d}\mu_g(X) = C \int_{\mathcal{S}_{shape}} f(s) w(s) \mathrm{d}s$$
where $C$ is a normalization constant, and $w(s)$ is the **Generalized Density Kernel**. This kernel completely captures the metric volume distortion and is defined as the product of the flat Adjoint orbit volume and the Riemannian curvature determinant:
$$w(s) = \text{vol}_{\mathfrak{m}}(Ad_H(V(s))) \cdot \det(d\text{Exp}_{V(s)})$$

**Proof.**

The proof rigorously maps the integration from the curved manifold $\mathcal{M}$ back to the flat tangent space $\mathfrak{m}$, and then applies the geometric Coarea formula to quotient out the rotational degrees of freedom governed by the continuous isometric action of the stabilizer $H$.

*Step 1: Pullback to the Tangent Space via Global Measure Equivalence.*

Let $\mathcal{M} = G/H$ be a compact homogeneous space equipped with a $G$-invariant Riemannian metric $g$. Let $\widehat{M}_k \in \mathcal{M}$ be the consensus point.

Let $\text{cut}(\widehat{M}_k) \subset \mathcal{M}$ denote the cut locus of $\widehat{M}_k$. By standard Riemannian geometry, the cut locus of any point on a compact manifold is a closed set of Riemannian measure zero, meaning $\mu_g(\text{cut}(\widehat{M}_k)) = 0$. Consequently, the integral of any measurable function $f$ over the entire manifold $\mathcal{M}$ is exactly equal to the integral over the dense, open set $\mathcal{M} \setminus \text{cut}(\widehat{M}_k)$:
$$\int_{\mathcal{M}} f(X) \mathrm{d}\mu_g(X) = \int_{\mathcal{M} \setminus \text{cut}(\widehat{M}_k)} f(X) \mathrm{d}\mu_g(X)$$

Within the tangent space $T_{\widehat{M}_k}\mathcal{M} \cong \mathfrak{m}$, let $\mathcal{D} \subset \mathfrak{m}$ be the open, star-shaped domain (the interior of the fundamental Weyl alcove) such that the Riemannian exponential map $\text{Exp}_{\widehat{M}_k}: \mathcal{D} \to \mathcal{M} \setminus \text{cut}(\widehat{M}_k)$ is a global diffeomorphism.

By the change of variables formula for diffeomorphisms between Riemannian manifolds, the volume form $\mathrm{d}\mu_g(X)$ pulls back to the Lebesgue measure $\mathrm{d}V$ on the vector space $\mathfrak{m}$ (induced by the $Ad(H)$-invariant inner product) multiplied by the Jacobian determinant of the exponential map:
$$\int_{\mathcal{M} \setminus \text{cut}(\widehat{M}_k)} f(X) \mathrm{d}\mu_g(X) = \int_{\mathcal{D}} f(\text{Exp}_{\widehat{M}_k}(V)) \det(d\text{Exp}_V) \mathrm{d}V$$

By hypothesis, $f$ is a class function, meaning its value depends exclusively on the Adjoint orbit of the tangent vector. By the definition of the shape parameter $s = \Phi(X)$, we have $f(\text{Exp}_{\widehat{M}_k}(V)) = f(s)$.

*Step 2: Decomposition of the Tangent Space (The Coarea Formula).*

The stabilizer group $H$ acts on $\mathfrak{m}$ via the restricted Adjoint representation $Ad(H)$. Because the metric $g$ is $G$-invariant, this action is strictly by linear isometries. The action foliates $\mathcal{D}$ into a continuous family of disjoint orbits. The shape space $\mathcal{S}_{shape} = \mathcal{D} / Ad(H)$ parametrizes these orbits.

By the geometric Coarea Formula (Integration over Fibers) applied to the quotient map $\pi: \mathcal{D} \to \mathcal{S}_{shape}$, the standard measure $\mathrm{d}V$ on the tangent space rigorously factorizes into a base measure $\mathrm{d}s$ over the quotient space $\mathcal{S}_{shape}$ and a family of conditional measures $\mathrm{d}\mu_{orbit}$ over the individual Adjoint orbits $\mathcal{O}_s = \{ Ad_h(V(s)) \mid h \in H \}$:
$$\mathrm{d}V = \mathrm{d}\mu_{orbit}(V) \wedge \mathrm{d}s$$

Applying Fubini's theorem via this disintegration, we split the integral over $\mathcal{D}$ into an outer integral over the shape parameters $s \in \mathcal{S}_{shape}$, and an inner integral over the specific Adjoint orbit $\mathcal{O}_s$:
$$\int_{\mathcal{D}} f(s) \det(d\text{Exp}_V) \mathrm{d}V = \int_{\mathcal{S}_{shape}} \left( \int_{\mathcal{O}_s} f(s) \det(d\text{Exp}_V) \mathrm{d}\mu_{orbit}(V) \right) \mathrm{d}s$$

*Step 3: Evaluating the Inner Orbit Integral via Isometric Invariance.*

Consider the inner integral over the orbit $\mathcal{O}_s$. Because $f(s)$ depends only on the orbit class $s$, it is strictly constant over $\mathcal{O}_s$ and factors out of the inner integral.

We must now evaluate the behavior of the Jacobian determinant $\det(d\text{Exp}_V)$ along the orbit. Because the Riemannian metric $g$ on $\mathcal{M}$ is strictly $G$-invariant, the left-action of any element $h \in H \subset G$ constitutes a global isometry on $\mathcal{M}$. By the geometric properties of isometries, they map geodesics to geodesics, yielding the identity:
$$\text{Exp}_{\widehat{M}_k}(Ad_h(V)) = h \cdot \text{Exp}_{\widehat{M}_k}(V)$$

Taking the differential of this equation with respect to $V$, we see that the pushforward $d\text{Exp}$ intertwines the linear Adjoint action on $\mathfrak{m}$ with the differential of the left-translation on $T\mathcal{M}$. Because both the Adjoint action on $\mathfrak{m}$ and the left-translation on $(\mathcal{M}, g)$ are volume-preserving isometries, the Riemannian curvature distortion (measured by the Jacobian determinant) is perfectly isotropic along the Adjoint orbit.

Therefore, for all $V \in \mathcal{O}_s$, the determinant is constant: $\det(d\text{Exp}_V) = \det(d\text{Exp}_{V(s)})$, where $V(s)$ is the canonical shape embedding. Factoring this out yields:
$$\int_{\mathcal{O}_s} f(s) \det(d\text{Exp}_V) \mathrm{d}\mu_{orbit}(V) = f(s) \det(d\text{Exp}_{V(s)}) \int_{\mathcal{O}_s} \mathrm{d}\mu_{orbit}(V)$$

The remaining integral $\int_{\mathcal{O}_s} \mathrm{d}\mu_{orbit}(V)$ evaluates exactly to the geometric volume of the Adjoint orbit within the flat space $\mathfrak{m}$, denoted as $\text{vol}_{\mathfrak{m}}(Ad_H(V(s)))$.

*Step 4: Assembly of the Generalized Kernel.*

Substituting the exactly evaluated inner integral back into the outer integral, we obtain:
$$\int_{\mathcal{M}} f(X) \mathrm{d}\mu_g(X) = \int_{\mathcal{S}_{shape}} f(s) \left[ \det(d\text{Exp}_{V(s)}) \cdot \text{vol}_{\mathfrak{m}}(Ad_H(V(s))) \right] \mathrm{d}s$$

We define the generalized density kernel as $w(s) = \det(d\text{Exp}_{V(s)}) \cdot \text{vol}_{\mathfrak{m}}(Ad_H(V(s)))$. Absorbing any necessary manifold-specific topological normalization factors into the global constant $C$, we arrive at the final exact, global factorization:
$$\int_{\mathcal{M}} f(X) \mathrm{d}\mu_g(X) = C \int_{\mathcal{S}_{shape}} f(s) w(s) \mathrm{d}s \qquad \blacksquare$$

---

#### **Definition 2.14: The Generalized Weyl Density Kernel**
Let $\mathcal{S}_{shape} = \mathfrak{m} / Ad(H)$ be the shape orbit space of a compact homogeneous space, and let $s \in \mathcal{S}_{shape}$ be the invariant shape parameter. The **Generalized Weyl Density Kernel**, denoted $w(s)$, represents the exact Riemannian volume of the unobserved Adjoint orientation fiber $\Phi^{-1}(s) \subset \mathcal{M}$ situated above the shape $s$.

For spaces where the active shape is parameterized by a set of principal angles $\theta$ interacting through a restricted root system $\Phi^+$, the density kernel evaluates to a product of trigonometric twists:
$$w(\theta) \propto \prod_{\alpha \in \Phi^+} \left| \sin\left( c_\alpha \alpha(\theta) \right) \right|^{m_\alpha}$$
where:
* $\alpha(\theta)$ is a positive root functional evaluating the angular interactions.
* $m_\alpha$ is the geometric multiplicity (the dimension of the unobserved subspace governed by that root).
* $c_\alpha$ is a structural scaling constant (typically $1/2$ or $1$) determined by the algebraic normalization of the root on the specific homogeneous space.

This kernel $w(s)$ acts as the Jacobian determinant for the Riemannian submersion, mathematically quantifying the entropic volume distortion that occurs when projecting the full, high-dimensional manifold dynamics down to the lower-dimensional shape space.

---

### Subsection 2.3: Projection of CBO Stochastic Dynamics

Having established the algebraic decomposition of the homogeneous space and the spectral map $\Phi$, we now project the high-dimensional manifold dynamics onto the low-dimensional Maximal Abelian Subspace $\mathfrak{a}$. This requires mapping the full Stratonovich SDE defined in Section 1.3 through the non-bijective coordinate system to derive an autonomous stochastic process for the principal angles.

#### **Definition 2.15: The Laplace-Beltrami Operator on the Orbit Space**
Let $f(s)$ be a twice-differentiable class function defined strictly on the local coordinates of the shape space $\mathcal{S}_{shape} = \mathfrak{m} / Ad(H)$. Because the quotient of a flat space by a continuous group action induces intrinsic curvature, $\mathcal{S}_{shape}$ is equipped with an induced quotient metric $g_{\mathcal{S}}$.

The action of the manifold's Laplace-Beltrami operator $\Delta_g$ on $f$ is mathematically equivalent to the weighted Laplace-Beltrami operator on this curved orbit space. In local coordinates, evaluating the Riemannian divergence rigorously requires the inclusion of the Riemannian volume element $\sqrt{|g_{\mathcal{S}}|}$, where $|g_{\mathcal{S}}| = \det(g_{\mathcal{S}})$. The operator is defined exactly as:
$$\Delta_{\mathcal{S}} f(s) = \frac{1}{w(s) \sqrt{|g_{\mathcal{S}}|}} \sum_{i,j} \frac{\partial}{\partial s^i} \left( w(s) \sqrt{|g_{\mathcal{S}}|} g_{\mathcal{S}}^{ij} \frac{\partial f}{\partial s^j} \right)$$
where $g_{\mathcal{S}}^{ij}$ is the inverse metric tensor of the orbit space, and $w(s)$ is the Generalized Density Kernel (Theorem 2.1). This operator rigorously accounts for both the intrinsic geometric curvature of the quotient space (via the metric determinant) and the volume distortion of the Adjoint fibers (via the density kernel).

#### **Definition 2.16: The Generalized Dyson Drift (Log-Kernel Gradient)**
The geometric transformation of the isotropic Brownian noise from the full manifold $\mathcal{M}$ down to the generalized shape space $\mathcal{S}_{shape}$ induces a deterministic displacement vector known as the **Generalized Dyson Drift**, denoted $\mathcal{D}(s)$. It is defined precisely as the Riemannian gradient of the log-density kernel:
$$\mathcal{D}(s) = \text{grad}_s \ln w(s)$$
Physically, this term represents an entropic restorative force. The "hidden" Brownian collisions within the unobserved orientation fibers push the shape coordinates away from the boundaries of the orbit space, preventing the singular collapse of the frame. For non-symmetric spaces like the Stiefel manifold, this drift dynamically couples the principal angles $\theta$ and the internal rotational degrees of freedom $\Omega_{int}$.

> **Remark** *Because $w(s) = w(\theta)$ is independent of the internal rotation $\Omega_{int}$ and the right-action twist $V_{right}$, the Dyson drift has zero components along those directions: $\mathcal{D}_{\Omega_{int}} = \mathbf{0}$ and $\mathcal{D}_{V_{right}} = \mathbf{0}$. The full drift vector on $\mathcal{S}_{shape}$ is therefore $\mathcal{D}(s) = (\mathbf{0}, \mathcal{D}(\theta), \mathbf{0})$.*


> **Example 2.16 (The Generalized Dyson Drift for the Stiefel Manifold):**
> For the Stiefel manifold $V(n, k)$ under the assumption $n \ge 2k$, the Weyl density kernel is governed by the rank-$k$ Grassmannian root system. Because the manifold is defined over the real field (Orthogonal groups), the Type $D_k$ roots strictly have a geometric multiplicity of $m_\alpha = 1$. Furthermore, because the projection maps to a symmetric space rather than evaluating a pure Lie group conjugacy class, the Riemannian volume distortion evaluates the roots at their full angles ($c_\alpha = 1$).
>
> Applying the generalized formula, the exact Stiefel volume density kernel is:
> $$w(\theta) = \left( \prod_{i < j}^k \sin(\theta_i - \theta_j) \sin(\theta_i + \theta_j) \right) \prod_{i=1}^k \sin^{n-2k}(\theta_i)$$
>
> The Generalized Dyson Drift $\mathcal{D}(\theta) = \text{grad}_{g_{\mathcal{S}}} \ln w(\theta)$ is computed via the partial derivatives of the log-kernel. Applying the logarithmic derivative rigorously to the full-angle cross-terms yields:
> $$\mathcal{D}_i(\theta) = \sum_{j \neq i} \left[ \cot(\theta_i - \theta_j) + \cot(\theta_i + \theta_j) \right] + (n-2k)\cot(\theta_i)$$
>
> This exact vector field defines the restorative geometric pressure that prevents the principal angles from colliding with each other (the cotangent cross-terms) or with the continuous stabilizer boundaries (the $(n-2k)\cot(\theta_i)$ term).

#### **Definition 2.17: The Local Time Process at the Orbit Boundary**
Because the spectral map confines the dynamics to the closure of the shape space $\overline{\mathcal{S}_{shape}}$, we must define a boundary regulator. Let $s_t^i$ be a continuous semi-martingale representing a shape coordinate, and let $\partial \mathcal{S}_{shape}$ be the boundary of the valid geometric domain (e.g., where angles collide or cross zero). The **local time process** $L_t$ is the unique, continuous, non-decreasing stochastic process supported strictly on the boundary times $\{ t : s_t \in \partial \mathcal{S}_{shape} \}$. It acts as the minimal instantaneous "push" required to keep the particle within the valid topological domain.

#### **Definition 2.18: The Aggregate Reflection Process**
For the shape vector $s_t \in \overline{\mathcal{S}_{shape}}$, the **aggregate reflection process** $\mathrm{d}L_t$ is the vector sum of the individual local times acting on the boundaries of the orbit space. This signed reflection term ensures that the state vector instantaneously bounces off the singular hyperplanes, preserving the geometric validity of the shape parameters at all times $t \ge 0$.

#### **Definition 2.19: Intrinsic Manifold Brownian Motion ($B_t$)**
The stochastic driving process $B_t$ in the Stratonovich sense ($\circ \mathrm{d}B_t$) represents the intrinsic **Brownian motion** on the manifold $\mathcal{M}$. It is mathematically defined as the diffusion process whose infinitesimal generator is exactly $\frac{1}{2}\Delta_g$, where $\Delta_g$ is the Laplace-Beltrami operator on $(\mathcal{M}, g)$. This ensures that the noise is isotropic within the tangent space $T_{X_t}\mathcal{M}$ and strictly respects the curvature of the homogeneous space.

#### **Theorem 2.2: The Autonomous Shape SDE and Coupled Drift**
Let $X_t$ be a particle evolving on a compact homogeneous space $\mathcal{M}$ according to the semi-discrete CBO equation:
$$\mathrm{d}X_t = \lambda \text{Log}_{X_t}(\widehat{M}_k) \mathrm{d}t + \delta \circ \mathrm{d}B_t$$
where $B_t$ is the intrinsic manifold Brownian motion (Definition 2.19) and $\text{Log}$ is the Riemannian Logarithm (Definition 2.5).

Applying the Meyer-Itô formula through the generalized spectral map $\Phi$, the evolution of the full shape parameter $s_t \in \overline{\mathcal{S}_{shape}}$ is governed entirely by the autonomous, reflected Stochastic Differential Equation in **Itô form** on the curved orbit space $(\mathcal{S}_{shape}, g_{\mathcal{S}})$:
$$\mathrm{d}s_t = \left( -\lambda \text{grad}_{g_{\mathcal{S}}} \Psi(s_t) + \frac{\delta^2}{2} \mathcal{D}(s_t) \right) \mathrm{d}t + \delta \, \mathrm{d}B_t^{\mathcal{S}} + \mathrm{d}L_t$$
where:
* $\Psi(s_t) = \frac{1}{2} d_{\mathcal{S}}^2(s_t, s_0)$ is the intrinsic half-squared distance from the origin of the shape space.
* $\text{grad}_{g_{\mathcal{S}}}$ is the Riemannian gradient operator with respect to the quotient metric $g_{\mathcal{S}}$.
* $\mathcal{D}(s_t) = \text{grad}_{g_{\mathcal{S}}} \ln w(s_t)$ is the Generalized Dyson Drift (Definition 2.16).
* $\mathrm{d}B_t^{\mathcal{S}}$ is the intrinsic Brownian motion increment on the curved shape space evaluated in the standard Itô sense.
* $\mathrm{d}L_t$ is the aggregate reflection process maintaining $s_t \in \overline{\mathcal{S}_{shape}}$.

**Proof.**

*Setup and Strategy.*
Let $\Phi: \mathcal{M} \to \mathcal{S}_{shape}$ be the Riemannian submersion from the manifold to the shape orbit space. We operate entirely in local geometric coordinates $s^j$ on the base space $\mathcal{S}_{shape}$. We map the continuous dynamics using the Meyer-Itô formula, which provides the rigorous geometric conversion from the Stratonovich integral on $\mathcal{M}$ to the Itô integral on $\mathcal{S}_{shape}$.

We must analyze three components: the projected deterministic CBO drift, the Itô second-order curvature correction (which will yield both the connection drift and the Dyson drift), and the martingale noise term.

*Step 1: The Meyer-Itô Formula on a Manifold.*
For a smooth function $f: \mathcal{M} \to \mathbb{R}$ and a semimartingale $X_t$ satisfying the Stratonovich SDE:
$$\mathrm{d}X_t = F(X_t)\mathrm{d}t + \delta \circ \mathrm{d}B_t, \quad \text{where } F(X_t) = \lambda\,\text{Log}_{X_t}(\widehat{M}_k)$$
The Meyer-Itô formula converts the Stratonovich integration to Itô integration by adding a second-order correction term equal to the infinitesimal generator of the driving process applied to $f$. Because $B_t$ is intrinsic manifold Brownian motion, its generator is exactly $\frac{1}{2}\Delta_g$, where $\Delta_g$ is the Laplace-Beltrami operator on $\mathcal{M}$. The strict Itô differential is:
$$\mathrm{d}f(X_t) = \langle \text{grad}_g f,\, F(X_t)\rangle_g \mathrm{d}t + \frac{\delta^2}{2}\Delta_g f(X_t)\mathrm{d}t + \delta\mathrm{d}M_t$$
where $M_t$ is a local Itô martingale on $\mathbb{R}$. We apply this vectorially by setting $f$ to the local shape coordinates $s^j = \Phi^j(X)$.

*Step 2: Projection of the Deterministic Drift.*
We evaluate the first-order drift term $\langle \text{grad}_g \Phi^j,\, \lambda\,\text{Log}_{X_t}(\widehat{M}_k)\rangle_g$.
By Lemma 2.1 (Part 2), the continuous CBO drift is exactly the negative Riemannian gradient of the full-manifold potential $V(X) = \frac{\lambda}{2} d_g^2(X, \widehat{M}_k)$. Furthermore, $V(X)$ is strictly the pullback of the base space potential $\lambda \Psi(s) = \frac{\lambda}{2} d_{\mathcal{S}}^2(s, s_0)$ via the spectral map $\Phi$.
Because $\Phi$ is a Riemannian submersion, the inner product of the gradient of a pulled-back function with the gradient of a coordinate function exactly evaluates to the base-space gradient. Therefore, the deterministic drift rigorously projects to:
$$\mathrm{d}\Phi \left( \lambda \text{Log}_{X_t}(\widehat{M}_k) \right) = -\lambda \text{grad}_{g_{\mathcal{S}}} \Psi(s_t)$$

*Step 3: The Itô Correction and the Christoffel Symbols on the Principal Stratum.*
We must evaluate the second-order correction $\frac{\delta^2}{2}\Delta_g \Phi^j(X_t)$.

Because $\mathcal{S}_{shape}$ is a stratified space, $\Phi$ is only a smooth Riemannian submersion on the open, dense principal stratum $\mathcal{S}_{reg}$. We therefore restrict the application of O'Neill's differential formulas strictly to this regular domain.

For a Riemannian submersion generated by an isometric group action, the Laplace-Beltrami operator on the total space $\mathcal{M}_{reg}$ acting on invariant functions reduces exactly to the weighted Laplace-Beltrami operator on the regular quotient space $\mathcal{S}_{reg}$:
$$\Delta_g (\Phi^* f) = \Phi^* (\Delta_{\mathcal{S}} f)$$
where the weighted Laplacian on the base space is defined as:
$$\Delta_{\mathcal{S}} f = \Delta_{g_{\mathcal{S}}} f + \langle \text{grad}_{g_{\mathcal{S}}} \ln w, \text{grad}_{g_{\mathcal{S}}} f \rangle_{g_{\mathcal{S}}}$$

Applying this operator to the local coordinate function $f(s) = s^j$ within $\mathcal{S}_{reg}$, we evaluate both terms. For a curved Riemannian manifold, the standard Laplacian of a coordinate function exposes the intrinsic curvature via the Christoffel symbols $\Gamma^j_{ab}$ of the Levi-Civita connection:
$$\Delta_{g_{\mathcal{S}}} s^j = -g_{\mathcal{S}}^{ab} \Gamma^j_{ab}$$
The second term evaluates to the $j$-th component of the log-volume gradient, which is exactly the Generalized Dyson Drift (Definition 2.16):
$$\langle \text{grad}_{g_{\mathcal{S}}} \ln w, \text{grad}_{g_{\mathcal{S}}} s^j \rangle_{g_{\mathcal{S}}} = \left( \text{grad}_{g_{\mathcal{S}}} \ln w \right)^j = \mathcal{D}^j(s)$$
Thus, strictly within the principal stratum, the full Stratonovich-to-Itô geometric correction is:
$$\frac{\delta^2}{2}\Delta_g \Phi^j = -\frac{\delta^2}{2} g_{\mathcal{S}}^{ab} \Gamma^j_{ab} + \frac{\delta^2}{2}\mathcal{D}^j(s_t)$$

*Step 4: The Noise Martingale and Assembly of Manifold Brownian Motion.*
The term $\delta \mathrm{d}M_t^j$ represents the local martingale part. Because $\Phi$ is a Riemannian submersion, its differential $\mathrm{d}\Phi$ acts as an isometry between the horizontal subspace of $T_X\mathcal{M}$ and the tangent space $T_s\mathcal{S}_{shape}$. The projected martingale has a quadratic variation process governed exactly by the inverse metric tensor of the base space: $\mathrm{d}\langle M^i, M^j \rangle_t = g_{\mathcal{S}}^{ij} \mathrm{d}t$.

In local coordinates, the intrinsic Itô Brownian motion $\mathrm{d}B_t^{\mathcal{S}}$ on a curved manifold is defined exactly as the combination of this local martingale and the Christoffel connection drift:
$$\delta \mathrm{d}(B_t^{\mathcal{S}})^j = -\frac{\delta^2}{2} g_{\mathcal{S}}^{ab} \Gamma^j_{ab} \mathrm{d}t + \delta \mathrm{d}M_t^j$$
Therefore, the Christoffel term generated in Step 3 perfectly absorbs into the geometric definition of the base-space Brownian motion. This leaves the Generalized Dyson Drift $\frac{\delta^2}{2}\mathcal{D}(s_t)$ as the sole explicit external drift added to the system by the quotient projection.

*Step 5: Boundary Reflection.*
The orbit space $\mathcal{S}_{shape}$ is bounded by geometric singularities (where the orbit dimensions collapse). To ensure the process remains within the topologically valid closure $\overline{\mathcal{S}_{shape}}$, the aggregate local time process $\mathrm{d}L_t$ is added. By Definition 2.18, this process acts only on $\partial \mathcal{S}_{shape}$ to provide instantaneous normal reflection, ensuring the shape SDE remains mathematically well-posed.

*Step 6: Final Assembly.*
Combining the projected CBO drift (Step 2), the unabsorbed Dyson drift and the intrinsically assembled curved Brownian motion (Steps 3 and 4), and the boundary condition (Step 5), the full vectorial Itô SDE for the shape parameter is exactly:
$$\mathrm{d}s_t = \left( -\lambda \text{grad}_{g_{\mathcal{S}}} \Psi(s_t) + \frac{\delta^2}{2} \mathcal{D}(s_t) \right) \mathrm{d}t + \delta \, \mathrm{d}B_t^{\mathcal{S}} + \mathrm{d}L_t \qquad \blacksquare$$

*Verification via Intrinsic Fokker-Planck Analysis.*

To rigorously confirm that this SDE captures the manifold dynamics without losing information, we verify that the exact stationary density of the marginalized shape space $p_\infty(s) \propto w(s)\exp\left(-\frac{\lambda}{\delta^2}\|V(s)\|_g^2\right)$ is indeed the stationary solution to the intrinsic Fokker-Planck equation of our derived SDE on the curved orbit space $(\mathcal{S}_{shape}, g_{\mathcal{S}})$.

Let $\Psi(s) = \frac{1}{2}\|V(s)\|_g^2$ be the intrinsic half-squared distance. We can rewrite the target stationary density as $p_\infty(s) \propto w(s)\exp\left(-\frac{2\lambda}{\delta^2}\Psi(s)\right)$.

On a Riemannian manifold, the probability current $\mathbf{J}$ is a vector field defined by the deterministic drift flow minus the metric-induced diffusion gradient:
$$\mathbf{J} = p_\infty(s) \mathbf{F}(s) - \frac{\delta^2}{2} \text{grad}_{g_{\mathcal{S}}} p_\infty(s)$$
where the derived intrinsic SDE drift vector field is $\mathbf{F}(s) = -\lambda \text{grad}_{g_{\mathcal{S}}} \Psi(s) + \frac{\delta^2}{2} \mathcal{D}(s)$.

We compute the Riemannian gradient of the stationary density $p_\infty(s)$ using the logarithmic derivative identity $\text{grad}_{g_{\mathcal{S}}} p = p \, \text{grad}_{g_{\mathcal{S}}} \ln p$:
$$\text{grad}_{g_{\mathcal{S}}} \ln p_\infty(s) = \text{grad}_{g_{\mathcal{S}}} \left( \ln w(s) - \frac{2\lambda}{\delta^2} \Psi(s) \right) = \text{grad}_{g_{\mathcal{S}}} \ln w(s) - \frac{2\lambda}{\delta^2} \text{grad}_{g_{\mathcal{S}}} \Psi(s)$$

Recognizing that the Generalized Dyson Drift is exactly $\mathcal{D}(s) = \text{grad}_{g_{\mathcal{S}}} \ln w(s)$ (Definition 2.16), we substitute this back into the diffusion gradient:
$$\frac{\delta^2}{2} \text{grad}_{g_{\mathcal{S}}} p_\infty(s) = \frac{\delta^2}{2} p_\infty(s) \left( \mathcal{D}(s) - \frac{2\lambda}{\delta^2} \text{grad}_{g_{\mathcal{S}}} \Psi(s) \right) = p_\infty(s) \left( \frac{\delta^2}{2} \mathcal{D}(s) - \lambda \text{grad}_{g_{\mathcal{S}}} \Psi(s) \right)$$

We now substitute both the drift vector field $\mathbf{F}(s)$ and the diffusion gradient into the probability current vector field $\mathbf{J}$:
$$\mathbf{J} = p_\infty(s) \left( -\lambda \text{grad}_{g_{\mathcal{S}}} \Psi(s) + \frac{\delta^2}{2} \mathcal{D}(s) \right) - p_\infty(s) \left( -\lambda \text{grad}_{g_{\mathcal{S}}} \Psi(s) + \frac{\delta^2}{2} \mathcal{D}(s) \right) = \mathbf{0}$$

Because the probability current vector field $\mathbf{J}$ identically vanishes everywhere on the manifold, its Riemannian divergence is zero ($\text{div}_{g_{\mathcal{S}}} \mathbf{J} = 0$). This strictly satisfies the stationary intrinsic Fokker-Planck equation, definitively proving that the Shape SDE governed by the Generalized Dyson Drift produces the exact target stationary measure. $\blacksquare$

---

#### **Theorem 2.3: Equivalence of the Stationary Marginals**
Let $p_{\infty}(s)$ denote the exact stationary probability density function (with respect to the induced Riemannian volume measure $\mathrm{d}\text{vol}_{g_{\mathcal{S}}}$) of the autonomous Shape SDE defined in Theorem 2.2.

This density is mathematically equivalent to the true intrinsic stationary distribution of the full manifold SDE, marginalized exactly over the unobserved orientation fibers via the Generalized Integration Formula (Theorem 2.1). Specifically, the density is given by the closed-form analytical expression:
$$p_{\infty}(s) = \frac{1}{Z_{\mathcal{S}}} w(s) \exp\left( -\frac{\lambda}{\delta^2} \|V(s)\|_g^2 \right) \quad \text{for } s \in \mathcal{S}_{shape}$$
where $Z_{\mathcal{S}}$ is the restricted partition function ensuring normalization over the orbit space, and $w(s)$ is the Generalized Density Kernel representing the geometric volume of the Adjoint fibers. This guarantees that no statistical information regarding the ensemble's optimal shape is lost during the spectral projection.

> **Example (Stiefel Stationary Factorization):** Because the generalized Stiefel shape parameter is $s = (\Omega_{int}, \theta, V_{right})$ and the right-action $V_{right}$ cancels in the trace metric, the squared intrinsic distance decomposes geometrically as $\|V(s)\|_g^2 = \|\theta\|_2^2 + \frac{1}{2}\|\Omega_{int}\|_F^2$. Furthermore, the metric twist kernel $w(s)$ depends solely on the principal angles $\theta$. Thus, the Stiefel stationary density factorizes perfectly into independent angular and internal rotation marginals:
> $$p_\infty(\theta, \Omega_{int}) \propto \left[ w(\theta) \exp\left(-\frac{\lambda}{\delta^2} \|\theta\|_2^2\right) \right] \times \left[ \exp\left(-\frac{\lambda}{2\delta^2} \|\Omega_{int}\|_F^2\right) \right]$$

**Proof.**

The proof establishes the density $p_\infty(s)$ from two independent directions utilizing rigorous Riemannian differential geometry: first from above (pushing forward the global Gibbs measure via the Riemannian submersion) and second from below (solving the stationary Fokker-Planck equation intrinsically on the curved shape space).

*Step 1: Stationary Distribution of the Full Manifold SDE.*
The full CBO equation $\mathrm{d}X_t = \lambda\,\text{Log}_{X_t}(\widehat{M}_k)\,\mathrm{d}t + \delta\circ \mathrm{d}B_t$ defines a Langevin diffusion on the compact Riemannian manifold $(\mathcal{M}, g)$.
Let $V(X) = \frac{\lambda}{2}d_g^2(X, \widehat{M}_k)$ be the potential energy. By Lemma 2.1, the drift $\lambda\text{Log}_{X_t}(\widehat{M}_k) = -\text{grad}_g V(X_t)$.
The infinitesimal generator of this process is $\mathcal{L}_{\mathcal{M}} = -\langle \text{grad}_g V, \text{grad}_g \cdot \rangle_g + \frac{\delta^2}{2}\Delta_g$, where $\Delta_g$ is the Laplace-Beltrami operator on $\mathcal{M}$.
The stationary measure $\mu_\infty = \rho_\infty \mathrm{d}\mu_g$ must satisfy the adjoint equation $\mathcal{L}_{\mathcal{M}}^* \rho_\infty = 0$. Because $\Delta_g$ is self-adjoint with respect to the Riemannian volume measure $\mathrm{d}\mu_g$, and the divergence of a gradient flow generates the standard invariant Gibbs measure, the unique stationary density is:
$$\rho_\infty(X) = \frac{1}{Z}\exp\!\left(-\frac{2V(X)}{\delta^2}\right) = \frac{1}{Z}\exp\!\left(-\frac{\lambda}{\delta^2}d_g^2(X,\widehat{M}_k)\right)$$

*Step 2: The Full Distribution is a Class Function.*
By Lemma 2.2 (Part 3), the intrinsic distance function satisfies $d_g(X, \widehat{M}_k) = \|V(s)\|_g$ and is completely invariant under the Adjoint action of the stabilizer $H$. Therefore, $\rho_\infty(X) = \frac{1}{Z}\exp(-\frac{\lambda}{\delta^2}\|V(s)\|_g^2)$ is a class function, remaining strictly constant on each orientation fiber.

*Step 3: Marginalization via Pushforward Measure.*
We map the global measure $\mu_\infty$ down to the shape space $\mathcal{S}_{shape}$ via the spectral map $\Phi$. Because $\rho_\infty(X)$ is a class function, the pushforward measure $\Phi_*(\mu_\infty)$ evaluates to the integration of the fiber volumes over the base space. By Theorem 2.1 (The Generalized Integration Formula), the volume of the fiber at shape $s$ is exactly encoded by the density kernel $w(s)$.
Therefore, the pushed-forward stationary density with respect to the intrinsic volume measure $\mathrm{d}\text{vol}_{g_{\mathcal{S}}}$ on the orbit space is:
$$p(s) = \frac{C}{Z} w(s) \exp\!\left(-\frac{\lambda}{\delta^2}\|V(s)\|_g^2\right)$$
Normalizing this restricted distribution defines $Z_{\mathcal{S}}$ and yields our target $p_\infty(s)$.

*Step 4: Intrinsic Fokker-Planck on the Curved Shape Space.*
We must now verify that this exact density is the stationary solution to the autonomous Shape SDE derived in Theorem 2.2. The infinitesimal generator of the shape process on $(\mathcal{S}_{shape}, g_{\mathcal{S}})$ is:
$$\mathcal{L}_{\mathcal{S}} f = -\lambda \langle \text{grad}_{g_{\mathcal{S}}} \Psi, \text{grad}_{g_{\mathcal{S}}} f \rangle_{g_{\mathcal{S}}} + \frac{\delta^2}{2} \Delta_{\mathcal{S}} f$$
where $\Psi(s) = \frac{1}{2}\|V(s)\|_g^2$, and $\Delta_{\mathcal{S}} f = \frac{1}{w} \text{div}_{g_{\mathcal{S}}}(w \, \text{grad}_{g_{\mathcal{S}}} f)$ is the weighted Laplacian inherited from the submersion.
To find the stationary density $p(s)$, we require $\int_{\mathcal{S}_{shape}} (\mathcal{L}_{\mathcal{S}} f) p \, \mathrm{d}\text{vol}_{g_{\mathcal{S}}} = 0$ for all smooth, compactly supported test functions $f$.
Expanding the integral:
$$\int_{\mathcal{S}_{shape}} \left[ -\lambda \langle \text{grad}_{g_{\mathcal{S}}} \Psi, \text{grad}_{g_{\mathcal{S}}} f \rangle_{g_{\mathcal{S}}} + \frac{\delta^2}{2} \frac{1}{w} \text{div}_{g_{\mathcal{S}}}(w \, \text{grad}_{g_{\mathcal{S}}} f) \right] p \, \mathrm{d}\text{vol}_{g_{\mathcal{S}}} = 0$$
We apply Riemannian integration by parts (Green's First Identity) to the second term. Assuming boundary fluxes vanish (see Remark below), we shift the divergence off the test function $f$:
$$\int_{\mathcal{S}_{shape}} \frac{p}{w} \text{div}_{g_{\mathcal{S}}}(w \, \text{grad}_{g_{\mathcal{S}}} f) \, \mathrm{d}\text{vol}_{g_{\mathcal{S}}} = - \int_{\mathcal{S}_{shape}} \left\langle \text{grad}_{g_{\mathcal{S}}} \left( \frac{p}{w} \right), w \, \text{grad}_{g_{\mathcal{S}}} f \right\rangle_{g_{\mathcal{S}}} \mathrm{d}\text{vol}_{g_{\mathcal{S}}}$$
Applying the quotient rule for gradients: $\text{grad}_{g_{\mathcal{S}}} (p/w) = \frac{w \text{grad}_{g_{\mathcal{S}}} p - p \text{grad}_{g_{\mathcal{S}}} w}{w^2}$. Substituting this back, the $w$ cancels, yielding:
$$- \int_{\mathcal{S}_{shape}} \left\langle \text{grad}_{g_{\mathcal{S}}} p - p \, \text{grad}_{g_{\mathcal{S}}} (\ln w), \text{grad}_{g_{\mathcal{S}}} f \right\rangle_{g_{\mathcal{S}}} \mathrm{d}\text{vol}_{g_{\mathcal{S}}}$$
For the total integral to be zero for any $f$, the vector field dotted with $\text{grad}_{g_{\mathcal{S}}} f$ must vanish almost everywhere:
$$-\lambda p \, \text{grad}_{g_{\mathcal{S}}} \Psi - \frac{\delta^2}{2} \left( \text{grad}_{g_{\mathcal{S}}} p - p \, \text{grad}_{g_{\mathcal{S}}} \ln w \right) = \mathbf{0}$$
Dividing by $\frac{\delta^2}{2} p$ and rearranging to solve for the gradient of the log-density:
$$\text{grad}_{g_{\mathcal{S}}} \ln p = \text{grad}_{g_{\mathcal{S}}} \ln w - \frac{2\lambda}{\delta^2} \text{grad}_{g_{\mathcal{S}}} \Psi$$
Integrating this equation along any curve in the connected principal stratum of the shape space yields the exact scalar relation:
$$\ln p(s) = \ln w(s) - \frac{2\lambda}{\delta^2} \Psi(s) + C$$
Exponentiating both sides and substituting $\Psi(s) = \frac{1}{2}\|V(s)\|_g^2$ perfectly recovers the target distribution:
$$p(s) \propto w(s) \exp\left( -\frac{\lambda}{\delta^2} \|V(s)\|_g^2 \right)$$
Since the Fokker-Planck equation on the curved orbit space produces the exact same density as the pushforward of the full manifold Gibbs measure, the theorem is proven. $\square$

---

### Subsection 2.4: Spectral Sampling and Global Reconstruction

Having successfully projected the manifold dynamics into the Weyl chamber to obtain the exact stationary marginals (Theorem 2.3), we now invert the procedure to generate globally valid samples on the target space. This subsection defines the reconstruction phase, where the sampled "shape" (the principal angles) is lifted back to the full manifold by reintroducing the unobserved "orientation" components.

#### **Definition 2.20: The Lie-Exponential Lift**
To map an element from the flat Maximal Abelian Subspace $\mathfrak{a}$ back to the Lie group $G$, we utilize the **Lie-Exponential Lift**. Given a vector of principal angles $\theta \in \mathcal{W}$, we first embed these angles into the canonical skew-symmetric matrix representation $A(\theta) \in \mathfrak{a}$. The lift to the group $G$ is then executed via the Lie group exponential map:
$$\Theta = \exp(A(\theta))$$
This operation strictly constructs a block-diagonal rotation matrix in $G$, representing the displacement along the geodesics specified by the principal angles.

#### **Definition 2.21: Isotropic Fiber Conjugation**
The spectral map $\Phi$ explicitly discarded the orientation of the reference frame by factoring out the stabilizer $H$. To reconstruct a valid sample on the full manifold that respects the isotropic nature of the Brownian noise while preserving the sampled distance to the consensus point, we must conjugate the lifted element $\Theta$ strictly by an orientation matrix drawn from the **isotropy subgroup** $H$. Let $h \sim \text{Haar}(H)$ be a group element drawn from the unique normalized Haar measure on $H$. The **conjugated group element** is defined as:
$$\tilde{g} = h \Theta h^\top$$
Because the reductive complement $\mathfrak{m}$ is $Ad(H)$-invariant (Definition 2.9), this conjugation uniformly scatters the block-diagonal planar rotations strictly across the valid tangent space $T_{\widehat{M}_k}\mathcal{M}$. This perfectly recovers the isotropic structure of the original stochastic process without rotating the trajectory into the stabilizer algebra $\mathfrak{h}$, thereby rigorously preserving the intrinsic Riemannian distance $\|\theta\|_2$.

#### **Definition 2.22: The Quotient Projection Operator and Stabilizer Invariance**
Let $\mathcal{M} = G/H$ be a homogeneous space, and let $\widehat{M}_k \in \mathcal{M}$ be the consensus point, representing the canonical origin (the identity coset $eH$). The group $G$ acts transitively on $\mathcal{M}$ strictly via the **left-action**: $\mu(g, X) = g \cdot X$ for any $g \in G$ and $X \in \mathcal{M}$.

By definition, the stabilizer $H \subset G$ is the isotropy subgroup at $\widehat{M}_k$. Therefore, for any element $h \in H$, its left-action on the consensus point leaves the point mathematically invariant:
$$h \cdot \widehat{M}_k = \widehat{M}_k$$

When mapping the canonical shape embedding $\Theta = \exp(V(s)) \in G$ back to the manifold, we apply an isotropic fiber conjugation using an element $h \in H$, generating the global transformation $\tilde{g} = h \Theta h^{-1}$. For orthogonal groups, the inverse is exactly the transpose, so $h^{-1} = h^\top$.

The application of this conjugated element to the consensus point evaluates via the associativity of the left group action. Importantly, the operation $h^\top \cdot \widehat{M}_k$ constitutes a **left-action** by the inverse element $h^\top$, despite $h^\top$ appearing on the right side of the exponential term in the matrix product:
$$\tilde{g} \cdot \widehat{M}_k = (h \Theta h^\top) \cdot \widehat{M}_k = h \cdot \left( \Theta \cdot \left( h^\top \cdot \widehat{M}_k \right) \right)$$
Because $h^\top \in H$, its strict left-action stabilizes the point ($h^\top \cdot \widehat{M}_k = \widehat{M}_k$). The projection rigorously collapses the inverse element, leaving $h \cdot (\Theta \cdot \widehat{M}_k)$, which perfectly preserves the geometry of the canonical embedding while isotropically randomizing the unobserved frame.

> **Example 2.22 (The Stiefel Quotient Projection):** For the Stiefel manifold $V(n, k)$, the group is $G = SO(n)$ and the stabilizer is $H \cong SO(n-k)$. The consensus point $\widehat{M}_k$ is a fixed $n \times k$ orthogonal frame. The quotient projection is executed via matrix multiplication using a block-diagonal stabilizer matrix $h \in H$:
> $$X = (h \exp(A(\theta)) h^\top) \widehat{M}_k$$
> Because $h^\top \widehat{M}_k = \widehat{M}_k$, the multiplication safely randomizes the unobserved dimensions within the $SO(n-k)$ fiber while outputting a valid $n \times k$ frame $X \in V(n, k)$ situated at the exact intrinsic distance $\|\theta\|_2$ from the consensus point.

#### **Lemma 2.3: Integrity of the Lie-Exponential Lift**
Let $\theta \in \mathcal{W}$ be a vector of principal angles drawn from the stationary marginal $p_{\infty}(\theta)$. The composite operation of Lie-exponential lifting, Haar conjugation, and quotient projection circumvents the mathematical singularities inherent in the direct Riemannian exponential map $\text{Exp}_{\widehat{M}_k}$ (Definition 2.5). Specifically, this procedure guarantees that the resulting sample $X \in \mathcal{M}$ is globally valid, analytically exact, and completely avoids the calculation of undefined gradients at the cut locus.

**Proof.**

*Part 1: Global Validity and Strict Distance Preservation.*

We must prove that the generated sample $\tilde{X} = \tilde{g} \cdot \widehat{M}_k$, where $\tilde{g} = h \Theta h^{-1}$, $\Theta = \exp(V(s))$, and $h \sim \text{Haar}(H)$, is a globally valid point on the manifold $\mathcal{M} = G/H$ and that its intrinsic Riemannian distance to the consensus point $\widehat{M}_k$ strictly equals the norm of the sampled shape embedding $\|V(s)\|_g$.

*Step 1.1: Global Topological Validity.*
The generalized shape parameter $s \in \mathcal{S}_{shape}$ is mapped to the canonical tangent vector $V(s) \in \mathfrak{m}$. Because $\mathfrak{m}$ is a linear subspace of the full Lie algebra $\mathfrak{g}$, the Lie group exponential map rigorously generates a valid global group element: $\Theta = \exp(V(s)) \in G$.
Since the conjugation element $h$ is drawn from the stabilizer subgroup $H \subset G$, its inverse $h^{-1}$ is also in $H \subset G$. By the closure of the group $G$ under multiplication, $\tilde{g} = h \exp(V(s)) h^{-1} \in G$.
Because $G$ acts transitively and smoothly on the homogeneous space $\mathcal{M}$, the left action of any group element on the consensus point generates a globally valid point on the manifold. Thus, $\tilde{X} = \tilde{g} \cdot \widehat{M}_k \in \mathcal{M}$ is topologically well-defined.

*Step 1.2: Strict Distance Preservation via Stabilizer Action.*
We now evaluate the intrinsic Riemannian distance $d_g(\tilde{X}, \widehat{M}_k)$.
Substitute the definition of $\tilde{X}$:
$$d_g(\tilde{X}, \widehat{M}_k) = d_g\left( (h \exp(V(s)) h^{-1}) \cdot \widehat{M}_k, \widehat{M}_k \right)$$
By the fundamental definition of the stabilizer subgroup $H$, any element $h \in H$ fixes the consensus point (the origin of the homogeneous space). Therefore, the left-action of the inverse element evaluates strictly to: $h^{-1} \cdot \widehat{M}_k = \widehat{M}_k$.
Substituting this topological identity into the distance function yields:
$$d_g(\tilde{X}, \widehat{M}_k) = d_g\left( h \exp(V(s)) \cdot \widehat{M}_k, \widehat{M}_k \right)$$
Because the homogeneous space $\mathcal{M} = G/H$ is equipped with a $G$-invariant (specifically, left-invariant) Riemannian metric $g$, the left action of any group element acts as a strict global isometry. We apply the left action of $h^{-1}$ to both arguments of the distance function:
$$d_g(\tilde{X}, \widehat{M}_k) = d_g\left( h^{-1} h \exp(V(s)) \cdot \widehat{M}_k, h^{-1} \cdot \widehat{M}_k \right)$$
This reduces to:
$$d_g(\tilde{X}, \widehat{M}_k) = d_g\left( \exp(V(s)) \cdot \widehat{M}_k, \widehat{M}_k \right)$$
On a reductive homogeneous space equipped with a bi-invariant metric, the Lie group exponential map applied to a vector in the reductive complement $\mathfrak{m}$ identically matches the Riemannian exponential map at the origin: $\text{Exp}_{\widehat{M}_k}(V(s)) = \exp(V(s)) \cdot \widehat{M}_k$.
By the definition of the Riemannian exponential map, the intrinsic distance along the geodesic is exactly the metric norm of the tangent vector:
$$d_g(\tilde{X}, \widehat{M}_k) = \|V(s)\|_g$$
This rigorously proves that while the conjugation by $h \sim \text{Haar}(H)$ uniformly randomizes the unobserved orientation fibers, it mathematically cannot alter the intrinsic distance prescribed by the shape sample $s$. $\square_1$

*Part 2: Analytical Exactness — No Approximation Error.*

A direct approach to sampling from $\rho_\infty(X) \propto \exp(-\frac{\lambda}{\delta^2}d_g^2(X,\widehat{M}_k))$ would attempt to use the Riemannian exponential map $\text{Exp}_{\widehat{M}_k}(v)$ to push samples $v$ from the tangent space $T_{\widehat{M}_k}\mathcal{M}$ onto $\mathcal{M}$. However, this requires computing the pushforward density through $\text{Exp}_{\widehat{M}_k}$, which involves the Riemannian Jacobian — a determinant factor that depends on the sectional curvatures and is analytically intractable in general. Any sampling scheme based on this approach would either require approximation or truncation of the Jacobian.

The Lie-Exponential Lift avoids this entirely. By working at the group level, the lift $A(\theta) \mapsto \exp(A(\theta))$ is the Lie group exponential (Definition 2.4), not the Riemannian exponential $\text{Exp}_{\widehat{M}_k}$. These coincide in direction (both follow the same geodesic through the identity of $G$), but the Lie group exponential is always globally smooth and requires no Jacobian correction: the change of variables has already been exactly accounted for by the Weyl kernel $w(\theta)$ in the density $p_\infty(\theta)$ (Theorem 2.3). The Weyl Integration Formula (Theorem 2.1) proves that $w(\theta)$ exactly encodes the Jacobian of this change of variables. Sampling $\theta \sim p_\infty$ and lifting via $\exp(A(\theta))$ therefore incurs no approximation error by construction. $\checkmark$

*Part 3: Cut Locus Avoidance — Bypassing the Singularities of $\text{Exp}_{\widehat{M}_k}$.*

By Definition 2.5, the Riemannian logarithm $\text{Log}_{\widehat{M}_k}$ is only a smooth diffeomorphism within the injectivity ball $B_{\text{inj}(\mathcal{M})}(0) \subset T_{\widehat{M}_k}\mathcal{M}$. Points on the cut locus — corresponding to $\theta \in \partial\mathcal{W}$ by Lemma 2.2 Part 2 — are outside the injectivity ball, and $\text{Exp}_{\widehat{M}_k}$ is neither injective nor smoothly invertible there. Any algorithm relying on $\text{Exp}_{\widehat{M}_k}$ or $\text{Log}_{\widehat{M}_k}$ must either restrict to the injectivity ball (truncating the support of $p_\infty$) or handle the singularity analytically (which is generally not possible in closed form).

The Lie-Exponential Lift bypasses this problem because the Lie group exponential $\exp: \mathfrak{g} \to G$ is globally defined for all $A(\theta)$, regardless of the magnitude of $\theta$. For any $\theta \in \mathcal{W}$ (including those near but interior to $\partial\mathcal{W}$), $\exp(A(\theta))$ is a well-defined smooth map. The only requirement for the procedure to give a valid sample in $\mathcal{M}$ is that $\theta$ lies in $\overline{\mathcal{W}}$, which is guaranteed by the spectral map (Definition 2.12) and the fact that $p_\infty$ has support exactly on $\mathcal{W}$ (the boundary $\partial\mathcal{W}$ has measure zero). Therefore, the procedure never encounters undefined gradients or singularities of $\text{Exp}_{\widehat{M}_k}$. $\checkmark$ $\square$

---

### **Algorithm 2.1: Exact Spectral Generative Sampling on Compact Homogeneous Spaces**

**Objective:** Generate independent and identically distributed (i.i.d.) samples $X \in \mathcal{M} = G/H$ strictly from the global invariant stationary measure of the CBO process, $\mu_\infty \propto \exp(-\frac{\lambda}{\delta^2} d_g^2(X, \widehat{M}_k))$, without any online stochastic simulation or numerical grid approximations.

**Inputs:**
* $\widehat{M}_k \in \mathcal{M}$: The current weighted Fréchet mean (the consensus point).
* $\lambda > 0$: The deterministic consensus drift rate.
* $\delta > 0$: The isotropic manifold noise diffusion coefficient.

---

#### **Phase I: Offline Precomputation (The Exact Conditional Splines)**
*This phase is executed exactly once prior to the online generation, entirely absorbing the computational burden of the coupled metric geometry.*

1. **Define the Abstract Joint Measure:** Construct the exact, coupled stationary density for the active shape parameters $s$ on the curved orbit space $\mathcal{S}_{shape}$. For an arbitrary compact homogeneous space $\mathcal{M} = G/H$, the volume distortion is strictly governed by its associated restricted root system $\Phi^+$. Utilizing the Generalized Weyl Density Kernel $w(s)$, the mathematically exact joint density is:
   $$p_\infty(s) \propto w(s) \exp\left( -\frac{\lambda}{\delta^2} \|V(s)\|_g^2 \right)$$
   defined strictly on the fundamental domain associated with the specific geometric space.

2. **Sequential Conditional Marginalization:** To preserve the exact geometric coupling without requiring online stochastic simulation, decompose the joint distribution using the chain rule of probability for the multidimensional shape parameter $s = (s_1, \dots, s_k)$:
   $$p_\infty(s_1, \dots, s_k) = p(s_1) p(s_2 \mid s_1) \dots p(s_k \mid s_1, \dots, s_{k-1})$$
   Using high-precision offline numerical integration, compute the family of conditional Cumulative Distribution Functions (CDFs):
   $$F_i(x \mid s_1, \dots, s_{i-1}) = \int_0^x p(\tau \mid s_1, \dots, s_{i-1}) \mathrm{d}\tau$$

3. **Construct the Inverse-CDF Splines:** Invert this family of functions and store them as a set of continuous, multi-dimensional interpolating splines: $F_i^{-1}(u_i \mid s_{1:i-1})$.

---

#### **Phase II: Online Generation (The Stochastic Update)**
*This phase is executed instantaneously for each required sample, operating entirely via algebraically exact Lie group operations and spline evaluations.*

1. **Draw the Generalized Shape ($s \in \mathcal{S}_{shape}$):**
   Draw a vector of independent uniform variables $u \sim \mathcal{U}[0, 1]^k$. Evaluate the shape coordinates sequentially using the precomputed conditional splines:
   $$s_1 = F_1^{-1}(u_1)$$
   $$s_2 = F_2^{-1}(u_2 \mid s_1)$$
   $$\dots$$
   $$s_k = F_k^{-1}(u_k \mid s_1, \dots, s_{k-1})$$
   *This mathematically enforces the strict geometric volume repulsion without any temporal discretization or rejection steps.*

2. **Construct the Canonical Embedding ($V(s)$):**
   Embed the sampled shape dimensions into the canonical tangent vector $V(s)$. This vector must be mapped strictly into the reductive complement $\mathfrak{m}$, where the Lie algebra admits the canonical reductive decomposition $\mathfrak{g} = \mathfrak{h} \oplus \mathfrak{m}$.

3. **Map to the Lie Group ($\Theta$):**
   Map the canonical tangent vector back to the continuous group $G$ using the Lie group exponential map:
   $$\Theta = \exp(V(s)) \in G$$

4. **Isotropic Fiber Conjugation ($h \in G$):**
   Draw a random orientation frame strictly from the unobserved stabilizer subgroup according to its normalized Haar measure: $h \sim \text{Haar}(H)$.
   Map this element into the ambient continuous group $G$ using the canonical topological inclusion map $\iota: H \hookrightarrow G$.
   Conjugate the canonical shape element $\Theta \in G$ to uniformly randomize the unobserved orientation fibers. Because the metric is $G$-invariant, this operation strictly preserves the intrinsic Riemannian distance:
   $$\tilde{g} = h \Theta h^{-1}$$

5. **Geometric Recombination ($X \in \mathcal{M}$):**
   Project the globally valid group element $\tilde{g} \in G$ onto the homogeneous quotient space via the canonical left-action $\mu: G \times \mathcal{M} \to \mathcal{M}$ applied to the consensus point:
   $$X = \tilde{g} \cdot \widehat{M}_k$$
   Return the rigorously assembled, mathematically exact sample $X \in \mathcal{M}$.

### **Algorithm 2.1: Exact Spectral Generative Sampling on the Stiefel Manifold $V(n, k)$**

**Objective:** Generate independent and identically distributed (i.i.d.) samples $X \in V(n, k)$ strictly from the global invariant stationary measure of the CBO process, $\mu_\infty \propto \exp(-\frac{\lambda}{\delta^2} d_g^2(X, \widehat{M}_k))$, without any online stochastic simulation or spatial/temporal grid approximations.

**Assumptions:** Let $n \ge 2k$. The manifold is represented as the quotient space $SO(n)/SO(n-k)$.
**Inputs:**
* $\widehat{M}_k \in V(n, k)$: The current weighted Fréchet mean (the $n \times k$ consensus frame).
* $\lambda > 0$: The deterministic consensus drift rate.
* $\delta > 0$: The isotropic manifold noise diffusion coefficient.

---

#### **Phase I: Offline Precomputation (The Exact Conditional Splines)**
*This phase is executed exactly once to algebraically resolve the Dyson gas eigenvalue repulsion induced by the Stiefel geometry.*

1. **Define the Joint Stiefel Measure:** Construct the exact coupled stationary density for the principal angles $\theta$ on the curved orbit space. Because the Stiefel manifold is defined over the real orthogonal group, the internal Type $D_k$ roots strictly have a geometric multiplicity of $m_\alpha = 1$. Utilizing the full-angle Riemannian differential for the symmetric base space projection, the mathematically exact density is:
   $$p_\infty(\theta_1, \dots, \theta_k) \propto \left( \prod_{i=1}^k \sin^{n-2k}(\theta_i) \prod_{i < j}^k \sin(\theta_i - \theta_j) \sin(\theta_i + \theta_j) \right) \exp\left( -\frac{\lambda}{\delta^2} \|\theta\|_2^2 \right)$$
   defined strictly on the ordered Weyl chamber: $\frac{\pi}{2} \ge \theta_1 > \theta_2 > \dots > \theta_k \ge 0$.
2. **Sequential Conditional Marginalization:** Decompose the joint distribution using the chain rule of probability to prepare for discretization-free sequential sampling:
   $$p_\infty(\theta_1, \dots, \theta_k) = p(\theta_1) p(\theta_2 \mid \theta_1) \dots p(\theta_k \mid \theta_1, \dots, \theta_{k-1})$$
   Using high-precision offline numerical integration, compute the family of conditional Cumulative Distribution Functions (CDFs):
   $$F_i(x \mid \theta_1, \dots, \theta_{i-1}) = \int_0^x p(\tau \mid \theta_1, \dots, \theta_{i-1}) \mathrm{d}\tau$$
3. **Construct the Inverse-CDF Splines:** Invert this family of functions and store them as a set of continuous, multi-dimensional interpolating splines: $F_i^{-1}(u_i \mid \theta_{1:i-1})$.

---

#### **Phase II: Online Generation (The Stochastic Update)**
*Executed instantaneously for each sample, utilizing pure linear algebra and spline evaluations.*

1. **Draw the Full Coupled Shape $s = (\Omega_{int}, \theta, V_{right})$:**
    * **The Principal Angles ($\theta \in \mathbb{R}^k$):** Draw independent uniforms $u \sim \mathcal{U}[0, 1]^k$. Evaluate the exact angles sequentially to rigorously enforce the geometric $\sin^2$ volume repulsion:
      $$\theta_1 = F_1^{-1}(u_1)$$
      $$\theta_2 = F_2^{-1}(u_2 \mid \theta_1)$$
      $$\dots$$
      $$\theta_k = F_k^{-1}(u_k \mid \theta_1, \dots, \theta_{k-1})$$
    * **The Internal Rotation ($\Omega_{int} \in \mathfrak{so}(k)$):** Construct a $k \times k$ skew-symmetric matrix. Sample independent normal variables $z_{ij} \sim \mathcal{N}\left(0, \frac{\delta^2}{2\lambda}\right)$ for the strictly lower-triangular entries ($i > j$), set $\Omega_{j,i} = -z_{ij}$ for the upper triangle, and zeros on the diagonal.
    * **The Right-Action Twist ($V_{right} \in SO(k)$):** Generate a $k \times k$ matrix $Z$ with i.i.d. standard normal entries. Compute the QR decomposition $Z = QR$. Force uniqueness via the diagonal sign matrix $\Lambda = \text{diag}(\text{sgn}(R_{1,1}), \dots, \text{sgn}(R_{k,k}))$ and define $Q' = Q\Lambda$. To strictly restrict from $O(k)$ to the Special Orthogonal group $SO(k)$, compute $\det(Q')$. If $\det(Q') = -1$, multiply the first column of $Q'$ by $-1$. Set $V_{right} = Q'$.

2. **Construct the Canonical Embedding $V(s) \in \mathfrak{so}(n)$:**
   Embed the active shape dimensions into the full $n \times n$ Lie algebra. Define the $(n-k) \times k$ rectangular padding matrix:
   $$\Sigma(\theta) = \begin{bmatrix} \text{diag}(\theta) \\ \mathbf{0}_{(n-2k) \times k} \end{bmatrix}$$
   Construct the skew-symmetric block matrix:
   $$V(s) = \begin{bmatrix} \Omega_{int} & -V_{right} \Sigma(\theta)^\top \\ \Sigma(\theta) V_{right}^\top & \mathbf{0}_{n-k} \end{bmatrix}$$

3. **Map to the Lie Group ($\Theta \in SO(n)$):**
   Map the canonical tangent vector to the continuous orthogonal group using the standard matrix exponential:
   $$\Theta = \exp(V(s))$$

4. **Isotropic Fiber Conjugation ($h \in SO(n)$):**
   Draw a random orientation frame strictly from the unobserved stabilizer subgroup $SO(n-k)$.
    * Generate an $(n-k) \times (n-k)$ matrix of i.i.d. standard normals, perform the QR decomposition, enforce positive diagonals via $\Lambda$, check the determinant, and flip the first column if the determinant is -1 to yield a valid rotation $h_{sub} \in SO(n-k)$.
    * Embed this sub-matrix into the full $n \times n$ identity structure to create the stabilizer element $h$:$$h = \begin{bmatrix} I_k & \mathbf{0}_{k \times (n-k)} \\ \mathbf{0}_{(n-k) \times k} & h_{sub} \end{bmatrix}$$
    * Conjugate the canonical shape matrix. Because we are implicitly operating at the canonical origin $E_k = [I_k \;\; \mathbf{0}]^\top$, this conjugation correctly randomizes the unobserved orientation fibers while mathematically locking in the exact intrinsic distance:$$\tilde{g} = h \Theta h^\top$$

5. **Geometric Recombination ($X \in V(n, k)$):**
   Because the sample $\tilde{g}$ is geometrically anchored to the canonical origin, it must be rigidly transported to the consensus point $\widehat{M}_k$.
   * Compute the canonical sample: $X_{canonical} = \tilde{g} E_k$.
   * Construct an orthogonal completion matrix $Q \in SO(n)$ such that its first $k$ columns are exactly $\widehat{M}_k$. This is achieved by finding an orthonormal basis for the null space of $\widehat{M}_k^\top$ to form $\widehat{M}_k^\perp \in \mathbb{R}^{n \times (n-k)}$, constructing $Q = [\widehat{M}_k \mid \widehat{M}_k^\perp]$, and flipping the sign of the last column if $\det(Q) = -1$.
   * Project the canonical sample onto the target geometry via the left-action of $Q$:$$X = Q X_{canonical}$$Return the rigorously assembled, mathematically exact sample $X$.

#### **Theorem 2.4: Global Exactness of the Spectral Sampler**
Let $\mathcal{M} = G/H$ be a compact homogeneous space equipped with a $G$-invariant Riemannian metric $g$. The continuous generative procedure defined in Algorithm 2.1 produces independent and identically distributed (i.i.d.) samples $X \in \mathcal{M}$ that are drawn exactly from the global intrinsic stationary distribution of the full manifold CBO process:
$$\mu_\infty(\mathrm{d}X) = \frac{1}{Z} \exp\left( -\frac{\lambda}{\delta^2} d_g^2(X, \widehat{M}_k) \right) \mathrm{d}\mu_g(X)$$
where $\mathrm{d}\mu_g$ is the Riemannian volume measure on $\mathcal{M}$. The algorithm achieves zero discretization error in the spatial domain, perfectly reproducing the full topological geometry of the manifold without dimensional collapse.

---

**Proof.**

*Step 1: Measure Factorization via Riemannian Submersion.*
Let $\Phi: \mathcal{M} \to \mathcal{S}_{shape}$ be the Riemannian submersion onto the generalized shape space. By the Principal Orbit Type Theorem, $\mathcal{S}_{shape}$ contains an open, dense principal stratum $\mathcal{S}_{reg}$ (the regular elements). The singular strata (the boundaries $\partial \mathcal{S}_{shape}$) have strictly lower dimensionality and therefore possess zero measure with respect to the induced Riemannian volume measure $\mathrm{d}\text{vol}_{g_{\mathcal{S}}}$. We restrict our integration to $\mathcal{S}_{reg}$ almost everywhere (a.e.).

By the generalized coarea formula for Riemannian submersions, any integrable class function $f(X) = \tilde{f}(\Phi(X))$ satisfies:
$$\int_{\mathcal{M}} f(X) \mathrm{d}\mu_g(X) = \int_{\mathcal{S}_{shape}} \tilde{f}(s) \text{vol}_g(\Phi^{-1}(s)) \mathrm{d}\text{vol}_{g_{\mathcal{S}}}(s)$$
By Theorem 2.1, the fiber volume is exactly the Generalized Weyl Density Kernel: $\text{vol}_g(\Phi^{-1}(s)) = w(s)$.
By Lemma 2.2, the density $\rho_\infty(X) \propto \exp(-\frac{\lambda}{\delta^2} d_g^2(X, \widehat{M}_k))$ is a class function strictly equal to $\tilde{\rho}_\infty(s) \propto \exp(-\frac{\lambda}{\delta^2} \|V(s)\|_g^2)$.
Thus, the global measure $\mu_\infty$ pushes forward via $\Phi$ to the base measure $\nu_\infty$ on $\mathcal{S}_{shape}$:
$$\nu_\infty(\mathrm{d}s) = \frac{1}{Z_{\mathcal{S}}} w(s) \exp\left( -\frac{\lambda}{\delta^2} \|V(s)\|_g^2 \right) \mathrm{d}\text{vol}_{g_{\mathcal{S}}}(s)$$

*Step 2: The Algorithmic Product Measure.*
Algorithm 2.1 draws the complete shape parameter $s \in \mathcal{S}_{shape}$ precisely from the marginal density defined above: $s \sim \nu_\infty(\mathrm{d}s)$.
Independently, the algorithm draws an orientation frame $h \in H$ from the normalized Haar measure $\mathrm{d}\mu_H$ on the compact stabilizer group $H$.
Therefore, the algorithm rigorously generates a joint sample $(s, h)$ from the product probability measure on the Cartesian space $\mathcal{S}_{shape} \times H$:
$$\mathbb{P}_{algo}(\mathrm{d}s, \mathrm{d}h) = \nu_\infty(\mathrm{d}s) \otimes \mathrm{d}\mu_H(h)$$

*Step 3: The Geometric Recombination Map.*
We define the algorithmic recombination map $\Psi: \mathcal{S}_{shape} \times H \to \mathcal{M}$ as:
$$\Psi(s, h) = \left( h \exp(V(s)) h^{-1} \right) \cdot \widehat{M}_k$$
By Lemma 2.3, the right-side conjugation collapses ($h^{-1} \cdot \widehat{M}_k = \widehat{M}_k$), yielding the left-action $\Psi(s, h) = h \cdot \text{Exp}_{\widehat{M}_k}(V(s))$.
We must prove that the pushforward of the algorithmic measure via this map, $\Psi_*(\mathbb{P}_{algo})$, exactly equals the target measure $\mu_\infty$.

*Step 4: Pushforward onto the Adjoint Fibers.*
Fix a regular shape $s \in \mathcal{S}_{reg}$. The map $\Psi_s(h) = h \cdot \text{Exp}_{\widehat{M}_k}(V(s))$ surjectively maps the group $H$ onto the specific fiber $\mathcal{F}_s = \Phi^{-1}(s) \subset \mathcal{M}$.
Because $H$ acts by strict isometries on $\mathcal{M}$ (the metric $g$ is $G$-invariant), the Riemannian volume measure on the submanifold $\mathcal{F}_s$ is inherently $H$-invariant.
By the uniqueness of invariant measures on compact homogeneous spaces (Haar's Theorem applied to quotients), pushing forward the Haar measure of $H$ via the transitive action $\Psi_s$ yields the unique, uniform Riemannian probability measure on the fiber $\mathcal{F}_s$:
$$(\Psi_s)_*(\mathrm{d}\mu_H) = \frac{1}{w(s)} \mathrm{d}\text{vol}_{\mathcal{F}_s}$$

*Step 5: Global Equivalence via the Disintegration Theorem.*
By the Disintegration Theorem of measures, the global Riemannian measure $\mu_\infty(\mathrm{d}X)$ on $\mathcal{M}$ can be uniquely written as the integration of its conditional fiber measures against its marginal base measure.
Substituting our results from Step 1 (the marginal base measure $\nu_\infty$) and Step 4 (the uniform conditional fiber measure), the pushforward of the joint algorithmic distribution is:
$$\Psi_*(\mathbb{P}_{algo}(\mathrm{d}s, \mathrm{d}h)) = \int_{\mathcal{S}_{shape}} \left( \frac{1}{w(s)} \mathrm{d}\text{vol}_{\mathcal{F}_s}(X) \right) \nu_\infty(\mathrm{d}s)$$
Substitute $\nu_\infty(\mathrm{d}s)$:
$$= \int_{\mathcal{S}_{shape}} \left( \frac{1}{w(s)} \mathrm{d}\text{vol}_{\mathcal{F}_s}(X) \right) \left[ \frac{1}{Z_{\mathcal{S}}} w(s) \exp\left( -\frac{\lambda}{\delta^2} \|V(s)\|_g^2 \right) \mathrm{d}\text{vol}_{g_{\mathcal{S}}}(s) \right]$$
The $w(s)$ terms algebraically cancel. Because the exponential term is a class function, it distributes evenly across the fiber $\mathcal{F}_s$. Reassembling the fiber and base volume measures via the coarea formula reconstructs the full global Riemannian volume measure $\mathrm{d}\mu_g(X)$:
$$= \frac{1}{Z_{\mathcal{S}}} \exp\left( -\frac{\lambda}{\delta^2} d_g^2(X, \widehat{M}_k) \right) \mathrm{d}\mu_g(X) \equiv \mu_\infty(\mathrm{d}X)$$
This proves that the algorithmic recombination mathematically synthesizes the exact global measure, verifying zero spatial discretization error. $\square$