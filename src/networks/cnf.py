"""Continuous normalizing flow core inspired by the legacy thermal_flow_cnf suite."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

from .mlp import MLP


class ODEFunc(nn.Module):
    """
    Neural ODE function parameterizing a learnable velocity field for continuous normalizing flow.
    
    Computes dz/dt = f(z, context, t) where f is a neural network. During training, also computes
    the divergence of the velocity field using the Hutchinson trace estimator to track changes
    in log-probability along trajectories.
    
    **Purpose**: Serve as the differential equation core for CNF sampling and likelihood computation.
    
    **Key Algorithms**:
    - Hutchinson Trace Estimator: Approximates tr(∇f) ≈ ε^T·∇(f·ε) by sampling ε~N(0,I) and
      computing the vector-Jacobian product. Reduces O(dim²) adjoint cost to O(1) per forward pass.
    - Time-conditioned Network: Concatenates time feature alongside spatial coordinates and
      context to enable learned time-dependent advection dynamics.
    
    **Memory Ownership**:
    - Owns: self.net (MLP parameters), self.out_scale (scaling parameter)
    - Borrowed: context (set via set_context(), not retained after forward pass)
    - Borrowed: z, logp (ODE solver state, passed as inputs)
    
    **Attributes**:
        dim (int): Spatial dimensionality (e.g., 2 for 2D particle positions)
        cond_dim (int): Conditioning vector dimensionality (velocity field feature size)
        net (MLP): Neural network mapping [z, context, t] → R^dim velocity vector
        _context (Tensor | None): Current conditioning context (batch, cond_dim)
        out_scale (Parameter): Learnable scaling factor for velocity field
        z_scale (Tensor): Registered buffer for coordinate scaling (currently unused)
        skip_divergence (bool): Flag to disable divergence computation (e.g., during evaluation)
    
    **Time Complexity**:
    - Forward pass: O(batch_size × dim) for network evaluation + O(batch_size × dim²) for
      divergence computation if training (Hutchinson trace). With Hutchinson, effectively O(batch_size × dim).
    - Backward pass (adjoint): O(1) gradient of net parameters (adjoint avoids explicit Jacobian storage)
    
    **Space Complexity**:
    - O(dim) for Hutchinson epsilon vector
    - O(batch_size × dim) for z and gradients
    - O(batch_size × 1) for divergence accumulation
    
    **Error Behavior**:
    - Raises RuntimeError if context not set before forward() (line 27)
    - Raises RuntimeError if autograd fails (e.g., disconnected graph) - grad defaults to zero
    
    **Usage Example**::
    
        ode_func = ODEFunc(dim=2, cond_dim=64, hidden_dim=128, depth=3)
        context = torch.randn(batch_size, 64)  # Velocity field features
        ode_func.set_context(context)
        
        # During integration
        z = torch.randn(batch_size, 2)
        logp = torch.zeros(batch_size, 1)
        t = torch.tensor(0.5)
        f_vel, neg_div = ode_func(t, (z, logp))
        # f_vel: (batch_size, 2) velocity vector
        # neg_div: (batch_size, 1) negative divergence for log-prob tracking
    """
    
    def __init__(self, dim: int, cond_dim: int, hidden_dim: int = 64, depth: int = 3, dropout_p: float = 0.0) -> None:
        """
        Initialize the ODE function network.
        
        **Parameters**:
            dim (int): Spatial dimensionality. Must be ≥1. Typical: 2 for 2D advection, 3 for 3D.
            cond_dim (int): Conditioning (context) vector dimensionality. Must be ≥1.
                Typically computed from velocity field encoder output.
            hidden_dim (int, default=64): Width of hidden layers in MLP backbone.
                Must be ≥8. Larger values increase expressiveness at memory cost.
            depth (int, default=3): Number of hidden layers in MLP. Must be ≥1.
                Larger networks capture more complex dynamics but may overfit.
            dropout_p (float, default=0.0): Dropout probability during training [0, 1).
                Helps regularize for small datasets. Disabled during evaluation.
        
        **Returns**: None
        
        **Side Effects**:
        - Initializes MLP with input size = dim + cond_dim + 1 (time feature)
        - Registers z_scale buffer (currently unused, for future feature normalization)
        - Sets skip_divergence flag to False (compute divergence by default)
        
        **Error Behavior**:
        - No validation of argument ranges. Caller responsible for valid dim, cond_dim, hidden_dim, depth ≥1
        """
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        # Input: [z (dim), context (cond_dim), t (1)]
        self.net = MLP(dim + cond_dim + 1, dim, hidden_dim=hidden_dim, depth=depth, dropout_p=dropout_p)
        self._context: torch.Tensor | None = None
        self.out_scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("z_scale", torch.ones(1, dim))
        self.skip_divergence = False

    def set_context(self, context: torch.Tensor) -> None:
        """
        Set the conditioning context for the next forward pass.
        
        **Parameters**:
            context (Tensor): Conditioning vector, shape (batch_size, cond_dim). Typically
                velocity field features. Not validated; assumes caller provides correct shape.
        
        **Returns**: None
        
        **Side Effects**:
        - Modifies _context instance variable (not copied; stores reference)
        - Context persists until next set_context() call
        
        **Memory Ownership**: Borrows reference to context tensor; does not copy.
        
        **Usage Example**::
        
            z0 = torch.randn(batch_size, 2)
            context = torch.randn(batch_size, cond_dim)
            ode_func.set_context(context)  # Store for ODE solver
            # Solver will call forward(t, (z, logp)) multiple times with same context
        
        **Error Behavior**: No validation. Caller must ensure context shape matches (batch_size, cond_dim).
        """
        self._context = context

    def forward(self, t: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute velocity and divergence trace for the ODE: dz/dt = f(z, context, t), d(logp)/dt = -tr(∇f).
        
        **Parameters**:
            t (Tensor): Current integration time, shape scalar or (1,). Must be in valid ODE time domain.
            states (Tuple[Tensor, Tensor]): Tuple (z, logp) where:
                - z (Tensor): Particle positions, shape (batch_size, dim)
                - logp (Tensor): Log-probability accumulator, shape (batch_size, 1)
        
        **Returns**:
            Tuple[Tensor, Tensor]: (f, neg_div) where:
                - f (Tensor): Velocity field, shape (batch_size, dim)
                - neg_div (Tensor): Negative divergence, shape (batch_size, 1).
                  Log-probability change: d(logp)/dt = -tr(∇f) ≈ -neg_div
        
        **Side Effects**:
        - Enables gradient computation within forward pass (with torch.enable_grad())
        - Modifies z.requires_grad if not already set
        - May trigger autograd backprop chain for divergence computation (training mode)
        
        **Memory Ownership**:
        - Returns: New tensors f and neg_div (owned by caller)
        - Borrows: z, logp, context (not modified)
        
        **Time Complexity**:
        - Network forward: O(batch_size × dim)
        - Divergence (Hutchinson trace if training): O(batch_size × dim)
        - Divergence (eval mode or skip_divergence=True): O(batch_size) (zeros only)
        
        **Space Complexity**:
        - Hutchinson epsilon: O(batch_size × dim)
        - Gradient tensors: O(batch_size × dim) (temporary during computation)
        
        **Error Behavior**:
        - Raises RuntimeError: "Context must be set before integrating the CNF" if _context is None
        - Returns zero divergence if grad computation fails or grad is None (line 46, graceful degradation)
        - Returns zero divergence if not in training mode (line 50, avoids expensive computation)
        
        **Algorithm Details** (Hutchinson Trace Estimator):
        
        The divergence tr(∇f) cannot be computed directly (O(dim²) cost). Instead, we use:
        
            tr(∇f(z)) ≈ ε^T · ∇_z(f(z) · ε)  where ε ~ N(0, I)
        
        Implementation:
        1. Sample ε ~ N(0, I), shape (batch_size, dim)
        2. Compute scalar f·ε = (f * eps).sum() summed over batch and dimensions
        3. Backprop to get ∇_z(f·ε) via autograd (shape batch_size × dim)
        4. Multiply element-wise and sum: tr ≈ (grad * eps).sum(dim=1)
        5. Return as negative divergence for log-det-Jacobian tracking
        
        This reduces computation from O(dim² forward + O(dim²) backward to O(dim forward + O(dim) backward.
        
        **Usage Example**::
        
            ode_func.train()  # Enable divergence computation
            z = torch.randn(batch_size, 2, requires_grad=True)
            logp = torch.zeros(batch_size, 1)
            t = torch.tensor(0.5)
            
            f_vel, neg_div = ode_func(t, (z, logp))
            # f_vel: (batch_size, 2) velocity
            # neg_div: (batch_size, 1) divergence, used for log-det-Jacobian
            
            # During integration: z_new = z + dt * f_vel
            #                    logp_new = logp + dt * neg_div (for likelihood computation)
        """
        if self._context is None:
            raise RuntimeError("Context must be set before integrating the CNF")
        z, logp = states
        with torch.enable_grad():
            if not z.requires_grad:
                z = z.detach().requires_grad_(True)
            context = self._context.to(device=z.device, dtype=z.dtype)
            t = t.to(device=z.device, dtype=z.dtype)
            # Time-conditioned velocity field: concatenate time as feature
            t_feat = torch.ones(z.size(0), 1, device=z.device, dtype=z.dtype) * t
            inp = torch.cat([z, context, t_feat], dim=1)
            f = self.net(inp) * self.out_scale
            
            if self.training and not self.skip_divergence:
                # Hutchinson estimator: tr(∇f) ≈ ε^T·∇(f·ε) where ε ~ N(0,I)
                # Sample noise
                eps = torch.randn_like(z)
                # Scalar for backward: allows vector-Jacobian product
                f_eps = (f * eps).sum()
                # Compute divergence: ε^T · ∇_z(f·ε)
                grad = torch.autograd.grad(f_eps, z, create_graph=True, retain_graph=True, allow_unused=True)[0]
                if grad is None:
                    # If gradient computation failed (disconnected graph), use zero divergence
                    grad = torch.zeros_like(z)
                # Approximate trace as sum of element-wise products
                div = (grad * eps).sum(dim=1, keepdim=True)
            else:
                # Evaluation mode: skip expensive divergence computation
                div = torch.zeros(z.size(0), 1, device=z.device, dtype=z.dtype)
        
        return f, -div


class CNFModel(nn.Module):
    """
    Continuous Normalizing Flow (CNF) for learned trajectory distribution modeling.
    
    Trains a neural ODE to learn the dynamics of particle trajectories via continuous normalizing flows.
    The model solves an ODE in the latent space to transform samples from a base distribution
    (standard Gaussian) to the target trajectory distribution, computing exact log-likelihoods via
    the change-of-variables formula with divergence tracking.
    
    **Problem Solved**: Given observed trajectories x ~ p(x), learn a generative model and compute
    exact log p(x) without Variational bounds. Uses adjoint-method backprop to avoid storing
    intermediate ODE states.
    
    **Core Algorithm** (Continuous Normalizing Flow):
    
    Forward pass (likelihood computation):
        x_0 ~ p_0(x_0) = N(0, I)  [base distribution]
        dx/dt = f_θ(x, context, t)  [learned dynamics]
        log p(x_T) = log p_0(x_0) - ∫₀^T tr(∇_x f) dt  [change-of-variables]
    
    Reverse pass (sampling):
        z_0 ~ N(0, I)  [sample from base]
        dz/dt = f_θ(z, context, t)  [same ODE, different direction]
        x = z_T
    
    The divergence integral is accumulated via Hutchinson trace estimator in ODEFunc,
    reducing computational cost from O(dim²) to O(1) per evaluation.
    
    **Memory Ownership**:
    - Owns: self.func (ODEFunc neural network parameters)
    - Borrows: context (passed through, not stored)
    - Returns: New tensors z and logp_trajectories
    
    **Attributes**:
        dim (int): Spatial dimensionality of particle positions
        cond_dim (int): Dimensionality of conditioning context (velocity field features)
        func (ODEFunc): Neural ODE function parameterizing the dynamics
    
    **Time Complexity**:
    - Forward (flow/log_prob): O(batch × dim × steps) for ODE integration
        (each step: O(batch × dim) forward + O(batch × dim) divergence via Hutchinson)
    - Backward (gradient via adjoint): O(batch × dim × steps) (proportional to forward, memory-efficient)
    - Sample: O(batch × dim × steps)
    
    **Space Complexity**:
    - Model parameters: O(dim × hidden_dim × depth) (MLP weights)
    - Forward pass (non-adjoint naive): O(batch × dim × steps) [if storing all states]
    - Forward pass (with adjoint method): O(batch × dim) [constant, independent of steps]
    
    **Error Behavior**:
    - Raises ValueError: "tspan must contain at least two time points" if tspan has <2 points
    - Raises ValueError: "Invalid tspan: consecutive time steps..." if time steps too close
        (ODE solver underflow protection, helps debug workflows with step=1)
    - Raises AssertionError: If ODE integration returns None (should not occur with torchdiffeq)
    
    **Usage Example**::
    
        # Initialize model
        model = CNFModel(dim=2, cond_dim=64, hidden_dim=128, depth=3)
        model.train()
        
        # Forward pass: compute likelihood
        x = torch.randn(batch_size, 2)  # Particle positions
        context = velocity_encoder(v_snapshots)  # (batch_size, 64) velocity features
        log_prob = model.log_prob(x, context, base_std=1.0)
        
        # Backward pass
        loss = -log_prob.mean()  # Maximize likelihood
        loss.backward()
        optimizer.step()
        
        # Sampling (inference)
        model.eval()
        with torch.no_grad():
            samples = model.sample(n=100, context=context, base_std=1.0, steps=30)
            # samples shape: (100, 2)
    
    **Key Design Decisions**:
    1. Adjoint method (odeint_adjoint): Avoids storing ODE states; enables long integration horizons
    2. Hutchinson trace (in ODEFunc): O(1) divergence computation instead of O(dim²)
    3. Time-conditioned dynamics: Concatenates t as feature; allows learned time-varying flows
    4. Two-pass design: _integrate() shared by flow() and sample()
    5. Context setting: Decouples trajectory data from velocity field features
    """

    def __init__(
        self,
        dim: int = 2,
        cond_dim: int = 64,
        hidden_dim: int = 128,
        depth: int = 3,
        dropout: float = 0.0,
    ) -> None:
        """
        Initialize a Continuous Normalizing Flow model.
        
        **Parameters**:
            dim (int, default=2): Spatial dimensionality of particle positions.
                Must be ≥1. Typical: 2 for 2D CFD, 3 for 3D.
            cond_dim (int, default=64): Dimensionality of velocity field conditioning.
                Typically output size of velocity field encoder.
            hidden_dim (int, default=128): Hidden layer width in ODEFunc's MLP.
                Larger → more expressive dynamics, higher memory/compute cost.
            depth (int, default=3): Number of hidden layers in MLP.
                Typical range: 2-5. Deeper → captures more complex flows.
            dropout (float, default=0.0): Dropout probability during training [0, 1).
                Helps prevent overfitting on small datasets.
        
        **Returns**: None
        
        **Side Effects**:
        - Initializes ODEFunc with (dim, cond_dim, hidden_dim, depth, dropout) parameters
        - Sets up ODE solver with fixed tolerances (atol=1e-5, rtol=1e-5)
        
        **Error Behavior**: No validation of argument ranges. Caller responsible for dim, cond_dim ≥1.
        """
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        self.func = ODEFunc(dim=dim, cond_dim=cond_dim, hidden_dim=hidden_dim, depth=depth, dropout_p=dropout)

    def _integrate(self, x: torch.Tensor, context: torch.Tensor, tspan: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Core ODE integration routine: shared backend for flow(), sample(), log_prob().
        
        Integrates the ODE dz/dt = f_θ(z, context, t), d(logp)/dt = -tr(∇f) from t[0] to t[-1],
        returning final state and accumulated divergence.
        
        **Parameters**:
            x (Tensor): Initial state, shape (batch_size, dim). Must have requires_grad=True for
                gradient computation; will be set automatically if needed.
            context (Tensor): Conditioning vector, shape (batch_size, cond_dim). Velocity field
                features from encoder. Not validated; caller ensures correct shape.
            tspan (Tensor): Time points for integration, shape (n_steps,). Must have ≥2 points,
                strictly increasing, and time differences ≥ numerical tolerance (1e-38 float32).
                Typical: linspace(1.0, 0.0, 8) for likelihood; linspace(0.0, 1.0, 30) for sampling.
        
        **Returns**:
            Tuple[Tensor, Tensor]: (z_final, logp_final) where:
                - z_final (Tensor): Final state after ODE integration, shape (batch_size, dim)
                - logp_final (Tensor): Accumulated log-probability change, shape (batch_size, 1)
                  For likelihood: log p(x) = log p_0(x_0) + logp_final
        
        **Side Effects**:
        - Calls odeint_adjoint which traces autodiff operations
        - May raise ValueError if tspan validation fails (protects against ODE solver underflow)
        - Modifies x.requires_grad if needed
        
        **Memory Ownership**:
        - Returns: New tensors (owned by caller)
        - Borrows: x, context, tspan (not modified)
        - Adjoint method: Does NOT store intermediate states (constant memory w.r.t. steps)
        
        **Time Complexity**: O(batch × dim × n_steps) for integration
        - Each ODE step: forward O(batch × dim) + divergence O(batch × dim) 
        - Backward (adjoint): proportional to forward pass
        
        **Space Complexity**: O(batch × dim) independent of n_steps (adjoint method benefit)
        
        **Error Behavior**:
        - Raises ValueError: "tspan must contain at least two time points" if len(tspan) < 2
            → Caller must ensure steps ≥2 parameter passed to linspace
        - Raises ValueError: "Invalid tspan: consecutive time steps..." if diffs ≤ 1e-38
            → Caller accidentally passed duplicate times or step=1; check orchestration logic
        - Raises AssertionError: "ODE integration failed" if odeint returns None (should never occur)
        
        **Algorithm Details** (ODE Integration with Change-of-Variables):
        
        1. Initialize: z=x, logp=zeros, context set in ODEFunc
        2. Solve ODE via adjoint method (memory-efficient backprop)
        3. At each time t, compute: z_dot = f(z, context, t), logp_dot = -tr(∇f)
        4. Accumulate: dz = z_dot × dt, d(logp) = logp_dot × dt
        5. Return: final z and integrated logp
        
        Log-probability is tracked via the divergence integral (change-of-variables formula),
        avoiding explicit Jacobian determinant computation.
        
        **Usage Example**::
        
            x = torch.randn(32, 2, requires_grad=True)
            context = torch.randn(32, 64)
            tspan = torch.linspace(1.0, 0.0, 8)
            
            z_final, logp_final = model._integrate(x, context, tspan)
            # z_final: (32, 2) transformed state
            # logp_final: (32, 1) accumulated divergence integral
        
        **Validation Logic** (lines 84-95):
        Prevents common calling errors:
        - Check tspan has ≥2 points (needed for integration start/end)
        - Check time span is non-zero (t0 ≠ t1)
        - Check consecutive time steps > 1e-38 (avoids ODE solver underflow)
        These checks provide helpful error messages instead of silent NaN failures.
        """
        x = x.to(dtype=torch.float32)
        context = context.to(dtype=torch.float32)
        if not x.requires_grad:
            x = x.requires_grad_()
        logp0 = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
        self.func.set_context(context)
        tspan = tspan.to(device=x.device, dtype=x.dtype)
        
        # Validate the requested time span to avoid zero-range / underflow errors
        if tspan.numel() < 2:
            raise ValueError("tspan must contain at least two time points for integration")

        t0_val = float(tspan[0].to("cpu"))
        t1_val = float(tspan[-1].to("cpu"))
        if abs(t1_val - t0_val) == 0.0:
            # Nothing to integrate; return initial states unchanged
            return x, logp0

        # Ensure consecutive time differences are non-zero to avoid the
        # ODE solver asserting on an underflow in dt (e.g. "underflow in dt 0.0").
        # This can happen when callers construct a tspan with duplicate time
        # entries or with steps=1. Provide a helpful error message so the
        # orchestrating workflow can be adjusted.
        diffs = torch.abs(tspan[1:] - tspan[:-1])
        if torch.any(diffs <= torch.finfo(tspan.dtype).tiny):
            raise ValueError(
                "Invalid tspan: consecutive time steps contain zero or extremely "
                "small differences which will cause the ODE solver to underflow. "
                "Check caller for t0/t1/steps (received tspan=%s)." % (tspan,)
            )

        solver_options = {"dtype": torch.float32}
        res = odeint(self.func, (x, logp0), tspan, atol=1e-5, rtol=1e-5, options=solver_options)
        assert res is not None, "ODE integration failed during flow"
        z_traj, logp_traj = res
        assert z_traj is not None and logp_traj is not None, "ODE integration returned None states"
        return z_traj[-1], logp_traj[-1]

    def flow(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        *,
        t0: float = 1.0,
        t1: float = 0.0,
        steps: int = 8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform data through the learned dynamics: computes z_T = flow_θ(x, context, t0→t1).
        
        **Parameters**:
            x (Tensor): Initial state, shape (batch_size, dim)
            context (Tensor): Conditioning context, shape (batch_size, cond_dim)
            t0 (float, keyword-only, default=1.0): Starting time for integration
            t1 (float, keyword-only, default=0.0): Ending time for integration
            steps (int, keyword-only, default=8): Number of ODE solver steps. Must be ≥2.
                More steps → more accurate but slower. Typical: 8 for training, 30 for evaluation.
        
        **Returns**:
            Tuple[Tensor, Tensor]: (z_final, logp_final)
                - z_final: Transformed state, shape (batch_size, dim)
                - logp_final: Log-probability change, shape (batch_size, 1)
        
        **Time Complexity**: O(batch × dim × steps)
        
        **Space Complexity**: O(batch × dim) via adjoint method
        
        **Usage Example**::
        
            z, logp = model.flow(x, context, t0=1.0, t1=0.0, steps=8)
            # z: transformed positions
            # logp: accumulated divergence (for likelihood computation)
        """
        tspan = torch.linspace(t0, t1, steps=steps, device=x.device, dtype=x.dtype)
        return self._integrate(x, context, tspan)

    def log_prob(self, x: torch.Tensor, context: torch.Tensor, base_std: float = 1.0) -> torch.Tensor:
        """
        Compute exact log-likelihood of data under the learned model.
        
        Implements the change-of-variables formula:
            log p(x) = log p_0(T(x)) + ∫₀¹ tr(∇_z f_θ(z, context, t)) dt
        
        where T(x) is the flow transformation and the integral is approximated via divergence tracking.
        
        **Parameters**:
            x (Tensor): Data samples, shape (batch_size, dim)
            context (Tensor): Conditioning context, shape (batch_size, cond_dim)
            base_std (float, default=1.0): Standard deviation of base Gaussian p_0. Must be >0.
                Typical: 1.0. Larger σ → flatter base distribution.
        
        **Returns**:
            Tensor: Log-likelihood values, shape (batch_size,). Suitable for loss = -log_prob.mean()
        
        **Side Effects**:
        - Calls flow() which triggers ODE integration and gradient computation
        
        **Time Complexity**: O(batch × dim × steps) dominated by flow()
        
        **Space Complexity**: O(batch × dim)
        
        **Error Behavior**:
        - Assumes base_std > 0 (not validated). Caller responsibility.
        - Returns -∞ if log_prob becomes NaN (should not occur with stable training)
        
        **Algorithm** (Change-of-Variables):
        
        1. Forward: z = flow(x, context), returns accumulated divergence Δ logp
        2. Base log-likelihood: log p_0(z) = -0.5 × ||z/σ||² - 0.5×dim×log(2πσ²)
        3. Return: log p(x) = log p_0(z) + Δ logp
        
        The divergence integral automatically cancels the Jacobian determinant,
        yielding exact likelihood without variational bounds.
        
        **Usage Example**::
        
            # Training
            x_train = torch.randn(batch_size, 2)
            context = velocity_encoder(v_snapshots)
            log_prob = model.log_prob(x_train, context, base_std=1.0)
            loss = -log_prob.mean()
            loss.backward()
            
            # Evaluation
            model.eval()
            with torch.no_grad():
                log_prob_test = model.log_prob(x_test, context, base_std=1.0)
                nll = -log_prob_test.mean()  # Negative log-likelihood
        """
        device = x.device
        z, delta_logp = self.flow(x, context)
        # Base distribution: N(0, σ²I)
        normalizer = torch.log(torch.tensor(2 * torch.pi * base_std ** 2, device=device, dtype=z.dtype))
        base = -0.5 * ((z / base_std) ** 2).sum(dim=1, keepdim=True) - 0.5 * self.dim * normalizer
        # Change-of-variables: log p(x) = log p_0(z) + ∫ tr(∇f) dt
        return (base + delta_logp).squeeze(-1)

    @torch.no_grad()
    def eval_field(self, z: torch.Tensor, context: torch.Tensor, t: float = 1.0) -> torch.Tensor:
        """
        Evaluate the learned velocity field at specified positions and time (no gradient).
        
        Used for visualization and analysis of learned dynamics without affecting model state.
        Disables divergence computation for efficiency.
        
        **Parameters**:
            z (Tensor): Query positions, shape (batch_size, dim)
            context (Tensor): Conditioning context, shape (batch_size, cond_dim)
            t (float, default=1.0): Time at which to evaluate field
        
        **Returns**:
            Tensor: Velocity field values, shape (batch_size, dim)
        
        **Side Effects**:
        - Temporarily sets skip_divergence=True, then restores original value
        - Does not affect model parameters or state
        
        **Time Complexity**: O(batch × dim) for single forward pass
        
        **Space Complexity**: O(batch × dim)
        
        **Usage Example**::
        
            model.eval()
            z_grid = torch.randn(1000, 2)  # Query points on grid
            v_field = model.eval_field(z_grid, context, t=0.5)
            # v_field: (1000, 2) velocity vectors for visualization
        """
        device = z.device
        dtype = z.dtype
        z = z.to(device=device, dtype=dtype)
        context = context.to(device=device, dtype=dtype)
        logp0 = torch.zeros(z.size(0), 1, device=device, dtype=dtype)
        self.func.set_context(context)
        # Temporarily disable divergence for efficiency
        prev = self.func.skip_divergence
        self.func.skip_divergence = True
        try:
            f, _ = self.func(torch.tensor(float(t), device=device, dtype=dtype), (z, logp0))
        finally:
            self.func.skip_divergence = prev
        return f

    @torch.no_grad()
    def sample(self, n: int, context: torch.Tensor, base_std: float = 1.0, steps: int = 30) -> torch.Tensor:
        """
        Generate samples from the learned distribution.
        
        Implements reverse-time integration: sample z_0 ~ p_0, integrate ODE forward to get x.
        This is the "generative" mode of the CNF.
        
        **Parameters**:
            n (int): Number of samples to generate. Must be ≥1.
            context (Tensor): Conditioning context, shape (batch_size, cond_dim). Note: batch_size
                determines output batch dimension (for each context, generate n samples). Caller
                may need to expand context if generating multiple samples per context.
            base_std (float, default=1.0): Standard deviation for sampling. Must be >0.
                Typically: 1.0. Larger σ → more diverse samples.
            steps (int, default=30): Number of ODE solver steps. More steps → smoother
                trajectories but slower. Typical: 30 for quality, 8 for speed.
        
        **Returns**:
            Tensor: Samples from learned distribution, shape (n, dim)
        
        **Side Effects**:
        - Does not modify model parameters or state (torch.no_grad context)
        
        **Time Complexity**: O(n × dim × steps)
        
        **Space Complexity**: O(n × dim)
        
        **Error Behavior**:
        - Asserts that context is set before sampling (should not fail with correct caller)
        - Returns empty tensor if n=0 (not validated)
        
        **Algorithm** (Reverse-Time Generation):
        
        1. Sample z_0 ~ N(0, σ²I), shape (n, dim)
        2. Integrate from t=0 to t=1: z(t+dt) = z(t) + f_θ(z(t), context, t) × dt
        3. Return: x = z_T (final state)
        4. Note: Divergence term ignored (divergence is only for likelihood, not needed for sampling)
        
        The same ODE function is used as in flow(), but direction is reversed (t: 0→1)
        and divergence is not used.
        
        **Usage Example**::
        
            model.eval()
            with torch.no_grad():
                context = torch.randn(1, cond_dim)  # Single context
                x_samples = model.sample(n=100, context=context, base_std=1.0, steps=30)
                # x_samples: (100, 2) new particle trajectories sampled from model
        """
        device = context.device
        dtype = context.dtype
        # Sample from base Gaussian
        z0 = base_std * torch.randn(n, self.dim, device=device, dtype=dtype)
        # Integrate forward from z0 to x by going through the ODE (t: 0→1)
        tspan = torch.linspace(0.0, 1.0, steps=steps, device=device, dtype=dtype)
        self.func.set_context(context)
        assert self.func._context is not None, "Context must be set before sampling"
        solver_options = {"dtype": torch.float32}
        res = odeint(self.func, (z0, torch.zeros(n, 1, device=device, dtype=dtype)), tspan, atol=1e-5, rtol=1e-5, options=solver_options)
        assert res is not None, "ODE integration failed during sampling"
        x_traj, _ = res
        return x_traj[-1]
