"""Incoherent / partly-incoherent thin film solver class.

User-facing solver for stacks containing thick incoherent layers (e.g.,
glass substrates beyond the source coherence length). Mirrors the
IsotropicFilmSolver UX and inherits most of its plumbing; the only
overrides are an added ``c_list`` argument, a substrate-friendly default
``thickness_max``, and a ``simulate`` that calls the incoherent TMM kernel.

Algorithm and helper functions (``create_intensity_RT_isotropic``,
``group_layers_by_coherence``, ``interface_power_RT``,
``coh_stack_power_RT_isotropic``) live in ``film_solver_isotropic`` and
are imported from there.

Copyright (c) 2026, Xinge Yang, Qingyuan Fan, Zhaocheng Liu.
"""

from __future__ import annotations

import torch

from .film_solver_isotropic import (
    IsotropicFilmSolver,
    create_intensity_RT_isotropic,
)


class IncoherentIsotropicFilmSolver(IsotropicFilmSolver):
    """Multi-layer coating solver with partly-incoherent layer support.

    Same UX as :class:`IsotropicFilmSolver` but additionally accepts a
    ``c_list`` argument that marks each interior layer as coherent
    (``'c'``) or incoherent (``'i'``). Returns real power coefficients
    ``(Rs, Rp, Ts, Tp)`` rather than complex amplitudes, because
    incoherent calculations discard phase by design.
    """

    def __init__(
        self,
        mat_n_in,
        mat_n_out,
        mat_n_ls,
        c_list,
        thickness_ls=None,
        thickness_min=0.0,
        thickness_max=1000.0,   # in um; default sized for thick substrates
        batch_size=1,
        sigmoid_param=False,
        device=torch.device("cuda"),
    ):
        """Initialize the incoherent isotropic film solver.

        Args:
            mat_n_in: scalar incident refractive index.
            mat_n_out: scalar exit refractive index.
            mat_n_ls: refractive indices of interior layers (list or tensor of length N).
            c_list: list of 'c'/'i' codes, length N. One per interior layer.
            thickness_ls: thicknesses of interior layers in um, length N. If None,
                          random.
            thickness_min: minimum thickness in um.
            thickness_max: maximum thickness in um. Defaults to 1000 (um) so
                substrate-sized layers fit without special configuration.
            batch_size: number of film stacks in the batch.
            sigmoid_param: if True, use sigmoid parameterization.
            device: torch device.

        Raises:
            ValueError: if ``c_list`` length does not match the number of
                interior layers, or contains anything other than ``'c'`` / ``'i'``.
        """
        # Validate c_list before delegating to the parent — fail fast with a
        # helpful error rather than letting the parent allocate state we then
        # discard.
        n_layers_provided = (
            mat_n_ls.shape[0] if torch.is_tensor(mat_n_ls) else len(mat_n_ls)
        )
        if len(c_list) != n_layers_provided:
            raise ValueError(
                f"c_list length {len(c_list)} does not match number of interior "
                f"layers {n_layers_provided}."
            )
        for code in c_list:
            if code not in ("c", "i"):
                raise ValueError("c_list entries must be 'c' or 'i'.")
        self.c_list = list(c_list)

        super().__init__(
            mat_n_in=mat_n_in,
            mat_n_out=mat_n_out,
            mat_n_ls=mat_n_ls,
            thickness_ls=thickness_ls,
            thickness_min=thickness_min,
            thickness_max=thickness_max,
            batch_size=batch_size,
            sigmoid_param=sigmoid_param,
            device=device,
        )

    def simulate(self, theta, wvln):
        """Compute (Rs, Rp, Ts, Tp) for the configured stack.

        Args:
            theta: angles in radians. 1D ``(n_angles,)`` or 2D ``(batch, n_angles)``.
            wvln: wavelengths in um. Scalar, list, or 1D tensor.

        Returns:
            Rs, Rp, Ts, Tp: real tensors of shape ``(batch, n_wvlns, n_angles)``.
        """
        theta, wv_batch, d_batch = self._prepare_simulate_inputs(theta, wvln)
        return create_intensity_RT_isotropic(
            self.refract_idx_layers, d_batch, wv_batch,
            self.mat_n_in, self.mat_n_out, theta, self.c_list,
        )

    # to(), get_film_thickness(), __call__() are inherited from IsotropicFilmSolver.
    # load_ckpt() / save_ckpt() are intentionally NOT overridden; the parent's
    # implementations work because the stored state (film_thickness, num_layers,
    # n_in, n_out, refract_idx_layers) is identical. c_list is not persisted,
    # which is consistent with treating it as part of the *stack design* rather
    # than the optimizable state.
