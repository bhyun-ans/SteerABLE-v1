"""Truncated raised-cosine alpha schedule for epitope-guided embedding steering.

This implements the flow-matching guidance-weight schedule

    omega(s) = omega_init * 0.5 * (1 + cos(pi * s))   for 0 <= s < tau_trunc
    omega(s) = 0                                       for tau_trunc <= s <= 1

with the convention-agnostic *progress* variable ``s in [0, 1]``:

    s = 0  -> generation START (pure noise, coarse assembly)
    s = 1  -> generation END   (clean structure)

Time-convention note (important!):
    The source paper is flow matching (t=0 noise, t=1 data, generation t: 0->1),
    so it writes cos(pi*t) and gets "strong guidance early". Standard diffusion
    (DDPM) runs the other way (t=T noise -> t=0 data), so plugging cos(pi*t) in
    naively would flip the schedule and make guidance strongest *late*.

    We sidestep this entirely by working in the progress variable ``s`` instead
    of any model-specific time ``t``. In THIS codebase the diffusion sampling
    loop's ``step_idx`` already counts UP from the noisiest step (step 0) to the
    cleanest (final step) -- see ``noise_schedule[0]`` being the largest noise
    level in generator.py. So progress is simply

        s = step_idx / (total_steps - 1)

    and cos(pi*0)=1 correctly places the maximum weight at the start (strong
    early guidance), decaying to 0 -- no convention flip needed here.

Truncation form -- the PAPER variant (faithful reproduction baseline):
    omega is DISCONTINUOUS at s=tau. With tau=0.5, cos(pi*0.5)=0 gives
    omega = 0.5*omega_init right before the cutoff, then it jumps to 0.
    To run the smooth ("rescaled") ablation instead -- where omega reaches
    exactly 0 at s=tau -- swap the single ``phase = ...`` line in
    :func:`cosine_trunc` (see the comment there). Both are one line.

``tau_trunc`` (a.k.a. "when to turn alpha off") is exposed on the CLI via
``--guidance.alpha_trunc``; with N_step=200, tau=0.5 turns guidance off at
step 100 (alpha applied for steps 0..99, zero for 100..199).
"""

import math


def cosine_trunc(
    step_idx: int,
    total_steps: int,
    omega_init: float = 1.0,
    tau_trunc: float = 0.5,
) -> float:
    """Truncated raised-cosine guidance weight at a given diffusion step.

    Args:
        step_idx: Current diffusion step (0-based; 0 = noisiest = generation start).
        total_steps: Total number of diffusion steps (e.g. 200).
        omega_init: Peak weight at s=0 (paper grid-search value: 1.0).
        tau_trunc: Progress fraction in [0, 1] at which the weight drops to 0.

    Returns:
        The guidance weight omega(s) as a float.
    """
    if total_steps <= 1:
        return 0.0
    s = step_idx / (total_steps - 1)  # 0.0 (noise/start) -> 1.0 (clean/end)
    if s >= tau_trunc:
        return 0.0
    phase = math.pi * s  # PAPER form (discontinuous at tau)
    # Smooth/"rescaled" ablation -- replace the line above with:
    #     phase = math.pi * s / tau_trunc   # reaches exactly 0 at s=tau
    return omega_init * 0.5 * (1.0 + math.cos(phase))


def make_cosine_trunc(
    omega_init: float = 1.0,
    tau_trunc: float = 0.5,
):
    """Return a ``(step_idx, total_steps) -> float`` closure with params baked in.

    This adapts :func:`cosine_trunc` to the schedule-callable signature expected
    by the sampling loop in ``generator.py``.
    """
    if not 0.0 < tau_trunc <= 1.0:
        raise ValueError(f"tau_trunc must be in (0, 1], got {tau_trunc}.")

    def _schedule(step_idx: int, total_steps: int) -> float:
        return cosine_trunc(
            step_idx, total_steps, omega_init=omega_init, tau_trunc=tau_trunc,
        )

    _schedule.__doc__ = f"cosine_trunc(omega_init={omega_init}, tau_trunc={tau_trunc})"
    return _schedule
