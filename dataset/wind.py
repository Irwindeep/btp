import numpy as np
from numpy.fft import fft2, ifft2, fftfreq


def generate_random_wind_field(
    N: int,
    timesteps: int,
    dx: float = 1.0,
    dt: float = 1.0,
    mean_wind: tuple | None = (5.0, 0.0),
    target_rms: float = 2.0,
    spatial_corr_len: float = 10.0,
    spectral_slope: float = 3.0,
    correlation_time: float = 20.0,
    divergence_frac: float = 0.0,
    anisotropy: float = 1.0,
    random_seed: int | None = None,
) -> np.ndarray:
    """
    Generate a random time-varying horizontal wind field.
    """

    rng = np.random.default_rng(random_seed)
    winds = np.zeros((timesteps, N, N, 2), dtype=np.float32)

    # Frequency grids
    kx_1d = 2 * np.pi * fftfreq(N, d=dx)
    ky_1d = 2 * np.pi * fftfreq(N, d=dx)
    kx, ky = np.meshgrid(kx_1d, ky_1d, indexing="xy")

    # apply anisotropy by scaling kx
    kx_aniso = kx * (1.0 / max(anisotropy, 1e-6))
    k2_aniso = kx_aniso**2 + ky**2

    # spectral envelope S(k) = (k^2 + k0^2)^(-spectral_slope/2)
    k0 = 2 * np.pi / max(spatial_corr_len, 1e-6)
    S = (k2_aniso + k0**2) ** (-spectral_slope / 2.0)
    S[0, 0] = 0.0  # remove mean/zero mode energy
    sqrtS = np.sqrt(S)

    # Temporal OU parameter
    if correlation_time is None or correlation_time <= 0:
        rho = 0.0
    else:
        rho = np.exp(-dt / float(correlation_time))
    noise_scale = np.sqrt(max(0.0, 1.0 - rho**2))

    # helper: make new scalar field in spectral domain with specified spectrum
    def new_scalar_hat():
        # generate real white noise in spatial domain, then FFT and color by sqrtS
        noise = rng.standard_normal((N, N))
        return fft2(noise) * sqrtS

    # initialize per-level coefficients (streamfunction and potential)
    psi_hat = np.zeros((N, N), dtype=np.complex128)
    phi_hat = np.zeros((N, N), dtype=np.complex128)  # potential component

    psi_hat = new_scalar_hat()
    if divergence_frac > 0:
        phi_hat = new_scalar_hat() * np.sqrt(divergence_frac)  # small init

    # spectral derivative multipliers
    ikx = 1j * kx
    iky = 1j * ky

    if mean_wind is None:
        mean_profile = np.zeros(2, dtype=float)
    else:
        mean_profile = np.array(mean_wind, dtype=float)

    # make spectrum steeper: fewer small scales
    sqrtS *= 0  # we'll rebuild S with larger slope
    large_slope = max(spectral_slope, 4.0)
    S2 = (k2_aniso + k0**2) ** (-large_slope / 2.0)
    S2[0, 0] = 0.0
    sqrtS = np.sqrt(S2)

    # If we changed sqrtS after initialization, we should rescale existing hats to match sqrtS
    psi_hat = fft2(np.real(ifft2(psi_hat))) * (sqrtS / (np.abs(sqrtS) + 1e-16))
    if divergence_frac > 0:
        phi_hat = fft2(np.real(ifft2(phi_hat))) * (sqrtS / (np.abs(sqrtS) + 1e-16))

    # main time loop
    for t in range(timesteps):
        # OU update in spectral space
        psi_hat = rho * psi_hat + noise_scale * new_scalar_hat()
        if divergence_frac > 0:
            phi_hat = rho * phi_hat + noise_scale * new_scalar_hat()

        # compute velocity components
        # Streamfunction (incompressible): u = d(psi)/dy, v = -d(psi)/dx
        u_hat_stream = iky * psi_hat
        v_hat_stream = -ikx * psi_hat

        # Potential (divergent) part from potential phi: u = d(phi)/dx, v = d(phi)/dy
        u_hat_pot = ikx * phi_hat if divergence_frac > 0 else 0.0
        v_hat_pot = iky * phi_hat if divergence_frac > 0 else 0.0

        # combine with divergence_frac weight (phi energy fraction)
        u_hat = (1.0 - divergence_frac) * u_hat_stream + divergence_frac * u_hat_pot
        v_hat = (1.0 - divergence_frac) * v_hat_stream + divergence_frac * v_hat_pot

        u = np.real(ifft2(u_hat))
        v = np.real(ifft2(v_hat))

        # enforce target RMS for fluctuating part
        curr_rms = np.sqrt(np.mean(u**2 + v**2))
        if curr_rms > 0:
            scale = target_rms / curr_rms
            u *= scale
            v *= scale

        # add mean wind for this level
        u += mean_profile[0]
        v += mean_profile[1]

        # put into output
        winds[t, :, :, 0] = u
        winds[t, :, :, 1] = v

    return winds
