"""Deterministic simulated fields for the interactive demo and its cover image."""
from io import BytesIO

import numpy as np


def field(epoch):
    rng = np.random.default_rng(41)
    y, x = np.mgrid[-1:1:400j, -1.45:1.45:580j]
    u = x * 0.89 - y * 0.46
    v = (x * 0.46 + y * 0.89) * 2.45
    radius = np.hypot(u, v)
    angle = np.arctan2(v, u)
    arms = (0.5 + 0.5 * np.cos(2 * angle - 6.5 * np.log(radius + 0.09))) ** 3
    dust = rng.lognormal(0, 0.32, x.shape)
    signal = 170 * np.exp(-4.3 * radius) + 430 * np.exp(-radius * 14)
    signal += 210 * np.exp(-3.2 * radius) * arms * dust
    for _ in range(360):
        cx, cy = rng.uniform(-1.45, 1.45), rng.uniform(-1, 1)
        width = rng.uniform(.002, .011)
        signal += rng.uniform(8, 170) * np.exp(-((x-cx)**2 + (y-cy)**2) / (2*width**2))
    noise = np.random.default_rng(50 + epoch).normal(0, 0.95, x.shape)
    return (20 + (1 + epoch * .15) * signal + noise).astype('float32')


def example_files():
    from astropy.io import fits

    files = []
    for epoch in range(2):
        image = fits.PrimaryHDU(field(epoch))
        image.header.update({'OBJECT': 'Simulated spiral field', 'BUNIT': 'adu',
                             'EXPTIME': 60 + epoch * 9, 'CTYPE1': 'RA---TAN',
                             'CTYPE2': 'DEC--TAN', 'CRPIX1': 290.5, 'CRPIX2': 200.5,
                             'CRVAL1': 154., 'CRVAL2': 22., 'CDELT1': -.00025, 'CDELT2': .00025})
        buf = BytesIO()
        image.writeto(buf)
        files.append((f'spiral-field-{epoch+1:02}.fits', buf.getvalue()))
    return files
