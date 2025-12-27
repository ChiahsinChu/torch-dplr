# SPDX-License-Identifier: LGPL-3.0-or-later
"""torch-dplr: DPLR in PyTorch backend."""

from ._version import __version__
from .dipole_charge import DipoleChargeModifier

__all__ = ["__version__", "DipoleChargeModifier"]
