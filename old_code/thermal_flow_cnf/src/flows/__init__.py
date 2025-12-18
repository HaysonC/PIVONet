from .uniform import uniform_flow
from .couette import couette_flow
from .poiseuille import poiseuille_flow
from .diffuser import diffuser_flow
from .compressor import compressor_flow
from .bend import bend_flow

__all__ = [
    "poiseuille_flow",
    "diffuser_flow",
    "compressor_flow",
    "bend_flow",
]
__all__ = ["uniform_flow", "couette_flow", "poiseuille_flow"]
