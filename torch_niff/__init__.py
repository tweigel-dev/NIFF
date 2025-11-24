"""
Neural Implicit Frequency Filters (NIFF) for PyTorch

This package implements NIFF layers that can be used as drop-in replacements
for standard convolution layers in PyTorch models.
"""

__version__ = "0.1.0"
__author__ = "Julia Grabinski, Janis Keuper, Margret Keuper"

# Try to import main NIFF modules, handle missing dependencies gracefully
try:
    from .niff_small import *
    from .niff_full import *
    from .niff_big import *
    from .misc_niff import *

    # Import network implementations
    from .mobilenet_niff import *
    from .resnet_niff_full import *
    from .resnet_niff_splitconv import *
    from .densenet_niff_full import *
    from .densenet_niff_splitconv import *
    from .convnext_niff import *
    
    _imports_successful = True
except ImportError as e:
    import warnings
    warnings.warn(
        f"Could not import all NIFF modules. Make sure PyTorch is installed. Error: {e}",
        ImportWarning
    )
    _imports_successful = False
