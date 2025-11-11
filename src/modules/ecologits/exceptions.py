"""
Custom exceptions for ecologits module.
"""


class EcologitsError(Exception):
    """Base exception for ecologits module."""
    pass


class ModelingError(EcologitsError):
    """Exception raised for errors in impact modeling."""
    pass
