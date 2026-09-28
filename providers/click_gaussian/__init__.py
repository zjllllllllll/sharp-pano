"""Public API, loaded lazily so runtime selection needs only Python's stdlib."""
from importlib import import_module

__all__ = ['ClickGaussianConfig', 'GaussianModel', 'ClickGaussianTrainer']
_MODULES = {
    'ClickGaussianConfig': '.config',
    'GaussianModel': '.gaussian_model',
    'ClickGaussianTrainer': '.trainer',
}


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    value = getattr(import_module(_MODULES[name], __name__), name)
    globals()[name] = value
    return value
