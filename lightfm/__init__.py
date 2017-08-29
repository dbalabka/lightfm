try:
    __LIGHTFM_SETUP__
except NameError:
    from .lightfm import LightFM
    from .lightfm import CYTHON_DTYPE, ID_DTYPE

__version__ = '1.13.6'

__all__ = ['LightFM', 'datasets', 'evaluation']
