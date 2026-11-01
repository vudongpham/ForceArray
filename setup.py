from setuptools import setup, find_packages

setup(
    name='forcearray',
    version='1.1',
    packages=find_packages(),
    install_requires=['numpy', 'rasterio', 'numba'],
    python_requires='>=3.6',
)