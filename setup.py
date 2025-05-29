from setuptools import setup, find_packages

setup(
    name='forcearray',
    version='1.0',
    packages=find_packages(),
    install_requires=['rasterio', 'dask[complete]'],
    python_requires='>=3.6',
)