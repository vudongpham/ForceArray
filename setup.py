from setuptools import setup, find_packages

setup(
    name='forcearray',
    version='1.0',
    packages=find_packages(),
    install_requires=['rasterio', 'fastnanquantile'],
    python_requires='>=3.6',
)