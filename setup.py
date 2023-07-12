from setuptools import setup, find_packages
from Cython.Build import cythonize

setup(
    name='example',
    version='0.1.0',
    packages=find_packages(include=['jelly', 'jelly.*']),
    ext_modules=cythonize(
        ["jelly/*.pyx"],
        compiler_directives={'language_level': "3"},
        annotate=True
    )
)
