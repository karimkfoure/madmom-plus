#!/usr/bin/env python
# encoding: utf-8
"""
This file contains the setup for setuptools to distribute everything as a
(PyPI) package.

"""

import glob
import os
from pathlib import Path
from runpy import run_path

import numpy as np
from Cython.Build import cythonize, build_ext
from setuptools import Extension, setup, find_packages

# define version
version = run_path('madmom/_version.py')['__version__']

if not Path('madmom/models/__init__.py').is_file():
    raise RuntimeError('Model files are missing. Run '
                       '`git submodule update --init --recursive` before '
                       'building, or use a complete source distribution.')

# define which extensions to compile
include_dirs = [np.get_include()]

extensions = [
    Extension(
        'madmom.audio.comb_filters',
        ['madmom/audio/comb_filters.pyx'],
        include_dirs=include_dirs,
    ),
    Extension(
        'madmom.features.beats_crf',
        ['madmom/features/beats_crf.pyx'],
        include_dirs=include_dirs,
    ),
    Extension('madmom.ml.hmm', ['madmom/ml/hmm.pyx'], include_dirs=include_dirs),
    Extension(
        'madmom.ml.nn.layers', ['madmom/ml/nn/layers.py'], include_dirs=include_dirs
    ),
]

# The streaming backend is opt-in so existing installations need no C++ compiler.
if os.environ.get('MADMOM_BUILD_NATIVE') == '1':
    extensions.append(Extension(
        'madmom._native',
        ['native/src/online.cpp', 'native/src/bindings.cpp'],
        include_dirs=['native/include'],
        language='c++',
        extra_compile_args=['-std=c++17', '-O3', '-ffp-contract=off'],
    ))

# define scripts to be installed by the PyPI package
scripts = glob.glob('bin/*')

# define the models to be included in the PyPI package
package_data = [
    'models/LICENSE',
    'models/README.rst',
    'models/beats/*/*',
    'models/chords/*/*',
    'models/chroma/*/*',
    'models/downbeats/*/*',
    'models/key/*/*',
    'models/notes/*/*',
    'models/onsets/*/*',
    'models/patterns/*/*',
]

# some PyPI metadata
classifiers = [
    'Development Status :: 3 - Beta',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3.12',
    'Programming Language :: Python :: 3.13',
    'Environment :: Console',
    'License :: OSI Approved :: BSD License',
    'License :: Free for non-commercial use',
    'Topic :: Multimedia :: Sound/Audio :: Analysis',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
]

# requirements
requirements = [
    'numpy>=1.23.5',
    'scipy>=1.13',
    'mido>=1.2.6',
]

# docs to be included
try:
    long_description = open('README.rst', encoding='utf-8').read()
    long_description += '\n' + open('CHANGES.rst', encoding='utf-8').read()
except TypeError:
    long_description = open('README.rst').read()
    long_description += '\n' + open('CHANGES.rst').read()

# the actual setup routine
setup(
    name='madmom-plus',
    version=version,
    description='Python audio signal processing library',
    long_description=long_description,
    author='Department of Computational Perception, Johannes Kepler '
    'University, Linz, Austria and Austrian Research Institute for '
    'Artificial Intelligence (OFAI), Vienna, Austria',
    author_email='madmom-users@googlegroups.com',
    url='https://github.com/karimkfoure/madmom-plus',
    project_urls={
        'Upstream': 'https://github.com/CPJKU/madmom',
        'Documentation': 'https://madmom.readthedocs.io/',
    },
    license='BSD, CC BY-NC-SA',
    packages=find_packages(exclude=['tests', 'docs']),
    ext_modules=cythonize(extensions, compiler_directives={'language_level': 3}),
    package_data={'madmom': package_data},
    exclude_package_data={'': ['tests', 'docs']},
    scripts=scripts,
    install_requires=requirements,
    cmdclass={'build_ext': build_ext},
    python_requires='>=3.9',
    extras_require={'test': ['pytest>=7']},
    classifiers=classifiers,
)
