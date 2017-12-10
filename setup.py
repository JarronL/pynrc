"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

root = path.abspath(path.dirname(__file__))

#with open(path.join(root, NAME, 'VERSION'), encoding='utf-8') as f:
#    version = f.read().strip()
from pynrc.version import __version__
version = __version__

# Get the long description from the README file
with open(path.join(root, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='pynrc',

    # Versions should comply with PEP440.
    version=version,

    description='JWST NIRCam ETC and simulation project',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/JarronL/pynrc',

    # Author details
    author='Jarron Leisenring',
    author_email='jarronl@email.arizona.edu',
    license='MIT',
    keywords='jwst nircam etc simulator',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Astronomy',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
      install_requires=[
          'numpy>=1.10.0',
          'matplotlib>=1.5.0',
          'scipy>=0.16.0',
          'astropy>=1.2.0',
          'pysynphot>=0.9',
          'poppy>=0.6.1',
          'webbpsf>=0.6.0',
          'jwxml>=0.3.0'
      ],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    #extras_require={
    #    'dev': ['check-manifest>=0.34', 'lxml>=3.6.4', 'pytest>=3.0.2'],
    #},

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    #package_data={
    #    'pynrc': ['package_data.dat'],
    #},
)