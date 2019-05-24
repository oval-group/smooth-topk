from setuptools import setup, find_packages

__version__ = "1.0"

setup(name='topk',
      description='Implementation of Smooth Loss functions for Deep Top-k Classification',
      author='Leonard Berrada',
      packages=find_packages(),
      license="GNU General Public License",
      url='https://github.com/oval-group/smooth-topk',
      version=str(__version__),
      install_requires=["torch>=1.0",
                        "numpy"])
