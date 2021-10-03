"""Setup script for rl_games"""

import sys
import os
import pathlib

from setuptools import setup, find_packages
# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()
print(find_packages())

setup(name='rl-games',
      long_description=README,
      long_description_content_type="text/markdown",
      url="https://github.com/Denys88/rl_games",
      #packages=[package for package in find_packages() if package.startswith('rl_games')],
      packages = ['.','rl_games','docs'],
      package_data={'rl_games':['*'],'docs':['*'],},
      version='1.1.1',
      author='Denys Makoviichuk, Viktor Makoviichuk',
      author_email='trrrrr97@gmail.com, victor.makoviychuk@gmail.com',
      license="MIT",
      classifiers=[
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
      ],
      #packages=["rlg"],
      include_package_data=True,
      install_requires=[
            # this setup is only for pytorch
            # 
            'gym>=0.17.2',
            # 'gym[atari]',
            # 'gym[box2d]',
            'torch>=1.7.0',
            'numpy>=1.16.0',
            'ray>=1.1.0',
            'tensorboard>=1.14.0',
            'tensorboardX>=1.6',
            'setproctitle',
            'psutil',
            'pyyaml'
            # Optional dependencies
            # 'opencv-python>=4.1.0.25',
            # 'tensorflow-gpu==1.14.0',
            # 'gym-super-mario-bros==7.1.6',
            # 'pybullet>=2.5.0',
            # 'smac',
            # 'dm_control',
            # 'dm2gym',
      ],
      )
