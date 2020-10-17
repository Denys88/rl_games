"""Setup script for rl_games"""

import sys
import os

from setuptools import setup, find_packages

print(find_packages())

setup(name='rl_games',
      packages=[package for package in find_packages()
                if package.startswith('rl_games')],
      version='1.0b',
      author='Denys Makoviichuk, Viktor Makoviichuk',
      author_email='trrrrr97@gmail.com, victor.makoviychuk@gmail.com',
      install_requires=[
            # this setup is only for pytorch
            # 
            'gym>=0.17.2',
            'torch>=1.6',
            'numpy>=1.16.0',
            'ray==1.0.0',
            'tensorboard>=1.14.0',
            'tensorboardX>=1.6',
            'opencv-python>=4.1.0.25',
            'setproctitle',
            'psutil',
            'pyyaml'
            # Optional dependencies
            # 'tensorflow-gpu==1.14.0',
            # 'gym-super-mario-bros==7.1.6',
            # 'pybullet>=2.5.0',
            # 'smac',
            # 'dm_control',
            # 'dm2gym',
      ],
      )
