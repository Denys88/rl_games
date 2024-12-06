"""Setup script for rl_games"""

import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(name='rl-games',
      version='1.6.5',
      author='Denys Makoviichuk, Viktor Makoviichuk',
      author_email='trrrrr97@gmail.com, victor.makoviychuk@gmail.com',
      description='Reinforcement learning framework for games and robotics',
      long_description=README,
      long_description_content_type="text/markdown",
      url="https://github.com/Denys88/rl_games",
      license="MIT",
      classifiers=[
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12"
      ],
      packages=find_packages(include=["rl_games", "rl_games.*"]),
      include_package_data=True,
      package_data={
            "rl_games": ["*.py", "**/*.py"],
            "docs": ["*.md", "*.rst"],
      },
      install_requires=[
            'gym>=0.17.2',
            'torch>=2.0.1',
            'numpy>=1.16.0',
            'tensorboard>=1.14.0',
            'tensorboardX>=1.6',
            'setproctitle',
            'psutil',
            'pyyaml',
            'watchdog>=2.1.9',  # for evaluation process
      ],
      extras_require={
            "wandb": ["wandb>=0.12.11"],
      },
      )
