from setuptools import setup

setup(name='opencog-gym',
      version='0.0.1',
      packages=['opencog_gym', 'envs', 'envs.gym_chase'],
      install_requires=['gym', 'orderedmultidict']
)
