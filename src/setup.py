from setuptools import setup, find_packages

setup(name='super-mario-dqn',
      version='1.0',
      description='DQN with Super Mario',
      author='laurenmoos',
      author_email='lauren.a.moos@gmail.com',
      url='https://github.com/laurenmoos/supermario',
      packages=find_packages(exclude=('tests', 'docs')))
