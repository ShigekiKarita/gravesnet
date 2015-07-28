from setuptools import setup, find_packages
import sys

sys.path.append('./src')
sys.path.append('./tests')

setup(
    name = 'gravesnet',
    version = '0.1',
    description='This is test codes for travis ci',
    packages = find_packages(),
    tests_require=['nose'],
)
