from setuptools import find_packages, setup

setup(
    name='pascal',
    version='0.1.0',
    description='PASCAL - Pan-Arctic Behavioural and Life-history Simulator for Calanus',
    packages=find_packages(include=['pascal', 'pascal.*']),
)
