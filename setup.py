#!/usr/bin/python
import setuptools

setuptools.setup(
    name='emma',
    version='0.1',
    url='http://github.com/allenai/ontoemma',
    packages=setuptools.find_packages(),
    install_requires=[
    ],
    tests_require=[
    ],
    zip_safe=False,
    test_suite='py.test',
    entry_points='',
)