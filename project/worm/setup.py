from setuptools import setup

setup(
    name='simple_worm',
    version='0.0.1',
    description='Python implementation of numerical method for visco-elastic rods.',
    author='Tom Ranner, Tom Ilett',
    url='https://gitlab.com/tom-ranner/simple-worm',
    packages=['simple_worm'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    install_requires=[
        'fenics == 2019.1.0',
        'numpy >= 1.19, <1.20'
    ],
    extras_require={
        'test': [
            'pytest'
        ],
        'inv': [
            'dolfin_adjoint @ git+https://github.com/dolfin-adjoint/pyadjoint.git@1c9c15c1fa2c1a470826143ce98b721ebd00facd',
            'torch >= 1.7, <= 1.8',
            'matplotlib >= 3.3',
            'scikit-learn >= 0.24',
            'tensorboard == 2.4.1',
        ]
    },
    python_requires='>=3.9, <3.10'
)
