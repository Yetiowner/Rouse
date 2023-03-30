from setuptools import setup, find_packages

setup(
    name='Rouse',
    version='0.1',
    description='A python package for my rouse research project for data-cleaning using mutual bootstrapping',
    author='Oliver Gibson',
    author_email='ojrgibson@perse.co.uk',
    url='https://github.com/Yetiowner/Rouse',
    packages=find_packages(),
    install_requires=[
        "keras>=2.9.0",
        "opencv-python>=4.5.1.48",
        "matplotlib>=3.2.2",
        "numpy>=1.21.6",
        "tensorflow>=2.9.1"
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'
    ],
    license='MIT'
)