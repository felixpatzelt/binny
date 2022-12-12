from setuptools import setup, find_packages

setup(
    name='binny',
    version='0.6.2',
    description=(
        'Tools for binning'
    ),
    long_description="""
        Bin one-dimensional data and calculate conditional expectation
        values.
    """,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    keywords=[
        'Binning', 'Conditional expectation'
    ],
    url='http://github.com/felixpatzelt/binny',
    download_url=(
      'https://github.com/felixpatzelt/binny/archive/0.6.1.tar.gz'
    ),
    author='Felix Patzelt',
    author_email='felix@neuro.uni-bremen.de',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
    ],
    include_package_data=True,
    zip_safe=False,
    #test_suite='tests',
)