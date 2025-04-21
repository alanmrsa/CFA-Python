from setuptools import setup, find_packages

setup(
    name='fair-cause',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas>=2.2.3',
        'numpy>=1.19.0',
        'scikit-learn>=1.2.2',
        'seaborn>=0.13.2', 
        'statsmodels>=0.14.4'
    ],
    author='Alan Ma',
    author_email='alanmrsa@gmail.com',
    description='{Python implementation of Plecko and Bareinboim Causal Fairness Analysis}',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='[https://github.com/your-username/fair-cause'](https://github.com/your-username/fair-cause',)
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.8',
    ],
    python_requires='>=3.8',
)
