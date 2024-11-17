from setuptools import setup, find_packages

setup(
    name="ninjax",
    version="0.0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    entry_points={
        'console_scripts': [
            'ninjax_analysis=ninjax.analysis:main',
        ],
    },
)