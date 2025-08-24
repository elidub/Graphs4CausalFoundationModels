from setuptools import setup, find_packages

setup(
    name="CausalPriorFitting",
    version="0.1.0",
    description="CausalPriorFitting package",
    author="Arik Reuter",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.7",
)
