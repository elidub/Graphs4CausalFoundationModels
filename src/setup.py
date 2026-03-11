from setuptools import setup, find_packages

setup(
    name="gcfm",
    version="0.1.0",
    package_dir={"gcfm": ""},
    packages=["gcfm"] + ["gcfm." + p for p in find_packages()],
)
