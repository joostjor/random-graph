from setuptools import setup

from __version__ import version

with open("requirements.txt") as req_file:
    requirements = req_file.readlines()

test_requirements = ["pytest>=5.0.0"]
setup(
    name="girg-spreading",
    version=version,
    packages=["spreading_lib", "graph_lib"],
    install_requires=requirements,
    tests_require = test_requirements
)