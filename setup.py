import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="steinerpy", # Replace with your own username
    version="2.0.0",
    author="Kenny Chour",
    author_email="ckennyc@tamu.edu, chour.kenny@yahoo.com ",
    description="Heuristic approach to approximate a steiner tree, \
        based on the Primal-Dual algorithm. Implemented as multi-directional search\
            with several path termination criteria",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Kchour/steinerpy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Linux",
    ],
    python_requires='>=3.6',
)