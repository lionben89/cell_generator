import setuptools

__version__ = '1.0.0'

with open("README.md", "r") as fh:
    readme = fh.read()

requirements = [
    "matplotlib",
    "numpy",
    "tifffile",
    "scikit-image>=0.15.0",
    "aicsimageio==3.0.7",
    "quilt3",
    "bumpversion",
    "twine",
    "setuptools>=42",
    "wheel",
    "pandas",
    "multipledispatch",
    "cell_imaging_utils",
    "tqdm",
    "sklearn",
    "opencv-python",
    "scipy",
    "seaborn"
]


setuptools.setup(
    author="Lion Ben Nedava",
    author_email="lionben89@gmail.com",
    install_requires=requirements,
    license="MIT",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="cell_generator",
    name="cell_generator",
    packages=setuptools.find_packages(exclude=["images"]),
    python_requires=">=3.7",
    test_suite="tests",
    url="https://github.com/lionben89/cell_generator",
    # Do not edit this string manually, always use bumpversion
    # Details in CONTRIBUTING.rst
    version=__version__,
    zip_safe=False
)
