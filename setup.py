from setuptools import setup


def readme():
    with open("README.md") as f:
        return f.read()


setup(
    name="image_super_resolution",
    version="0.0.1",
    description="Implementation of super-resolution algorithms",
    long_description=readme(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    url="https://github.com/AnasEss/fast-single-image-super-resolution-using-new-analytical-solution",
    author="Anas ESSOUNAINI",
    author_email="essounaini97@gmail.com ",
    keywords="computer vision - image processing - super resolution ",
    license="MIT",
    packages=[
        "image_super_resolution",
    ],
    install_requires=[],
    include_package_data=True,
)