from setuptools import setup, find_packages

setup(
    name="e2e-iris",
    version="0.0.1",
    author="Rasheed Ibraheem",
    author_email="ibraheem.abdulrasheed@gmail.com",
    maintainer="ibraheem.abdulrasheed@gmail.com",
    maintainer_email="ibraheem.abdulrasheed@gmail.com",
    description="End-to-end classification problem using the iris dataset",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires=">=3.10",
)
