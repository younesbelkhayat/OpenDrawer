from setuptools import find_packages, setup

setup(
    name="opendrawer",
    version="0.1.0",
    description="Photo → page de coloriage pour enfant, 100 % local (agent + CLI)",
    packages=find_packages(exclude=("tests", "tests.*")),
    python_requires=">=3.10",
    install_requires=[
        "pyyaml>=6.0",
        "pydantic>=2.0",
        "jsonschema>=4.20",
        "python-dotenv>=1.0",
        "Pillow>=10.0",
        "numpy>=1.24",
        "scipy>=1.10",
        "scikit-image>=0.22",
        "opencv-python-headless>=4.9,<5",
        "requests>=2.31",
        "huggingface_hub>=0.24",
    ],
    entry_points={"console_scripts": ["opendrawer=opendrawer.cli:main"]},
)
