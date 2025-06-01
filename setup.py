"""Setup script for LLM Counting Mechanisms package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="llm-counting-mechanisms",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Behavioral analysis and causal mediation for LLM counting tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/llm-counting-mechanisms",
    project_urls={
        "Bug Tracker": "https://github.com/your-username/llm-counting-mechanisms/issues",
        "Documentation": "https://github.com/your-username/llm-counting-mechanisms",
        "Source Code": "https://github.com/your-username/llm-counting-mechanisms",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.812",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "llm-counting-benchmark=scripts.run_benchmark:main",
            "llm-counting-causal=scripts.run_causal_analysis:main",
            "llm-counting-plots=scripts.generate_plots:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords="llm transformer counting causal-analysis mechanistic-interpretability",
    zip_safe=False,
)
