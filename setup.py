from setuptools import setup

extra_requirements = {
    "tests": ["pytest", "coverage", "pytest-cov"],
    "docs": [
        "sphinx",
        "sphinx-autodoc-typehints",
        "furo",
        "sphinx-copybutton",
        "nbsphinx",
        "ipython",
    ],
}

setup(
    name="obara-saika-debug",
    author="Sarai D. Folkestad",
    description="Code to debug integrals",
    install_requires=["numpy", "scipy", "mpmath"],
    extras_require=extra_requirements,
)
