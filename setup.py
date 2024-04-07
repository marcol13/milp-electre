from setuptools import setup, find_packages

VERSION = '0.0.2'
DESCRIPTION = 'Mixed-integer Linear Programming (MILP) for ELECTRE method.'

classifiers = [
    "Development Status :: 1 - Planning",
    "Intended Audience :: Developers",
    "Programming Language :: Python :: 3",
    "Operating System :: Unix",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: Microsoft :: Windows",
]

# Setting up
setup(
    name="pyElectreMILP",
    version=VERSION,
    author="marcol13 (Marcin Krueger)",
    author_email="<marcinkrueger@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=open("README.md").read() + '\n\n' + open("CHANGELOG.md").read(),
    packages=find_packages(),
    license='MIT',
    keywords=['python', 'electre', 'milp', 'ranking', 'multi-criteria', 'decision-making'],
    classifiers=classifiers,
    # TODO add requirements
    install_requires=[],
    extras_require={
        "dev": []
    },
    python_requires=">=3.9"
)