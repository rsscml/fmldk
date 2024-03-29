
from pathlib import Path
from setuptools import setup

# The directory containing this file
this_directory = Path(__file__).parent
long_description  = (this_directory / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="fmldk",
    version="1.2.5",
    description="Forecast ML library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Rahul Sinha",
    author_email="rahul.sinha@unilever.com",
    packages=["sage","sage_gpu","tfr","tft","tft_gpu","stctn","ctfr","ctfrv2","ctfrv2_gpu","eda"],
    include_package_data=True,
    install_requires=["tensorflow", "tensorflow_probability", "numpy", "statsmodels", "pandas", "joblib", "pathlib", "bokeh", "holoviews", "hvplot", "ennemi"]
)