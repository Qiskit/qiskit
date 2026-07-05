
"""
Module to expose more detailed version info for the installed `numpy`
"""
version = "2.5.1"
__version__ = version
full_version = version

git_revision = "5e1d03ffac5f2c0a9c39bfcaa9fc853b2b83151e"
release = 'dev' not in version and '+' not in version
short_version = version.split("+")[0]
