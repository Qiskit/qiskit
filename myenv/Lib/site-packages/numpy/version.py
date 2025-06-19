
"""
Module to expose more detailed version info for the installed `numpy`
"""
version = "2.3.0"
__version__ = version
full_version = version

git_revision = "0532af47d6a815298b7841de00bdbc547104b237"
release = 'dev' not in version and '+' not in version
short_version = version.split("+")[0]
