#Initialization file for Qconfig

#Tries to import Qtoken and load parameters from there into Qconfig.
#If unable to find Qtoken, prompts user for credentials and offers to save/install them in Qtoken

#TO DO: Add methods to support additional features: display/update Qtoken info, switch between different Qtokens, specify additional parameters like default values for backend, shots, timeout

#First try to import Qtoken
try:
    import Qtoken
    config = Qtoken.config
    APItoken = Qtoken.APItoken
    #TO DO: consider adding optional "quiet" flag to Qtoken to suppress the following print statement
    print('\nQconfig: loaded API token from %s for access to %s' % (Qtoken.__file__, config['url']))
#If unable to load Qtoken, prompt user for API token and URL. Create Qtoken with this info if desired
except ModuleNotFoundError:
    import getpass
    import os
    import pip
    print('\nQconfig: unable to find a token.')
    DEFAULT_API_URL = 'https://quantumexperience.ng.bluemix.net/api'
    print('\nPlease enter the API access URL below, or simply press enter to use the default URL for the IBM Quantum Experience, %s' % DEFAULT_API_URL)
    actual_api_url = input()
    if len(actual_api_url) == 0:
        actual_api_url = DEFAULT_API_URL
    config = dict()
    config['url'] = actual_api_url
    print('\nUsing API URL %s' % actual_api_url)
    print('\nPlease paste your API token below. For security, your token will not be displayed. Any leading or trailing whitespace will be removed. IBM Quantum Experience users can obtained an API token from the "API Token" section of the page at https://quantumexperience.ng.bluemix.net/qx/account')
    APItoken = getpass.getpass('').strip()
    #TO DO: validate API URL and token before proceeding; if unable to validate, inform user and ask permission before proceeding
    print('\nPlease indicate how you would like to use this URL and token:\n 0 (default) = Create Qtoken package and install in active environment (this option is recommended for most users)\n 1 = Create Qtoken.py file in current working directory (%s)\n 2 = Use these credentials for the current session only' % os.getcwd())
    token_option = input()
    if token_option != '2': #if option 2, no further action needed; otherwise, make a Qtoken.py file and possibly pip install it
        #if installing Qtoken in current environment, create a folder for the Qtoken package. Otherwise just create a local Qtoken.py file.
        if token_option == '1':
            qt_file_loc = 'Qtoken.py'
        else:
            qt_file_loc = 'Qtoken/Qtoken/__init__.py'
            if not os.path.exists('Qtoken'):
                os.mkdir('Qtoken')
            if not os.path.exists('Qtoken/Qtoken'):
                os.mkdir('Qtoken/Qtoken')
        if os.path.exists(qt_file_loc):
            print('File %s exists; overwrite? (y/n)' % qt_file_loc)
            overwrite = input()
            if not 'y' in str.lower(overwrite):
                print('Aborting Qconfig setup. Move the file at %s elsewhere before trying again, or choose a different Qconfig setup option' % qt_file_loc)
                exit
        with open(qt_file_loc,'w') as qt_file:
            qt_file.write('APItoken = "%s"\n' % APItoken)
            qt_file.write('config = {"url": "%s"}\n' % config['url'])
        print('\nAPI parameters saved in %s. To update them, edit this file.' % qt_file_loc) #this is true even if the pip install option is 
        #if option 1, we're done; otherwise add a setup.py file and run pip to install the Qtoken package
        if token_option != '1':
            #make a setup.py file in that folder
            with open('Qtoken/setup.py','w') as qt_file:
                qt_file.write('from setuptools import setup\n')
                qt_file.write('setup(name="Qtoken",version="1.0",py_modules=["Qtoken"])')
            #do the installation with pip
            pip_result = pip.main(['install','-e','Qtoken'])
            if pip_result == 0:
                print('\nInstalled Qtoken package in current environment. This package remains linked to (and dependent on) the file at %s' % qt_file_loc)
            else:
                print('\nAn error occurred installing the Qtoken package.')