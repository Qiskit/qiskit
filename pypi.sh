#!/bin/bash
# This is a helper script to upload new versions of the SDK
# Author: Juan Gomez

# Please provide a valid username and password to connect to PyPi for uploading
# the QISKit SDK distributable packaged
USERNAME=""
PASSWORD=""



usage(){
    echo "Usage:"
    echo "$0 [OPTION]"
    echo "Helper script to create and upload QISKit SDK distributable package"
    echo "to PyPi severs."
    echo ""
    echo "Options:"
    echo "-u <username>   Specify a registered PyPi username"
    echo "-p <password>   The password of the username"
    echo "-h              Shows this help"
    echo ""
    exit 1
}

while getopts "u:p:h" optname
  do
    case "$optname" in
      "h")
        usage
        ;;
      "u")
        USERNAME=$OPTARG
        ;;
      "p")
        PASSWORD=$OPTARG
        ;;
      ":")
        echo "No argument value for option $OPTARG"
        usage
        ;;
      *)
      # Should not occur
        echo "Unknown error while processing options"
        usage
        ;;
    esac
  done

self=$0
date > $self.log
echo "**********************" >> $self.log

confirm() {
    # call with a prompt string or use a default
    read -r -p "${1:-Are you sure? [y/N]} " response
    case "$response" in
        [yY])
            return 1
            ;;
        *)
            return 0
            ;;
    esac
}

check_twine() {
    twine upload -h &>> $self.log
    if [[ $? != 0 ]]
    then
        return 0
    else
        return 1
    fi
}

install_twine() {
    pip install twine &>> $self.log
    if [[ $? != 0 ]]
    then
        return 0
    else
        return 1
    fi
}

diagnose() {
    echo "Let's see if I can figure out what happened..."
    echo -n "Is there a USERNAME?..."
    if [[ $USERNAME == "" ]]
    then
        echo -e "[NO]"
        echo "-----------------------"
        echo "Seems like you haven't provide a USERNAME!"
        echo "Please edit $self and add a registered username and password"
        echo "or use the command options: -u <USERNAME> -p <PASSWORD>"
        return
    fi
    echo -e "[YES]"
    echo -n "Is there a PASSWORD?..."
    if [[ $PASSWORD == "" ]]
    then
        echo -e "[NO]"
        echo "-----------------------"
        echo "Ok, so you provided a USERNAME but I can't seem to find the"
        echo "the corresponding PASSWORD."
        echo "Please edit $self and add the correct password"
        echo "or use the command options: -u <USERNAME> -p <PASSWORD>"
        return
    fi

    echo -e "[YES]"
    echo -n "Is there connection to upload.pypi.org?..."
    wget -q https://upload.pypi.org/ -O /dev/null &>> $self.log
    if [[ $? != 0 ]]
    then
        echo -e "[NO]"
        echo "-----------------------"
        echo "Seems like there's no connection to PyPi, or PyPi is down."
        echo "Please make sure there's a working internet connection."
        echo "If you have internet connection, the PyPi site may be down."
        echo "Please try again later."
        return

    fi
    echo -e "[YES]"
    echo "Seems like your credentials may be incorrect."
    echo "Please make sure you have introduced the correct username and "
    echo "password in the $self script."
}

echo -n "Clobbering build..."
# Clobbering build to rebuild it later
rm -rf qiskit.egg-info build dist
echo -e "[OK]"

echo -n "Building distributable package..."
# Let's build the wheel package
python setup.py sdist bdist_wheel -p manylinux1_x86_64 &>> $self.log
if [ $? != 0 ]
then
    echo -e "[ERROR]"
    echo "--------------------------"
    echo "Something wrong has happened!!. Please make sure that you are in the "
    echo "project root directory and there's a setup.py file in there."
    exit 2
fi
echo -e "[OK]"

echo -n "Checking for twine tool..."
check_twine
if [[ $? == 0 ]] # if it's ... false :)
then
    echo -e "[ERROR]"
    echo "Seems like you don't have Twine installed in yout system."
    echo "Twine tool is necessary for uploading the package to PyPi."
    echo ""
    confirm "Do you want to install Twine? [Y/n]"
    if [[ $? == 1 ]]
    then
        install_twine
        if [[ $? == 0 ]]
        then
            echo "Couldn't install Twine, please try to install twine by yourself:"
            echo "$ pip install twine"
            exit 3
        else
            echo "Twine successfully installed!"
        fi
    else
        echo "Install Twine by yourself and try again later!"
        exit 4
    fi
fi
echo -e "[OK]"

echo -n "Uploading distributable package to PyPi..."
twine upload -u $USERNAME -p $PASSWORD dist/* &>> $self.log
if [[ $? != 0 ]]
then
    echo -e "[ERROR]"
    echo "--------------------------"
    echo "Error! Couldn't upload the distributed package to PyPi!!"
    diagnose
    exit 5
fi

echo -e "[OK]"
echo "Done!"
echo "Remember to create a TAG in the repo with the same version you just uploaded!"
rm -f $self.log
exit 0
