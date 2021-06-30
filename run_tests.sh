#! /bin/bash


# if one test fails, exit immediately without continue for other tests
set -e

# output formatting
separator=""
blue=""
green=""
red=""
noColor=""

if [[ -t 1 ]] # if stdout is a terminal, use colorful format
then
    separator=$'--------------------------------------------------------------------------------\n'
    blue="$(tput bold; tput setaf 4)"
    green="$(tput bold; tput setaf 2)"
    red="$(tput bold; tput setaf 1)"
    noColor="$(tput sgr0)"
fi

function print_usage {
    echo "run_tests.sh [--unittests] [--coverage] "
    echo ""
    echo "ssc_scoring unit testing utilities."
    echo ""
    echo "Examples:"
    echo "./run_tests.sh -u --coverage  # run style checks, full tests, print code coverage (${green}recommended for pull requests${noColor})."
    echo "./run_tests.sh -u             # run unit tests."
    echo "./run_tests.sh --unittests    # run unit tests."
    exit 1  # raise error
}

function print_error_msg() {
    echo "${red}Error: $1.${noColor}"
    echo ""
}


doCoverage=false
doUnitTests=false

if [ -z "$1" ]
then
    print_error_msg "Too few arguments to $0"
    print_usage
fi

# parse arguments
while [[ $# -gt 0 ]]  # if number of arguments is greater than 0
do
    key="$1"
    case $key in
        --coverage)
            doCoverage=true
        ;;
        -u|--u*)  # allow --unittest | --unittests | --unittesting  etc.
            doUnitTests=true
        ;;
        *)
            print_error_msg "Incorrect commandline provided, invalid key: $key"
            print_usage
        ;;
    esac  # corresponding to case
    shift
done



# home directory
homedir="$( cd -P "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$homedir"

# add home directory to python path
export PYTHONPATH="$homedir:$PYTHONPATH"
echo "PYTHONPATH: $PYTHONPATH"

# by default do nothing
cmdPrefix=""


# testing command to run
PY_EXE="python"
cmd="${PY_EXE}"


# set coverage command
if [ $doCoverage = true ]
then
    echo "${separator}${blue}coverage${noColor}"
    cmd="${PY_EXE} -m coverage run --append"
fi


# unit tests
if [ $doUnitTests = true ]
then
    echo "${separator}${blue}unittests${noColor}"
    ${cmdPrefix}${cmd} ./tests/runner.py
fi

# report on coverage
if [ $doCoverage = true ]
then
    echo "${separator}${blue}coverage${noColor}"
    ${cmdPrefix}${PY_EXE} -m coverage combine --append .coverage/
    ${cmdPrefix}${PY_EXE} -m coverage report
fi
