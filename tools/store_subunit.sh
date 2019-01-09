#!/usr/bin/env bash

# If stestr is not run we don't have anything to populate in the db
if [ ! -d ".stestr" ] ; then
    exit 0
fi

pip install -U psycopg2-binary
# clone subunit2sql until release with fix for xfail happens
git clone git://git.openstack.org/openstack-infra/subunit2sql
pip install -U ./subunit2sql

METADATA="build_num:$TRAVIS_BUILD_NUMBER,event_type:$TRAVIS_EVENT_TYPE,job_number:$TRAVIS_JOB_NUMBER,commit:$TRAVIS_COMMIT,job_name:$TRAVIS_JOB_NAME"

if [ "$TRAVIS_PULL_REQUEST" != "false" ] ; then
    METADATA+=",pr_num:$TRAVIS_PULL_REQUEST,sha1:$TRAVIS_PULL_REQUEST_SHA,pr_origin:$TRAVIS_PULL_REQUEST_SLUG"
fi

echo "$SUBUNIT_DB_URI"
stestr last --subunit | subunit2sql --database-connection="$SUBUNIT_DB_URI" --artifacts="$TRAVIS_JOB_WEB_URL" --run_meta="$METADATA"
