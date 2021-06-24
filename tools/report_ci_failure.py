# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Utility module to open an issue on the repository when CIs fail."""

import os
import re

from github import Github


class CIFailureReporter:
    """Instances of this class can report to GitHub that the CI is failing."""

    stable_branch_regex = re.compile(r"^stable/\d+\.\d+")

    def __init__(self, repository, token):
        """
        Args:
            repository (str): a string in the form 'owner/repository-name'
                indicating the GitHub repository to report against.
            token (str): a GitHub token obtained following:
                https://help.github.com/articles/creating-a-personal-access-token-for-the-command-line/
        """
        self._repo = repository
        self._api = Github(token)

    def report(self, branch, commit, infourl=None, job_name=None):
        """Report on GitHub that the specified branch is failing to build at
        the specified commit. The method will open an issue indicating that
        the branch is failing. If there is an issue already open, it will add a
        comment avoiding to report twice about the same failure.

        Args:
            branch (str): branch name to report about.
            commit (str): commit hash at which the build fails.
            infourl (str): URL with extra info about the failure such as the
                build logs.
            job_name (str): name of the failed ci job.
        """
        if branch != "main" and not self.stable_branch_regex.search(branch):
            return
        key_label = self._key_label(branch, job_name)
        issue_number = self._get_report_issue_number(key_label)
        if issue_number:
            self._report_as_comment(issue_number, branch, commit, infourl)
        else:
            self._report_as_issue(branch, commit, infourl, job_name)

    def _key_label(self, branch_name, job_name):
        if job_name == "Randomized tests":
            return "randomized test"
        elif job_name == "Benchmarks":
            return "benchmarks failing"
        elif branch_name == "main":
            return "main failing"
        elif branch_name.startswith("stable/"):
            return "stable failing"
        else:
            return ""

    def _get_report_issue_number(self, key_label):
        query = f'state:open label:"{key_label}" repo:{self._repo}'
        results = self._api.search_issues(query=query)
        try:
            return results[0].number
        except IndexError:
            return None

    def _report_as_comment(self, issue_number, branch, commit, infourl):
        stamp = _branch_is_failing_stamp(branch, commit)
        report_exists = self._check_report_existence(issue_number, stamp)
        if not report_exists:
            _, body = _branch_is_failing_template(branch, commit, infourl)
            message_body = f"{stamp}\n{body}"
            self._post_new_comment(issue_number, message_body)

    def _check_report_existence(self, issue_number, target):
        repo = self._api.get_repo(self._repo)
        issue = repo.get_issue(issue_number)
        if target in issue.body:
            return True

        for comment in issue.get_comments():
            if target in comment.body:
                return True

        return False

    def _report_as_issue(self, branch, commit, infourl, key_label):
        repo = self._api.get_repo(self._repo)
        stamp = _branch_is_failing_stamp(branch, commit)
        title, body = _branch_is_failing_template(branch, commit, infourl)
        message_body = f"{stamp}\n{body}"
        repo.create_issue(title=title, body=message_body, labels=[key_label])

    def _post_new_comment(self, issue_number, body):
        repo = self._api.get_repo(self._repo)
        issue = repo.get_issue(issue_number)
        issue.create_comment(body)


def _branch_is_failing_template(branch, commit, infourl):
    title = f"Branch `{branch}` is failing"
    body = f"Trying to build `{branch}` at commit {commit} failed."
    if infourl:
        body += f"\nMore info at: {infourl}"
    return title, body


def _branch_is_failing_stamp(branch, commit):
    return f"<!-- commit {commit}@{branch} -->"


_REPOSITORY = "Qiskit/qiskit-terra"
_GH_TOKEN = os.getenv("GH_TOKEN")


def _get_repo_name():
    return os.getenv("TRAVIS_REPO_SLUG") or os.getenv("APPVEYOR_REPO_NAME")


def _get_branch_name():
    return os.getenv("TRAVIS_BRANCH") or os.getenv("APPVEYOR_REPO_BRANCH")


def _get_commit_hash():
    return os.getenv("TRAVIS_COMMIT") or os.getenv("APPVEYOR_REPO_COMMIT")


def _get_job_name():
    return os.getenv("TRAVIS_JOB_NAME") or os.getenv("APPVEYOR_JOB_NAME")


def _get_info_url():
    if os.getenv("TRAVIS"):
        job_id = os.getenv("TRAVIS_JOB_ID")
        return f"https://travis-ci.com/{_REPOSITORY}/jobs/{job_id}"

    if os.getenv("APPVEYOR"):
        build_id = os.getenv("APPVEYOR_BUILD_ID")
        return f"https://ci.appveyor.com/project/{_REPOSITORY}/build/{build_id}"

    return None


if __name__ == "__main__":
    if os.getenv("TRAVIS_EVENT_TYPE", "") == "push":
        _REPORTER = CIFailureReporter(_get_repo_name(), _GH_TOKEN)
        _REPORTER.report(_get_branch_name(), _get_commit_hash(), _get_info_url(), _get_job_name())
