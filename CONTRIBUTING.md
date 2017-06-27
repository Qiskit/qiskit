# Contributing

**We appreciate all kinds of help, so thank you!** :clap: :kissing_heart:

You can contribute in many ways to this project.

## Issue reporting

:fire: This is a goog point to start, when you find a problem please add it to the [issue traker](https://github.ibm.com/IBMQuantum/qiskit-sdk-py-dev/issues). The ideal report should include the steps to reproduce it.

## Doubts solving

:two_women_holding_hands: To help less advanced users is another wonderful way to start. You can help us to close some opened issues. This kind of tickets should be labeled as `question`.

## Improvement proposal

:smiling_imp: If you have an idea for a new feature please open a ticket labeled as `enhancement`. If you could also add a piece of code with the idea or a partial implementation it would be awesome.

## Documentation

:eyes: We all know the doc always need fixes/upgrades :wink:, so please feel free to send a PR (see next point) with what you found.

## Code

:star: This section include some tips that will help you to push source code.

### Dependencies

Needed libraries should be installed in this way:

```sh
# Depending on the system and setup to append "sudo -H" before could be needed.
pip3 install -r requires-dev.txt
```

### Test

New features often imply changes in the existent tests or new ones are needed. Once they're updated/added run this be sure they keep passing:

```sh
make test
```

### Style guide

Submit clean code and please make effort to follow existing conventions in order to keep it as readable as possible. We use [Pylint](https://www.pylint.org) and [PEP 8](https://www.python.org/dev/peps/pep-0008) style guide, to confirm the new stuff respects it run the next command:

```sh
make lint
```

### Doc

Review the parts of the documentation regarding the new changes and update it if it's needed.

### Pull requests

We use [GitHub pull requests](https://help.github.com/articles/about-pull-requests) to accept the contributions. You can explain yourself as much as you want :smile:. Please follow the next rules for the commit messages:

- It should be formed by a one-line subject, followed by one line of white space. Followed by one or more descriptive paragraphs, each separated by one line of white space. All of them finished by a dot.
- If it fixes an issue, it should include a reference to the issue ID in the first line of the commit.
- It should provide enough information for a reviewer to understand the changes and their relation to the rest of the code.
