# DEVELOPMENT CYCLE
Our development cycle is straightforward, we define a roadmap with milestones for releases, and features that we want
to include in these releases. The roadmap is not public at the moment, but it’s a committed project in our community and we are working to make parts of it public in a way that can be beneficial for everyone. Whenever a new release is close to be launched, we'll announce it and detail what has changed since the latest version.
The channels we'll use to announce new releases are still being discussed, but for now you can [follow us](https://twitter.com/qiskit) on Twitter!

## BRANCH MODEL
There are two main branches in the repository:

* `master`
    * This is the development branch.
    * Next release is going to be developed here. For example, if the current latest release version is r1.0.3, the
    master branch version will point to r1.1.0 (or r2.0.0).
    * You should expect this branch to be updated very frequently.
    * Even though we are always doing our best to not push code that breaks things, is more likely to eventually push
    code that breaks something... we will fix it ASAP, promise :).
    * This should not be considered as an stable branch to use in production environments.
    * The API of the SDK could change without prior notice.

* `stable`
    * This is our stable release branch.
    * It's always synchronized with the latest distributed package, as for now, the package you can download from pip.
    * The code in this branch is well tested and should be free of errors (unfortunately sometimes it's not).
    * This is an stable branch (as the name suggest), meaning that you can expect an stable software ready for production
    enviornments.
    * All the tags from the release versions are created from this branch

## RELEASE CYCLE
From time to time, we will release brand new versions of the QISKit SDK. These are well-tested versions of the software.
When the time for a new release has come, we will:
1. Merge the `master` branch with the `stable` branch.
2. Create a new tag with the version number in the `stable` branch.
3. Crate and distribute the pip package.
4. Change the `master` version to the next release version.
5. Announce the new version to the world!

The `stable` branch should only receive changes in the form of bug fixes, so the third version number (the maintanance
number: <major>.<minor>.<maintanance>) will increase on every new change.

## WHAT VERSION SHOULD I USE: DEVELOPMENT OR STABLE?
It depends on your needs as a user.
If you want to use QISKit for building Apps which goal is to run Quantum programs, we encourge you to use the latest
released version, installing it via Pip.

`$ pip install qiskit`

If you found out that the release version doesn't fit your needs, and you are thinking about extending the functionallity
of the SDK, you are more likely to use the `master` branch and thinking seriously about contributing with us :)

