# Contributing

Thank you for investing your time in contributing to SteerABLE-v1!

Read our [Code of Conduct](./CODE_OF_CONDUCT.md) to keep our community approachable and respectable.

This guide details how to use issues and pull requests to improve SteerABLE-v1.

SteerABLE-v1 is a fork of [Protenix](https://github.com/bytedance/Protenix) (v1 lineage); its Protenix-v2 sibling is [SteerABLE](https://github.com/bhyun-ans/SteerABLE). If a change belongs upstream -- anything outside the epitope-steering code listed in the README's repository-layout table -- please send it to Protenix instead, so both projects get it.

## General Guidelines

### Pull Requests

Make sure to keep Pull Requests small and functional to make them easier to review, understand, and look up in commit history. This repository uses "Squash and Commit" to keep our history clean and make it easier to revert changes based on PR.

Adding the appropriate documentation, unit tests and e2e tests as part of a feature is the responsibility of the feature owner, whether it is done in the same Pull Request or not.

Pull Requests should follow the "subject: message" format, where the subject describes what part of the code is being modified.

Refer to the template for more information on what goes into a PR description.

### Design Docs

A contributor proposes a design with a PR on the repository to allow for revisions and discussions. If a design needs to be discussed before formulating a document for it, make use of Google doc and GitHub issue to involve the community on the discussion. 

### GitHub Issues

GitHub Issues are used to file bugs, work items, and feature requests with actionable items/issues (Please refer to the "Reporting Bugs/Feature Requests" section below for more information).

### Reporting Bugs/Feature Requests

We welcome you to use the GitHub issue tracker to report bugs or suggest features that have actionable items/issues (as opposed to introducing a feature request on GitHub Discussions).

When filing an issue, please check existing open, or recently closed, issues to make sure somebody else hasn't already reported the issue. Please try to include as much information as you can. Details like these are incredibly useful:

- A reproducible test case or series of steps
- The version of the code being used
- Any modifications you've made relevant to the bug
- Anything unusual about your environment or deployment

## Contributing via Pull Requests

### Find interesting issue

If you spot a problem, [search whether an issue already exists](https://github.com/bhyun-ans/SteerABLE-v1/issues). If a related issue doesn't exist, open a new one.

### Solve an issue

`examples/steerable/7yds/` is the quickest way to get a working run in front of you before changing anything.

### Open a Pull request.

When you're done making the changes, open a pull request and fill PR template so we can better review your PR. The template helps reviewers understand your changes and the purpose of your pull request. 

Don't forget to link PR to the issue if you are solving one.

If you run into any merge issues, checkout this [git tutorial](https://lab.github.com/githubtraining/managing-merge-conflicts) to help you resolve merge conflicts and other issues.


## Finding contributions to work on

Looking at the existing issues is a great way to find something to contribute on. As our projects, by default, use the default GitHub issue labels (enhancement/bug/duplicate/help wanted/invalid/question/wontfix), looking at any 'help wanted' and 'good first issue' issues are a great place to start.
