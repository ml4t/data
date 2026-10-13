Read the [ML4T contribution policy](https://github.com/ml4t/ecosystem/blob/main/CONTRIBUTING.md)
before submitting.

## Owning issue

Closes #

The issue and this pull request each describe one problem or coherent change. Separate unrelated
fixes. Automated dependency updates and administrative metadata changes are exempt from the issue
requirement.

## Outcome

Describe the problem, expected behavior, and what this change does.

## Compatibility and numerical behavior

- [ ] No public compatibility impact
- [ ] Compatibility impact is documented in the owning issue and migration guidance
- [ ] No numerical result or documented definition changes
- [ ] A numerical defect is reproduced below and corrected by this change
- [ ] An intended definition or convention change was agreed in the owning issue and documented

## Verification

For a bug fix, run the same regression through the affected public API or user workflow on both
revisions:

- Base revision, command, and failing result:
- Pull-request revision, same command, and passing result:

For other changes, state the relevant commands and results.

- [ ] Verification exercises the affected public API or user workflow
- [ ] Ruff lint and format checks pass
- [ ] `ty` passes
- [ ] The applicable test lanes pass
- [ ] The package build passes when applicable
- [ ] Strict MkDocs passes when documentation is affected
- [ ] Ecosystem qualification passes

## AI assistance

State whether AI tools helped produce code or prose and identify the affected parts.

- [ ] I reviewed and tested this submission and can explain its code, prose, and evidence

## Documentation and release

State the documentation, release-note, and patch-release work required after merge.
