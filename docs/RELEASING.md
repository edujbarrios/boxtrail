# Releasing BoxTrail

BoxTrail uses **automatic publication to PyPI** on GitHub Actions, with
OpenID Connect (OIDC) Trusted Publishing. No static PyPI API token is needed.

## Automatic publication

For every new version:

1. Update the version in **both** `pyproject.toml` (`[project].version`) and
   `src/boxtrail/__init__.py` (`__version__`), for example from `0.2.0` to `0.3.0`.
2. Update the `CHANGELOG.md` and any relevant docs/code/tests.
3. Merge the changes to `main`.
4. A push to `main` that changes one of the version files triggers
   `.github/workflows/release.yml` automatically.
5. The workflow validates that the two versions match and checks whether
   that exact version already exists on PyPI. If it already exists, upload
   jobs are skipped (they are **not** rerun for an existing version).
6. For a new version, the workflow tests on Python 3.10–3.14 (including
   optional NumPy tests), builds wheel + sdist, validates package metadata
   and publishes the artifacts to PyPI.
7. Once PyPI upload succeeds, the workflow creates a matching GitHub Release
   and tag such as `v0.3.0`.

**No manual GitHub Release, tag or PyPI upload is required.**

Ordinary changes to README, code or tests do not trigger publishing
unless one of the version files also changes. Updates to the release
workflow itself trigger a preflight check, but re-upload is skipped if
the current version is already on PyPI.

The workflow also offers a manual `workflow_dispatch` trigger for recovery
if the initial publication of a new version fails. It will not overwrite an
already-published version.

## Trusted Publisher (existing configuration)

- PyPI distribution: `boxtrail`
- GitHub owner/repository: `edujbarrios/boxtrail`
- Workflow filename: `release.yml`
- GitHub Actions environment: `pypi`

The workflow uses `pypa/gh-action-pypi-publish@release/v1` with
`id-token: write`, and requires that GitHub and PyPI Trusted Publisher
settings match.

## Security and limits

- PyPI release versions are immutable; never reuse a version number for a
  modified wheel or sdist.
- A CI failure blocks automatic publication.
- If PyPI is unreachable, the preflight fails rather than assuming a new
  version. Retry after the service is available.
- Changes merged to `main` with a new version will **publish without
  further confirmation**. Protect the main branch with reviews and required
  CI checks if you want approval before deployment.
- The license remains AGPL-3.0-only. Copyright and attribution notices are
  included in distributed artifacts.

To build locally without uploading:

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```
