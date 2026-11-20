# Releasing BoxTrail

BoxTrail publishes to PyPI through GitHub Actions with **Trusted Publishing (OIDC)**.
No PyPI API token is stored in the repository.

## Trusted Publisher settings

- PyPI distribution: `boxtrail`
- GitHub owner: `edujbarrios`
- GitHub repository: `boxtrail`
- Workflow: `.github/workflows/release.yml`
- GitHub environment: `pypi`

The environment name must match the publisher registration at PyPI.

## Release procedure

1. Update versions in `pyproject.toml` and `src/boxtrail/__init__.py`.
2. Update `CHANGELOG.md` and README if needed.
3. Run `python -m unittest discover -s tests -v` and check CI on pull requests.
4. Publish a GitHub Release with a matching version tag, e.g. `v0.3.0`.
5. The `release.published` workflow builds the wheel and sdist, runs tests and uploads distributions to PyPI.
6. Verify the action is green and confirm the new version is available at https://pypi.org/project/boxtrail/.

For initial 0.2.0 publication, a narrowly gated push trigger is included for the
exact commit message `release: publish boxtrail 0.2.0`. The publish action
runs only after successful tests/build. On successful PyPI publication the
workflow also creates GitHub tag and release `v0.2.0` if GitHub Actions has
write permission to do so. This one-time push trigger is not a general
auto-release-on-commit mechanism.

To build and inspect artifacts without uploading:

```bash
python -m pip install --upgrade build
python -m build
```

Each PyPI version is immutable: increment the version for subsequent uploads.
