# Publishing BoxTrail to PyPI

BoxTrail uses GitHub Actions Trusted Publishing (OIDC) and the official
`pypa/gh-action-pypi-publish@release/v1` action. No PyPI API key or password
is stored in GitHub.

The publication workflow is **.github/workflows/release.yml**.

## Trusted Publisher configuration on PyPI

The PyPI publisher must identify:
- Owner: `edujbarrios`
- Repository: `boxtrail`
- Workflow filename: `release.yml`
- Environment: `pypi` (when configured, it must match both sides)
- Package name: `boxtrail`

The same `pypi` environment is used by the `publish` job in GitHub Actions.
An environment may require deployment approval if configured accordingly.

## Initial release (0.1.0)

The initial workflow setup includes a **one-time gated push trigger**, restricted
to the exact commit message `release: publish boxtrail 0.1.0`, to publish the
first version when the workflow is installed. Subsequent ordinary pushes do
not publish; follow-up releases use GitHub Releases (published event).

Check the Actions tab and the PyPI package page to verify successful upload.
If the Trusted Publisher configuration differs from the values above, the
publish job will fail authentication and PyPI will not receive the version.

## Future releases

1. Update `pyproject.toml` and `src/boxtrail/__init__.py` versions together.
2. Update CHANGELOG.md and run `python -m unittest discover -s tests -v`.
3. Commit the code, create a matching Git tag and publish a GitHub Release.
4. The `release.published` event triggers test/build/upload.
5. Check the job result and https://pypi.org/project/boxtrail/.

To build locally without uploading:

```bash
python -m pip install --upgrade build
python -m build
```

**Important:** uploading an existing version to PyPI a second time is not
supported. Increase the version for each new publication.
