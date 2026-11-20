# Release readiness (maintainer checklist)

**Do not publish without explicit approval from the maintainer.** The repository intentionally has no PyPI credentials, Trusted Publisher configuration, automatic publication action, or release job.

## Before any future release

1. Review API, examples and tests; confirm chosen license and copyright notices.
2. Decide whether to tag `v0.1.0` and whether the name `boxtrail` is available on PyPI.
3. Run `python -m unittest discover -s tests -v`.
4. If you want to inspect package artifacts locally, install the build tool with `python -m pip install --upgrade build` and run `python -m build`; this creates `dist/` files **without uploading them**.
5. Add a test-only CI workflow if desired. A publishing workflow should be added only after a separate maintainer decision.
6. Review the final `README.md`, `CHANGELOG.md`, `LICENSE`, `NOTICE` and package metadata.

No release to PyPI or GitHub Releases is part of this initial repository setup.
