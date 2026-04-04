# Releasing omniscore

This repository supports three release modes:

1. Automatic release to PyPI when a Git tag matching `v*` is pushed.
2. Manual release from GitHub Actions with `workflow_dispatch`.
3. Optional local manual release with `python -m build` and `twine upload`.

The GitHub workflow is at `.github/workflows/release.yml`.

## One-time setup

### 1. GitHub repository

Create these GitHub environments in `firojalam/omniscore_package`:

- `pypi`
- `testpypi`

The workflow does not require stored `PYPI_TOKEN` secrets when Trusted Publishing is configured. If you want extra protection, add required reviewers to the `pypi` environment.

### 2. PyPI Trusted Publisher

In the PyPI project settings for `omniscore`, add a Trusted Publisher with these exact values:

- Owner: `firojalam`
- Repository name: `omniscore_package`
- Workflow name: `release.yml`
- Environment name: `pypi`

Do the same in TestPyPI for the `omniscore` project, but use:

- Owner: `firojalam`
- Repository name: `omniscore_package`
- Workflow name: `release.yml`
- Environment name: `testpypi`

If `omniscore` does not exist on TestPyPI yet, create a pending Trusted Publisher first instead of doing a one-off bootstrap upload. PyPI documents that flow here:

- [Creating a PyPI project with a Trusted Publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)

References:

- [PyPI Trusted Publishers](https://docs.pypi.org/trusted-publishers/using-a-publisher/)
- [Packaging guide for GitHub Actions publishing](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)

## Automatic release with a tag

1. Update the package version in `pyproject.toml`.
2. Update `omniscore/__init__.py` to the same version.
3. Commit and push your changes to `main`.
4. Create and push a tag such as `v0.1.1`.

Example:

```bash
git checkout main
git pull --rebase origin main
git tag v0.1.1
git push origin main
git push origin v0.1.1
```

The workflow will:

- run `pytest -q`
- build the wheel and source distribution
- run `twine check`
- publish to PyPI through Trusted Publishing

The workflow also checks that `v0.1.1` matches `version = "0.1.1"` in `pyproject.toml`.

## Manual release from GitHub Actions

Open the `Release` workflow in GitHub Actions and choose `Run workflow`.

You can select the branch or tag in the GitHub UI, then choose one of these `publish_target` values:

- `none`: test the release path only, without publishing
- `testpypi`: publish to TestPyPI
- `pypi`: publish to PyPI

Recommended usage:

- Use `testpypi` for release-candidate validation.
- Use `pypi` only from a release tag or from `main` after the version bump is committed.
- Use `none` when you want a manual build-and-test dry run.

## Local manual release

Local uploads are optional and separate from Trusted Publishing. For a local upload you need a PyPI or TestPyPI API token.

### Build and check

```bash
python -m pip install --upgrade build twine
rm -rf build/ dist/
python -m build
python -m twine check dist/*
```

### Upload to TestPyPI

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=<your-testpypi-token>
python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

### Upload to PyPI

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=<your-pypi-token>
python -m twine upload dist/*
```

## Release checklist

- The version in `pyproject.toml` and `omniscore/__init__.py` matches.
- `pytest -q` passes locally.
- `README.md` reflects the current release.
- The tag uses the form `vX.Y.Z`.
- You are not trying to re-upload an existing version. PyPI rejects reused version numbers.
