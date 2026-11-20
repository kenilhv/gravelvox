# Releasing

`gravelvox-engine` is published to PyPI by [`.github/workflows/release.yml`](../.github/workflows/release.yml) when a version tag is pushed. PyPI trusts that workflow through [trusted publishing](https://docs.pypi.org/trusted-publishers/), so no API token is stored in GitHub or on anyone's machine.

## One-time setup

Do these in order. If a workflow names an environment that does not exist, GitHub creates it with no protection rules, and the publish job would then run without asking anyone. So the environment comes first.

1. **GitHub environment `pypi`** (Settings → Environments). This repository already has it. If it is ever deleted, recreate it before any release tag is pushed, with these settings:
   - **Required reviewers:** `kenilhv`. Leave **Prevent self-review** off: with one maintainer, the person who pushes the tag is the only one who can approve it.
   - **Allow administrators to bypass configured protection rules:** off.
   - **Deployment branches and tags:** "Selected branches and tags", with one rule of ref type **Tag** and the pattern `gravelvox-engine-v*`. A branch rule with the same pattern would reject every release tag.
2. **PyPI pending publisher.** At <https://pypi.org/manage/account/publishing/>, add a pending publisher for GitHub Actions. PyPI needs a verified primary e-mail address on the account for this.

   | Field | Value |
   |---|---|
   | PyPI Project Name | `gravelvox-engine` |
   | Owner | `kenilhv` |
   | Repository name | `gravelvox` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

   PyPI matches these values exactly; if one is wrong, the publish job fails with `invalid-publisher` and nothing is uploaded. A pending publisher does not reserve the name: if someone else registers `gravelvox-engine` before the first upload, it stops working. So publish the first release soon after adding it.

## Cutting a release

Releases are cut from `main`. Write the version with three numbers in its normalized form, for example `0.1.0`, `0.1.0rc1` or `0.1.0.post1`: the tag has to repeat the version in `packages/engine/pyproject.toml` exactly, and a tag without three numbers starts nothing.

1. **Prepare the release commit.** Set the version in `packages/engine/pyproject.toml`, run `uv lock`, and get the change onto `main`.

   For the first release, update the two READMEs in the same commit. `packages/engine/README.md` becomes the PyPI project page, and PyPI never lets that page be edited for a version once it is uploaded:
   - `packages/engine/README.md`: make the Install section say `pip install gravelvox-engine` (with Python 3.11 to 3.14), and keep the checkout instructions as the way to work on the code.
   - `README.md`: replace "The package is not on PyPI yet, so install it from a checkout as shown." with a line pointing to `uv tool install gravelvox-engine` or `pip install gravelvox-engine`.

   The release workflow refuses to publish while the package README still says the package is not on PyPI, or has no `pip install gravelvox-engine` line.
2. **Rehearse.** Once that commit is on `main`, run the Release workflow by hand: Actions → Release → Run workflow, branch `main`. It runs CI and builds, checks and smoke-tests the files exactly as a tag run would, but it skips the tag check and never publishes. Do not tag until it passes.
3. **Tag the rehearsed commit and push the tag,** with your version in place of `0.1.0`:

   ```bash
   git switch main
   git pull --ff-only
   git tag -a gravelvox-engine-v0.1.0 -m "gravelvox-engine 0.1.0"
   git push origin gravelvox-engine-v0.1.0
   ```

4. **Approve.** The publish job asks for approval only after CI, the build and the checks have passed on the tagged commit. Open the run, check that the build log shows the version you expect, then choose **Review deployments**, tick `pypi` and choose **Approve and deploy**.
5. **Confirm** that the version is listed at <https://pypi.org/project/gravelvox-engine/> and installs:

   ```bash
   uv run --no-project --with gravelvox-engine==0.1.0 -- gravelvox --version
   ```

## What the workflow checks before publishing

- CI (lint, and the test suite on Linux, Windows and macOS) passes on the exact commit being published.
- The tag matches the version in `packages/engine/pyproject.toml`, and the tagged commit is on `main`.
- The package README is ready to be the PyPI project page.
- The build backend is installed from hash-pinned versions in [`.github/build-constraints.txt`](../.github/build-constraints.txt), so a newly released build tool cannot change what gets published.
- `twine check --strict` accepts the metadata of both files.
- [`scripts/smoke_dist.py`](../scripts/smoke_dist.py) passes against the wheel and against the source distribution, each installed alone in a clean environment: the package imports from the installed copy, `py.typed` is present, the `gravelvox` command runs, and PyAV decodes a generated WAV file.

The build job uploads the files as soon as they are built, before any other code runs beside them, and the publish job downloads exactly that upload by its ID. The publish job holds only the right to prove its identity to PyPI. It does not check out the repository or run project code, and it attaches signed [PEP 740](https://peps.python.org/pep-0740/) publish attestations to both files.

## If something goes wrong

A file on PyPI can never be replaced, and a file name can never be used again, even after the file is deleted. No file reaches PyPI before the publish job prints "Uploading distributions"; a failure before that costs nothing, and the version is still free.

- **The CI, build or check job fails** (the tag and the version differ, the commit is not on `main`, a test or check fails). Fix the cause on `main`, delete the tag and tag again:

  ```bash
  git push --delete origin gravelvox-engine-v0.1.0
  git tag -d gravelvox-engine-v0.1.0
  ```

- **The publish job fails with `invalid-publisher`.** The pending publisher on PyPI does not match the run. Correct it on PyPI to match the values shown in the job log, then choose **Re-run failed jobs** and approve again.
- **The upload stops partway,** for example with the wheel on PyPI and the source distribution missing. Choose **Re-run failed jobs** (not "Re-run all jobs") and approve again. It uploads the same stored files: PyPI accepts an identical copy of a file it already has and adds the missing one.
- **A release run shows as Cancelled.** Nothing was uploaded. Re-run it.
- **A published release is broken.** Yank it on PyPI (the project's Manage page → Releases → Options → Yank) and release a fixed version. A yanked version stays installable only when someone asks for that exact version.

## Refreshing the pinned build backend

Dependabot does not update `.github/build-constraints.txt`. To move to a newer hatchling, regenerate it, commit it, and rehearse before the next release:

```bash
uv pip compile .github/build-requirements.in --universal --python-version 3.11 --generate-hashes -o .github/build-constraints.txt
```
