# Code Signing Policy

This document describes how release artifacts of **webai2api** are built and
signed. It is referenced by our
[SignPath Foundation](https://signpath.org) open-source signing application.

## 1. Scope

Signed artifacts: the Windows release archive (`webai2api-win64.zip`) that
contains `webai2api.exe`, published as GitHub Release assets.

## 2. Release authority

Only the project maintainer (GitHub account
[`duruoxian`](https://github.com/duruoxian)) may create releases. Releases are
cut exclusively from tagged commits (`v*`) on the `main` branch.

## 3. Builds happen only in CI

Every signed artifact is built exclusively by GitHub Actions
(`.github/workflows/build-release.yml`) from a clean checkout of the tagged
commit. No local or manually assembled build is ever signed or published.

## 4. Manual approval required for every signature

Each signing request is submitted to SignPath from CI and requires a **manual
approval by the maintainer** (SignPath "Approver" role) before the signature
is applied. This step is a deliberate anti-tampering gate: no release can be
signed without the maintainer explicitly approving it in the SignPath console.

## 5. Release flow

1. Maintainer pushes tag `vX.Y.Z` on `main`.
2. CI builds the executable, packages the archive, and uploads it as a GitHub
   Actions artifact.
3. CI submits a SignPath signing request for that artifact.
4. Maintainer reviews and approves the signing request in the SignPath
   console.
5. CI downloads the signed archive and publishes it to the GitHub Release.

## 6. Third-party components disclosure

The release archive bundles third-party components:

- **`sha3_wasm_v1.wasm`** — a proof-of-work (PoW) solver WebAssembly module
  obtained from DeepSeek's public web client. It is redistributed inside the
  archive through the third-party PyPI package `deepseek-pow` (MIT-licensed
  community package, declared as a dependency in `pyproject.toml`; the module
  itself is DeepSeek-origin code). This is the only binary component in the
  archive that was not authored for this project.
- All other bundled binaries and libraries are open-source components under
  OSI-approved licenses (`wasmtime`, `curl_cffi`, `pywebview`, `fastapi`,
  `uvicorn`, etc.), as declared in `pyproject.toml`.

The project itself is MIT-licensed open-source software. It contains no
malware, adware, or spyware, performs no telemetry or data collection, and
runs entirely on the user's local machine.

## 7. Contact

Maintainer: DuRuoxian — GitHub: [duruoxian](https://github.com/duruoxian)
