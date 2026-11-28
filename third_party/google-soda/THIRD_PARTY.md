# Third-party SODA model files

The full source checkout may contain files under
`third_party/google-soda/vi-VN/1.3071.0/` for the Vietnamese Google Chrome SODA
language component used by this project. Its `SODAModels` directory is an
optional local cache-seeding optimization; the application downloads the pack
when it is absent.

- Locale: `vi-VN`
- Component version: `1.3071.0`
- Chrome component ID: `bpnckojighgibleldbofdjmfdipfeiak`
- Source mechanism: Google Chrome component-update/Omaha service

These third-party binaries and model weights are **not covered by this
repository's MIT license**. Google/Chrome retains all applicable rights to the
SODA component, model weights, trademarks, and related material.

The component itself did not contain a LICENSE or NOTICE file when obtained
from Chrome's component-update service. Users are responsible for ensuring that
their use and redistribution comply with the applicable Google/Chrome terms.

`SHA256SUMS` records the exact hashes of the source-checkout files. Lightweight
transfer ZIPs created by `ops/create-portable-zip.ps1` exclude every
`SODAModels` payload under `third_party/google-soda/`; fresh installs download
Vietnamese and all other configured language packs from Google's Chrome
component-update infrastructure on first run. No model binaries are
redistributed in the lightweight package.
