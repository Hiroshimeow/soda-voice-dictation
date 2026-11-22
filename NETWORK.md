# Network access

The backend binds to loopback by default at:

    http://127.0.0.1:18765/

Do not expose the unauthenticated backend directly to the public Internet.

## Optional direct Tailscale-IP access

If Tailscale is installed and signed in, run:

    powershell -File ops\enable-tailnet.ps1

The script discovers the current Tailscale IPv4 address and creates a tailnet-only TCP forward on port 53219 by default:

    http://100.x.y.z:53219/

The local SODA backend remains on 127.0.0.1:18765. Tailscale forwards only tailnet traffic from the selected TCP port to that local backend.

The script removes the old SODA HTTPS Serve endpoint on port 18765 but leaves unrelated Tailscale services untouched. It does not enable Funnel/public Internet exposure.

To use another tailnet port:

    powershell -File ops\enable-tailnet.ps1 -Port 54000

To disable this SODA TCP forward:

    tailscale serve --tcp=53219 off

### Browser microphone limitation

Direct access by Tailscale IP uses plain HTTP. Modern browsers normally require HTTPS or localhost for microphone capture, so the web microphone UI may be blocked when opened as http://100.x.y.z:53219/.

The direct Tailscale-IP endpoint remains useful for:

- health/API access;
- file and non-browser clients;
- private tailnet connectivity tests.

The Windows desktop app does not depend on the Tailscale endpoint.

## Health

Local:

    Invoke-RestMethod http://127.0.0.1:18765/health

Tailnet example:

    Invoke-RestMethod http://100.x.y.z:53219/health

Origin checks protect browser WebSocket/sample-ingest paths, but Origin is not authentication. Tailnet access control is provided by Tailscale policy.

Normal dictation audio is not persisted by the backend. Corrected Vietnamese samples are stored only after explicit consent.
