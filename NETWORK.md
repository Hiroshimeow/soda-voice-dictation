# Network access

The backend binds to loopback by default at http://127.0.0.1:18765/ . Do not expose the unauthenticated backend directly to the public Internet.

## Optional Tailscale Serve

If Tailscale is installed and signed in, run:

    powershell -File ops\enable-tailnet.ps1

The script discovers the current Tailscale DNS name, adds that HTTPS origin to the ignored data/network.json file, configures Tailscale Serve on port 18765, and keeps access inside the tailnet. It does not enable Funnel.

Browser microphone capture requires a secure context, so use the HTTPS Tailscale DNS URL printed by the script rather than an insecure HTTP tailnet IP.

## Health

    Invoke-RestMethod http://127.0.0.1:18765/health

Origin checks protect browser WebSocket/sample-ingest paths, but Origin is not authentication. Tailnet access control is provided by Tailscale policy.

Normal dictation audio is not persisted by the backend. Corrected Vietnamese samples are stored only after explicit consent.
