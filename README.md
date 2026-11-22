# SODA Voice Dictation

Local-first Windows voice dictation powered by Google Chrome's on-device SODA speech engine.

Hold a global hotkey, speak, watch live partial transcription, release the hotkey, and insert the final transcript into the original text field. The app can also transcribe local audio files and keeps final results in the Windows clipboard.

> Independent interoperability/research project. Not an official Google product, Google Live Transcribe, or ML Kit Speech Recognition.

## Features

- Windows tray app with Ctrl+Space hold-to-talk by default.
- Recording starts immediately on hotkey press, including initial silence.
- Thin, non-activating live transcript and waveform overlay.
- Final transcript copied to the clipboard and optionally auto-inserted.
- Local file transcription: WAV, MP3, M4A, AAC, FLAC, OGG.
- Three SODA language modes: Vietnamese (vi-VN), Japanese (ja-JP), English US (en-US).
- Vietnamese SODA model bundled in the repository; Japanese and English packs are downloaded/cached automatically on first run.
- Vietnamese personal vocabulary and deterministic post-corrections.
- Consent-based local collection of corrected Vietnamese samples.
- Evaluation/calibration tooling with train/holdout gates and rollback.
- Optional browser/mobile UI and optional private Tailscale TCP access by Tailscale IP.

## Windows quick start

Prerequisites:

- Windows 11.
- WSL2 with Debian or Ubuntu installed.
- Internet access on first run.
- winget is recommended so the launcher can install FFmpeg automatically.

Install WSL once if needed:

    wsl --install -d Debian

Then clone and run:

    git clone https://github.com/Hiroshimeow/soda-voice-dictation.git
    cd soda-voice-dictation
    .\run-desktop-windows.bat

On first run the launcher auto-detects a usable WSL distro (Ubuntu 22.04, Debian, Ubuntu, then another non-Docker distro), creates Python 3.12 environments for Windows and WSL, installs dependencies, installs FFmpeg if missing, seeds the bundled Vietnamese model into the local SODA cache, downloads/caches the SODA engine plus Japanese and English language packs, installs a user-scoped WSL service pointing at the current clone path, and starts the tray app. Set `SODA_WSL_DISTRO` to override the detected distro. Later launches reuse the environments and cached models.

## Dictation

1. Focus a text field.
2. Hold Ctrl+Space.
3. Recording starts immediately.
4. Speak while watching the live partial transcript.
5. Release Ctrl+Space.
6. SODA finalizes the utterance.
7. The final transcript is copied to the clipboard and, when enabled, inserted into the original target.

Partial results are intentionally not inserted directly because SODA may revise earlier partial hypotheses. If auto-insert cannot safely target the original field, the transcript remains in the clipboard for manual Ctrl+V.

## Languages

| Mode | Locale | Download |
| --- | --- | --- |
| Vietnamese | vi-VN | bundled in this repository (v1.3071.0) |
| Japanese | ja-JP | automatic first-run download |
| English (US) | en-US | automatic first-run download |

The desktop selector controls both microphone dictation and file transcription. Vietnamese personal vocabulary/context biasing is applied only to Vietnamese. Japanese and English use baseline SODA.

## File transcription

Click Transcribe file... and choose WAV, MP3, M4A, AAC, FLAC, or OGG. FFmpeg decodes the file to 16 kHz mono PCM and streams it to the same local SODA backend. Partial results appear in the app; the final transcript is copied to clipboard/history. The current UI accepts files up to four hours.

## Vietnamese personalization

Open Personal vocabulary in the desktop app. Priority phrases are sent as recognition context for new Vietnamese sessions; deterministic heard => write corrections are applied only after final recognition. Local vocabulary is stored under data/user_vocabulary.json and is ignored by Git.

The UI can store up to 64 phrases. Current SODA testing found that large bias profiles fail, so only the first 40 phrases are sent to SODA.

## Training and adaptation

The project separates context biasing, deterministic post-correction, and experimental model adaptation. Corrected Vietnamese training samples are stored only after explicit consent. Normal dictation audio is not persisted.

The training workflow includes labeled sample collection, deterministic train/holdout splitting, WER/CER evaluation, contextual-bias calibration, guarded promotion, rollback, and experimental SODA graph/weight research. Experimental weight changes have not yet demonstrated production-quality accuracy improvement.

See docs/TRAINING_WORKFLOW.md.

## Browser/mobile UI

The local web UI is available at http://127.0.0.1:18765/ . For private phone access, run:

    powershell -File ops\enable-tailnet.ps1

The script discovers the current Tailscale IPv4 address and exposes the local backend as a tailnet-only TCP forward on port 53219 by default, for example `http://100.x.y.z:53219/`. It removes the old SODA HTTPS Serve endpoint and does not use Funnel. Direct HTTP IP access may not satisfy browser secure-context rules for microphone capture; see NETWORK.md.

## Privacy

Normal dictation audio is not saved. Local/private artifacts such as data/, samples/, Python environments, runtime SODA caches, and recordings are ignored by Git. The bundled Vietnamese model under third_party/ is intentionally tracked.

## Limitations

- Windows insertion is subject to normal focus and integrity-level restrictions.
- The SODA runtime is Linux x86-64; Windows uses WSL2.
- Vietnamese contextual biasing is experimental.
- Chrome component IDs can change in future Chrome releases.
- This project does not establish that Chrome SODA uses the same model/runtime as Android Live Transcribe or ML Kit Speech Recognition.

## License and third-party components

Project-authored source code is licensed under the MIT License in LICENSE.

Project-authored code is MIT-licensed, but Google SODA binaries, language packs, model weights, trademarks, and other third-party components are not relicensed by this repository. The Vietnamese v1.3071.0 language component is bundled under `third_party/google-soda/`; the SODA engine plus Japanese/English packs are obtained from Google's Chrome component-update infrastructure. See `third_party/google-soda/THIRD_PARTY.md`.
