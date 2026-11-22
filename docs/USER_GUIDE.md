# User guide

## Live dictation

Run run-desktop-windows.bat. The app starts in the system tray.

1. Choose Vietnamese, Japanese, or English (US).
2. Focus a target text field.
3. Hold Ctrl+Space. Recording starts immediately, including initial silence.
4. Speak and watch the live partial transcript in the thin overlay.
5. Release Ctrl+Space to stop recording.
6. The final transcript is copied to the clipboard and, if Auto-insert is enabled, inserted into the original target.

Esc cancels the active session or pending insertion. Partial hypotheses are display-only because SODA may revise earlier words.

If Ctrl+Space conflicts with an IME or another application, choose one of the alternate hotkeys in the desktop window.

## File transcription

Click Transcribe file... and choose WAV, MP3, M4A, AAC, FLAC, or OGG. The selected language mode is used for file transcription as well as the microphone. The final text is copied to the clipboard/history.

## Vietnamese personal vocabulary

Personal vocabulary is currently Vietnamese-only. Open Personal vocabulary and edit:

- Priority phrases: one phrase per line. Up to 64 can be stored; the first 40 are sent to SODA.
- Deterministic corrections: one rule per line using heard => write.

Only add deterministic corrections after observing the actual recurring recognition error. Broad or ambiguous rules can damage correct transcripts.

The ignored local file is data/user_vocabulary.json.

## Corrected samples

The adaptation dataset is currently Vietnamese-only.

1. Select Vietnamese.
2. Record a complete 0.5-30 second utterance from the desktop UI.
3. Stop and edit the transcript so it exactly matches what was spoken.
4. Enable the local-save consent checkbox.
5. Save the corrected sample.

Normal dictation audio is not saved. Only explicitly consented corrected samples enter the local dataset.

## Local state

Application settings and a short transcript history are stored under %LOCALAPPDATA%\SodaDictation. Private training data and vocabulary live under the ignored data/ directory in the repository.
