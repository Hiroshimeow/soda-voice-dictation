# Windows desktop client

The Windows client is a tray-first dictation app.

## Daily use

1. Run run-desktop-windows.bat.
2. Focus a text field.
3. Hold Ctrl+Space.
4. Speak while watching the live transcript overlay.
5. Release the hotkey.
6. The final transcript is copied to the clipboard and optionally inserted into the original target.

Closing the main window hides it to the system tray. Use the tray menu to show or exit the app.

## Language modes

The language selector controls both microphone dictation and file transcription: Vietnamese (vi-VN), Japanese (ja-JP), and English US (en-US). Vietnamese alone currently uses the personal vocabulary/training pipeline.

## File transcription

Use Transcribe file... for WAV/MP3/M4A/AAC/FLAC/OGG input. The app decodes through FFmpeg and uses the same local SODA backend.

## Local data

Final transcripts are copied to the clipboard and the most recent entries are stored under %LOCALAPPDATA%\SodaDictation. Normal microphone dictation is not recorded to disk. Corrected Vietnamese samples are saved only after explicit consent.
