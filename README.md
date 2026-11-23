# Auto Mix / Auto Master (VST3)

Two VST3 plugins: **Auto Mix** (per-track/bus) and **Auto Master** (master bus).
Auto EQ (spectral balance), auto compressor, loudness targeting, stereo width, limiter.

## Build (no install needed): GitHub Actions
1. Create a new GitHub repo, upload all of these files (keep the folder structure).
2. Open the Actions tab -> "Build VST3" -> wait ~10 min.
3. Download the artifact (Windows). Unzip it.

## Install in FL Studio
1. Copy the `Auto Mix.vst3` and `Auto Master.vst3` folders to `C:\Program Files\Common Files\VST3`
2. FL Studio: Options -> Manage plugins -> Find installed plugins (or "Verify plugins").
3. Add from the plugin database under Installed -> Effects.

Put Auto Mix on tracks, Auto Master on the Master channel (last in chain).
