# DeskOS — local desktop simulator

A custom, interactive operating-system simulator with a second AI cursor. It is a simulated environment, not a replacement for macOS or Windows.

## Features

- Desktop, taskbar, start menu, draggable/resizable app windows
- Simulated Files, Notes, Terminal, Settings, and a closed fake browser
- Virtual files and notes stored in this browser's local storage
- Separate AI cursor, chat panel, speech output, and tool execution
- AI can inspect the desktop through an explicitly shared browser-tab capture
- Local Ollama model picker populated from the models installed on this computer
- Ollama requests go through the local Python bridge; no cloud AI service is used
- No external website navigation or real shell execution

## AI model

Default: `qwen2.5vl:7b`. This is a vision-language model and was previously confirmed working locally. The model picker reads the actual installed Ollama tags at runtime. If it is not installed anymore, choose another installed model; vision requires a vision-capable model.

The AI is not literally equivalent to ChatGPT and cannot do arbitrary real-world actions. Its tools are intentionally limited to the simulated desktop. The fake browser returns fictional local results and never fetches the public internet.

## Run

1. Install and start Ollama.
2. Make sure `qwen2.5vl:7b` is available (or choose another installed model in the selector).
3. In Terminal:

```bash
git clone https://github.com/How2FixBug/test
cd test
python3 app.py
```

4. Open http://127.0.0.1:8000.
5. Click the AI button, choose an installed model, and start chatting.
6. To let the model actually see the rendered desktop, click **Share tab for vision** and select this browser tab in the browser's screen-share dialog. This permission is optional and can be stopped at any time.

No Python packages are required. The app uses Python's standard library and your already-running local Ollama service.

## Privacy and boundaries

- The server binds only to `127.0.0.1`.
- The Ollama bridge only contacts `127.0.0.1:11434`.
- App state stays in browser local storage.
- Shared screenshots are sent only to the local Ollama instance.
- The simulated terminal does not execute operating-system commands.
- The simulated browser has no external network access.
