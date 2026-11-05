# Ultillama

A minimal browser chat interface for local Ollama models.

## Run it

1. Start Ollama on your computer.
2. Open this repository through a local static web server, or deploy it to a static host.
3. Choose an installed model and send a message.

Ultillama connects to Ollama at `http://localhost:11434`.

## No context by design

Every request contains exactly one message: the current user message. Previous messages remain visible in the page so you can read the conversation, but they are never sent to Ollama. There is no memory, retrieval, system prompt, or conversation history.

## Browser access

If the page cannot connect, Ollama may be blocking the page's origin. Configure Ollama's allowed origins to include the origin serving this page, then restart Ollama. For a locally served page, that is usually a localhost origin.

Ultillama does not include a backend or send prompts to a third-party service.