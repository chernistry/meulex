# Slack Integration Setup (Meulex)

This guide explains how Slack integration works in Meulex and how to configure a Slack App to receive events and send responses.

## 1) How it works (in code)
- Endpoint: POST `/slack/events` handles Slack Events API requests.
- Signature verification: `X-Slack-Signature` + `X-Slack-Request-Timestamp` are validated with your `SLACK_SIGNING_SECRET`.
- URL verification: Slack sends `type = url_verification`; API returns `{ "challenge": ... }` to confirm the endpoint.
- Events handled: `app_mention` in channels and `message` in direct messages (DM).
- Response model: API ACKs within ≤3s, then posts an async reply via Slack Web API (`chat.postMessage`) using `SLACK_BOT_TOKEN`.
- Loop prevention: messages from your bot are ignored via `SLACK_BOT_USER_ID`.

Code map:
- `meulex/api/routes/slack.py`: endpoint, initialization, URL verification, ACK + async send
- `meulex/integrations/slack/auth.py`: signature verification (HMAC-SHA256)
- `meulex/integrations/slack/processor.py`: event handling (mentions/DM), formatting, idempotency cache
- `meulex/integrations/slack/models.py`: payload/response models and message formatting

## 2) Local API and public URL
1. Start Meulex API:
   ```bash
   docker compose up -d
   uvicorn meulex.api.app:app --host 0.0.0.0 --port 8000
   ```
2. Expose a public URL (e.g., ngrok):
   ```bash
   ngrok http 8000
   ```
   Use: `https://<subdomain>.ngrok.io/slack/events` for Slack Request URL.

## 3) Create and configure Slack App
1. Go to `https://api.slack.com/apps` → Create New App → From scratch.
2. Basic Information:
   - Copy Signing Secret → set as `SLACK_SIGNING_SECRET` in your `.env`.
3. OAuth & Permissions → Bot Token Scopes (minimum):
   - `app_mentions:read` (read mentions in channels)
   - `im:history` (read bot DMs)
   - `chat:write` (post messages)
   - (optional) `chat:write.public`, `channels:history` if needed
   Click “Install to Workspace” → copy `Bot User OAuth Token` (starts with `xoxb-...`) → set `SLACK_BOT_TOKEN`.
4. Event Subscriptions:
   - Enable Events: ON
   - Request URL: `https://<subdomain>.ngrok.io/slack/events` (wait for green check)
   - Subscribe to bot events: `app_mention`, `message.im`
   - Save changes
5. App Home:
   - Enable “Allow users to send a message to the app” (for DMs)
6. Invite the bot to channels where you want it active: `/invite @<botname>`

## 4) Environment variables
Set in `.env`:
```bash
SLACK_SIGNING_SECRET=...
SLACK_BOT_TOKEN=xoxb-...
SLACK_BOT_USER_ID=UXXXXXXXX   # Bot user ID to skip its own messages
```
How to get `SLACK_BOT_USER_ID`:
- In Slack, open App Home → About → More → Copy member ID (format `U...`), or open DM with the bot and copy from profile.

## 5) Verify
- DM the bot: `help` → should return a help message.
- In a channel (where bot is invited): `@<botname> <your question>` → should reply with an answer + citations.
- Check API logs for `slack_events` traces and signature verification success.

## 6) Common issues
- 401 Invalid signature → check `SLACK_SIGNING_SECRET`, server time drift (≤5 min), and correct Request URL.
- No response in channels → ensure the bot is invited and has `app_mentions:read`, `chat:write` scopes.
- DM not working → enable App Home messages and add `im:history`; set `SLACK_BOT_USER_ID`.
- ngrok URL changed → update Request URL in Slack → Save.
- Rate limiting → avoid high RPS in tests.

## 7) Notes
- Slash Commands: the repo has flags for slash support, but current route focuses on Events API. To add slash commands, enable Interactivity & Shortcuts, define a command, and implement a handler that uses `response_url` (can be added later).

