# Add the Emotion Companion to the BeyondWords sign-language site

`EmotionCompanion.jsx` is a self-contained React page for the BeyondWords site
(`ai-sign-language-beyond-words-final.vercel.app`). It records the user's voice and calls
the Beyond Words voice backend for speech-to-text, the vocal emotion cue, and a Gemini
companion reply. It uses only what the site already has: **React, Tailwind CSS and
lucide-react**.

## 1. Copy the page

Copy `EmotionCompanion.jsx` into the site, for example:

```
src/pages/speech/EmotionCompanion.jsx
```

## 2. Add the route

Next to the existing Speech Hub routes (`/speech/to-text`, `/speech/text-to-speech`),
add a protected route inside the same dashboard layout:

```jsx
import EmotionCompanion from "./pages/speech/EmotionCompanion";

// inside the dashboard/protected routes
<Route path="/speech/emotion-companion" element={<EmotionCompanion />} />
```

## 3. Add the sidebar entry

Under **Speech Hub**, after "Text to Speech":

```jsx
import { HeartHandshake } from "lucide-react";

{ label: "Emotion Companion", to: "/speech/emotion-companion", icon: HeartHandshake }
```

Match the shape of the other Speech Hub items in the site's sidebar config, and add
"Emotion Companion" to the breadcrumb and search lists if the site keeps those separately.

## 4. Point it at the backend

Set this environment variable (locally in `.env`, and in **Vercel → Project → Settings →
Environment Variables**, then redeploy):

```
VITE_COMPANION_API_URL=https://<your-backend-url>
```

`<your-backend-url>` is the Beyond Words FastAPI server: the Cloud Run URL
(`https://beyond-words-xxxx.us-central1.run.app`) for production, or
`http://localhost:8080` while developing.

The backend must allow the site's origin (CORS). It allows
`https://ai-sign-language-beyond-words-final.vercel.app` and `http://localhost:5173`
by default; set `CORS_ALLOW_ORIGINS` on the backend (comma-separated) for other domains,
such as Vercel preview URLs.

## What the page does

| Feature | Backend endpoint |
|---|---|
| Record (up to 12 s) or upload a clip, live level meter and timer | — |
| Transcript + vocal emotion chip + Gemini reply | `POST /v1/turn` |
| Type the words if speech-to-text can't understand the clip | `POST /v1/turn` with `manual_transcript` |
| Play the reply aloud | `POST /v1/speak` |
| Side-by-side reply with vs. without the emotion cue | `POST /v1/compare` |

Microphone recording needs HTTPS (Vercel provides it) or `localhost`.
