# Beyond Words

Beyond Words is a voice studio for customer support and supportive conversations. It analyzes acoustic cues with a trained Emotion2Vec+ classifier, uses Gemini on Vertex AI to generate a response, and can play it back with Google Cloud Text-to-Speech.

[Open the Cloud Run demo](https://beyondwords-283299931293.us-central1.run.app/) or see the [architecture and API overview](ARCHITECTURE.md).

## Run locally

The Docker image installs the CPU inference dependencies, downloads the base encoder during the build, and uses the classifier checkpoint in `model_weights/`. The server entry point is `main:app`.

Set `GOOGLE_CLOUD_PROJECT` to your authorized project and configure Application Default Credentials. Enable Vertex AI and Cloud Text-to-Speech in that project. The example mounts local Application Default Credentials read-only.

```sh
docker build -t beyondwords .
docker run --rm -p 127.0.0.1:8080:8080 \
  -e GOOGLE_CLOUD_PROJECT \
  -e GOOGLE_APPLICATION_CREDENTIALS=/run/google-adc.json \
  -v "${HOME}/.config/gcloud/application_default_credentials.json:/run/google-adc.json:ro" \
  beyondwords
```

Open `http://localhost:8080/` for the studio, `/docs` for the API, and `/api/v1/status` for runtime status. Set `GEMINI_MODELS` to a comma-separated list if your project needs different Gemini model IDs.

## Implementation history

The merged application uses the deployed Cloud Run implementation from `harshad`, with the response guidance from `samay` adapted to its Gemini prompts. The earlier local prototype and its test results remain available on [samay](https://github.com/saisamay/Beyond_Words/tree/samay).
