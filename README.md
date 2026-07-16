# ExtraTrees Online Calculator

This repository is ready to deploy on Streamlit Community Cloud.

It bundles three saved ExtraTrees pipelines:

- `assets/mce`: MCE prediction
- `assets/diagnosis`: 3-month poor functional outcome prediction
- `assets/diagnosis_mce`: joint 3-month outcome model that includes `MCE`

The web app outputs:

- MCE probability
- 3-month poor functional outcome probability
- 3-month poor functional outcome probability from the joint model

## Why this project targets Python 3.10

The bundled `.pkl` model files were verified locally under Python `3.9` and `3.10`. Streamlit Community Cloud currently supports Python `3.10` through `3.14`, so choose Python `3.10` in `Advanced settings` when you deploy.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repository root.
2. Open the Streamlit deployment page:

   `https://share.streamlit.io/`

3. Select your repository and set the main file path to:

   ```text
   app.py
   ```

4. In `Advanced settings`, choose Python `3.10`.
5. Deploy and wait for the app to build.

For Chinese step-by-step instructions, see `DEPLOY_GUIDE_ZH.md`.

## Repository contents

- `app.py`: Streamlit UI
- `predictor.py`: model loading and prediction logic
- `requirements.txt`: Python dependencies
- `packages.txt`: Linux system package dependencies for Streamlit Cloud
- `assets/`: bundled model files and metadata

## Notes

- The `Diagnosis+MBE` model requires an `MCE` input.
- If the true `MCE` status is unknown, the app can auto-fill `MCE` using the predicted MCE class.
- This tool is for research and decision support only. It is not a standalone clinical decision maker.
