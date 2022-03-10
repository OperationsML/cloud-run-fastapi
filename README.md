# Deeplearning + FASTAPI

The following code inside of "app" can be used to package a keras transformer model into FASTAPI and deploy it on google cloud run.

Steps:

Define your inference code in main.py along with your FastAPI definition.

Create the Dockerfile to package your API

Build the image (command line)

```
gcloud builds submit --tag gcr.io/${GOOGLE_CLOUD_PROJECT}/deeplearnapi:latest --timeout 1800
```

Deploy on cloud run with 1 min instance, 4 CPUs, and 2 GB of memory
```
gcloud beta run deploy deeplearn  --platform=managed --allow-unauthenticated --image=gcr.io/${GOOGLE_CLOUD_PROJECT}/deeplearnapi:latest --region=us-central1 --cpu=4 --memory=2Gi --min-instances=1
```
