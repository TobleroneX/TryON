#!/bin/bash

# Build Docker image
docker build -t virtual-tryon-api .

# Push to registry
docker tag virtual-tryon-api registry.heroku.com/$APP_NAME/web
docker push registry.heroku.com/$APP_NAME/web

# Release to Heroku
heroku container:release web -a $APP_NAME

# Run deployment tests
pytest tests/test_deployment.py -v 