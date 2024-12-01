#!/bin/bash

# Build and push images
docker-compose -f docker-compose.prod.yml build
docker-compose -f docker-compose.prod.yml push

# Deploy to production
docker stack deploy -c docker-compose.prod.yml tryon

# Run migrations if needed
docker-compose -f docker-compose.prod.yml exec web flask db upgrade

# Run tests
docker-compose -f docker-compose.yml run tests

# Check deployment
docker service ls
docker service logs tryon_web 