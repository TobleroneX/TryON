from prometheus_flask_exporter import PrometheusMetrics
from flask import request

metrics = PrometheusMetrics(app)

# Request latency
@metrics.histogram('request_latency_seconds', 'Request latency')
def measure_latency():
    return request.endpoint

# Image processing time
@metrics.histogram('image_processing_seconds', 'Image processing time')
def measure_processing():
    return request.endpoint

# Error rate
@metrics.counter('errors_total', 'Total errors')
def count_errors():
    return request.endpoint 