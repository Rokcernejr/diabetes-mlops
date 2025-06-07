import time
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Request, Response
from functools import wraps

# Metrics
REQUEST_COUNT = Counter(
    'api_requests_total', 
    'Total API requests', 
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'api_request_duration_seconds', 
    'API request duration in seconds',
    ['method', 'endpoint']
)

PREDICTION_COUNT = Counter(
    'ml_predictions_total',
    'Total ML predictions made',
    ['model_version']
)

MODEL_ACCURACY = Gauge(
    'ml_model_accuracy',
    'Current model accuracy score',
    ['model_name', 'model_version']
)

ACTIVE_CONNECTIONS = Gauge(
    'api_active_connections',
    'Number of active connections'
)

class MetricsMiddleware:
    """Middleware to collect request metrics"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        
        request = Request(scope, receive)
        start_time = time.time()
        
        # Increment active connections
        ACTIVE_CONNECTIONS.inc()
        
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_code = message["status"]
                duration = time.time() - start_time
                
                # Record metrics
                REQUEST_COUNT.labels(
                    method=request.method,
                    endpoint=request.url.path,
                    status_code=status_code
                ).inc()
                
                REQUEST_DURATION.labels(
                    method=request.method,
                    endpoint=request.url.path
                ).observe(duration)
                
                # Decrement active connections
                ACTIVE_CONNECTIONS.dec()
            
            await send(message)
        
        await self.app(scope, receive, send_wrapper)

def record_prediction(model_version: str = "1.0.0"):
    """Record a prediction event"""
    PREDICTION_COUNT.labels(model_version=model_version).inc()

def update_model_accuracy(accuracy: float, model_name: str = "diabetes", model_version: str = "1.0.0"):
    """Update model accuracy metric"""
    MODEL_ACCURACY.labels(model_name=model_name, model_version=model_version).set(accuracy)

def generate_metrics():
    """Generate Prometheus metrics"""
    return generate_latest()