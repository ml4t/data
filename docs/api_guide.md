# ML4T Data API Guide

## Overview

The ML4T Data API provides programmatic access to all data management functionality through a RESTful interface with WebSocket support for real-time data streams. The API is built with FastAPI and provides automatic OpenAPI documentation.

## Getting Started

### Starting the Server

```bash
# Start the API server
ml4t-data server

# Start with custom host and port
ml4t-data server --host 0.0.0.0 --port 8080

# Development mode with auto-reload
ml4t-data server --reload
```

### Authentication

All data modification endpoints require API key authentication using Bearer tokens:

```bash
curl -H "Authorization: Bearer your_api_key_here" \
     http://localhost:8000/api/v1/data/list
```

Default demo API key: `demo_api_key_12345`

### Interactive Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Spec**: http://localhost:8000/openapi.json

## API Endpoints

### Data Management

#### Load Data
Load new data from a provider:

```bash
POST /api/v1/data/load
Content-Type: application/json
Authorization: Bearer demo_api_key_12345

{
  "provider": "yahoo",
  "symbol": "AAPL",
  "start": "2024-01-01",
  "end": "2024-01-31",
  "frequency": "daily",
  "asset_class": "equities"
}
```

#### Get Data
Retrieve stored data:

```bash
GET /api/v1/data/equities/daily/AAPL?include_data=true&limit=10
Authorization: Bearer demo_api_key_12345
```

#### Update Data
Perform incremental update:

```bash
PUT /api/v1/data/equities/daily/AAPL/update
Content-Type: application/json
Authorization: Bearer demo_api_key_12345

{
  "symbol": "AAPL",
  "frequency": "daily",
  "asset_class": "equities",
  "lookback_days": 7,
  "fill_gaps": true
}
```

#### List Data
List all stored datasets:

```bash
GET /api/v1/data/list?prefix=equities
Authorization: Bearer demo_api_key_12345
```

#### Delete Data
Remove stored data:

```bash
DELETE /api/v1/data/equities/daily/AAPL
Authorization: Bearer demo_api_key_12345
```

### Export Operations

#### Export Data
Export data to various formats:

```bash
POST /api/v1/export
Content-Type: application/json
Authorization: Bearer demo_api_key_12345

{
  "key": "equities/daily/AAPL",
  "format_type": "csv",
  "compression": "gzip",
  "columns": ["timestamp", "close", "volume"],
  "start_date": "2024-01-01",
  "end_date": "2024-01-31",
  "add_returns": true,
  "add_volatility": false
}
```

#### Batch Export
Export multiple datasets:

```bash
POST /api/v1/export/batch
Content-Type: application/json
Authorization: Bearer demo_api_key_12345

{
  "pattern": "equities/daily/*",
  "output_dir": "/tmp/exports",
  "format_type": "csv",
  "compression": "gzip"
}
```

### Health & Metadata

#### System Health
Get overall system health (no auth required):

```bash
GET /api/v1/health?stale_days=7
```

#### Dataset Health
Get specific dataset health:

```bash
GET /api/v1/health/equities/daily/AAPL?stale_days=7
```

#### Dataset Metadata
Get detailed metadata:

```bash
GET /api/v1/metadata/equities/daily/AAPL
```

#### Available Providers
List supported data providers:

```bash
GET /api/v1/providers
```

### WebSocket Real-time Data

**🔐 Authentication Required**: WebSocket connections now require API key authentication via query parameter.

Connect to the WebSocket endpoint for real-time data streams:

```javascript
// IMPORTANT: Include API key in connection URL
const ws = new WebSocket('ws://localhost:8000/ws/data?api_key=demo_api_key_12345');

// Subscribe to symbols (after successful authentication)
ws.send(JSON.stringify({
  type: 'subscribe',
  symbols: ['AAPL', 'GOOGL', 'MSFT'],  // Max 100 symbols, 50 chars each
  frequency: '1minute'
}));

// Connection will be closed if:
// - No api_key query parameter provided
// - Invalid API key
// - More than 10 connections from same IP
// - Message size exceeds 8KB

// Handle real-time updates
ws.onmessage = (event) => {
  const message = JSON.parse(event.data);

  if (message.type === 'data_update') {
    console.log('Price update:', message.symbol, message.data);
  }
};

// Unsubscribe from symbols
ws.send(JSON.stringify({
  type: 'unsubscribe',
  symbols: ['AAPL']
}));
```

## Python Client Example

```python
import httpx
import asyncio
import websockets
import json

class ML4T DataClient:
    def __init__(self, base_url="http://localhost:8000", api_key="demo_api_key_12345"):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.client = httpx.Client(base_url=base_url, headers=self.headers)

    def load_data(self, provider, symbol, start, end, frequency="daily", asset_class="equities"):
        """Load data from a provider."""
        response = self.client.post("/api/v1/data/load", json={
            "provider": provider,
            "symbol": symbol,
            "start": start,
            "end": end,
            "frequency": frequency,
            "asset_class": asset_class
        })
        response.raise_for_status()
        return response.json()

    def get_data(self, key, include_data=False, limit=None):
        """Get stored data."""
        params = {}
        if include_data:
            params["include_data"] = True
        if limit:
            params["limit"] = limit

        response = self.client.get(f"/api/v1/data/{key}", params=params)
        response.raise_for_status()
        return response.json()

    def export_data(self, key, format_type="csv", **options):
        """Export data."""
        response = self.client.post("/api/v1/export", json={
            "key": key,
            "format_type": format_type,
            **options
        })
        response.raise_for_status()
        return response.json()

    def get_health(self):
        """Get system health."""
        response = self.client.get("/api/v1/health")
        response.raise_for_status()
        return response.json()

    async def subscribe_to_updates(self, symbols, callback):
        """Subscribe to real-time updates via WebSocket."""
        # SECURITY: Include API key in WebSocket URL
        uri = f"ws://localhost:8000/ws/data?api_key={self.api_key}"

        async with websockets.connect(uri) as websocket:
            # Subscribe to symbols
            await websocket.send(json.dumps({
                "type": "subscribe",
                "symbols": symbols,
                "frequency": "1minute"
            }))

            # Listen for updates
            async for message in websocket:
                data = json.loads(message)
                await callback(data)

# Usage example
client = ML4T DataClient()

# Load some data
result = client.load_data("yahoo", "AAPL", "2024-01-01", "2024-01-31")
print(f"Loaded: {result['key']}")

# Get the data
data = client.get_data("equities/daily/AAPL", include_data=True, limit=5)
print(f"Rows: {data['rows']}")

# Export to CSV
export_result = client.export_data("equities/daily/AAPL", "csv", compression="gzip")
print(f"Export: {export_result['output_path']}")

# Check health
health = client.get_health()
print(f"System status: {health['status']}")

# Real-time updates
async def handle_update(message):
    if message["type"] == "data_update":
        print(f"Update: {message['symbol']} = {message['data']['close']}")

# asyncio.run(client.subscribe_to_updates(["AAPL"], handle_update))
```

## Rate Limits

The API implements rate limiting to prevent abuse:

- **Data operations**: 10-20 requests/minute
- **Queries**: 50-100 requests/minute
- **Health checks**: 100 requests/minute
- **Exports**: 5-20 requests/minute

Rate limits are applied per IP address.

## Error Handling

All errors return standardized JSON responses:

```json
{
  "error": "Data not found",
  "detail": "No data exists for key: equities/daily/INVALID",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

Common HTTP status codes:
- **200**: Success
- **400**: Bad Request (invalid parameters)
- **401**: Unauthorized (invalid/missing API key)
- **404**: Not Found (resource doesn't exist)
- **429**: Too Many Requests (rate limit exceeded)
- **500**: Internal Server Error

## WebSocket Security & Limits

### Authentication
- **Required**: API key must be provided as query parameter `?api_key=your_key`
- **Validation**: Connection rejected immediately if API key is missing or invalid
- **Error Codes**:
  - `1008`: Authentication required (no API key)
  - `1008`: Invalid API key

### Connection Limits
- **Global Limit**: Maximum 1,000 concurrent connections
- **Per-IP Limit**: Maximum 10 connections per IP address
- **Error Code**: `1008` when limits exceeded

### Message Validation
- **Size Limit**: Maximum 8KB per message
- **Symbol Validation**:
  - Format: `^[A-Z0-9\-_.:/]{1,50}$`
  - Maximum 50 characters per symbol
  - Maximum 100 symbols per subscription
- **Error Codes**:
  - `1009`: Message too large
  - Validation error response for invalid symbols

### Error Handling
```json
{
  "type": "error",
  "error": "Validation error",
  "detail": "Invalid symbol format: INVALID!",
  "timestamp": "2025-01-19T12:00:00Z"
}
```

## WebSocket Message Types

### Client to Server

#### Subscribe
```json
{
  "type": "subscribe",
  "symbols": ["AAPL", "GOOGL"],  // Max 100 symbols, 50 chars each
  "frequency": "1minute"
}
```

**Validation Rules**:
- `symbols`: 1-100 valid symbol strings
- Symbol format: Letters, numbers, hyphens, underscores, dots, colons, slashes
- Max 50 characters per symbol

#### Unsubscribe
```json
{
  "type": "unsubscribe",
  "symbols": ["AAPL"]
}
```

#### Heartbeat
```json
{
  "type": "heartbeat"
}
```

### Server to Client

#### Data Update
```json
{
  "type": "data_update",
  "symbol": "AAPL",
  "data": {
    "timestamp": "2024-01-15T10:30:00Z",
    "open": 185.20,
    "high": 186.15,
    "low": 184.80,
    "close": 185.95,
    "volume": 1250000
  },
  "provider": "yahoo"
}
```

#### Connection Status
```json
{
  "type": "connection_status",
  "status": "connected",
  "subscriptions": ["AAPL", "GOOGL"]
}
```

#### Error
```json
{
  "type": "error",
  "error": "Invalid symbol",
  "detail": "Symbol 'INVALID' is not supported"
}
```

## Deployment

### Production Considerations

1. **Security**:
   - Use strong API keys
   - Enable HTTPS/TLS
   - Configure CORS appropriately
   - Set up proper firewall rules

2. **Performance**:
   - Use a production ASGI server (uvicorn with workers)
   - Configure connection pooling
   - Enable compression
   - Set up caching for frequently accessed data

3. **Monitoring**:
   - Log all API requests
   - Monitor rate limits and errors
   - Set up health check endpoints
   - Use structured logging

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN pip install -e .

EXPOSE 8000
CMD ["ml4t-data", "server", "--host", "0.0.0.0", "--port", "8000"]
```

### Systemd Service

```ini
[Unit]
Description=ML4T Data API Server
After=network.target

[Service]
Type=simple
User=ml4t-data
WorkingDirectory=/opt/ml4t-data
ExecStart=/opt/ml4t-data/venv/bin/ml4t-data server --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

## Future Enhancements

Sprint 007 provides the foundation for future API enhancements:

- **Authentication**: OAuth2, JWT tokens, user management
- **Analytics**: Real-time technical indicators via WebSocket
- **Scalability**: Redis caching, database backends, horizontal scaling
- **Monitoring**: Prometheus metrics, distributed tracing
- **Documentation**: Interactive tutorials, SDK generation
