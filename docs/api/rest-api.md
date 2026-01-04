# REST API Reference

Complete reference for the ML4T Data REST API.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

The API supports API key authentication when enabled:

```http
X-API-Key: your-api-key-here
```

## Response Format

All responses follow a consistent format:

### Success Response
```json
{
  "status": "success",
  "data": {...},
  "meta": {
    "timestamp": "2024-06-15T10:30:00Z",
    "version": "1.0.0"
  }
}
```

### Error Response
```json
{
  "status": "error",
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid symbol format",
    "details": {...}
  },
  "meta": {
    "timestamp": "2024-06-15T10:30:00Z"
  }
}
```

## Endpoints

### Health Check

Check API server health status.

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime": 3600,
  "database": "connected",
  "providers": {
    "yahoo": "active",
    "binance": "active",
    "cryptocompare": "inactive"
  }
}
```

### Get Symbol Data

Retrieve market data for a specific symbol.

```http
GET /data/{symbol}
```

**Parameters:**
| Parameter | Type | Location | Required | Description |
|-----------|------|----------|----------|-------------|
| symbol | string | path | Yes | Symbol identifier (e.g., AAPL) |
| start | string | query | No | Start date (YYYY-MM-DD) |
| end | string | query | No | End date (YYYY-MM-DD) |
| frequency | string | query | No | Data frequency (1m, 5m, 1h, 1d) |
| provider | string | query | No | Specific provider |

**Example Request:**
```bash
curl -X GET "http://localhost:8000/api/v1/data/AAPL?start=2024-01-01&end=2024-06-30"
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "symbol": "AAPL",
    "provider": "yahoo",
    "frequency": "daily",
    "rows": [
      {
        "timestamp": "2024-01-01T00:00:00Z",
        "open": 150.25,
        "high": 152.30,
        "low": 149.50,
        "close": 151.75,
        "volume": 45678900
      },
      ...
    ]
  },
  "meta": {
    "total_rows": 125,
    "start_date": "2024-01-01",
    "end_date": "2024-06-30"
  }
}
```

### Update Symbol Data

Update data for a specific symbol.

```http
POST /update/{symbol}
```

**Parameters:**
| Parameter | Type | Location | Required | Description |
|-----------|------|----------|----------|-------------|
| symbol | string | path | Yes | Symbol to update |
| provider | string | body | No | Specific provider |
| force | boolean | body | No | Force full refresh |
| fill_gaps | boolean | body | No | Fill detected gaps |

**Example Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/update/AAPL" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"fill_gaps": true}'
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "symbol": "AAPL",
    "provider": "yahoo",
    "rows_before": 250,
    "rows_after": 252,
    "new_rows": 2,
    "gaps_filled": 0
  }
}
```

### List Symbols

Get list of all available symbols.

```http
GET /symbols
```

**Parameters:**
| Parameter | Type | Location | Required | Description |
|-----------|------|----------|----------|-------------|
| provider | string | query | No | Filter by provider |
| search | string | query | No | Search term |
| limit | integer | query | No | Maximum results (default: 100) |
| offset | integer | query | No | Pagination offset |

**Example Request:**
```bash
curl -X GET "http://localhost:8000/api/v1/symbols?provider=yahoo&limit=10"
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "symbols": [
      {
        "symbol": "AAPL",
        "name": "Apple Inc.",
        "provider": "yahoo",
        "last_updated": "2024-06-15T09:30:00Z",
        "row_count": 252,
        "date_range": {
          "start": "2024-01-01",
          "end": "2024-06-15"
        }
      },
      ...
    ]
  },
  "meta": {
    "total": 150,
    "limit": 10,
    "offset": 0
  }
}
```

### Export Data

Export symbol data in various formats.

```http
POST /export
```

**Request Body:**
```json
{
  "symbol": "AAPL",
  "format": "csv",
  "start": "2024-01-01",
  "end": "2024-06-30",
  "columns": ["timestamp", "close", "volume"]
}
```

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| symbol | string | Yes | Symbol to export |
| format | string | Yes | Export format (csv, json, excel, parquet) |
| start | string | No | Start date |
| end | string | No | End date |
| columns | array | No | Specific columns to include |

**Response (CSV):**
```
timestamp,close,volume
2024-01-01,151.75,45678900
2024-01-02,152.50,43210000
...
```

**Response (JSON):**
```json
{
  "status": "success",
  "data": {
    "url": "/downloads/export_abc123.csv",
    "expires_at": "2024-06-15T11:00:00Z"
  }
}
```

### Validate Data

Validate data quality for a symbol.

```http
POST /validate/{symbol}
```

**Parameters:**
| Parameter | Type | Location | Required | Description |
|-----------|------|----------|----------|-------------|
| symbol | string | path | Yes | Symbol to validate |
| checks | array | body | No | Specific validation checks |
| fix | boolean | body | No | Attempt to fix issues |

**Example Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/validate/AAPL" \
  -H "Content-Type: application/json" \
  -d '{"checks": ["nulls", "consistency"], "fix": false}'
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "symbol": "AAPL",
    "passed": true,
    "issues": [],
    "summary": {
      "total_rows": 252,
      "null_count": 0,
      "duplicate_count": 0,
      "consistency_errors": 0
    }
  }
}
```

### Detect Gaps

Analyze data gaps for a symbol.

```http
GET /gaps/{symbol}
```

**Parameters:**
| Parameter | Type | Location | Required | Description |
|-----------|------|----------|----------|-------------|
| symbol | string | path | Yes | Symbol to analyze |
| threshold | integer | query | No | Gap threshold in hours |

**Response:**
```json
{
  "status": "success",
  "data": {
    "symbol": "AAPL",
    "gaps": [
      {
        "start": "2024-03-15T16:00:00Z",
        "end": "2024-03-18T09:30:00Z",
        "duration_hours": 65.5,
        "missing_periods": 2
      }
    ],
    "total_gaps": 1,
    "total_missing_periods": 2
  }
}
```

### Get Statistics

Get statistical summary for symbol data.

```http
GET /stats/{symbol}
```

**Parameters:**
| Parameter | Type | Location | Required | Description |
|-----------|------|----------|----------|-------------|
| symbol | string | path | Yes | Symbol identifier |
| start | string | query | No | Start date |
| end | string | query | No | End date |

**Response:**
```json
{
  "status": "success",
  "data": {
    "symbol": "AAPL",
    "period": {
      "start": "2024-01-01",
      "end": "2024-06-30"
    },
    "statistics": {
      "close": {
        "mean": 175.50,
        "median": 174.25,
        "std": 8.75,
        "min": 160.00,
        "max": 195.00
      },
      "volume": {
        "mean": 45000000,
        "median": 43000000,
        "total": 5625000000
      },
      "returns": {
        "total": 0.0850,
        "daily_mean": 0.0007,
        "volatility": 0.0175,
        "sharpe_ratio": 1.45
      }
    }
  }
}
```

## WebSocket API

Real-time data streaming via WebSocket.

### Connection

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
```

### Subscribe to Symbol

```json
{
  "action": "subscribe",
  "symbols": ["AAPL", "MSFT"],
  "provider": "yahoo"
}
```

### Unsubscribe

```json
{
  "action": "unsubscribe",
  "symbols": ["AAPL"]
}
```

### Data Stream

```json
{
  "type": "data",
  "symbol": "AAPL",
  "timestamp": "2024-06-15T10:30:00Z",
  "data": {
    "open": 175.50,
    "high": 176.25,
    "low": 175.00,
    "close": 176.00,
    "volume": 1234567
  }
}
```

## Rate Limiting

API rate limits when authentication is enabled:

| Tier | Requests/Minute | Requests/Hour | Concurrent Connections |
|------|----------------|---------------|------------------------|
| Free | 60 | 1000 | 2 |
| Basic | 300 | 5000 | 10 |
| Pro | 1000 | 20000 | 50 |

Rate limit headers:
```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1718446200
```

## Error Codes

| Code | Description |
|------|-------------|
| `SYMBOL_NOT_FOUND` | Symbol does not exist |
| `PROVIDER_ERROR` | Provider API error |
| `VALIDATION_ERROR` | Request validation failed |
| `RATE_LIMIT_EXCEEDED` | Rate limit exceeded |
| `AUTHENTICATION_FAILED` | Invalid API key |
| `INTERNAL_ERROR` | Internal server error |
| `DATA_NOT_AVAILABLE` | No data for requested period |

## Client Examples

### Python

```python
import requests

class ML4T DataClient:
    def __init__(self, base_url="http://localhost:8000/api/v1", api_key=None):
        self.base_url = base_url
        self.headers = {"X-API-Key": api_key} if api_key else {}

    def get_data(self, symbol, start=None, end=None):
        params = {}
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        response = requests.get(
            f"{self.base_url}/data/{symbol}",
            params=params,
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()["data"]

    def update_symbol(self, symbol, fill_gaps=False):
        response = requests.post(
            f"{self.base_url}/update/{symbol}",
            json={"fill_gaps": fill_gaps},
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()["data"]

# Usage
client = ML4T DataClient(api_key="your-api-key")
data = client.get_data("AAPL", start="2024-01-01")
print(f"Retrieved {len(data['rows'])} rows")
```

### JavaScript

```javascript
class ML4T DataClient {
  constructor(baseUrl = 'http://localhost:8000/api/v1', apiKey = null) {
    this.baseUrl = baseUrl;
    this.headers = apiKey ? { 'X-API-Key': apiKey } : {};
  }

  async getData(symbol, start = null, end = null) {
    const params = new URLSearchParams();
    if (start) params.append('start', start);
    if (end) params.append('end', end);

    const response = await fetch(
      `${this.baseUrl}/data/${symbol}?${params}`,
      { headers: this.headers }
    );

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data.data;
  }

  async updateSymbol(symbol, fillGaps = false) {
    const response = await fetch(
      `${this.baseUrl}/update/${symbol}`,
      {
        method: 'POST',
        headers: {
          ...this.headers,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ fill_gaps: fillGaps })
      }
    );

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data.data;
  }
}

// Usage
const client = new ML4T DataClient('http://localhost:8000/api/v1', 'your-api-key');
client.getData('AAPL', '2024-01-01').then(data => {
  console.log(`Retrieved ${data.rows.length} rows`);
});
```

### cURL

```bash
# Get data
curl -X GET "http://localhost:8000/api/v1/data/AAPL?start=2024-01-01" \
  -H "X-API-Key: your-api-key"

# Update symbol
curl -X POST "http://localhost:8000/api/v1/update/AAPL" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"fill_gaps": true}'

# Export to CSV
curl -X POST "http://localhost:8000/api/v1/export" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "format": "csv"}' \
  -o aapl.csv
```
