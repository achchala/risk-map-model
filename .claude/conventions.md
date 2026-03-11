# Coding Conventions & Standards

## General Principles

1. **Simplicity over cleverness** - Prefer clear, readable code
2. **Avoid over-engineering** - Don't add features not explicitly requested
3. **Consistent naming** - Follow language/framework conventions
4. **Error handling** - Always handle errors gracefully with user feedback
5. **Performance awareness** - Consider memory and CPU for large datasets

---

## Python (Backend & ML Pipeline)

### Style
- **PEP 8** compliant
- Snake_case for functions and variables
- CamelCase for classes
- SCREAMING_SNAKE_CASE for constants

### Code Organization
```python
# Order: stdlib, third-party, local
import os
import json

import pandas as pd
from flask import Flask

from src.models.model_trainer import ModelTrainer
```

### Configuration
- Centralize config in `/config.py`
- Use `pathlib.Path` for file paths
- Environment variables for secrets (when applicable)

### Error Handling
```python
try:
    result = risky_operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}")
    return jsonify({"error": str(e)}), 500
```

### Logging
```python
import logging
logger = logging.getLogger(__name__)
logger.info("Starting process...")
```

### Flask API Patterns

**Endpoint naming:**
- Plural for collections: `/api/risk-predictions`
- Singular for single item: `/api/risk-prediction`

**Response structure:**
```python
# Success
return jsonify(data), 200

# Error
return jsonify({"error": "descriptive message"}), 400
```

**CORS:**
- Always enable for mobile app development
- Restrict in production

---

## Swift (iOS App)

### Style
- **Swift API Design Guidelines**
- camelCase for functions and variables
- PascalCase for types (structs, classes, enums)

### SwiftUI Patterns

**View Structure:**
```swift
struct MyView: View {
    // MARK: - Properties
    @StateObject private var service = MyService()
    @State private var isLoading = false

    // MARK: - Body
    var body: some View {
        content
    }

    // MARK: - View Components
    private var content: some View {
        VStack { /* ... */ }
    }

    // MARK: - Actions
    private func handleAction() { }
}
```

**State Management:**
- `@State` for local view state
- `@StateObject` for view-owned observable objects
- `@ObservedObject` for passed-in observable objects
- `@Published` in ObservableObject for reactive properties

### Service Pattern
```swift
class MyService: ObservableObject {
    @Published var data: [Item] = []
    @Published var isLoading = false
    @Published var errorMessage: String?

    func fetchData() async { }
}
```

### Error Handling
```swift
// Use Result type
func fetchData() -> Result<Data, APIError> { }

// Or throws
func fetchData() async throws -> Data { }

// Always show user-friendly errors
.alert("Error", isPresented: $showError) {
    Button("OK") { }
} message: {
    Text(errorMessage ?? "Unknown error")
}
```

### Networking
- 30-second timeout for all requests
- Always decode with `Codable`
- Use `CodingKeys` for JSON mapping when needed
```swift
struct RoadSegment: Codable {
    let id: String
    let linearName: String

    enum CodingKeys: String, CodingKey {
        case id
        case linearName = "LINEAR_NAME"
    }
}
```

### MapKit Conventions
- Always validate coordinates (filter NaN/infinite)
- Use `CLLocationCoordinate2D` for points
- Use `MKPolyline` for routes
- Handle location permissions properly

---

## Data Models

### Shared Concepts (Backend & iOS)

**Risk Level:**
- Always use enum/string: "low", "medium", "high"
- Never use integers (0, 1, 2) for risk levels

**Coordinates:**
- Always latitude/longitude (never reversed)
- Format: `{"latitude": 43.6532, "longitude": -79.3832}`
- Validate: lat ∈ [-90, 90], lng ∈ [-180, 180]

**Segment IDs:**
- Use string identifiers
- Format: Original dataset IDs or generated UUIDs

---

## File Organization

### Python
```
module/
├── __init__.py
├── core_logic.py      # Main functionality
├── utils.py           # Helper functions
└── config.py          # Module configuration
```

### Swift
```
Feature/
├── FeatureView.swift        # Main view
├── FeatureViewModel.swift   # If complex logic
├── FeatureService.swift     # API/business logic
└── FeatureModels.swift      # Data models
```

---

## API Design

### REST Conventions
- Use POST for data queries (with body)
- Use GET for simple queries (with params)
- Return 200 for success, 400 for client errors, 500 for server errors

### Request/Response Patterns

**Bounding Box Query:**
```json
POST /api/risk-predictions
{
  "north": 43.7,
  "south": 43.6,
  "east": -79.3,
  "west": -79.4
}
```

**Point Query:**
```json
POST /api/risk-prediction
{
  "latitude": 43.6532,
  "longitude": -79.3832
}
```

---

## Spatial Data Conventions

### Coordinate Systems
- **Always use EPSG:4326 (WGS84)** for lat/lng
- Never mix coordinate systems

### Buffer Distances
- Backend spatial join: 20 meters
- iOS route matching: 50 meters
- Adjustable in config.py

### Geometry Types
- Roads: LineString or MultiLineString
- Points: Point
- Bounding boxes: Polygon

---

## Performance Guidelines

### Backend
- Limit API responses: 500 segments max
- Limit coordinates per segment: 50 points
- Use pre-computed data when possible
- Log slow operations (>1 second)

### iOS
- Limit route sample points: 100 max
- Limit segment checks: 1000 max
- Debounce user input: 300ms
- Cancel previous requests when new ones start

### ML Pipeline
- Use R-tree for spatial joins (O(log n))
- Process in batches for large datasets
- Cache intermediate results

---

## Comments & Documentation

### When to Comment
- Complex algorithms (e.g., risk scoring)
- Non-obvious design decisions
- Performance optimizations
- Workarounds for bugs/limitations

### When NOT to Comment
- Self-explanatory code
- Obvious function names
- Standard patterns

### Good Comment Example
```python
# Use R-tree spatial indexing for O(log n) performance
# instead of brute-force O(n²) comparison
spatial_index = STRtree(geometries)
```

### Documentation
- Add docstrings for public APIs
- Update README when changing setup
- Keep INTEGRATION.md current

---

## Testing Approach

### Manual Testing Workflow
1. Run ML pipeline to generate fresh data
2. Start backend API
3. Test API with curl/Postman
4. Run iOS app and test features
5. Verify map, routing, and risk display

### What to Test
- Backend: Health endpoint, both prediction endpoints
- iOS: Map loading, location search, route calculation
- ML: Model training, output generation

---

## Git Practices

### Commit Messages
- Descriptive and concise
- Present tense: "Add feature" not "Added feature"
- Include co-author for AI assistance:
  ```
  Add safer route comparison feature

  Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
  ```

### Branching
- Main branch: `main`
- Feature branches: descriptive names (e.g., `adriel-ui-branch`)

### What to Commit
- Source code
- Configuration files
- Documentation
- Requirements/dependencies

### What NOT to Commit
- Generated outputs (models, GeoJSON)
- Large data files (use .gitignore)
- API keys or secrets
- Xcode user-specific files

---

## Security Considerations

### Current State (Development)
- No authentication on backend
- CORS wide open
- Running on localhost only

### Production Requirements
- Add API key authentication
- Restrict CORS to specific origins
- Use HTTPS
- Rate limiting
- Input validation

### Data Privacy
- No user data collected currently
- Historical crash data is public
- Consider anonymization for future features

---

## Code Review Checklist

Before merging:
- [ ] Code follows style conventions
- [ ] Error handling implemented
- [ ] Performance considerations addressed
- [ ] No hardcoded values (use config)
- [ ] Documentation updated if needed
- [ ] Tested manually
- [ ] No commented-out code
- [ ] No debug print statements
