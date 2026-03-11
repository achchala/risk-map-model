# System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   OFFLINE PIPELINE                      │
│                                                         │
│  ┌────────────┐    ┌──────────────┐    ┌────────────┐ │
│  │ Raw Data   │───>│ ML Pipeline  │───>│ Outputs    │ │
│  │ 618K rows  │    │ (Python)     │    │ Model+JSON │ │
│  └────────────┘    └──────────────┘    └────────────┘ │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   ONLINE SERVICES                       │
│                                                         │
│  ┌────────────┐    ┌──────────────┐                    │
│  │ Backend    │<───│ Precomputed  │                    │
│  │ API        │    │ Predictions  │                    │
│  │ (Flask)    │    └──────────────┘                    │
│  └─────┬──────┘                                         │
│        │                                                │
│        │ HTTP/JSON                                      │
│        ▼                                                │
│  ┌────────────┐                                         │
│  │ iOS App    │                                         │
│  │ (SwiftUI)  │                                         │
│  └────────────┘                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### 1. ML Pipeline (`/src/`)

**Purpose:** Offline data processing and model training

**Flow:**
```
Raw Data → Load → Clean → Spatial Join → Feature Engineering
→ Label Generation → Model Training → Evaluation → Export
```

**Components:**

#### Data Processing (`/src/data_processing/`)
- **data_loader.py**: Load collision, KSI, and road network data
  - Input: Excel (collisions), CSV (KSI), GeoJSON (roads)
  - Output: Pandas/GeoPandas DataFrames

- **spatial_join_fast.py**: Match crashes to road segments
  - Uses R-tree spatial indexing (O(log n))
  - 20-meter buffer distance
  - Processing time: 2-3 minutes for 618K crashes

#### Feature Engineering (`/src/feature_engineering/`)
- **feature_creator.py**: Generate 15+ features
  - Temporal: time_of_day, season, weekend_ratio
  - Road: road_class, segment_length
  - Crash: crash_density, severity_index, ksi_ratio

- **label_generator.py**: Assign risk labels
  - High: >2 KSI OR >10 crashes
  - Medium: ≥1 KSI OR >5 crashes
  - Low: everything else

#### Model Training (`/src/models/`)
- **model_trainer.py**: Train Random Forest classifier
  - 100 trees, max depth 10
  - SMOTE for class balancing
  - 80/20 train/test split
  - Grid search for hyperparameters

- **model_evaluator.py**: Calculate metrics
  - Accuracy, precision, recall, F1 per class
  - Confusion matrix
  - Feature importance

#### Visualization (`/src/visualization/`)
- **risk_mapper.py**: Generate HTML maps
  - Folium for interactive maps
  - Color-coded by risk level
  - Clickable segments with details

**Outputs:**
```
outputs/
├── models/
│   └── toronto_risk_model.joblib      # Trained model + metadata
├── reports/
│   ├── toronto_road_risk.geojson      # Pre-computed predictions
│   ├── toronto_road_risk.csv
│   └── risk_analysis_summary.html
└── maps/
    ├── toronto_risk_map.html
    └── toronto_risk_analysis_dashboard.html
```

---

### 2. Backend API (`/backend-api/`)

**Purpose:** REST API serving ML predictions

**Architecture:** Single-file Flask application (`app.py`)

**Startup Sequence:**
```python
1. Load trained model (joblib)
2. Load pre-processed GeoJSON
3. Load raw road network (fallback)
4. Start Flask server on port 8000
```

**Endpoints:**

#### GET `/api/health`
```
Purpose: System health check
Response: {
  "model_loaded": bool,
  "preprocessed_data_available": bool,
  "road_network_available": bool,
  "num_preprocessed_segments": int
}
```

#### POST `/api/risk-predictions`
```
Purpose: Get risk segments in bounding box
Input: {north, south, east, west}
Processing:
  1. Parse bounding box coordinates
  2. Create Shapely box geometry
  3. Query pre-processed GeoJSON
  4. Filter segments intersecting bbox
  5. Prioritize high-risk segments
  6. Limit to 500 segments
  7. Extract coordinates (max 50 per segment)
Output: Array of RoadSegment objects
Performance: 10-50ms (pre-processed)
```

#### POST `/api/risk-prediction`
```
Purpose: Get risk for specific location
Input: {latitude, longitude}
Processing:
  1. Parse coordinates
  2. Find nearest segment (within 1km)
  3. Return segment risk data
Output: Single RoadSegment with prediction
Performance: 50-200ms
```

**Data Sources:**
1. **Primary:** Pre-processed GeoJSON (fastest)
2. **Secondary:** ML model prediction (accurate)
3. **Fallback:** Raw road network (basic info)

**Error Handling:**
- Try-except blocks for all endpoints
- Logging with Python logging module
- Return JSON errors with appropriate status codes

---

### 3. iOS App (`/ios-app/RiskMapApp/`)

**Purpose:** Mobile interface for risk visualization and routing

**Architecture:** SwiftUI with MVVM pattern

**Structure:**
```
RiskMapApp/
├── App Entry
│   ├── RiskMapApp.swift          # @main entry point
│   └── ContentView.swift         # TabView navigation
├── Models/
│   └── RiskModels.swift          # Data structures
├── Services/
│   ├── RiskService.swift         # Backend API client
│   └── RouteService.swift        # Route calculation + risk analysis
└── Views/
    ├── MapView.swift             # Main risk map
    ├── RiskDetailView.swift      # Segment details modal
    ├── RiskListView.swift        # High-risk list
    ├── NavigationView.swift      # Route planning
    └── SettingsView.swift        # App settings
```

#### Navigation Flow
```
App Launch
  ↓
TabView (ContentView.swift)
  ├── Map Tab → MapView
  ├── High Risk Tab → RiskListView
  ├── Navigation Tab → NavigationView
  └── Settings Tab → SettingsView
```

#### Service Layer

**RiskService.swift:**
- Communicates with backend API
- ObservableObject with @Published properties
- Handles network requests and errors
- Decodes JSON to Swift models

**RouteService.swift:**
- Calculates routes using MapKit
- Performs risk analysis on routes
- Spatial matching of route points to segments
- Generates route comparisons

#### View Layer

**MapView.swift:**
```
Responsibilities:
- Display MapKit map
- Fetch risk segments for visible region
- Render colored polylines (green/orange/red)
- Handle user interactions (tap segment)
- Show loading/error states

Data Flow:
1. onAppear → fetch risk data
2. Map region changes → update visible area
3. Receive segments → render polylines
4. User taps → show detail sheet
```

**NavigationView.swift:**
```
Responsibilities:
- Location search with autocomplete
- Route calculation (optimal + safer)
- Route visualization on map
- Route comparison card
- Export to Google Maps

User Flow:
1. Enter start/destination
2. Tap "Find Safest Route"
3. View both routes on map
4. Compare metrics in card
5. Select preferred route
6. Export to Google Maps
```

---

## Data Flow Diagrams

### Map View Data Flow
```
User Opens Map Tab
        ↓
    MapView.onAppear()
        ↓
Get visible map region (bbox)
        ↓
RiskService.fetchRiskPredictions(bbox)
        ↓
POST /api/risk-predictions
        ↓
Backend queries GeoJSON
        ↓
Returns array of RoadSegments
        ↓
Decode JSON → Swift models
        ↓
Update @Published property
        ↓
SwiftUI re-renders view
        ↓
Draw colored polylines on map
```

### Route Planning Data Flow
```
User Enters Start/Destination
        ↓
Tap "Find Safest Route"
        ↓
RouteService.calculateRoutes()
        ↓
┌─────────────────┬─────────────────┐
│                 │                 │
MKDirections      MKDirections      RiskService
(optimal)         (alternates)      (risk data)
│                 │                 │
└─────────────────┴─────────────────┘
        ↓
Receive routes + risk segments
        ↓
For each route:
  - Sample points (every 100m)
  - Match to nearest segments
  - Count high/medium/low risk
  - Calculate risk score
        ↓
Select safest route (lowest score)
        ↓
Generate comparison
        ↓
Update @Published properties
        ↓
Display routes on map + comparison card
```

### ML Pipeline Data Flow
```
Run run_risk_analysis.py
        ↓
Load Raw Data
  - Collisions (618K rows)
  - KSI crashes (18K rows)
  - Road network (65K segments)
        ↓
Spatial Join (R-tree)
  - Match crashes to segments
  - Aggregate by segment
        ↓
Feature Engineering
  - Create 15+ features
  - Normalize/scale
        ↓
Label Generation
  - Apply risk rules
  - Assign High/Medium/Low
        ↓
Train Model
  - Random Forest
  - SMOTE balancing
  - Cross-validation
        ↓
Evaluate Model
  - Metrics
  - Confusion matrix
        ↓
Generate Predictions
  - Predict all segments
        ↓
Export Outputs
  - Model (joblib)
  - GeoJSON (predictions)
  - HTML (visualizations)
```

---

## Key Algorithms

### 1. Route Risk Scoring
```
Input: MKRoute, Array<RoadSegment>
Output: Route with risk metrics

Algorithm:
1. Extract route coordinates
2. Sample points (every 100m or 50 total)
3. For each point:
   a. Find nearest segment (within 50m)
   b. Record risk level if found
4. Count segments by risk level
5. Calculate weighted score:
   score = (high×3.0 + medium×2.0 + low×1.0) / total
6. Return route with metrics
```

### 2. Spatial Matching (Nearest Segment)
```
Input: Coordinate, Array<RoadSegment>
Output: Nearest segment or nil

Algorithm:
1. Initialize minDistance = infinity
2. For each segment (limit 1000):
   a. For each coordinate in segment:
      - Calculate distance to point
      - Track minimum distance
3. If minDistance < 50m:
   Return segment
4. Else:
   Return nil
```

### 3. Safer Route Selection
```
Input: Array<MKRoute>, Array<RoadSegment>
Output: Safest route

Algorithm:
1. Analyze each route (get risk scores)
2. Sort by risk score (ascending)
3. If tie, prefer shorter distance
4. Return route with lowest score
```

---

## Performance Characteristics

### Backend API
| Operation | Time | Bottleneck |
|-----------|------|------------|
| Health check | <10ms | In-memory check |
| Regional query (cached) | 10-50ms | GeoJSON query |
| Regional query (model) | 100-500ms | ML inference |
| Point query | 50-200ms | Linear search |

### iOS App
| Operation | Time | Bottleneck |
|-----------|------|------------|
| Map load | 1-2s | Network + rendering |
| Route calculation | 2-5s | MapKit + risk analysis |
| Location search | 300-800ms | MKLocalSearchCompleter |
| Segment tap response | <500ms | UI update |

### ML Pipeline
| Operation | Time | Bottleneck |
|-----------|------|------------|
| Data loading | 30-60s | File I/O |
| Spatial join | 2-3min | R-tree indexing |
| Feature engineering | 1-2min | Pandas operations |
| Model training | 3-5min | Random Forest fit |
| Total pipeline | 10-15min | Combined |

---

## Scalability Considerations

### Current Limitations
- **Backend:** Single-threaded, no caching
- **iOS:** All data in memory
- **ML:** Runs on single machine

### Scaling Path
1. **Backend:**
   - Add Redis caching layer
   - Use PostGIS for spatial queries
   - Deploy with gunicorn (multi-worker)
   - Add load balancer

2. **iOS:**
   - Implement progressive loading
   - Cache segments locally
   - Background refresh

3. **ML Pipeline:**
   - Distributed training (Spark MLlib)
   - Incremental updates
   - Cloud deployment (AWS SageMaker)

---

## Technology Decisions

### Why Flask?
- Lightweight and simple
- Easy integration with Python ML libraries
- Sufficient for MVP and local development
- Can upgrade to Django/FastAPI later

### Why SwiftUI?
- Modern declarative UI framework
- Native performance
- Built-in reactive patterns
- Future-proof for iOS

### Why Random Forest?
- Handles non-linear relationships
- Feature importance insights
- Robust to outliers
- Good accuracy without tuning

### Why GeoJSON?
- Standard format for spatial data
- Human-readable
- Works with GeoPandas and Shapely
- Easy to visualize

### Why MapKit?
- Native iOS integration
- Free (no API keys needed)
- Good routing quality
- Built-in geocoding

---

## Security Architecture

### Current (Development)
- No authentication
- CORS: `*` (allow all)
- HTTP (not HTTPS)
- No rate limiting
- No input validation

### Production Requirements
```
┌────────────┐
│ iOS App    │
└─────┬──────┘
      │ HTTPS + API Key
      ▼
┌────────────┐
│ API Gateway│ ← Rate Limiting
│            │ ← Authentication
└─────┬──────┘
      │
      ▼
┌────────────┐
│ Backend API│ ← Input Validation
│            │ ← Logging
└────────────┘
```

---

## Deployment Architecture (Future)

### Development
```
localhost:8000 (Backend)
    ↑
iOS Simulator
```

### Production
```
┌─────────────┐
│ iOS App     │
│ (App Store) │
└──────┬──────┘
       │ HTTPS
       ▼
┌──────────────┐
│ Load Balancer│
└──────┬───────┘
       │
  ┌────┴────┐
  │         │
┌─▼───┐ ┌──▼──┐
│ API │ │ API │
│ #1  │ │ #2  │
└─┬───┘ └──┬──┘
  │        │
  └────┬───┘
       ▼
┌─────────────┐
│ PostgreSQL  │
│ + PostGIS   │
└─────────────┘
```
