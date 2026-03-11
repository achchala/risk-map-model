# Common Workflows

This document describes standard workflows for common development tasks.

---

## 1. Setting Up Development Environment

### Initial Setup
```bash
# Clone repository
git clone <repo-url>
cd risk-map-model

# Install Python dependencies
pip install -r requirements.txt

# Install backend dependencies
cd backend-api
pip install -r requirements.txt
cd ..

# Open iOS project in Xcode
open ios-app/RiskMapApp/RiskMapApp.xcodeproj
```

### Running Full Stack Locally

**Terminal 1 - Backend API:**
```bash
cd backend-api
python app.py
# Server runs on http://localhost:8000
```

**Terminal 2 - iOS Simulator:**
```bash
# In Xcode, press ⌘R to build and run
# Or use xcodebuild from command line
```

---

## 2. Updating ML Model

### When to Update
- New crash data available
- Model hyperparameters changed
- Feature engineering updates
- Risk labeling rules changed

### Workflow
```bash
# 1. Update data in /data/ folder
cp new_collision_data.xlsx data/

# 2. Update config.py if needed
vim config.py

# 3. Run ML pipeline (takes 5-10 minutes)
python run_risk_analysis.py

# 4. Verify outputs generated
ls -lh outputs/models/toronto_risk_model.joblib
ls -lh outputs/reports/toronto_road_risk.geojson

# 5. Restart backend API to load new model
# Stop app.py and restart

# 6. Test in iOS app
# Should see updated risk predictions
```

### Validation
- Check console logs for model accuracy metrics
- Open `outputs/maps/toronto_risk_analysis_dashboard.html`
- Verify confusion matrix and feature importance
- Test API endpoints with curl

---

## 3. Adding New Backend API Endpoint

### Steps
1. **Define endpoint in `backend-api/app.py`:**
```python
@app.route('/api/new-endpoint', methods=['POST'])
def new_endpoint():
    try:
        # Parse request
        data = request.get_json()

        # Validate input
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Process request
        result = process_data(data)

        # Return response
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"error": str(e)}), 500
```

2. **Test with curl:**
```bash
curl -X POST http://localhost:8000/api/new-endpoint \
  -H "Content-Type: application/json" \
  -d '{"test": "data"}'
```

3. **Update iOS service:**
```swift
// In RiskService.swift
func fetchNewData() async throws -> NewData {
    guard let url = URL(string: "\(baseURL)/new-endpoint") else {
        throw APIError.invalidURL
    }

    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")

    let (data, _) = try await URLSession.shared.data(for: request)
    let decoded = try JSONDecoder().decode(NewData.self, from: data)
    return decoded
}
```

4. **Document in `backend-api/README.md`**

---

## 4. Adding New iOS View

### Steps
1. **Create new Swift file:**
```bash
# In Xcode: File → New → File → SwiftUI View
# Name: FeatureView.swift
```

2. **Implement view:**
```swift
import SwiftUI

struct FeatureView: View {
    @StateObject private var service = FeatureService()

    var body: some View {
        NavigationView {
            content
                .navigationTitle("Feature")
        }
    }

    private var content: some View {
        VStack {
            // Your UI here
        }
    }
}

#Preview {
    FeatureView()
}
```

3. **Add to tab navigation in `ContentView.swift`:**
```swift
TabView {
    // ... existing tabs

    FeatureView()
        .tabItem {
            Label("Feature", systemImage: "icon.name")
        }
}
```

4. **Test in simulator**

---

## 5. Debugging Navigation Issues

### Check Route Calculation
```swift
// Add debug prints in RouteService.swift

print("📍 Calculating routes from \(start) to \(destination)")

// After route calculation
print("✅ Found \(routes.count) alternate routes")
for (index, route) in routes.enumerated() {
    print("Route \(index): \(route.distance)m, \(route.expectedTravelTime)s")
}

// After risk analysis
print("🔴 High risk: \(highRiskCount), 🟠 Medium: \(mediumRiskCount), 🟢 Low: \(lowRiskCount)")
print("📊 Risk score: \(riskScore)")
```

### Check Backend API
```bash
# Test risk predictions endpoint
curl -X POST http://localhost:8000/api/risk-predictions \
  -H "Content-Type: application/json" \
  -d '{"north": 43.72, "south": 43.65, "east": -79.35, "west": -79.42}' \
  | jq '.'

# Check health endpoint
curl http://localhost:8000/api/health | jq '.'
```

### Check Network Requests in iOS
```swift
// In RiskService.swift, add logging
print("🌐 Fetching risk predictions for bbox: \(north), \(south), \(east), \(west)")

do {
    let decoded = try JSONDecoder().decode([RoadSegment].self, from: data)
    print("✅ Received \(decoded.count) segments")
} catch {
    print("❌ Decoding error: \(error)")
}
```

---

## 6. Performance Optimization

### Backend API
```python
# Add timing to endpoints
import time

@app.route('/api/risk-predictions', methods=['POST'])
def risk_predictions():
    start_time = time.time()

    # ... process request

    elapsed = time.time() - start_time
    logger.info(f"Request completed in {elapsed:.3f}s")
    return jsonify(results)
```

### iOS App
```swift
// Profile route calculation
let startTime = Date()

await routeService.calculateRoutes(from: start, to: destination)

let elapsed = Date().timeIntervalSince(startTime)
print("⏱️ Route calculation took \(elapsed)s")
```

### Identify Bottlenecks
1. Backend logs show slow operations
2. iOS Instruments (Time Profiler)
3. Network request timing
4. Memory usage (Allocations instrument)

---

## 7. Testing Complete Flow

### Full Integration Test
```bash
# 1. Generate fresh ML predictions
python run_risk_analysis.py

# 2. Start backend
cd backend-api && python app.py

# 3. Test API
curl http://localhost:8000/api/health

# 4. Test iOS app
# - Open app in simulator
# - View map (should show colored segments)
# - Enter start/destination
# - Calculate route
# - Verify both routes display
# - Check comparison card metrics
# - Export to Google Maps

# 5. Verify in logs
# - Backend shows requests
# - iOS console shows responses
# - No errors in either
```

---

## 8. Adding New Feature to Config

### Update config.py
```python
# Add new configuration parameter
NEW_FEATURE_ENABLED = True
NEW_FEATURE_THRESHOLD = 0.75

# Document it
"""
NEW_FEATURE_ENABLED: Enable experimental feature
NEW_FEATURE_THRESHOLD: Confidence threshold for feature
"""
```

### Use in backend
```python
from config import NEW_FEATURE_ENABLED, NEW_FEATURE_THRESHOLD

if NEW_FEATURE_ENABLED:
    result = experimental_feature(threshold=NEW_FEATURE_THRESHOLD)
```

### Use in iOS (if needed)
```swift
// Add to RiskService.swift or settings
struct AppConfig {
    static let featureEnabled = true
    static let threshold = 0.75
}
```

---

## 9. Updating Documentation

### After Code Changes
```bash
# Update relevant docs
vim backend-api/README.md       # API changes
vim backend-api/INTEGRATION.md  # ML pipeline changes
vim README.md                   # General project changes
vim .claude/project_context.md  # Context updates
```

### Documentation Checklist
- [ ] API endpoint changes documented
- [ ] New configuration parameters explained
- [ ] Architecture diagrams updated (if applicable)
- [ ] Examples provided for new features
- [ ] Known limitations documented

---

## 10. Git Workflow

### Feature Development
```bash
# Create feature branch
git checkout -b feature/new-navigation-ui

# Make changes
# ... edit files ...

# Stage changes
git add specific_files.swift  # Prefer specific files over git add .

# Commit with descriptive message
git commit -m "Add route comparison card with risk metrics

Implements collapsible card showing:
- Time and distance for each route
- Risk segment breakdown
- Safety explanation
- Export to Google Maps button

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"

# Push to remote
git push -u origin feature/new-navigation-ui
```

### Code Review
```bash
# Update from main
git checkout main
git pull origin main

# Merge feature branch
git checkout feature/new-navigation-ui
git merge main

# Resolve conflicts if any
# ... fix conflicts ...
git add resolved_files
git commit -m "Merge main into feature branch"

# Push
git push
```

### Creating PR
```bash
# Using GitHub CLI
gh pr create \
  --title "Add route comparison card with risk metrics" \
  --body "## Summary
- Implements collapsible route comparison card
- Shows safety metrics for each route
- Adds export to Google Maps

## Test Plan
- [x] Routes display correctly
- [x] Comparison card shows accurate metrics
- [x] Export to Google Maps works
- [x] No crashes or errors

🤖 Generated with Claude Code"
```

---

## 11. Troubleshooting Common Issues

### Backend won't start
```bash
# Check if port 8000 is in use
lsof -ti:8000

# Kill process if needed
kill -9 $(lsof -ti:8000)

# Check if model files exist
ls -lh outputs/models/toronto_risk_model.joblib
ls -lh outputs/reports/toronto_road_risk.geojson

# If missing, regenerate
python run_risk_analysis.py
```

### iOS app can't connect to backend
```swift
// Check URL in RiskService.swift:19
private let baseURL = "http://127.0.0.1:8000/api"

// For simulator, use localhost
// For physical device, use computer's IP
private let baseURL = "http://192.168.1.100:8000/api"
```

### Routes not displaying
1. Check backend logs for errors
2. Verify API returns data (curl test)
3. Check iOS console for decoding errors
4. Verify coordinates are valid (not NaN)
5. Check map region is correct (Toronto area)

### No alternate routes found
- MapKit may only return 1 route
- Normal behavior for some origin/destination pairs
- App handles this by showing same route twice

---

## 12. Deployment Preparation

### Pre-Production Checklist
- [ ] Remove debug print statements
- [ ] Add authentication to backend
- [ ] Configure HTTPS
- [ ] Set up monitoring/logging
- [ ] Create production config
- [ ] Test with production data
- [ ] Update API base URL in iOS app
- [ ] Create App Store assets
- [ ] Test on physical devices
- [ ] Performance testing under load

### Backend Deployment
```bash
# Use production WSGI server
pip install gunicorn

# Run with multiple workers
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

### iOS Deployment
1. Update version and build number
2. Archive app in Xcode (Product → Archive)
3. Distribute to TestFlight or App Store
4. Configure push notifications (if applicable)

---

## Quick Reference

### Most Common Commands
```bash
# Start backend
cd backend-api && python app.py

# Run ML pipeline
python run_risk_analysis.py

# Test API
curl http://localhost:8000/api/health

# Git commit
git add files && git commit -m "message"

# iOS build and run
open ios-app/RiskMapApp/RiskMapApp.xcodeproj
# Then press ⌘R in Xcode
```
