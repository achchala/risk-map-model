# quick start to run in Xcode!!

## 1: open the project in xcode

```bash
cd ios-app/RiskMapApp
open RiskMapApp.xcodeproj
```

or:
1. Open Xcode
2. File → Open
3. Navigate to `ios-app/RiskMapApp/RiskMapApp.xcodeproj`
4. Click Open

## step 2: configure your development team

1. In Xcode, select the **RiskMapApp** project in the navigator (top item)
2. Select the **RiskMapApp** target
3. Go to **Signing & Capabilities** tab
4. Under **Team**, select your Apple Developer account
   - If you don't have one, you can use "Personal Team" (free, but requires Apple ID)
5. Xcode will automatically manage provisioning

## step 3: update API endpoint 

Before running, you need to set the backend API URL:

1. Open `RiskMapApp/Services/RiskService.swift`
2. Find line ~18: `private let baseURL = "https://your-api-domain.com/api"`
3. Update it to your backend URL:

**for local development (iOS simulator):**
```swift
private let baseURL = "http://localhost:8000/api"
```

**for production:** NOT READY YET
```swift
private let baseURL = "https://your-api-domain.com/api"
```

## Step 4: Start the Backend API

**In a separate terminal window:**

```bash
cd backend-api
pip install -r requirements.txt
python app.py
```

You should see:
```
INFO: Model loaded successfully
INFO: Loaded X pre-processed road segments
 * Running on http://0.0.0.0:8000
```

**Keep this terminal running** - the API needs to be running for the iOS app to work.

## Step 5: Build and Run in Xcode

1. **Select a simulator or device:**
   - Click the device selector at the top (next to the play button)
   - Choose iPhone 15 Pro (or any iOS 17+ device)

2. **Build and run:**
   - Press `⌘ + R` (Command + R)
   - Or click the Play button ▶️

3. **First run:**
   - Xcode will build the app (may take 1-2 minutes first time)
   - The simulator will launch
   - The app will open

## Step 6: Test the App

1. **Grant location permissions** when prompted
2. **Check the Map tab** - you should see Toronto map
3. **Check the High Risk tab** - should show high-risk roads
4. **Tap a segment** on the map to see details

## Troubleshooting


**"Network error" or "No data"**
- Check that backend API is running (`python app.py`)
- Check the API URL in `RiskService.swift`
- For physical device, make sure Mac and iPhone are on same WiFi
- Check backend logs for errors

**Map not loading**
- Grant location permissions in Settings
- Check that backend is returning data (test `/api/health` endpoint)

**App crashes on launch**
- Check Xcode console for error messages
- Make sure all required files are in the project

### Testing the Backend API

Before running the iOS app, test the backend:

```bash
# Test health endpoint
curl http://localhost:8000/api/health

# Test predictions endpoint
curl -X POST http://localhost:8000/api/risk-predictions \
  -H "Content-Type: application/json" \
  -d '{"north": 43.7, "south": 43.6, "east": -79.3, "west": -79.4}'
```

## Project Structure in Xcode

```
RiskMapApp (Project)
└── RiskMapApp (Target)
    ├── RiskMapApp.swift          # App entry
    ├── ContentView.swift         # Main tabs
    ├── Models/
    │   └── RiskModels.swift      # Data models
    ├── Services/
    │   └── RiskService.swift     # API service
    └── Views/
        ├── MapView.swift         # Map screen
        ├── RiskDetailView.swift  # Detail screen
        ├── RiskListView.swift    # List screen
        └── SettingsView.swift    # Settings screen
```
