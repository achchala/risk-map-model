# Testing Guide

This document explains how to run the unit tests for both the Python backend and iOS app.

## Prerequisites

### Python Backend
- Python 3.9+ installed
- Virtual environment activated

### iOS App
- Xcode installed with Command Line Tools
- iOS Simulator available

---

## Python Backend Tests (No Xcode Required)

### 1. Install Dependencies

```bash
pip install -r backend-api/requirements.txt
```

This installs pytest, pytest-cov, and pytest-mock along with other dependencies.

### 2. Run All Tests

```bash
pytest backend-api/tests/ -v
```

### 3. Run Tests with Coverage Report

```bash
pytest backend-api/tests/ -v --cov=backend-api --cov-report=term-missing
```

### 4. Run Specific Test Files

```bash
# Test only endpoints
pytest backend-api/tests/test_endpoints.py -v

# Test only helper functions
pytest backend-api/tests/test_helpers.py -v

# Test only validation
pytest backend-api/tests/test_validation.py -v
```

### Test Coverage

The Python backend tests cover:
- ✅ API endpoints (/api/health, /api/risk-predictions, /api/risk-prediction, etc.)
- ✅ Helper functions (_extract_coordinates, _predict_segment_risk, _convert_to_json_serializable)
- ✅ Input validation (coordinates, risk labels, pagination)
- ✅ Error handling

**Expected Coverage: ~70-75%**

---

## iOS App Tests (Requires Xcode)

### Current Test Status: ✅ 14 Passing | 3 Skipped | 0 Failing

### 1. Run Tests from Terminal

#### Option A: Colorized Output with xcpretty (Recommended)

Install xcpretty (one-time setup):
```bash
gem install xcpretty
```

Run tests with beautiful colored output:
```bash
# Clean, colorized output with test summary
xcodebuild test -scheme RiskMapApp -destination 'platform=iOS Simulator,name=iPhone 17' | xcpretty --test --color

# Extra clean (tests only, no build steps)
xcodebuild test -scheme RiskMapApp -destination 'platform=iOS Simulator,name=iPhone 17' | xcpretty --test --color --simple
```

**What you'll see:**
- ✅ Green checkmarks for passing tests
- ❌ Red X marks for failing tests
- ⏭️ Yellow/gray for skipped tests
- Clean summary at the end
- No build noise

#### Option B: Standard Output (No Installation)

```bash
# Use iPhone 17 simulator (or any available simulator)
xcodebuild test -scheme RiskMapApp -destination 'platform=iOS Simulator,name=iPhone 17'
```

**Note:** The simulator name must match one available on your system. Check available simulators with:
```bash
xcodebuild -showdestinations -scheme RiskMapApp | grep "iPhone"
```

Common available simulators:
- iPhone 17, iPhone 17 Pro, iPhone 17 Pro Max
- iPhone 16e
- iPhone Air

### 2. Run Tests in Xcode

1. Open `ios-app/RiskMapApp/RiskMapApp.xcodeproj` in Xcode
2. Select Product → Test (or press Cmd+U)
3. View test results in the Test Navigator

### 3. Filter Test Output

#### With xcpretty (Recommended):
```bash
# Clean, colored output
xcodebuild test -scheme RiskMapApp -destination 'platform=iOS Simulator,name=iPhone 17' | xcpretty --test --color
```

#### Without xcpretty (grep-based filtering):
```bash
# Highlight test results with grep colors
xcodebuild test -scheme RiskMapApp -destination 'platform=iOS Simulator,name=iPhone 17' 2>&1 | \
  grep --color=always -E "Test Case.*passed|Test Case.*failed|Test Case.*skipped|passed|failed|^\*\* TEST" | tail -30
```

### Test Coverage

The iOS tests cover:
- ✅ RiskLevel enum (display names, colors, system images, Codable, string decoding) - **5 tests**
- ✅ RoadSegment model (JSON decoding with custom CodingKeys, coordinates) - **2 tests**
- ✅ RiskService (initialization, filtering, sorting, empty state) - **4 tests**
- ✅ RiskPredictionResponse decoding (with/without optional fields) - **2 tests**
- ✅ API error types - **1 test**
- ⏭️ Route model (safety explanations, properties) - **3 tests skipped** (MKRoute mocking limitation)

**Total: 17 tests (14 passing, 3 intentionally skipped)**

### Known Limitations

**Route Model Tests Skipped:**
Three tests are intentionally disabled because `MKRoute` from MapKit cannot be mocked in unit tests:
- `testSafetyExplanationAvoidsHighRisk`
- `testSafetyExplanationReducesHighRisk`
- `testRouteProperties`

These tests require integration testing with real MapKit calls or a protocol-based abstraction layer. They are marked with `.disabled()` and won't cause test failures.

---

## Pre-commit Hook

A pre-commit hook automatically runs Python backend tests before each commit.

### How It Works

When you run `git commit`, the hook:
1. Runs all Python backend tests
2. Checks code coverage (minimum 50%)
3. Blocks the commit if tests fail

### Bypass Hook (Not Recommended)

If you need to commit without running tests:

```bash
git commit --no-verify -m "your message"
```

**Note:** iOS tests are NOT included in the pre-commit hook because they're slower (~30 seconds). Run them manually when changing iOS code.

---

## Troubleshooting

### Python Tests

**Issue:** `ModuleNotFoundError: No module named 'pytest'`
```bash
pip install -r backend-api/requirements.txt
```

**Issue:** Tests fail due to missing data files
- Some tests expect preprocessed data or model files
- Tests are designed to gracefully handle missing data (return 500 or skip)

**Issue:** Import errors for `backend-api.app`
- Make sure you're running pytest from the project root directory

### iOS Tests

**Issue:** `xcodebuild: command not found`
```bash
xcode-select --install
```

**Issue:** `Unable to find a destination matching the provided destination specifier`
- Open Xcode → Window → Devices and Simulators
- Check available simulators with: `xcodebuild -showdestinations -scheme RiskMapApp`
- Adjust the destination to match an available simulator (e.g., iPhone 17 instead of iPhone 15)

**Issue:** Route tests are skipped
- Route model tests that require MKRoute mocking are intentionally disabled with `.disabled()` trait
- This is a known limitation of unit testing MapKit - these require integration tests
- Tests will show as "skipped" rather than "failed"

**Issue:** `main actor-isolated initializer cannot be called from outside of the actor`
- This has been fixed by marking `RiskService` with `@MainActor` annotation
- Tests properly use `await` when accessing MainActor-isolated types

---

## Continuous Integration

### GitHub Actions (Optional Setup)

To run tests on every push, create `.github/workflows/test.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  backend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - run: pip install -r backend-api/requirements.txt
      - run: pytest backend-api/tests/ -v --cov=backend-api

  ios-tests:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install xcpretty
        run: gem install xcpretty
      - name: Run tests
        run: |
          xcodebuild test -scheme RiskMapApp -destination 'platform=iOS Simulator,name=iPhone 17' | xcpretty --test --color
```

---

## Summary Commands

**Quick reference for copy-paste:**

```bash
# Python backend tests (no Xcode)
pytest backend-api/tests/ -v --cov=backend-api --cov-report=term-missing

# iOS tests - Colorized (requires xcpretty: gem install xcpretty)
xcodebuild test -scheme RiskMapApp -destination 'platform=iOS Simulator,name=iPhone 17' | xcpretty --test --color

# iOS tests - Standard output
xcodebuild test -scheme RiskMapApp -destination 'platform=iOS Simulator,name=iPhone 17'

# Run both sequentially (from project root) - with colors
pytest backend-api/tests/ -v && xcodebuild test -scheme RiskMapApp -destination 'platform=iOS Simulator,name=iPhone 17' | xcpretty --test --color

# iOS tests with grep-based colored filtering (no installation)
xcodebuild test -scheme RiskMapApp -destination 'platform=iOS Simulator,name=iPhone 17' 2>&1 | \
  grep --color=always -E "Test Case.*passed|Test Case.*failed|Test Case.*skipped|passed|failed|^\*\* TEST"
```
