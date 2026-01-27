# RiskMapApp Testing Documentation

Comprehensive testing guide for the RiskMap iOS application.

## Table of Contents

- [Overview](#overview)
- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Code Linting](#code-linting)
- [Code Coverage](#code-coverage)
- [Test Categories](#test-categories)
- [Mock Objects](#mock-objects)
- [Writing New Tests](#writing-new-tests)
- [CI/CD Pipeline](#cicd-pipeline)
- [Troubleshooting](#troubleshooting)

## Overview

This project uses **XCTest** for unit testing with a focus on:

- **Unit Tests**: Testing models, services, and business logic in isolation
- **Mock Infrastructure**: Simulating network calls and dependencies
- **Code Coverage**: Aiming for 80%+ coverage on critical components
- **Continuous Integration**: Automated testing on every push/PR

### Test Statistics

- **Total Test Files**: 3
- **Total Test Cases**: 75+
- **Mock Classes**: 3
- **Test Helpers**: TestData utility
- **Coverage Goal**: 80%+ overall

## Test Structure

```
RiskMapAppTests/
├── Mocks/
│   ├── TestData.swift              # Test data helpers
│   ├── MockURLProtocol.swift       # HTTP response mocking
│   ├── MockRiskService.swift       # Mock risk API service
│   └── MockRouteService.swift      # Mock route calculation
├── ModelTests/
│   └── RiskModelsTests.swift       # Model unit tests (30+ tests)
└── ServiceTests/
    ├── RiskServiceTests.swift      # RiskService tests (25+ tests)
    └── RouteServiceTests.swift     # RouteService tests (20+ tests)
```

## Running Tests

### In Xcode

1. **Run All Tests**
   - Press `⌘U` or select `Product > Test`

2. **Run Specific Test Suite**
   - Click the ▶️ icon next to the test class/method in the test navigator
   - Or use `^⌥⌘U` to run tests from the current file

3. **Run Single Test**
   - Click the ▶️ icon next to the test method
   - Or place cursor in test method and press `^⌥⌘U`

4. **Debug Test**
   - Click the breakpoint, then click the test's ▶️ icon
   - Or use `^⌥⌘U` while in debug mode

### Command Line

```bash
# Navigate to project directory
cd ios-app/RiskMapApp

# Run all tests
xcodebuild test \
  -scheme RiskMapApp \
  -destination 'platform=iOS Simulator,name=iPhone 15 Pro,OS=17.2' \
  -only-testing:RiskMapAppTests

# Run specific test suite
xcodebuild test \
  -scheme RiskMapApp \
  -destination 'platform=iOS Simulator,name=iPhone 15 Pro,OS=17.2' \
  -only-testing:RiskMapAppTests/RiskModelsTests

# Run with code coverage
xcodebuild test \
  -scheme RiskMapApp \
  -destination 'platform=iOS Simulator,name=iPhone 15 Pro,OS=17.2' \
  -enableCodeCoverage YES

# Pretty output with xcpretty (install with: gem install xcpretty)
xcodebuild test \
  -scheme RiskMapApp \
  -destination 'platform=iOS Simulator,name=iPhone 15 Pro,OS=17.2' \
  | xcpretty --color --test
```

## Code Linting

We use **SwiftLint** to enforce code style and catch common issues.

### Installation

```bash
# Install via Homebrew
brew install swiftlint

# Or use Mint
mint install realm/SwiftLint
```

### Running SwiftLint

```bash
# Lint all files
cd ios-app/RiskMapApp
swiftlint

# Auto-fix issues
swiftlint --fix

# Strict mode (fail on warnings)
swiftlint lint --strict

# Lint specific files
swiftlint lint RiskMapApp/Models/RiskModels.swift
```

### Configuration

SwiftLint rules are configured in [.swiftlint.yml](ios-app/RiskMapApp/.swiftlint.yml):

- **Line length**: 150 warning, 200 error
- **File length**: 500 warning, 1000 error
- **Function body**: 80 warning, 150 error (relaxed for SwiftUI)
- **Custom rules**: No print statements, private @State, etc.

## Code Coverage

### Viewing Coverage in Xcode

1. Run tests with coverage enabled (`⌘U`)
2. Open Report Navigator (`⌘9`)
3. Select latest test run
4. Click **Coverage** tab
5. Drill down into specific files/functions

### Command Line Coverage

```bash
# Generate coverage report
xcodebuild test \
  -scheme RiskMapApp \
  -destination 'platform=iOS Simulator,name=iPhone 15 Pro' \
  -enableCodeCoverage YES \
  -derivedDataPath DerivedData

# View coverage summary
xcrun xccov view --report DerivedData/Logs/Test/*.xcresult

# Export as JSON
xcrun xccov view --report --json DerivedData/Logs/Test/*.xcresult > coverage.json
```

### Coverage Goals

| Component | Current | Goal |
|-----------|---------|------|
| RiskModels | ~95% | 95%+ |
| RiskService | ~80% | 80%+ |
| RouteService | ~75% | 80%+ |
| **Overall** | **~80%** | **80%+** |

## Test Categories

### Model Tests (RiskModelsTests.swift)

**30+ tests** covering:

- **RiskLevel Enum** (5 tests)
  - Display names, colors, icons
  - Raw values and Codable conformance

- **RoadSegment Model** (7 tests)
  - Initialization with various parameters
  - JSON decoding with custom keys
  - Coordinate handling

- **Route Model** (10 tests)
  - Safety explanations
  - Risk scoring calculations
  - Route comparison logic

- **RouteComparison** (5 tests)
  - Time differences
  - Safety improvements
  - Detailed explanations

- **APIError** (3 tests)
  - Error descriptions for all cases

### Service Tests (RiskServiceTests.swift)

**25+ tests** covering:

- **API Requests** (6 tests)
  - Success responses
  - Network errors
  - Server errors
  - Decoding errors
  - Empty responses
  - Timeouts

- **State Management** (4 tests)
  - Loading state transitions
  - Error message handling
  - Published properties
  - Main actor dispatch

- **Region Handling** (4 tests)
  - Small regions
  - Large regions
  - Request body formatting

- **High-Risk Roads** (4 tests)
  - Filtering by risk level
  - Sorting by crash count

### Route Service Tests (RouteServiceTests.swift)

**20+ tests** covering:

- **Route Calculation** (4 tests)
  - Success scenarios
  - Failure handling
  - Loading states
  - Location tracking

- **Route Analysis** (4 tests)
  - High-risk segment detection
  - Low-risk segment preference
  - Mixed risk scenarios
  - Risk score calculation

- **Route Comparison** (4 tests)
  - Safer vs optimal routes
  - Time deltas
  - Risk avoidance

- **Extensions** (3 tests)
  - MKPolyline coordinate extraction
  - MKRoute detailed coordinates
  - Invalid coordinate filtering

- **Error Handling** (3 tests)
  - RouteError descriptions
  - Error propagation

- **Mock Services** (2 tests)
  - Mock tracking
  - Error simulation

## Mock Objects

### TestData

Helper methods for creating test data:

```swift
// Road segments
let segment = TestData.sampleRoadSegment()
let highRisk = TestData.highRiskSegment()
let lowRisk = TestData.lowRiskSegment()

// Locations
let coord = TestData.torontoCoordinate()
let region = TestData.torontoRegion()

// API responses
let apiData = TestData.sampleAPIResponseData()
```

### MockURLProtocol

Mock HTTP responses for network testing:

```swift
// Success response
MockURLProtocol.requestHandler = { request in
    let response = HTTPURLResponse(url: request.url!, statusCode: 200, ...)
    return (response, mockData)
}

// Error response
MockURLProtocol.error = NSError(domain: ..., code: ...)

// Reset
MockURLProtocol.reset()
```

### MockRiskService

Mock risk service for testing:

```swift
let mockService = MockRiskService()

// Configure response
mockService.setMockSegments([segment1, segment2])

// Simulate error
mockService.simulateError(APIError.networkError(...))

// Verify calls
XCTAssertEqual(mockService.fetchRiskPredictionsCallCount, 1)
XCTAssertEqual(mockService.lastRegion?.center.latitude, 43.65, accuracy: 0.01)
```

### MockRouteService

Mock route service for testing:

```swift
let mockService = MockRouteService(riskService: mockRiskService)

// Configure routes
mockService.setMockRoutes(safer: saferRoute, optimal: optimalRoute)

// Simulate error
mockService.simulateError(RouteError.noRouteFound)

// Verify calls
XCTAssertEqual(mockService.calculateRoutesCallCount, 1)
```

## Writing New Tests

### Test Structure

```swift
import XCTest
@testable import RiskMapApp

final class MyFeatureTests: XCTestCase {
    var sut: MyFeature!  // System Under Test

    override func setUpWithError() throws {
        try super.setUpWithError()
        sut = MyFeature()
        // Setup mocks, dependencies
    }

    override func tearDownWithError() throws {
        sut = nil
        // Clean up mocks
        try super.tearDownWithError()
    }

    func testFeatureBehavior() {
        // Given (Arrange)
        let input = "test"

        // When (Act)
        let result = sut.process(input)

        // Then (Assert)
        XCTAssertEqual(result, "expected")
    }

    func testAsyncFeature() async throws {
        // Given
        let input = TestData.sampleInput()

        // When
        let result = try await sut.asyncProcess(input)

        // Then
        XCTAssertNotNil(result)
    }
}
```

### Best Practices

1. **Test One Thing**: Each test should verify a single behavior
2. **Use Descriptive Names**: `testFetchRiskPredictionsSuccess` not `testFetch`
3. **Arrange-Act-Assert**: Structure tests with clear Given/When/Then
4. **Use Mocks**: Don't hit real APIs or databases
5. **Test Edge Cases**: Empty arrays, nil values, errors
6. **Async/Await**: Use `async throws` for async tests
7. **Cleanup**: Reset mocks in `tearDown`

### Common Assertions

```swift
// Equality
XCTAssertEqual(actual, expected)
XCTAssertNotEqual(actual, notExpected)

// Nil checks
XCTAssertNil(value)
XCTAssertNotNil(value)

// Boolean
XCTAssertTrue(condition)
XCTAssertFalse(condition)

// Numeric
XCTAssertGreaterThan(a, b)
XCTAssertLessThan(a, b)
XCTAssertEqual(a, b, accuracy: 0.01)

// Collections
XCTAssertEqual(array.count, 3)
XCTAssertTrue(array.contains(element))

// Errors
XCTAssertThrowsError(try function())
XCTAssertNoThrow(try function())

// Async
let value = try await asyncFunction()
XCTAssertEqual(value, expected)
```

## CI/CD Pipeline

Tests run automatically on GitHub Actions for every push and pull request.

### Workflow Triggers

- Push to `main`, `develop`, `adriel-routes-ui`, `unit-tests`
- Pull requests to `main`, `develop`
- Only when iOS code changes (`ios-app/**`)

### Pipeline Steps

1. **Checkout code**
2. **Select Xcode 15.2**
3. **Install SwiftLint**
4. **Run SwiftLint** (strict mode, fails on warnings)
5. **Build and Test** (iPhone 15 Pro simulator)
6. **Generate Coverage Report**
7. **Check Coverage Threshold** (70% minimum)
8. **Upload Artifacts** (test results, coverage report)

### Viewing Results

- **GitHub Actions Tab**: See test runs and logs
- **Pull Request Checks**: Status badges show pass/fail
- **Artifacts**: Download test results and coverage reports

### Local CI Simulation

```bash
# Run the same checks as CI
cd ios-app/RiskMapApp

# 1. Lint
swiftlint lint --strict

# 2. Build and test
xcodebuild test \
  -scheme RiskMapApp \
  -destination 'platform=iOS Simulator,name=iPhone 15 Pro' \
  -enableCodeCoverage YES

# 3. Check coverage
xcrun xccov view --report DerivedData/Logs/Test/*.xcresult
```

## Troubleshooting

### Common Issues

#### Tests Don't Run

**Problem**: "No tests found" or tests are grayed out

**Solution**:
1. Ensure test files are part of the test target
2. Check that test class inherits from `XCTestCase`
3. Clean build folder (`⌘⇧K`)
4. Reset simulator (`Device > Erase All Content and Settings`)

#### Mock Not Working

**Problem**: Real network calls instead of mock responses

**Solution**:
1. Ensure `MockURLProtocol.reset()` is called in `tearDown`
2. Configure `requestHandler` or `error` before making requests
3. Check URLSession is using mock protocol class

#### Async Tests Failing

**Problem**: Async tests timeout or fail unexpectedly

**Solution**:
1. Add `async throws` to test method signature
2. Use `await` for async calls
3. Increase timeout if needed
4. Check for race conditions in concurrent code

#### Code Coverage Not Showing

**Problem**: Coverage tab is empty

**Solution**:
1. Enable code coverage in scheme settings:
   - Edit Scheme > Test > Options > Code Coverage
2. Run tests again (`⌘U`)
3. Check Report Navigator (`⌘9`)

#### SwiftLint Errors

**Problem**: SwiftLint fails with many warnings

**Solution**:
1. Run auto-fix: `swiftlint --fix`
2. Check `.swiftlint.yml` for rule configuration
3. Disable specific rules if needed (add to `disabled_rules`)

### Getting Help

- **Xcode Console**: Check for error messages
- **Test Logs**: View detailed test output
- **Coverage Reports**: Identify untested code
- **GitHub Issues**: Report bugs or ask questions

## Maintenance

### Regular Tasks

- **Weekly**: Review and update coverage reports
- **Monthly**: Update dependencies and Xcode version
- **Per Feature**: Write tests for new code (aim for 80%+ coverage)
- **Per Bug Fix**: Add regression test

### Updating Tests

When modifying production code:

1. Update relevant tests
2. Add new tests for new functionality
3. Ensure all tests pass
4. Verify coverage hasn't decreased
5. Update documentation if needed

### Adding New Test Suites

```bash
# 1. Create test file in appropriate directory
# 2. Add to test target in Xcode
# 3. Import @testable import RiskMapApp
# 4. Follow existing test patterns
# 5. Update this documentation
```

---

## Quick Reference

### Keyboard Shortcuts (Xcode)

- `⌘U` - Run all tests
- `^⌥⌘U` - Run tests in current file
- `⌘9` - Show Report Navigator
- `⌘⇧K` - Clean Build Folder
- `⌘B` - Build
- `⌘R` - Run app

### Command Cheat Sheet

```bash
# Run tests
xcodebuild test -scheme RiskMapApp -destination 'platform=iOS Simulator,name=iPhone 15 Pro'

# Lint code
swiftlint

# View coverage
xcrun xccov view --report DerivedData/Logs/Test/*.xcresult

# Clean
rm -rf ~/Library/Developer/Xcode/DerivedData
```

---

**Last Updated**: January 26, 2025
**Maintainer**: RiskMap Team
**Version**: 1.0
