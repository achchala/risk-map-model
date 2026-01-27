//
//  MockURLProtocol.swift
//  RiskMapAppTests
//
//  Mock URL protocol for intercepting network requests in tests
//

import Foundation

class MockURLProtocol: URLProtocol {
    // Static properties to configure mock responses
    static var requestHandler: ((URLRequest) throws -> (HTTPURLResponse, Data?))?
    static var error: Error?

    override class func canInit(with request: URLRequest) -> Bool {
        return true
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest {
        return request
    }

    override func startLoading() {
        // Check if we should return an error
        if let error = MockURLProtocol.error {
            client?.urlProtocol(self, didFailWithError: error)
            return
        }

        // Check if we have a request handler configured
        guard let handler = MockURLProtocol.requestHandler else {
            client?.urlProtocol(self, didFailWithError: NSError(
                domain: "MockURLProtocol",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "No request handler configured"]
            ))
            return
        }

        do {
            // Call the request handler to get response
            let (response, data) = try handler(request)

            // Send response to client
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)

            // Send data if available
            if let data = data {
                client?.urlProtocol(self, didLoad: data)
            }

            // Mark as finished
            client?.urlProtocolDidFinishLoading(self)
        } catch {
            // Handle any errors from the handler
            client?.urlProtocol(self, didFailWithError: error)
        }
    }

    override func stopLoading() {
        // Required method - nothing to clean up
    }

    // MARK: - Helper Methods

    static func mockResponse(for url: URL, statusCode: Int, data: Data?) -> (HTTPURLResponse, Data?) {
        let response = HTTPURLResponse(
            url: url,
            statusCode: statusCode,
            httpVersion: nil,
            headerFields: ["Content-Type": "application/json"]
        )!
        return (response, data)
    }

    static func reset() {
        requestHandler = nil
        error = nil
    }
}
