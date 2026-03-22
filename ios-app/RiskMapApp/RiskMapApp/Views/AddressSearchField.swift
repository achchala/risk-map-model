//
//  AddressSearchField.swift
//  RiskMapApp
//
//  Address autocomplete using MKLocalSearchCompleter (matches feature/weather-hourly-model)
//

import SwiftUI
import MapKit

struct AddressSearchField: View {
    let placeholder: String
    let iconName: String
    let iconColor: Color
    @Binding var text: String
    var onCoordinateSelected: (CLLocationCoordinate2D) -> Void

    @StateObject private var completer = LocalSearchCompleter()
    @FocusState private var isFocused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 12) {
                Image(systemName: iconName)
                    .foregroundColor(iconColor)
                    .font(.title3)

                TextField(placeholder, text: $text)
                    .textFieldStyle(.plain)
                    .focused($isFocused)
                    .onChange(of: text) { _, newValue in
                        completer.search(query: newValue)
                    }
                    .autocorrectionDisabled()
            }
            .padding()
            .background(Color(UIColor.systemBackground))

            // Suggestions dropdown — show when we have results (not tied to focus so tap registers before dropdown hides)
            if !completer.suggestions.isEmpty {
                VStack(alignment: .leading, spacing: 0) {
                    ForEach(Array(completer.suggestions.enumerated()), id: \.offset) { index, suggestion in
                        Button {
                            selectSuggestion(suggestion)
                        } label: {
                            HStack(spacing: 12) {
                                Image(systemName: "mappin.circle")
                                    .foregroundColor(.brandTertiary)
                                    .font(.body)
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(suggestion.title)
                                        .font(.subheadline)
                                        .foregroundColor(.primary)
                                    if !suggestion.subtitle.isEmpty {
                                        Text(suggestion.subtitle)
                                            .font(.caption)
                                            .foregroundColor(.secondary)
                                    }
                                }
                                Spacer()
                            }
                            .padding(.horizontal, 16)
                            .padding(.vertical, 10)
                        }
                        .buttonStyle(.plain)

                        if index < completer.suggestions.count - 1 {
                            Divider()
                                .padding(.leading, 40)
                        }
                    }
                }
                .background(Color(UIColor.secondarySystemBackground))
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(Color.brandTertiary.opacity(0.18), lineWidth: 1)
                )
                .cornerRadius(8)
                .padding(.horizontal, 4)
                .padding(.top, 4)
            }
        }
    }

    private func selectSuggestion(_ suggestion: MKLocalSearchCompletion) {
        completer.suggestions = []
        let request = MKLocalSearch.Request(completion: suggestion)
        let search = MKLocalSearch(request: request)

        search.start { response, error in
            guard let response = response, let item = response.mapItems.first else { return }
            let coord = item.placemark.coordinate
            let title = item.name ?? suggestion.title

            DispatchQueue.main.async {
                text = title
                onCoordinateSelected(coord)
            }
        }
    }
}

/// Observable object that manages MKLocalSearchCompleter for SwiftUI (matches feature branch)
final class LocalSearchCompleter: NSObject, ObservableObject {
    @Published var suggestions: [MKLocalSearchCompletion] = []

    private let completer = MKLocalSearchCompleter()

    // Toronto region bias for better local results (same as feature branch)
    private let torontoRegion = MKCoordinateRegion(
        center: CLLocationCoordinate2D(latitude: 43.6532, longitude: -79.3832),
        span: MKCoordinateSpan(latitudeDelta: 0.3, longitudeDelta: 0.3)
    )

    override init() {
        super.init()
        completer.delegate = self
        completer.region = torontoRegion
        completer.resultTypes = [.address, .pointOfInterest]
    }

    func search(query: String) {
        guard query.count >= 2 else {
            suggestions = []
            return
        }
        completer.queryFragment = query
    }
}

extension LocalSearchCompleter: MKLocalSearchCompleterDelegate {
    func completerDidUpdateResults(_ completer: MKLocalSearchCompleter) {
        DispatchQueue.main.async {
            self.suggestions = Array(completer.results.prefix(5))
        }
    }

    func completer(_ completer: MKLocalSearchCompleter, didFailWithError error: Error) {
        DispatchQueue.main.async {
            self.suggestions = []
        }
    }
}
