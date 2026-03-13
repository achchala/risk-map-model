//
//  AddressSearchField.swift
//  RiskMapApp
//
//  Address autocomplete using MKLocalSearchCompleter for routing
//

import SwiftUI
import MapKit

struct AddressSearchField: View {
    let placeholder: String
    let iconName: String
    let iconColor: Color
    @Binding var text: String
    var onCoordinateSelected: (CLLocationCoordinate2D) -> Void

    @StateObject private var completer = AddressSearchCompleter()

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 12) {
                Image(systemName: iconName).foregroundColor(iconColor)
                TextField(placeholder, text: $text)
                    .textFieldStyle(.plain)
                    .onChange(of: text) { _, newValue in
                        completer.queryFragment = newValue
                    }
            }
            .padding()

            if !completer.results.isEmpty && !text.isEmpty {
                Divider().padding(.leading, 44)
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 0) {
                        ForEach(Array(completer.results.enumerated()), id: \.offset) { _, completion in
                            Button {
                                selectCompletion(completion)
                            } label: {
                                HStack(spacing: 8) {
                                    Image(systemName: "mappin.circle")
                                        .foregroundColor(.secondary)
                                        .font(.caption)
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text(completion.title)
                                            .font(.subheadline)
                                            .foregroundColor(.primary)
                                        if !completion.subtitle.isEmpty {
                                            Text(completion.subtitle)
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
                        }
                    }
                }
                .frame(maxHeight: 160)
            }
        }
    }

    private func selectCompletion(_ completion: MKLocalSearchCompletion) {
        text = completion.title
        if !completion.subtitle.isEmpty {
            text += ", \(completion.subtitle)"
        }
        completer.queryFragment = ""
        completer.results = []

        let request = MKLocalSearch.Request(completion: completion)
        request.region = MKCoordinateRegion(
            center: CLLocationCoordinate2D(latitude: 43.6532, longitude: -79.3832),
            span: MKCoordinateSpan(latitudeDelta: 0.2, longitudeDelta: 0.2)
        )
        let search = MKLocalSearch(request: request)
        search.start { response, _ in
            guard let item = response?.mapItems.first,
                  let coord = item.placemark.location?.coordinate else { return }
            DispatchQueue.main.async {
                onCoordinateSelected(coord)
            }
        }
    }
}

final class AddressSearchCompleter: NSObject, ObservableObject {
    @Published var queryFragment: String = "" {
        didSet {
            completer.queryFragment = queryFragment
        }
    }
    @Published var results: [MKLocalSearchCompletion] = []

    private let completer = MKLocalSearchCompleter()

    override init() {
        super.init()
        completer.delegate = self
        completer.resultTypes = .address
        // Bias suggestions to Toronto/GTA
        completer.region = MKCoordinateRegion(
            center: CLLocationCoordinate2D(latitude: 43.6532, longitude: -79.3832),
            span: MKCoordinateSpan(latitudeDelta: 0.25, longitudeDelta: 0.25)
        )
        if #available(iOS 18.0, *) {
            completer.regionPriority = .required
        }
    }
}

extension AddressSearchCompleter: MKLocalSearchCompleterDelegate {
    func completerDidUpdateResults(_ completer: MKLocalSearchCompleter) {
        DispatchQueue.main.async {
            self.results = completer.results
        }
    }

    func completer(_ completer: MKLocalSearchCompleter, didFailWithError error: Error) {
        // Ignore
    }
}
