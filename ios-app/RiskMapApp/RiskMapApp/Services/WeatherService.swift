//
//  WeatherService.swift
//  RiskMapApp
//
//  Fetches current weather from WeatherAPI.com for real-time risk adjustment.
//  Add your API key at https://www.weatherapi.com/ (free tier available).
//

import Foundation
import CoreLocation

struct WeatherData {
    let condition: WeatherCondition
    let temperature: Double
    let visibility: Double?
    let windSpeed: Double?
    let precipitation: Double?

    enum WeatherCondition: String, Codable {
        case clear = "clear"
        case cloudy = "cloudy"
        case rain = "rain"
        case heavyRain = "heavy_rain"
        case snow = "snow"
        case heavySnow = "heavy_snow"
        case fog = "fog"
        case mist = "mist"
        case thunderstorm = "thunderstorm"
        case sleet = "sleet"

        var displayName: String {
            switch self {
            case .clear: return "Clear"
            case .cloudy: return "Cloudy"
            case .mist: return "Mist"
            case .rain: return "Rain"
            case .heavyRain: return "Heavy Rain"
            case .snow: return "Snow"
            case .heavySnow: return "Heavy Snow"
            case .fog: return "Fog"
            case .thunderstorm: return "Thunderstorm"
            case .sleet: return "Sleet"
            }
        }
    }

    /// Backend expects format: { "condition": "clear" | "rain" | ... }
    func toBackendDict() -> [String: Any] {
        ["condition": condition.rawValue]
    }
}

class WeatherService: ObservableObject {
    @Published var currentWeather: WeatherData?
    @Published var isLoading = false
    @Published var errorMessage: String?

    // API key from feature/weather-hourly-model branch
    private let apiKey: String? = ProcessInfo.processInfo.environment["WEATHER_API_KEY"]
        ?? "ae916e90e524463987013350262701" // fallback for development

    func getWeatherData(for location: CLLocationCoordinate2D) async -> WeatherData {
        if let apiKey = apiKey, !apiKey.isEmpty {
            do {
                let weather = try await fetchWeatherFromAPI(location: location, apiKey: apiKey)
                await MainActor.run {
                    self.currentWeather = weather
                    self.errorMessage = nil
                }
                return weather
            } catch {
                await MainActor.run { self.errorMessage = error.localizedDescription }
            }
        }
        let fallback = estimateWeatherFromTimeAndLocation(location: location)
        await MainActor.run {
            self.currentWeather = fallback
        }
        return fallback
    }

    private func fetchWeatherFromAPI(location: CLLocationCoordinate2D, apiKey: String) async throws -> WeatherData {
        let query = "\(location.latitude),\(location.longitude)"
        let urlString = "https://api.weatherapi.com/v1/current.json?key=\(apiKey)&q=\(query)&aqi=no"
        guard let url = URL(string: urlString) else { throw WeatherError.invalidURL }

        let (data, response) = try await URLSession.shared.data(from: url)
        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            if let err = try? JSONDecoder().decode(WeatherAPIErrorResponse.self, from: data) {
                throw WeatherError.serverErrorWithMessage(err.error.message)
            }
            throw WeatherError.serverError
        }

        let decoded = try JSONDecoder().decode(WeatherAPIResponse.self, from: data)
        return decoded.toWeatherData()
    }

    private func estimateWeatherFromTimeAndLocation(location: CLLocationCoordinate2D) -> WeatherData {
        let now = Date()
        let hour = Calendar.current.component(.hour, from: now)
        let month = Calendar.current.component(.month, from: now)
        let isWinter = month >= 11 || month <= 3
        let isEarlyMorning = (hour >= 6 && hour < 9) || (hour >= 17 && hour < 20)

        let condition: WeatherData.WeatherCondition = isWinter ? .cloudy : (isEarlyMorning ? .cloudy : .clear)
        let temperature: Double = isWinter ? 0 : 15
        return WeatherData(condition: condition, temperature: temperature, visibility: 10, windSpeed: 15, precipitation: 0)
    }
}

// MARK: - WeatherAPI.com Response
private struct WeatherAPIResponse: Codable {
    let current: CurrentWeather
    struct CurrentWeather: Codable {
        let temp_c: Double
        let condition: ConditionInfo
        let vis_km: Double?
        let precip_mm: Double?
        let wind_kph: Double?
    }
    struct ConditionInfo: Codable {
        let text: String
        let code: Int
    }

    func toWeatherData() -> WeatherData {
        let condition = mapCondition(current.condition.text, code: current.condition.code)
        return WeatherData(
            condition: condition,
            temperature: current.temp_c,
            visibility: current.vis_km,
            windSpeed: current.wind_kph,
            precipitation: current.precip_mm
        )
    }

    private func mapCondition(_ text: String, code: Int) -> WeatherData.WeatherCondition {
        if code == 1000 { return .clear }
        if code >= 1003 && code <= 1006 { return .cloudy }
        if (code >= 1063 && code <= 1087) || (code >= 1150 && code <= 1201) {
            return (code >= 1195 && code <= 1201) ? .heavyRain : .rain
        }
        if (code >= 1210 && code <= 1225) || (code >= 1255 && code <= 1264) {
            return (code >= 1219 && code <= 1225) || (code >= 1258 && code <= 1264) ? .heavySnow : .snow
        }
        if code == 1030 || code == 1135 { return .mist }
        if code == 1147 { return .fog }
        if code >= 1273 && code <= 1282 { return .thunderstorm }
        if (code >= 1066 && code <= 1114) || (code >= 1213 && code <= 1216) { return .sleet }

        let t = text.lowercased()
        if t.contains("clear") || t.contains("sunny") { return .clear }
        if t.contains("cloud") { return .cloudy }
        if t.contains("rain") { return t.contains("heavy") ? .heavyRain : .rain }
        if t.contains("snow") { return t.contains("heavy") ? .heavySnow : .snow }
        if t.contains("fog") { return .fog }
        if t.contains("mist") || t.contains("haze") { return .mist }
        if t.contains("thunder") { return .thunderstorm }
        if t.contains("sleet") || t.contains("ice") { return .sleet }
        return .clear
    }
}

private struct WeatherAPIErrorResponse: Codable {
    let error: ErrorInfo
    struct ErrorInfo: Codable {
        let code: Int
        let message: String
    }
}

enum WeatherError: Error, LocalizedError {
    case invalidURL
    case serverError
    case serverErrorWithMessage(String)
    var errorDescription: String? {
        switch self {
        case .invalidURL: return "Invalid weather API URL"
        case .serverError: return "Weather service unavailable"
        case .serverErrorWithMessage(let m): return "Weather: \(m)"
        }
    }
}
