//
//  WeatherService.swift
//  RiskMapApp
//
//  service for fetching current weather data
//

import Foundation
import CoreLocation
import Combine

// WeatherData is defined in RiskModels.swift

class WeatherService: ObservableObject {
    @Published var currentWeather: WeatherData?
    @Published var isLoading = false
    @Published var errorMessage: String?
    
    // Using WeatherAPI.com 
    private let apiKey: String? = "ae916e90e524463987013350262701"
    
    // fallback: Use time-based and location-based heuristics if API unavailable
    func getWeatherData(for location: CLLocationCoordinate2D) async -> WeatherData {
        if let apiKey = apiKey {
            print("[WeatherService] Attempting to fetch weather from API for location: \(location.latitude), \(location.longitude)")
            do {
                let weather = try await fetchWeatherFromAPI(location: location, apiKey: apiKey)
                print("[WeatherService] Successfully fetched weather from API:")
                print("[WeatherService]   Condition: \(weather.condition.displayName)")
                print("[WeatherService]   Temperature: \(weather.temperature)°C")
                print("[WeatherService]   Visibility: \(weather.visibility?.description ?? "N/A") km")
                print("[WeatherService]   Precipitation: \(weather.precipitation?.description ?? "N/A") mm")
                print("[WeatherService]   Wind Speed: \(weather.windSpeed?.description ?? "N/A") km/h")
                await MainActor.run {
                    self.currentWeather = weather
                    self.errorMessage = nil
                }
                return weather
            } catch {
                print("[WeatherService] API error: \(error.localizedDescription)")
                await MainActor.run {
                    self.errorMessage = error.localizedDescription
                }
            }
        } else {
            print("[WeatherService] No API key provided, using fallback estimation")
        }
        
        // fallback: Use time-based and seasonal heuristics
        print("[WeatherService] Using fallback weather estimation")
        let weather = estimateWeatherFromTimeAndLocation(location: location)
        print("[WeatherService] Estimated weather:")
        print("[WeatherService]   Condition: \(weather.condition.displayName)")
        print("[WeatherService]   Temperature: \(weather.temperature)°C")
        await MainActor.run {
            self.currentWeather = weather
        }
        return weather
    }
    
    private func fetchWeatherFromAPI(location: CLLocationCoordinate2D, apiKey: String) async throws -> WeatherData {
        // WeatherAPI.com format: q=lat,lon
        let query = "\(location.latitude),\(location.longitude)"
        let urlString = "https://api.weatherapi.com/v1/current.json?key=\(apiKey)&q=\(query)&aqi=no"
        
        print("[WeatherService] API URL: \(urlString.replacingOccurrences(of: apiKey, with: "***"))")
        
        guard let url = URL(string: urlString) else {
            print("[WeatherService] Invalid URL")
            throw WeatherError.invalidURL
        }
        
        print("[WeatherService] Sending API request...")
        let (data, response) = try await URLSession.shared.data(from: url)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            print("[WeatherService] Invalid HTTP response")
            throw WeatherError.serverError
        }
        
        print("[WeatherService] Response status code: \(httpResponse.statusCode)")
        
        guard httpResponse.statusCode == 200 else {
            // Try to decode error message if available
            if let errorResponse = try? JSONDecoder().decode(WeatherAPIErrorResponse.self, from: data) {
                print("[WeatherService] API Error - Code: \(errorResponse.error.code), Message: \(errorResponse.error.message)")
                throw WeatherError.serverErrorWithMessage(errorResponse.error.message)
            }
            
            // Log raw response for debugging
            if let responseString = String(data: data, encoding: .utf8) {
                print("[WeatherService] Error response body: \(responseString)")
            }
            throw WeatherError.serverError
        }
        
        // Log raw response for debugging (first 500 chars)
        if let responseString = String(data: data, encoding: .utf8) {
            let preview = String(responseString.prefix(500))
            print("[WeatherService] Response preview: \(preview)...")
        }
        
        let weatherResponse = try JSONDecoder().decode(WeatherAPIResponse.self, from: data)
        print("[WeatherService] Successfully decoded API response")
        print("[WeatherService]   Location: \(weatherResponse.location.name), \(weatherResponse.location.country)")
        print("[WeatherService]   Raw condition text: \(weatherResponse.current.condition.text)")
        print("[WeatherService]   Condition code: \(weatherResponse.current.condition.code)")
        
        return weatherResponse.toWeatherData()
    }
    
    private func estimateWeatherFromTimeAndLocation(location: CLLocationCoordinate2D) -> WeatherData {
        let calendar = Calendar.current
        let now = Date()
        let hour = calendar.component(.hour, from: now)
        let month = calendar.component(.month, from: now)
        
        // toronto area (latitude ~43.65)
        let isWinter = month >= 11 || month <= 3
        let isSpring = month >= 4 && month <= 5
        let isSummer = month >= 6 && month <= 8
        let isFall = month >= 9 && month <= 10
        
        // time-based conditions
        let isNight = hour >= 20 || hour < 6
        let isEarlyMorning = hour >= 6 && hour < 9
        let isEvening = hour >= 17 && hour < 20
        
        // estimate weather based on season and time
        var condition: WeatherData.WeatherCondition = .clear
        var precipitation: Double? = nil
        
        if isWinter {
            // winter: higher chance of snow, especially in early morning/evening
            if isEarlyMorning || isEvening {
                condition = Double.random(in: 0...1) < 0.3 ? .snow : .cloudy
                if condition == .snow {
                    precipitation = Double.random(in: 0.5...3.0)
                }
            } else if isNight {
                condition = Double.random(in: 0...1) < 0.2 ? .snow : .clear
            } else {
                condition = Double.random(in: 0...1) < 0.15 ? .snow : .cloudy
            }
        } else if isSpring || isFall {
            // spring/Fall: chance of rain
            if isEarlyMorning || isEvening {
                condition = Double.random(in: 0...1) < 0.25 ? .rain : .cloudy
                if condition == .rain {
                    precipitation = Double.random(in: 0.5...5.0)
                }
            } else {
                condition = Double.random(in: 0...1) < 0.15 ? .rain : .clear
            }
        } else {
            // summer: mostly clear, occasional rain
            if isEarlyMorning || isEvening {
                condition = Double.random(in: 0...1) < 0.2 ? .rain : .clear
                if condition == .rain {
                    precipitation = Double.random(in: 0.5...10.0)
                }
            } else {
                condition = .clear
            }
        }
        
        // add fog/mist in early morning
        if isEarlyMorning && Double.random(in: 0...1) < 0.15 {
            condition = .mist
        }
        
        // estimate temperature based on season
        let temperature: Double
        if isWinter {
            temperature = Double.random(in: -10...5)
        } else if isSpring || isFall {
            temperature = Double.random(in: 5...20)
        } else {
            temperature = Double.random(in: 20...30)
        }
        
        // Estimate visibility (lower in fog, rain, snow)
        let visibility: Double?
        switch condition {
        case .fog, .mist:
            visibility = Double.random(in: 0.5...2.0)
        case .heavyRain, .heavySnow:
            visibility = Double.random(in: 1.0...3.0)
        case .rain, .snow:
            visibility = Double.random(in: 3.0...8.0)
        default:
            visibility = Double.random(in: 10.0...20.0)
        }
        
        return WeatherData(
            condition: condition,
            temperature: temperature,
            visibility: visibility,
            windSpeed: Double.random(in: 5...25),
            precipitation: precipitation
        )
    }
    
    // get time-based risk multiplier
    func getTimeOfDayRiskMultiplier() -> Double {
        let calendar = Calendar.current
        let now = Date()
        let hour = calendar.component(.hour, from: now)
        let weekday = calendar.component(.weekday, from: now)
        let isWeekend = weekday == 1 || weekday == 7
        
        // rush hour periods (higher risk)
        let isMorningRush = hour >= 7 && hour <= 9
        let isEveningRush = hour >= 17 && hour <= 19
        
        // night time (higher risk due to visibility)
        let isNight = hour >= 22 || hour < 6
        let isLateNight = hour >= 23 || hour < 5
        
        if isLateNight {
            return 1.4 // highest risk - very late night/early morning
        } else if isNight {
            return 1.25 // night time risk
        } else if isMorningRush || isEveningRush {
            return isWeekend ? 1.1 : 1.3 // rush hour, higher on weekdays
        } else if hour >= 9 && hour < 17 {
            return 1.0 // daytime, normal risk
        } else {
            return 1.1 // other times
        }
    }
}

// MARK: - WeatherAPI.com Response Models
private struct WeatherAPIResponse: Codable {
    let current: CurrentWeather
    let location: LocationInfo
    
    struct CurrentWeather: Codable {
        let temp_c: Double
        let condition: ConditionInfo
        let vis_km: Double?
        let precip_mm: Double?
        let wind_kph: Double?
        let wind_mph: Double?
    }
    
    struct ConditionInfo: Codable {
        let text: String
        let code: Int
    }
    
    struct LocationInfo: Codable {
        let name: String
        let region: String?
        let country: String
    }
    
    func toWeatherData() -> WeatherData {
        let conditionText = current.condition.text.lowercased()
        let condition = mapWeatherCondition(conditionText, code: current.condition.code)
        
        return WeatherData(
            condition: condition,
            temperature: current.temp_c,
            visibility: current.vis_km,
            windSpeed: current.wind_kph ?? (current.wind_mph.map { $0 * 1.60934 }), // Convert mph to km/h if needed
            precipitation: current.precip_mm
        )
    }
    
    private func mapWeatherCondition(_ text: String, code: Int) -> WeatherData.WeatherCondition {
        // use condition code for more accurate mapping
        // WeatherAPI.com condition codes: https://www.weatherapi.com/docs/weather_conditions.json
        if code == 1000 {
            return .clear
        } else if code >= 1003 && code <= 1006 {
            return .cloudy
        } else if code >= 1063 && code <= 1087 || code >= 1150 && code <= 1201 {
            // rain codes (1063-1087, 1150-1201)
            if code >= 1195 && code <= 1201 {
                return .heavyRain
            }
            return .rain
        } else if code >= 1210 && code <= 1225 || code >= 1255 && code <= 1264 {
            // snow codes (1210-1225, 1255-1264)
            if code >= 1219 && code <= 1225 || code >= 1258 && code <= 1264 {
                return .heavySnow
            }
            return .snow
        } else if code == 1030 || code == 1135 {
            return .mist
        } else if code == 1147 {
            return .fog
        } else if code >= 1273 && code <= 1282 {
            return .thunderstorm
        } else if code >= 1066 && code <= 1114 || code >= 1213 && code <= 1216 || code >= 1237 && code <= 1249 || code >= 1261 && code <= 1271 {
            // sleet codes
            return .sleet
        }
        
        // fallback to text-based mapping
        if text.contains("clear") || text.contains("sunny") {
            return .clear
        } else if text.contains("cloud") {
            return .cloudy
        } else if text.contains("rain") {
            if text.contains("heavy") || text.contains("torrential") {
                return .heavyRain
            }
            return .rain
        } else if text.contains("snow") {
            if text.contains("heavy") || text.contains("blizzard") {
                return .heavySnow
            }
            return .snow
        } else if text.contains("fog") {
            return .fog
        } else if text.contains("mist") || text.contains("haze") {
            return .mist
        } else if text.contains("thunder") {
            return .thunderstorm
        } else if text.contains("sleet") || text.contains("ice") {
            return .sleet
        }
        
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
    case decodingError
    
    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "Invalid weather API URL"
        case .serverError:
            return "Weather service unavailable"
        case .serverErrorWithMessage(let message):
            return "Weather service error: \(message)"
        case .decodingError:
            return "Failed to parse weather data"
        }
    }
}
