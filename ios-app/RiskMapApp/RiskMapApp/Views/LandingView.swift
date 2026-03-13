//
//  LandingView.swift
//  RiskMapApp
//
//  StreetSmart landing shown on app open from home screen
//

import SwiftUI

struct LandingView: View {
    var onContinue: () -> Void

    var body: some View {
        ZStack {
            LinearGradient(
                colors: [
                    Color(red: 0.11, green: 0.36, blue: 0.42),
                    Color(red: 0.06, green: 0.22, blue: 0.28)
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()

            VStack(spacing: 0) {
                Spacer()

                Image(systemName: "map.fill")
                    .font(.system(size: 56))
                    .foregroundStyle(.white.opacity(0.9))
                    .padding(.bottom, 20)

                Text("StreetSmart")
                    .font(.system(size: 34, weight: .bold, design: .rounded))
                    .foregroundColor(.white)
                    .padding(.bottom, 8)

                Text("See Risk Before It Happens")
                    .font(.system(size: 18, weight: .semibold, design: .rounded))
                    .foregroundColor(.white.opacity(0.95))
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 32)
                    .padding(.bottom, 6)

                Text("AI-Powered Risk-Aware Navigation")
                    .font(.subheadline)
                    .foregroundColor(.white.opacity(0.75))
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 40)
                    .padding(.bottom, 48)

                Button(action: onContinue) {
                    Text("Get Started")
                        .font(.headline)
                        .foregroundColor(Color(red: 0.11, green: 0.36, blue: 0.42))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 16)
                        .background(Color.white)
                        .cornerRadius(14)
                }
                .buttonStyle(.plain)
                .padding(.horizontal, 32)
                .padding(.bottom, 52)
            }
        }
    }
}

struct LandingView_Previews: PreviewProvider {
    static var previews: some View {
        LandingView(onContinue: {})
    }
}
