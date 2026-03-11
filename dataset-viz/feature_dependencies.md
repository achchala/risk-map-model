# Feature Dependencies — Logic Flowchart

```mermaid
flowchart TD
    subgraph RAW["Raw Fields (Source Data)"]
        A[centreline_id]
        B[segment_length]
        C[crash_count]
        D[avg_daily_vol]
        E[avg_weekday_daily_vol]
        F[avg_weekend_daily_vol]
        G[avg_wkdy_am_peak_vol]
        H[avg_wkdy_pm_peak_vol]
        I[avg_speed]
        J[avg_85th_percentile_speed]
        K[avg_95th_percentile_speed]
        L[avg_heavy_pct]
    end

    subgraph DERIVED["Derived Fields (Computed)"]
        M[log_volume]
        N[exposure]
        O[log_exposure]
        P[crash_rate]
    end

    D -->|"log(avg_daily_vol)"| M
    D -->|"avg_daily_vol × segment_length"| N
    B -->|"avg_daily_vol × segment_length"| N
    N -->|"log(exposure)"| O
    C -->|"crash_count / exposure"| P
    N -->|"crash_count / exposure"| P
```

## Column Inventory

| Column | Type | Notes |
|---|---|---|
| `centreline_id` | Raw | Segment identifier |
| `segment_length` | Raw | Metres |
| `crash_count` | Raw | Observed crashes |
| `avg_daily_vol` | Raw | All-day average |
| `avg_weekday_daily_vol` | Raw | Weekday subset |
| `avg_weekend_daily_vol` | Raw | Often missing |
| `avg_wkdy_am_peak_vol` | Raw | AM peak hour |
| `avg_wkdy_pm_peak_vol` | Raw | PM peak hour |
| `avg_speed` | Raw | Mean speed (km/h) |
| `avg_85th_percentile_speed` | Raw | Often missing |
| `avg_95th_percentile_speed` | Raw | Often missing |
| `avg_heavy_pct` | Raw | Heavy vehicle %; frequently missing |
| `log_volume` | Derived | log(avg_daily_vol) |
| `exposure` | Derived | avg_daily_vol × segment_length |
| `log_exposure` | Derived | log(exposure) |
| `crash_rate` | Derived | crash_count / exposure |
