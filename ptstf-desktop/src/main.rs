#[cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use ptstf_core::flow::TriadicFlowCurve;
use nalgebra::Vector2;
use serde::Serialize;

#[derive(Serialize)]
struct RenderCurveResponse {
    bending_energy: f64,
    crosses: bool,
    points: Vec<(f64, f64)>,
}

#[tauri::command]
fn analyze_curve(p0: (f64, f64), p1: (f64, f64), p2: (f64, f64), a: f64, b: f64) -> RenderCurveResponse {
    let curve = TriadicFlowCurve::new(
        Vector2::new(p0.0, p0.1),
        Vector2::new(p1.0, p1.1),
        Vector2::new(p2.0, p2.1),
    );

    let mut points = Vec::new();
    for i in 0..=50 {
        let s = i as f64 / 50.0;
        let pt = curve.evaluate(s);
        points.push((pt.x, pt.y));
    }

    RenderCurveResponse {
        bending_energy: curve.bending_energy(a, b),
        crosses: curve.detect_crossing(),
        points,
    }
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![analyze_curve])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
