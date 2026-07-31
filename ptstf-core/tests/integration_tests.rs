use ptstf_core::flow::TriadicFlowCurve;
use ptstf_core::optimization::{GradientFlowOptimizer, smooth_crossing_probability};
use nalgebra::Vector2;

#[test]
fn test_bending_energy_stability() {
    let stagnant_curve = TriadicFlowCurve::new(
        Vector2::new(0.0, 0.0),
        Vector2::new(0.0, 0.0),
        Vector2::new(0.0, 0.0),
    );
    let energy = stagnant_curve.bending_energy(1.0, 1.0);
    assert!(energy.is_finite());
    assert_eq!(energy, 0.0);
}

#[test]
fn test_analytical_crossing_accuracy() {
    let crossing_curve = TriadicFlowCurve::new(
        Vector2::new(-1.0, 0.0),
        Vector2::new(0.0, 1.0),
        Vector2::new(1.0, 0.0),
    );
    assert!(crossing_curve.detect_crossing());

    let safe_curve = TriadicFlowCurve::new(
        Vector2::new(1.0, 0.0),
        Vector2::new(2.0, 1.0),
        Vector2::new(3.0, 0.0),
    );
    assert!(!safe_curve.detect_crossing());
}

#[test]
fn test_theorem_9_1_monotonicity() {
    // Setup initial representation state for 3 semantic nodes in R^2
    let mut y = vec![
        Vector2::new(-0.5, 0.2),
        Vector2::new(0.1, 0.8),
        Vector2::new(0.6, -0.3),
    ];
    
    // Stable learning rate
    let optimizer = GradientFlowOptimizer::new(0.0001, 10.0, 1.5);
    let steps = [0.0, 0.25, 0.5, 0.75, 1.0];
    
    // Evaluation closure of global Corpus Objective functional J(Y)
    let eval_objective = |coords: &[Vector2<f64>]| -> f64 {
        let curve = TriadicFlowCurve::new(coords[0], coords[1], coords[2]);
        let energy = curve.bending_energy(1.0, 1.0);
        let p_cross = smooth_crossing_probability(&curve, &steps, optimizer.tau);
        - optimizer.mu * 1.0 * energy * p_cross
    };

    let initial_obj = eval_objective(&y);
    let mut last_obj = initial_obj;
    
    // Execute 10 sequential representation gradient ascent steps
    for step in 1..=10 {
        // Curve defined by index linkage to nodes 0, 1, and 2
        let curve_indices = vec![(0, 1, 2)];
        let alphas = vec![1.0];
        
        optimizer.step_representation_flow(&mut y, &curve_indices, &alphas, 1.0, 1.0);
        
        let current_obj = eval_objective(&y);
        
        // Check strict monotonic ascent: J(Y_t) must be non-decreasing
        assert!(
            current_obj >= last_obj,
            "Monotonicity Violation on step {}! Prior J(Y): {}, Current J(Y): {}",
            step, last_obj, current_obj
        );
        
        last_obj = current_obj;
    }
    
    assert!(last_obj > initial_obj, "Objective should have climbed from initial state");
}
