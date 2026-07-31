use nalgebra::Vector2;

#[derive(Clone, Copy)]
pub struct TriadicFlowCurve {
    pub p0: Vector2<f64>,
    pub p1: Vector2<f64>,
    pub p2: Vector2<f64>,
}

impl TriadicFlowCurve {
    pub fn new(p0: Vector2<f64>, p1: Vector2<f64>, p2: Vector2<f64>) -> Self {
        Self { p0, p1, p2 }
    }

    pub fn evaluate(&self, s: f64) -> Vector2<f64> {
        let t = 1.0 - s;
        t * t * self.p0 + 2.0 * t * s * self.p1 + s * s * self.p2
    }

    /// Exact, highly stable closed-form solution for physical bending energy [2].
    pub fn bending_energy(&self, a: f64, b: f64) -> f64 {
        let ax = self.p0.x - 2.0 * self.p1.x + self.p2.x;
        let ay = self.p0.y - 2.0 * self.p1.y + self.p2.y;
        let bx = self.p1.x - self.p0.x;
        let by = self.p1.y - self.p0.y;

        let vec_a = Vector2::new(2.0 * ax, 2.0 * ay);
        let vec_b = Vector2::new(2.0 * bx, 2.0 * by);

        let norm_a_sq = vec_a.norm_squared();
        let norm_b_sq = vec_b.norm_squared();
        let dot_ab = vec_a.dot(&vec_b);

        a * ((1.0 / 3.0) * norm_a_sq + dot_ab + norm_b_sq) + b * norm_a_sq
    }

    /// Rigorously corrected quadratic crossing solver over interval s in [0, 1] [2].
    pub fn detect_crossing(&self) -> bool {
        let ax = self.p0.x - 2.0 * self.p1.x + self.p2.x;
        let bx = 2.0 * (self.p1.x - self.p0.x);
        let cx = self.p0.x;

        const EPSILON: f64 = 1e-12;

        if ax.abs() < EPSILON {
            if bx.abs() < EPSILON {
                cx.abs() < EPSILON
            } else {
                let s = -cx / bx;
                s >= 0.0 && s <= 1.0
            }
        } else {
            let discriminant = bx * bx - 4.0 * ax * cx;
            if discriminant < 0.0 {
                false
            } else {
                let sqrt_disc = discriminant.sqrt();
                let s1 = (-bx + sqrt_disc) / (2.0 * ax);
                let s2 = (-bx - sqrt_disc) / (2.0 * ax);
                (s1 >= 0.0 && s1 <= 1.0) || (s2 >= 0.0 && s2 <= 1.0)
            }
        }
    }
}
