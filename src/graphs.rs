/// Represents data for particle tracking graphs
#[derive(Debug, Clone)]
pub struct GraphData {
    pub time: Vec<f32>,
    pub displacement_x: Vec<f32>,
    pub displacement_y: Vec<f32>,
    pub velocity_x: Vec<f32>,
    pub velocity_y: Vec<f32>,
    pub acceleration_x: Vec<f32>,
    pub acceleration_y: Vec<f32>,
    pub max_points: usize,
    pub last_update_time: f32,
}

impl GraphData {
    pub fn new(max_points: usize) -> Self {
        Self {
            time: Vec::with_capacity(max_points),
            displacement_x: Vec::with_capacity(max_points),
            displacement_y: Vec::with_capacity(max_points),
            velocity_x: Vec::with_capacity(max_points),
            velocity_y: Vec::with_capacity(max_points),
            acceleration_x: Vec::with_capacity(max_points),
            acceleration_y: Vec::with_capacity(max_points),
            max_points,
            last_update_time: 0.0,
        }
    }

    pub fn add_point(
        &mut self,
        time: f32,
        pos_x: f32,
        pos_y: f32,
        vel_x: f32,
        vel_y: f32,
        acc_x: f32,
        acc_y: f32,
    ) {
        // Only add points if enough time has passed to avoid overwhelming the graph
        if time - self.last_update_time < 0.016 && !self.time.is_empty() {
            return;
        }

        self.last_update_time = time;

        if self.time.len() >= self.max_points {
            // Remove oldest point efficiently
            self.time.remove(0);
            self.displacement_x.remove(0);
            self.displacement_y.remove(0);
            self.velocity_x.remove(0);
            self.velocity_y.remove(0);
            self.acceleration_x.remove(0);
            self.acceleration_y.remove(0);
        }

        self.time.push(time);
        self.displacement_x.push(pos_x);
        self.displacement_y.push(pos_y);
        self.velocity_x.push(vel_x);
        self.velocity_y.push(vel_y);
        self.acceleration_x.push(acc_x);
        self.acceleration_y.push(acc_y);
    }

    pub fn clear(&mut self) {
        self.time.clear();
        self.displacement_x.clear();
        self.displacement_y.clear();
        self.velocity_x.clear();
        self.velocity_y.clear();
        self.acceleration_x.clear();
        self.acceleration_y.clear();
        self.last_update_time = 0.0;
    }

    pub fn get_stats(&self) -> GraphStats {
        if self.time.is_empty() {
            return GraphStats::default();
        }

        let vel_mag: Vec<f32> = self
            .velocity_x
            .iter()
            .zip(self.velocity_y.iter())
            .map(|(vx, vy)| (vx * vx + vy * vy).sqrt())
            .collect();

        let acc_mag: Vec<f32> = self
            .acceleration_x
            .iter()
            .zip(self.acceleration_y.iter())
            .map(|(ax, ay)| (ax * ax + ay * ay).sqrt())
            .collect();

        GraphStats {
            max_velocity: vel_mag.iter().fold(0.0f32, |a, &b| a.max(b)),
            max_acceleration: acc_mag.iter().fold(0.0f32, |a, &b| a.max(b)),
            total_distance: self.calculate_total_distance(),
            duration: self.time.last().unwrap_or(&0.0) - self.time.first().unwrap_or(&0.0),
        }
    }

    fn calculate_total_distance(&self) -> f32 {
        if self.displacement_x.len() < 2 {
            return 0.0;
        }

        let mut total = 0.0;
        for i in 1..self.displacement_x.len() {
            let dx = self.displacement_x[i] - self.displacement_x[i - 1];
            let dy = self.displacement_y[i] - self.displacement_y[i - 1];
            total += (dx * dx + dy * dy).sqrt();
        }
        total
    }

    /// Calculate optimal plot bounds for data with padding
    pub fn calculate_plot_bounds(data: &[f32]) -> (f64, f64) {
        if data.is_empty() {
            return (0.0, 1.0);
        }

        let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        if (max_val - min_val).abs() < f32::EPSILON {
            // Handle case where all values are the same
            return ((min_val - 0.1) as f64, (max_val + 0.1) as f64);
        }

        let range = max_val - min_val;
        let padding = range * 0.1; // 10% padding
        ((min_val - padding) as f64, (max_val + padding) as f64)
    }

    /// Calculate time bounds for the graph
    pub fn calculate_time_bounds(&self) -> (f64, f64) {
        if self.time.is_empty() {
            return (0.0, 1.0);
        }
        Self::calculate_plot_bounds(&self.time)
    }
}

/// Statistics derived from graph data
#[derive(Debug, Default)]
pub struct GraphStats {
    pub max_velocity: f32,
    pub max_acceleration: f32,
    pub total_distance: f32,
    pub duration: f32,
}
