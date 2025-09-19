use serde::{Deserialize, Serialize};
use std::error::Error;

/// Configuration for the physics simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationConfig {
    pub gravity: [f32; 2],
    pub particle_mass: f32,
    pub target_tps: f32,
    pub target_fps: f32,
    pub show_velocity_vectors: bool,
    pub show_acceleration_vectors: bool,
    pub vector_display_multiplier: f32,
    pub walls_enabled: bool,
    pub wall_bounce: bool,
    pub max_particles: usize,          // Kept for backward compatibility
    pub particle_limit: Option<usize>, // None means infinite
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            gravity: [0.0, -9.80665], // Standard Earth gravity in m/s²
            particle_mass: 0.1,       // 100 grams in kg
            target_tps: 60.0,
            target_fps: 60.0,
            show_velocity_vectors: false,
            show_acceleration_vectors: false,
            vector_display_multiplier: 0.5,
            walls_enabled: true,
            wall_bounce: false,
            max_particles: 10000,
            particle_limit: None, // Infinite by default
        }
    }
}

/// Configuration management functionality
pub struct ConfigManager;

impl ConfigManager {
    /// Save configuration to a JSON file via file dialog
    pub fn save_config(config: &SimulationConfig) -> Result<(), Box<dyn Error>> {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("JSON", &["json"])
            .set_file_name("physics_config.json")
            .save_file()
        {
            let json = serde_json::to_string_pretty(config)?;
            std::fs::write(path, json)?;
        }
        Ok(())
    }

    /// Load configuration from a JSON file via file dialog
    pub fn load_config() -> Result<Option<SimulationConfig>, Box<dyn Error>> {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("JSON", &["json"])
            .pick_file()
        {
            let json = std::fs::read_to_string(path)?;
            let config: SimulationConfig = serde_json::from_str(&json)?;
            Ok(Some(config))
        } else {
            Ok(None)
        }
    }

    /// Create a config from current app state
    pub fn create_config(
        gravity: [f32; 2],
        particle_mass: f32,
        target_tps: f32,
        target_fps: f32,
        show_velocity_vectors: bool,
        show_acceleration_vectors: bool,
        vector_display_multiplier: f32,
        walls_enabled: bool,
        wall_bounce: bool,
        particle_limit: Option<usize>,
    ) -> SimulationConfig {
        SimulationConfig {
            gravity,
            particle_mass,
            target_tps,
            target_fps,
            show_velocity_vectors,
            show_acceleration_vectors,
            vector_display_multiplier,
            walls_enabled,
            wall_bounce,
            max_particles: 10000, // Could be made configurable
            particle_limit,
        }
    }
}
