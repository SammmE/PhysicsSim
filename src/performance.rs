use std::time::Instant;

/// Performance monitoring functionality
#[derive(Debug)]
pub struct PerformanceMonitor {
    pub current_fps: f32,
    pub render_time: f32,
    pub update_time: f32,
    pub physics_time: f32,
    pub ui_time: f32,
    pub memory_usage: usize,
    
    // Internal tracking
    frame_count: u32,
    fps_timer: Instant,
    uniforms_dirty: bool,
    max_ticks_per_frame: u32,
}

impl PerformanceMonitor {
    pub fn new() -> Self {
        Self {
            current_fps: 0.0,
            render_time: 0.0,
            update_time: 0.0,
            physics_time: 0.0,
            ui_time: 0.0,
            memory_usage: 0,
            frame_count: 0,
            fps_timer: Instant::now(),
            uniforms_dirty: true,
            max_ticks_per_frame: 5,
        }
    }

    pub fn update_metrics(&mut self, _frame_time: f32) {
        self.frame_count += 1;
        
        if self.fps_timer.elapsed().as_secs_f32() >= 1.0 {
            self.current_fps = self.frame_count as f32 / self.fps_timer.elapsed().as_secs_f32();
            self.frame_count = 0;
            self.fps_timer = Instant::now();
        }
    }

    pub fn update_memory_usage(&mut self, particle_count: usize) {
        // Estimate memory usage based on particles and other data structures
        const PARTICLE_SIZE: usize = std::mem::size_of::<crate::engine::Particle>();
        const BASE_MEMORY: usize = 1024 * 1024; // 1MB base
        
        self.memory_usage = BASE_MEMORY + (particle_count * PARTICLE_SIZE);
    }

    pub fn mark_uniforms_dirty(&mut self) {
        self.uniforms_dirty = true;
    }

    pub fn uniforms_dirty(&self) -> bool {
        self.uniforms_dirty
    }

    pub fn clear_uniforms_dirty(&mut self) {
        self.uniforms_dirty = false;
    }

    pub fn max_ticks_per_frame(&self) -> u32 {
        self.max_ticks_per_frame
    }
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}