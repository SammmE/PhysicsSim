mod config;
mod engine;
mod graphs;
mod performance;
mod renderer;

use config::{ConfigManager, SimulationConfig};
use egui_plot::{Line, Plot, PlotPoints};
use engine::ParticleSystem;
use graphs::GraphData;
use performance::PerformanceMonitor;
use renderer::ParticleRenderer;
use std::time::Instant;
use winit::{
    event::*,
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowBuilder},
};

struct App {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    window: &'static Window,
    particle_system: ParticleSystem,
    particle_renderer: ParticleRenderer,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    last_update: Instant,

    // Configuration
    config_data: SimulationConfig,

    // Performance monitoring
    performance: PerformanceMonitor,

    // UI state
    gravity: [f32; 2],
    particle_mass: f32,
    spawn_position: [f32; 2],

    // Tick rate system
    target_tps: f32,
    tick_accumulator: f32,

    // Vector visualization
    show_velocity_vectors: bool,
    show_acceleration_vectors: bool,
    vector_display_multiplier: f32,

    // FPS control
    target_fps: f32,
    frame_time_accumulator: f32,

    // Mouse drag controls
    mouse_drag_start: Option<[f32; 2]>,
    is_dragging: bool,
    drag_button: Option<winit::event::MouseButton>,
    simulation_paused: bool,
    drag_preview_vector: Option<[f32; 2]>,

    // Enhanced mouse controls
    mouse_sensitivity: f32,
    scroll_zoom: f32,
    camera_offset: [f32; 2],

    // Boundary system
    walls_enabled: bool,
    wall_bounce: bool,
    world_bounds: [f32; 4], // left, right, bottom, top

    // Enhanced boundary options
    wall_damping: f32,
    gravity_strength: f32,

    // Particle limit system
    particle_limit: Option<usize>, // None means infinite

    // Particle graphing system
    selected_particle_index: Option<usize>,
    graph_data: GraphData,
    simulation_time: f32,
    show_graphs: bool,
    waiting_for_first_particle: bool,
}

impl App {
    async fn new(window: &'static Window) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let surface = instance.create_surface(window).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let present_mode = surface_caps
            .present_modes
            .iter()
            .find(|&&mode| mode == wgpu::PresentMode::Mailbox)
            .or_else(|| {
                surface_caps
                    .present_modes
                    .iter()
                    .find(|&&mode| mode == wgpu::PresentMode::Immediate)
            })
            .copied()
            .unwrap_or(wgpu::PresentMode::Fifo);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let particle_system = ParticleSystem::new();
        let particle_renderer = ParticleRenderer::new(&device, config.format, 10000);

        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            None,
            None,
        );
        let egui_renderer = egui_wgpu::Renderer::new(&device, config.format, None, 1);

        Self {
            surface,
            device,
            queue,
            config,
            size,
            window,
            particle_system,
            particle_renderer,
            egui_ctx,
            egui_state,
            egui_renderer,
            last_update: Instant::now(),
            config_data: SimulationConfig::default(),

            performance: PerformanceMonitor::new(),

            gravity: [0.0, -9.80665], // Standard Earth gravity in m/s²
            particle_mass: 0.1,       // 100 grams in kg
            spawn_position: [0.0, 0.0],
            target_tps: 60.0,
            tick_accumulator: 0.0,
            show_velocity_vectors: false,
            show_acceleration_vectors: false,
            vector_display_multiplier: 0.5,
            target_fps: 60.0,
            frame_time_accumulator: 0.0,
            mouse_drag_start: None,
            is_dragging: false,
            drag_button: None,
            simulation_paused: false,
            drag_preview_vector: None,
            mouse_sensitivity: 1.0,
            scroll_zoom: 1.0,
            camera_offset: [0.0, 0.0],
            walls_enabled: true,
            wall_bounce: false,
            world_bounds: Self::calculate_world_bounds(size),
            wall_damping: 0.8,
            gravity_strength: 1.0,
            particle_limit: None, // Infinite by default
            selected_particle_index: None,
            graph_data: GraphData::new(1000), // Store last 1000 data points
            simulation_time: 0.0,
            show_graphs: false,
            waiting_for_first_particle: false,
        }
    }

    fn calculate_world_bounds(size: winit::dpi::PhysicalSize<u32>) -> [f32; 4] {
        let aspect = size.width as f32 / size.height as f32;
        // Create a world space of approximately 20m wide x 15m tall
        let world_height = 15.0; // meters
        let world_width = world_height * aspect;

        let left = -world_width / 2.0;
        let right = world_width / 2.0;
        let bottom = -world_height / 2.0;
        let top = world_height / 2.0;

        [left, right, bottom, top]
    }

    fn window(&self) -> &Window {
        &self.window
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.performance.mark_uniforms_dirty();
            self.world_bounds = Self::calculate_world_bounds(new_size);
        }
    }

    fn select_next_particle(&mut self) {
        if self.particle_system.particles.is_empty() {
            self.selected_particle_index = None;
            self.show_graphs = false;
            return;
        }

        self.selected_particle_index = match self.selected_particle_index {
            None => Some(0),
            Some(current) => {
                let next = (current + 1) % self.particle_system.particles.len();
                Some(next)
            }
        };

        self.show_graphs = true;
        self.graph_data.clear();
    }

    fn clear_graphs(&mut self) {
        self.graph_data.clear();
    }

    fn check_and_select_first_particle(&mut self) {
        // If we were waiting for the first particle after clearing, select it
        if self.waiting_for_first_particle && !self.particle_system.particles.is_empty() {
            self.selected_particle_index = Some(0);
            self.show_graphs = true;
            self.graph_data.clear();
            self.waiting_for_first_particle = false;
        }
    }

    fn update_graph_data(&mut self) {
        if let Some(index) = self.selected_particle_index {
            if index < self.particle_system.particles.len() {
                let particle = &self.particle_system.particles[index];
                self.graph_data.add_point(
                    self.simulation_time,
                    particle.position.x,
                    particle.position.y,
                    particle.velocity.x,
                    particle.velocity.y,
                    particle.acceleration.x,
                    particle.acceleration.y,
                );
            } else {
                // Particle was removed, reset selection
                self.selected_particle_index = None;
                self.show_graphs = false;
            }
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        // Handle enhanced keyboard shortcuts first
        if self.add_keyboard_shortcuts(event) {
            return true;
        }

        if let WindowEvent::MouseInput { state, button, .. } = event {
            match state {
                ElementState::Pressed => {
                    if self.drag_button.is_none() {
                        self.drag_button = Some(*button);
                        self.is_dragging = false;
                        self.simulation_paused = true;
                    }
                }
                ElementState::Released => {
                    if let Some(drag_button) = self.drag_button {
                        if self.is_dragging && self.mouse_drag_start.is_some() {
                            self.create_particle_from_drag(drag_button);
                        } else if self.mouse_drag_start.is_some() {
                            let pos = self.mouse_drag_start.unwrap();
                            self.particle_system.add_particle_at(
                                pos[0],
                                pos[1],
                                self.particle_mass,
                                self.particle_limit,
                            );
                            self.check_and_select_first_particle();
                        }
                    }

                    self.mouse_drag_start = None;
                    self.is_dragging = false;
                    self.drag_button = None;
                    self.simulation_paused = false;
                    self.drag_preview_vector = None;
                }
            }
        }

        if let WindowEvent::CursorMoved { position, .. } = event {
            let world_pos = self.screen_to_world(position.x as f32, position.y as f32);

            if self.drag_button.is_some() {
                if self.mouse_drag_start.is_none() {
                    self.mouse_drag_start = Some(world_pos);
                } else {
                    let start = self.mouse_drag_start.unwrap();
                    let distance = ((world_pos[0] - start[0]).powi(2)
                        + (world_pos[1] - start[1]).powi(2))
                    .sqrt();

                    if distance > 2.0 && !self.is_dragging {
                        self.is_dragging = true;
                    }

                    if self.is_dragging {
                        self.drag_preview_vector = Some([
                            (world_pos[0] - start[0]) * 5.0,
                            (world_pos[1] - start[1]) * 5.0,
                        ]);
                    }
                }
            }
        }

        self.egui_state
            .on_window_event(&self.window, event)
            .consumed
    }

    fn screen_to_world(&self, screen_x: f32, screen_y: f32) -> [f32; 2] {
        let screen_center_x = self.size.width as f32 / 2.0;
        let screen_center_y = self.size.height as f32 / 2.0;

        // Convert screen coordinates to world coordinates in meters
        // Assume the world is 20m wide (from world bounds calculation)
        let aspect = self.size.width as f32 / self.size.height as f32;
        let world_height = 15.0; // meters
        let world_width = world_height * aspect;

        let world_x = ((screen_x - screen_center_x) / screen_center_x) * (world_width / 2.0);
        let world_y = ((screen_center_y - screen_y) / screen_center_y) * (world_height / 2.0);

        [world_x, world_y]
    }

    fn create_particle_from_drag(&mut self, button: winit::event::MouseButton) {
        if let (Some(start_pos), Some(preview_vector)) =
            (self.mouse_drag_start, self.drag_preview_vector)
        {
            match button {
                winit::event::MouseButton::Left => {
                    self.particle_system.add_particle_with_velocity(
                        start_pos[0],
                        start_pos[1],
                        preview_vector[0],
                        preview_vector[1],
                        self.particle_mass,
                        self.particle_limit,
                    );
                    self.check_and_select_first_particle();
                }
                winit::event::MouseButton::Right => {
                    self.particle_system.add_particle_with_acceleration(
                        start_pos[0],
                        start_pos[1],
                        preview_vector[0],
                        preview_vector[1],
                        self.particle_mass,
                        self.particle_limit,
                    );
                    self.check_and_select_first_particle();
                }
                _ => {}
            }
        }
    }

    fn update(&mut self) {
        let now = Instant::now();
        let frame_dt = (now - self.last_update).as_secs_f32().min(1.0 / 30.0);
        self.last_update = now;

        let target_frame_time = 1.0 / self.target_fps;
        self.frame_time_accumulator += frame_dt;

        if self.frame_time_accumulator < target_frame_time {
            return;
        }

        self.frame_time_accumulator -= target_frame_time;

        self.performance.update_metrics(frame_dt);

        let tick_dt = 1.0 / self.target_tps;
        self.tick_accumulator += target_frame_time;

        if !self.simulation_paused {
            let mut ticks_processed = 0;
            while self.tick_accumulator >= tick_dt
                && ticks_processed < self.performance.max_ticks_per_frame()
            {
                self.particle_system.apply_gravity(
                    engine::Vector2f::new(self.gravity[0], self.gravity[1]),
                    self.gravity_strength,
                    tick_dt,
                );
                self.particle_system.update(tick_dt);

                if self.walls_enabled {
                    self.particle_system.apply_wall_constraints(
                        self.world_bounds,
                        self.wall_bounce,
                        self.wall_damping,
                    );
                }

                self.simulation_time += tick_dt;
                self.update_graph_data();

                self.tick_accumulator -= tick_dt;
                ticks_processed += 1;
            }

            if ticks_processed >= self.performance.max_ticks_per_frame() {
                self.tick_accumulator = 0.0;
            }
        }

        if self.performance.uniforms_dirty() {
            self.particle_renderer.update_uniforms(
                &self.queue,
                self.size.width as f32,
                self.size.height as f32,
            );
            self.performance.clear_uniforms_dirty();
        }

        if !self.particle_system.particles.is_empty() {
            self.particle_renderer
                .update_particles(&self.queue, &self.particle_system.particles);

            if self.show_velocity_vectors || self.show_acceleration_vectors {
                self.particle_renderer.update_vector_lines(
                    &self.queue,
                    &self.particle_system.particles,
                    self.show_velocity_vectors,
                    self.show_acceleration_vectors,
                    1.0 * self.vector_display_multiplier, // Velocity vectors: 1 meter per 1 m/s
                    5.0 * self.vector_display_multiplier, // Acceleration vectors: 5 meters per 1 m/s²
                );
            }
        }

        // Update preview particle if dragging
        if let (Some(start_pos), Some(preview_vector)) =
            (self.mouse_drag_start, self.drag_preview_vector)
        {
            let mut preview_particle =
                engine::Particle::new(start_pos[0], start_pos[1], self.particle_mass);

            if let Some(button) = self.drag_button {
                match button {
                    winit::event::MouseButton::Left => {
                        // Preview velocity
                        preview_particle.velocity =
                            engine::Vector2f::new(preview_vector[0], preview_vector[1]);
                    }
                    winit::event::MouseButton::Right => {
                        // Preview acceleration
                        preview_particle.acceleration =
                            engine::Vector2f::new(preview_vector[0], preview_vector[1]);
                    }
                    _ => {}
                }
            }

            self.particle_renderer
                .update_preview_particle(&self.queue, &preview_particle);

            // Update preview vector lines
            let show_vel = self.drag_button == Some(winit::event::MouseButton::Left);
            let show_acc = self.drag_button == Some(winit::event::MouseButton::Right);

            if show_vel || show_acc {
                self.particle_renderer.update_preview_vector_lines(
                    &self.queue,
                    &preview_particle,
                    show_vel,
                    show_acc,
                    1.0 * self.vector_display_multiplier, // Velocity vectors: 1 meter per 1 m/s
                    5.0 * self.vector_display_multiplier, // Acceleration vectors: 5 meters per 1 m/s²
                );
            }
        }
    }

    fn save_config(&self) -> Result<(), Box<dyn std::error::Error>> {
        let config = ConfigManager::create_config(
            self.gravity,
            self.particle_mass,
            self.target_tps,
            self.target_fps,
            self.show_velocity_vectors,
            self.show_acceleration_vectors,
            self.vector_display_multiplier,
            self.walls_enabled,
            self.wall_bounce,
            self.particle_limit,
        );
        ConfigManager::save_config(&config)
    }

    fn load_config(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(config) = ConfigManager::load_config()? {
            self.gravity = config.gravity;
            self.particle_mass = config.particle_mass;
            self.target_tps = config.target_tps;
            self.target_fps = config.target_fps;
            self.show_velocity_vectors = config.show_velocity_vectors;
            self.show_acceleration_vectors = config.show_acceleration_vectors;
            self.vector_display_multiplier = config.vector_display_multiplier;
            self.walls_enabled = config.walls_enabled;
            self.wall_bounce = config.wall_bounce;
            self.particle_limit = config.particle_limit;
        }
        Ok(())
    }

    fn update_performance_metrics(&mut self, _frame_time: f32) {
        self.performance
            .update_memory_usage(self.particle_system.particle_count());
    }

    fn add_keyboard_shortcuts(&mut self, event: &WindowEvent) -> bool {
        if let WindowEvent::KeyboardInput {
            event: key_event, ..
        } = event
        {
            if key_event.state == ElementState::Pressed {
                match key_event.physical_key {
                    PhysicalKey::Code(KeyCode::Tab) => {
                        self.select_next_particle();
                        return true;
                    }
                    PhysicalKey::Code(KeyCode::Space) => {
                        self.clear_graphs();
                        return true;
                    }
                    PhysicalKey::Code(KeyCode::KeyP) => {
                        self.simulation_paused = !self.simulation_paused;
                        return true;
                    }
                    PhysicalKey::Code(KeyCode::KeyR) => {
                        self.particle_system.clear();
                        self.graph_data.clear();
                        self.selected_particle_index = None;
                        self.show_graphs = false;
                        self.waiting_for_first_particle = false;
                        return true;
                    }
                    PhysicalKey::Code(KeyCode::F1) => {
                        let _ = self.save_config(); // F1 to save
                        return true;
                    }
                    PhysicalKey::Code(KeyCode::F2) => {
                        let _ = self.load_config(); // F2 to load
                        return true;
                    }
                    _ => {}
                }
            }
        }
        false
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        self.particle_renderer
            .render(&mut encoder, &view, self.particle_system.particle_count());

        if (self.show_velocity_vectors || self.show_acceleration_vectors)
            && !self.particle_system.particles.is_empty()
        {
            let mut line_count = 0;
            if self.show_velocity_vectors {
                line_count += self.particle_system.particle_count();
            }
            if self.show_acceleration_vectors {
                line_count += self.particle_system.particle_count();
            }

            self.particle_renderer
                .render_vector_lines(&mut encoder, &view, line_count);
        }

        // Render preview particle and its vectors if dragging
        if let (Some(_start_pos), Some(_preview_vector)) =
            (self.mouse_drag_start, self.drag_preview_vector)
        {
            if self.is_dragging {
                // Render preview particle
                self.particle_renderer
                    .render_preview_particle(&mut encoder, &view);

                // Render preview vectors
                let show_vel = self.drag_button == Some(winit::event::MouseButton::Left);
                let show_acc = self.drag_button == Some(winit::event::MouseButton::Right);

                if show_vel || show_acc {
                    let preview_line_count = if show_vel && show_acc { 2 } else { 1 };
                    self.particle_renderer.render_preview_vector_lines(
                        &mut encoder,
                        &view,
                        preview_line_count,
                    );
                }
            }
        }

        let mut should_clear = false;
        let mut should_add_particle = false;
        let mut new_gravity = self.gravity;
        let mut new_particle_mass = self.particle_mass;
        let mut new_spawn_position = self.spawn_position;
        let mut new_target_tps = self.target_tps;
        let mut new_target_fps = self.target_fps;
        let mut new_show_velocity = self.show_velocity_vectors;
        let mut new_show_acceleration = self.show_acceleration_vectors;
        let mut new_vector_multiplier = self.vector_display_multiplier;
        let mut new_walls_enabled = self.walls_enabled;
        let mut new_wall_bounce = self.wall_bounce;

        // Get references to avoid borrowing issues in the closure
        let current_fps = self.performance.current_fps;
        let memory_usage = self.performance.memory_usage;
        let render_time = self.performance.render_time;
        let particle_count = self.particle_system.particle_count();
        let tick_accumulator = self.tick_accumulator;
        let simulation_paused = self.simulation_paused;
        let mut gravity_strength_local = self.gravity_strength;
        let mut save_config_clicked = false;
        let mut load_config_clicked = false;

        let raw_input = self.egui_state.take_egui_input(&self.window);
        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            // Calculate responsive sizes based on screen dimensions
            let screen_rect = ctx.screen_rect();
            let controls_width = (screen_rect.width() * 0.25).clamp(300.0, 400.0);
            let graph_width = (screen_rect.width() * 0.4).clamp(500.0, 800.0);
            let graph_height = (screen_rect.height() * 0.6).clamp(400.0, 600.0);

            egui::Window::new("Physics Controls")
                .default_width(controls_width)
                .resizable(true)
                .show(ctx, |ui| {
                    ui.heading("PhysicsSim - Physics Simulation");

                    ui.separator();
                    ui.strong("Frame Rate Control");
                    ui.add(
                        egui::Slider::new(&mut new_target_fps, 1.0..=120.0)
                            .text("Target FPS")
                            .suffix(" fps"),
                    );
                    ui.label("Lower FPS = Slow Motion Effect");

                    ui.separator();
                    ui.strong("Tick Rate Control");
                    ui.add(
                        egui::Slider::new(&mut new_target_tps, 1.0..=240.0)
                            .text("Target TPS")
                            .suffix(" ticks/sec"),
                    );
                    ui.label(format!(
                        "Fixed timestep: {:.4}s per tick",
                        1.0 / new_target_tps
                    ));

                    ui.separator();
                    ui.strong("Physics Parameters");

                    ui.horizontal(|ui| {
                        ui.label("Gravity (m/s²):");
                        ui.add(
                            egui::DragValue::new(&mut new_gravity[0])
                                .prefix("X: ")
                                .suffix(" m/s²")
                                .speed(0.1),
                        );
                        ui.add(
                            egui::DragValue::new(&mut new_gravity[1])
                                .prefix("Y: ")
                                .suffix(" m/s²")
                                .speed(0.1),
                        );
                    });

                    ui.add(
                        egui::Slider::new(&mut new_particle_mass, 0.001..=10.0)
                            .text("Particle Mass (kg)")
                            .suffix(" kg")
                            .logarithmic(true),
                    );

                    // Particle limit slider with infinity option
                    ui.horizontal(|ui| {
                        ui.label("Particle Limit:");
                        let mut is_infinite = self.particle_limit.is_none();

                        if ui.checkbox(&mut is_infinite, "Infinite").changed() {
                            if is_infinite {
                                self.particle_limit = None;
                            } else {
                                self.particle_limit = Some(10000); // Default to 10000 when switching from infinite
                            }
                        }

                        if !is_infinite {
                            let mut current_limit = self.particle_limit.unwrap_or(10000);
                            if ui
                                .add(
                                    egui::Slider::new(&mut current_limit, 1..=50000)
                                        .text("Max particles")
                                        .logarithmic(true),
                                )
                                .changed()
                            {
                                self.particle_limit = Some(current_limit);
                            }
                        }
                    });

                    ui.horizontal(|ui| {
                        ui.label("Spawn Position (m):");
                        ui.add(
                            egui::DragValue::new(&mut new_spawn_position[0])
                                .prefix("X: ")
                                .suffix(" m")
                                .speed(0.1),
                        );
                        ui.add(
                            egui::DragValue::new(&mut new_spawn_position[1])
                                .prefix("Y: ")
                                .suffix(" m")
                                .speed(0.1),
                        );
                    });

                    ui.separator();
                    ui.strong("Visualization Options");

                    ui.checkbox(&mut new_show_velocity, "Show Velocity Vectors");
                    ui.checkbox(&mut new_show_acceleration, "Show Acceleration Vectors");
                    ui.add(
                        egui::Slider::new(&mut new_vector_multiplier, 0.1..=2.0)
                            .text("Vector Display Scale")
                            .step_by(0.1),
                    );
                    ui.small("Green: Velocity (m/s), Red: Acceleration (m/s²)");

                    ui.separator();
                    ui.strong("Boundary System");

                    ui.checkbox(&mut new_walls_enabled, "Enable Walls");
                    ui.add_enabled(
                        new_walls_enabled,
                        egui::Checkbox::new(&mut new_wall_bounce, "Wall Bounce"),
                    );
                    if new_walls_enabled && new_wall_bounce {
                        // TODO: Add wall damping slider
                        ui.small("Wall damping: 0.8 (configurable in future update)");
                    }
                    if !new_walls_enabled {
                        ui.small("Particles can move freely");
                    } else if new_wall_bounce {
                        ui.small("Particles bounce off screen edges");
                    } else {
                        ui.small("Particles stop at screen edges");
                    }

                    ui.separator();
                    ui.strong("Performance Info");

                    ui.label(format!("FPS: {:.1}", current_fps));
                    ui.label(format!(
                        "Frame Time: {:.2}ms",
                        1000.0 / current_fps.max(1.0)
                    ));
                    ui.label(format!(
                        "Memory Usage: {:.1}KB",
                        memory_usage as f32 / 1024.0
                    ));
                    ui.label(format!("Render Time: {:.2}ms", render_time * 1000.0));

                    ui.separator();
                    ui.strong("Enhanced Gravity");

                    ui.add(
                        egui::Slider::new(&mut gravity_strength_local, 0.0..=3.0)
                            .text("Gravity Multiplier")
                            .step_by(0.1),
                    );
                    ui.small("Multiplies gravity acceleration (Earth = 1.0)");

                    ui.separator();
                    ui.strong("Configuration");

                    ui.horizontal(|ui| {
                        if ui.button("💾 Save Config (F1)").clicked() {
                            save_config_clicked = true;
                        }
                        if ui.button("📁 Load Config (F2)").clicked() {
                            load_config_clicked = true;
                        }
                    });

                    ui.separator();
                    ui.strong("Performance Info");

                    ui.label(format!("FPS: {:.1}", current_fps));
                    ui.label(format!(
                        "Frame Time: {:.2}ms",
                        1000.0 / current_fps.max(1.0)
                    ));
                    let limit_text = if let Some(limit) = self.particle_limit {
                        format!("Active Particles: {} / {}", particle_count, limit)
                    } else {
                        format!("Active Particles: {} / ∞", particle_count)
                    };
                    ui.label(limit_text);
                    ui.label(format!("Tick Accumulator: {:.4}s", tick_accumulator));
                    ui.label(format!(
                        "Memory Usage: {:.1}KB",
                        memory_usage as f32 / 1024.0
                    ));
                    ui.label(format!("Render Time: {:.2}ms", render_time * 1000.0));
                    if simulation_paused {
                        ui.label("🚫 SIMULATION PAUSED");
                    }

                    ui.separator();

                    if ui.button("Add Particle").clicked() {
                        should_add_particle = true;
                    }

                    if ui.button("Clear All Particles").clicked() {
                        should_clear = true;
                    }
                    ui.small("Note: Clearing keeps graphs open for next particle");

                    ui.separator();
                    ui.small("Keyboard Controls:");
                    ui.small("• Tab: Switch particle tracking");
                    ui.small("• Space: Clear graphs");
                    ui.small("• P: Pause/Resume simulation");
                    ui.small("• R: Reset everything");
                    ui.small("• F1: Save configuration");
                    ui.small("• F2: Load configuration");
                    ui.separator();
                    ui.small("Mouse Controls:");
                    ui.small("• Left drag: Set velocity vector (m/s, green preview)");
                    ui.small("• Right drag: Set acceleration vector (m/s², red preview)");
                    ui.small("• Simple click: Normal particle");
                    ui.small("• Preview shows particle + vector while dragging");
                    ui.separator();
                    ui.small("World space: ~20m × 15m, Earth gravity: 9.81 m/s²");
                });

            // Particle graphs window
            if self.show_graphs && self.selected_particle_index.is_some() {
                egui::Window::new("Particle Graphs")
                    .default_width(graph_width)
                    .default_height(graph_height)
                    .resizable(true)
                    .show(ctx, |ui| {
                        let particle_index = self.selected_particle_index.unwrap();
                        ui.heading(format!("Particle {} Tracking", particle_index));

                        if !self.graph_data.time.is_empty() {
                            ui.horizontal(|ui| {
                                ui.label(format!("Data points: {}", self.graph_data.time.len()));
                                ui.label(format!("Time: {:.2}s", self.simulation_time));
                            });

                            let plot_height = (graph_height * 0.25).clamp(100.0, 150.0);

                            // Create plot points for displacement
                            let displacement_x_points: PlotPoints = self
                                .graph_data
                                .time
                                .iter()
                                .zip(self.graph_data.displacement_x.iter())
                                .map(|(&t, &x)| [t as f64, x as f64])
                                .collect();

                            let displacement_y_points: PlotPoints = self
                                .graph_data
                                .time
                                .iter()
                                .zip(self.graph_data.displacement_y.iter())
                                .map(|(&t, &y)| [t as f64, y as f64])
                                .collect();

                            // Create plot points for velocity
                            let velocity_x_points: PlotPoints = self
                                .graph_data
                                .time
                                .iter()
                                .zip(self.graph_data.velocity_x.iter())
                                .map(|(&t, &vx)| [t as f64, vx as f64])
                                .collect();

                            let velocity_y_points: PlotPoints = self
                                .graph_data
                                .time
                                .iter()
                                .zip(self.graph_data.velocity_y.iter())
                                .map(|(&t, &vy)| [t as f64, vy as f64])
                                .collect();

                            // Create plot points for acceleration
                            let acceleration_x_points: PlotPoints = self
                                .graph_data
                                .time
                                .iter()
                                .zip(self.graph_data.acceleration_x.iter())
                                .map(|(&t, &ax)| [t as f64, ax as f64])
                                .collect();

                            let acceleration_y_points: PlotPoints = self
                                .graph_data
                                .time
                                .iter()
                                .zip(self.graph_data.acceleration_y.iter())
                                .map(|(&t, &ay)| [t as f64, ay as f64])
                                .collect();

                            // Displacement plot
                            ui.separator();
                            ui.strong("Displacement vs Time (m)");
                            let (time_min, time_max) = self.graph_data.calculate_time_bounds();
                            let (x_min, x_max) =
                                GraphData::calculate_plot_bounds(&self.graph_data.displacement_x);
                            let (y_min, y_max) =
                                GraphData::calculate_plot_bounds(&self.graph_data.displacement_y);
                            Plot::new("displacement_plot")
                                .include_x(time_min)
                                .include_x(time_max)
                                .include_y(x_min.min(y_min))
                                .include_y(x_max.max(y_max))
                                .auto_bounds([true, true].into())
                                .height(plot_height)
                                .show(ui, |plot_ui| {
                                    plot_ui.line(
                                        Line::new(displacement_x_points)
                                            .name("X Position (m)")
                                            .color(egui::Color32::from_rgb(255, 165, 0)), // Orange for X
                                    );
                                    plot_ui.line(
                                        Line::new(displacement_y_points)
                                            .name("Y Position (m)")
                                            .color(egui::Color32::from_rgb(0, 255, 255)), // Cyan for Y
                                    );
                                });

                            // Velocity plot
                            ui.separator();
                            ui.strong("Velocity vs Time (m/s)");
                            let (vx_min, vx_max) =
                                GraphData::calculate_plot_bounds(&self.graph_data.velocity_x);
                            let (vy_min, vy_max) =
                                GraphData::calculate_plot_bounds(&self.graph_data.velocity_y);
                            Plot::new("velocity_plot")
                                .include_x(time_min)
                                .include_x(time_max)
                                .include_y(vx_min.min(vy_min))
                                .include_y(vx_max.max(vy_max))
                                .auto_bounds([true, true].into())
                                .height(plot_height)
                                .show(ui, |plot_ui| {
                                    plot_ui.line(
                                        Line::new(velocity_x_points)
                                            .name("X Velocity (m/s)")
                                            .color(egui::Color32::from_rgb(255, 165, 0)), // Orange for X
                                    );
                                    plot_ui.line(
                                        Line::new(velocity_y_points)
                                            .name("Y Velocity (m/s)")
                                            .color(egui::Color32::from_rgb(0, 255, 255)), // Cyan for Y
                                    );
                                });

                            // Acceleration plot
                            ui.separator();
                            ui.strong("Acceleration vs Time (m/s²)");
                            let (ax_min, ax_max) =
                                GraphData::calculate_plot_bounds(&self.graph_data.acceleration_x);
                            let (ay_min, ay_max) =
                                GraphData::calculate_plot_bounds(&self.graph_data.acceleration_y);
                            Plot::new("acceleration_plot")
                                .include_x(time_min)
                                .include_x(time_max)
                                .include_y(ax_min.min(ay_min))
                                .include_y(ax_max.max(ay_max))
                                .auto_bounds([true, true].into())
                                .height(plot_height)
                                .show(ui, |plot_ui| {
                                    plot_ui.line(
                                        Line::new(acceleration_x_points)
                                            .name("X Acceleration (m/s²)")
                                            .color(egui::Color32::from_rgb(255, 165, 0)), // Orange for X
                                    );
                                    plot_ui.line(
                                        Line::new(acceleration_y_points)
                                            .name("Y Acceleration (m/s²)")
                                            .color(egui::Color32::from_rgb(0, 255, 255)), // Cyan for Y
                                    );
                                });
                        } else {
                            ui.label("No data collected yet. Wait for simulation to run...");
                        }
                    });
            }
        });

        self.gravity = new_gravity;
        self.particle_mass = new_particle_mass;
        self.spawn_position = new_spawn_position;
        self.target_tps = new_target_tps;
        self.target_fps = new_target_fps;
        self.show_velocity_vectors = new_show_velocity;
        self.show_acceleration_vectors = new_show_acceleration;
        self.vector_display_multiplier = new_vector_multiplier;
        self.walls_enabled = new_walls_enabled;
        self.wall_bounce = new_wall_bounce;
        self.gravity_strength = gravity_strength_local;

        // Handle click-to-add particles only if UI is not using pointer input
        if !simulation_paused {
            // Check if egui wants pointer input (hovering over UI elements)
            // Also check if we're not hovering over any windows (including graphs)
            let pointer_over_area =
                self.egui_ctx.wants_pointer_input() || self.egui_ctx.is_pointer_over_area();

            if !pointer_over_area {
                if self.egui_ctx.input(|i| i.pointer.primary_clicked()) {
                    if let Some(pos) = self.egui_ctx.input(|i| i.pointer.interact_pos()) {
                        // Convert screen position to world position
                        new_spawn_position[0] = pos.x;
                        new_spawn_position[1] = pos.y;
                        should_add_particle = true;
                    }
                }
            }
        }

        // Handle configuration save/load
        if save_config_clicked {
            let _ = self.save_config();
        }
        if load_config_clicked {
            let _ = self.load_config();
        }

        // Update performance metrics
        let now = Instant::now();
        let frame_dt = (now - self.last_update).as_secs_f32().min(1.0 / 30.0);
        self.update_performance_metrics(frame_dt);

        if should_add_particle {
            // Convert screen coordinates to world coordinates if needed
            if new_spawn_position[0] > 1.0 || new_spawn_position[1] > 1.0 {
                // This looks like screen coordinates, convert them
                let world_pos = self.screen_to_world(new_spawn_position[0], new_spawn_position[1]);
                new_spawn_position[0] = world_pos[0];
                new_spawn_position[1] = world_pos[1];
            }

            self.particle_system.add_particle_at(
                new_spawn_position[0],
                new_spawn_position[1],
                self.particle_mass,
                self.particle_limit,
            );

            // If we were waiting for the first particle after clearing, select it
            if self.waiting_for_first_particle {
                self.selected_particle_index = Some(0);
                self.show_graphs = true;
                self.graph_data.clear();
                self.waiting_for_first_particle = false;
            }
        }
        if should_clear {
            let was_showing_graphs = self.show_graphs;
            self.particle_system.clear();

            // If we were showing graphs, keep them open and wait for first new particle
            if was_showing_graphs {
                self.selected_particle_index = None;
                self.graph_data.clear();
                self.waiting_for_first_particle = true;
                // Keep show_graphs = true to maintain the graph window
            } else {
                self.waiting_for_first_particle = false;
            }
        }

        self.egui_state
            .handle_platform_output(&self.window, full_output.platform_output);

        let tris = self
            .egui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);
        for (id, image_delta) in &full_output.textures_delta.set {
            self.egui_renderer
                .update_texture(&self.device, &self.queue, *id, image_delta);
        }

        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: self.window.scale_factor() as f32,
        };

        self.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            &tris,
            &screen_descriptor,
        );

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Egui Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        self.egui_renderer
            .render(&mut render_pass, &tris, &screen_descriptor);
        drop(render_pass);

        for x in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(x);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new()
        .with_title("PhysicsSim - Advanced Particle Physics Simulation")
        .with_inner_size(winit::dpi::LogicalSize::new(1200, 800))
        .build(&event_loop)
        .unwrap();

    let window: &'static Window = Box::leak(Box::new(window));
    let mut app = pollster::block_on(App::new(window));

    event_loop
        .run(move |event, control_flow| match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == app.window().id() => {
                if !app.input(event) {
                    match event {
                        WindowEvent::CloseRequested
                        | WindowEvent::KeyboardInput {
                            event:
                                KeyEvent {
                                    state: ElementState::Pressed,
                                    physical_key: PhysicalKey::Code(KeyCode::Escape),
                                    ..
                                },
                            ..
                        } => control_flow.exit(),
                        WindowEvent::Resized(physical_size) => {
                            app.resize(*physical_size);
                        }
                        WindowEvent::RedrawRequested => {
                            app.update();
                            match app.render() {
                                Ok(_) => {}
                                Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                                    app.resize(app.size)
                                }
                                Err(wgpu::SurfaceError::OutOfMemory) => control_flow.exit(),
                                Err(wgpu::SurfaceError::Timeout) => log::warn!("Surface timeout"),
                            }
                        }
                        _ => {}
                    }
                }
            }
            Event::AboutToWait => {
                app.window().request_redraw();
            }
            _ => {}
        })
        .unwrap();
}
