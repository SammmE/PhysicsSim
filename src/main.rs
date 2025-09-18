mod engine;
mod renderer;

use egui_plot::{Line, Plot, PlotPoints};
use engine::ParticleSystem;
use renderer::ParticleRenderer;
use std::time::Instant;
use winit::{
    event::*,
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowBuilder},
};

#[derive(Debug, Clone)]
struct GraphData {
    pub time: Vec<f32>,
    pub displacement_x: Vec<f32>,
    pub displacement_y: Vec<f32>,
    pub velocity_x: Vec<f32>,
    pub velocity_y: Vec<f32>,
    pub acceleration_x: Vec<f32>,
    pub acceleration_y: Vec<f32>,
    pub max_points: usize,
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
        if self.time.len() >= self.max_points {
            // Remove oldest point
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
    }
}

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

    // UI state
    gravity: [f32; 2],
    particle_mass: f32,
    spawn_position: [f32; 2],

    // Tick rate system
    target_tps: f32,
    tick_accumulator: f32,

    // Performance optimization
    frame_count: u32,
    fps_timer: Instant,
    current_fps: f32,
    uniforms_dirty: bool,
    max_ticks_per_frame: u32,

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

    // Boundary system
    walls_enabled: bool,
    wall_bounce: bool,
    world_bounds: [f32; 4], // left, right, bottom, top

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
            gravity: [0.0, -98.0],
            particle_mass: 1.0,
            spawn_position: [0.0, 0.0],
            target_tps: 60.0,
            tick_accumulator: 0.0,
            frame_count: 0,
            fps_timer: Instant::now(),
            current_fps: 0.0,
            uniforms_dirty: true,
            max_ticks_per_frame: 10,
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
            walls_enabled: true,
            wall_bounce: false,
            world_bounds: Self::calculate_world_bounds(size),
            selected_particle_index: None,
            graph_data: GraphData::new(1000), // Store last 1000 data points
            simulation_time: 0.0,
            show_graphs: false,
            waiting_for_first_particle: false,
        }
    }

    fn calculate_world_bounds(size: winit::dpi::PhysicalSize<u32>) -> [f32; 4] {
        let aspect = size.width as f32 / size.height as f32;
        let scale = 0.01;

        let left = -aspect / scale;
        let right = aspect / scale;
        let bottom = -1.0 / scale;
        let top = 1.0 / scale;

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
            self.uniforms_dirty = true;
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
        // Handle keyboard input first
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
                    _ => {}
                }
            }
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

        let world_x = (screen_x - screen_center_x) / 5.0;
        let world_y = (screen_center_y - screen_y) / 5.0;

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

        self.frame_count += 1;
        if (now - self.fps_timer).as_secs_f32() >= 1.0 {
            self.current_fps = self.frame_count as f32;
            self.frame_count = 0;
            self.fps_timer = now;
        }

        let tick_dt = 1.0 / self.target_tps;
        self.tick_accumulator += target_frame_time;

        if !self.simulation_paused {
            let mut ticks_processed = 0;
            while self.tick_accumulator >= tick_dt && ticks_processed < self.max_ticks_per_frame {
                self.apply_gravity_batched(tick_dt);
                self.particle_system.update(tick_dt);

                if self.walls_enabled {
                    self.apply_wall_constraints();
                }

                self.simulation_time += tick_dt;
                self.update_graph_data();

                self.tick_accumulator -= tick_dt;
                ticks_processed += 1;
            }

            if ticks_processed >= self.max_ticks_per_frame {
                self.tick_accumulator = 0.0;
            }
        }

        if self.uniforms_dirty {
            self.particle_renderer.update_uniforms(
                &self.queue,
                self.size.width as f32,
                self.size.height as f32,
            );
            self.uniforms_dirty = false;
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
                    10.0 * self.vector_display_multiplier,
                    1.0 * self.vector_display_multiplier,
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
                    10.0 * self.vector_display_multiplier,
                    1.0 * self.vector_display_multiplier,
                );
            }
        }
    }

    fn apply_gravity_batched(&mut self, dt: f32) {
        let gravity_force = engine::Vector2f::new(self.gravity[0], self.gravity[1]);

        for particle in &mut self.particle_system.particles {
            particle.apply_force(
                engine::Vector2f::new(
                    gravity_force.x * particle.mass,
                    gravity_force.y * particle.mass,
                ),
                dt,
            );
        }
    }

    fn apply_wall_constraints(&mut self) {
        let [left, right, bottom, top] = self.world_bounds;

        for particle in &mut self.particle_system.particles {
            if particle.position.x < left {
                if self.wall_bounce {
                    particle.position.x = left;
                    particle.velocity.x = -particle.velocity.x * 0.8;
                } else {
                    particle.position.x = left;
                    particle.velocity.x = 0.0;
                }
            } else if particle.position.x > right {
                if self.wall_bounce {
                    particle.position.x = right;
                    particle.velocity.x = -particle.velocity.x * 0.8;
                } else {
                    particle.position.x = right;
                    particle.velocity.x = 0.0;
                }
            }

            if particle.position.y < bottom {
                if self.wall_bounce {
                    particle.position.y = bottom;
                    particle.velocity.y = -particle.velocity.y * 0.8;
                } else {
                    particle.position.y = bottom;
                    particle.velocity.y = 0.0;
                }
            } else if particle.position.y > top {
                if self.wall_bounce {
                    particle.position.y = top;
                    particle.velocity.y = -particle.velocity.y * 0.8;
                } else {
                    particle.position.y = top;
                    particle.velocity.y = 0.0;
                }
            }
        }
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
                        ui.label("Gravity:");
                        ui.add(
                            egui::DragValue::new(&mut new_gravity[0])
                                .prefix("X: ")
                                .speed(1.0),
                        );
                        ui.add(
                            egui::DragValue::new(&mut new_gravity[1])
                                .prefix("Y: ")
                                .speed(1.0),
                        );
                    });

                    ui.add(
                        egui::Slider::new(&mut new_particle_mass, 0.1..=10.0).text("Particle Mass"),
                    );

                    ui.horizontal(|ui| {
                        ui.label("Spawn Position:");
                        ui.add(
                            egui::DragValue::new(&mut new_spawn_position[0])
                                .prefix("X: ")
                                .speed(1.0),
                        );
                        ui.add(
                            egui::DragValue::new(&mut new_spawn_position[1])
                                .prefix("Y: ")
                                .speed(1.0),
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
                    ui.small("Green: Velocity, Red: Acceleration");

                    ui.separator();
                    ui.strong("Boundary System");

                    ui.checkbox(&mut new_walls_enabled, "Enable Walls");
                    ui.add_enabled(
                        new_walls_enabled,
                        egui::Checkbox::new(&mut new_wall_bounce, "Wall Bounce"),
                    );
                    if !new_walls_enabled {
                        ui.small("Particles can move freely");
                    } else if new_wall_bounce {
                        ui.small("Particles bounce off screen edges");
                    } else {
                        ui.small("Particles stop at screen edges");
                    }

                    ui.separator();
                    ui.strong("Performance Info");

                    ui.label(format!("FPS: {:.1}", self.current_fps));
                    ui.label(format!(
                        "Frame Time: {:.2}ms",
                        1000.0 / self.current_fps.max(1.0)
                    ));
                    ui.label(format!(
                        "Active Particles: {}",
                        self.particle_system.particle_count()
                    ));
                    ui.label(format!("Tick Accumulator: {:.4}s", self.tick_accumulator));
                    if self.simulation_paused {
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
                    ui.small("Mouse Controls:");
                    ui.small("• Left drag: Set velocity vector (green preview)");
                    ui.small("• Right drag: Set acceleration vector (red preview)");
                    ui.small("• Simple click: Normal particle");
                    ui.small("• Preview shows particle + vector while dragging");
                    ui.separator();
                    ui.small("Keyboard Controls:");
                    ui.small("• Tab: Switch particle tracking");
                    ui.small("• Space: Clear graphs");
                    ui.separator();
                    ui.small("Walls are at screen edges");
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
                            ui.strong("Displacement vs Time");
                            Plot::new("displacement_plot")
                                .view_aspect(2.0)
                                .height(plot_height)
                                .show(ui, |plot_ui| {
                                    plot_ui.line(
                                        Line::new(displacement_x_points)
                                            .name("X Position")
                                            .color(egui::Color32::from_rgb(255, 165, 0)), // Orange for X
                                    );
                                    plot_ui.line(
                                        Line::new(displacement_y_points)
                                            .name("Y Position")
                                            .color(egui::Color32::from_rgb(0, 255, 255)), // Cyan for Y
                                    );
                                });

                            // Velocity plot
                            ui.separator();
                            ui.strong("Velocity vs Time");
                            Plot::new("velocity_plot")
                                .view_aspect(2.0)
                                .height(plot_height)
                                .show(ui, |plot_ui| {
                                    plot_ui.line(
                                        Line::new(velocity_x_points)
                                            .name("X Velocity")
                                            .color(egui::Color32::from_rgb(255, 165, 0)), // Orange for X
                                    );
                                    plot_ui.line(
                                        Line::new(velocity_y_points)
                                            .name("Y Velocity")
                                            .color(egui::Color32::from_rgb(0, 255, 255)), // Cyan for Y
                                    );
                                });

                            // Acceleration plot
                            ui.separator();
                            ui.strong("Acceleration vs Time");
                            Plot::new("acceleration_plot")
                                .view_aspect(2.0)
                                .height(plot_height)
                                .show(ui, |plot_ui| {
                                    plot_ui.line(
                                        Line::new(acceleration_x_points)
                                            .name("X Acceleration")
                                            .color(egui::Color32::from_rgb(255, 165, 0)), // Orange for X
                                    );
                                    plot_ui.line(
                                        Line::new(acceleration_y_points)
                                            .name("Y Acceleration")
                                            .color(egui::Color32::from_rgb(0, 255, 255)), // Cyan for Y
                                    );
                                });
                        } else {
                            ui.label("No data collected yet. Wait for simulation to run...");
                        }
                    });
            }

            // Legacy click-to-add when not dragging
            if !self.simulation_paused && ctx.input(|i| i.pointer.primary_clicked()) {
                if let Some(pos) = ctx.input(|i| i.pointer.interact_pos()) {
                    let world_pos = self.screen_to_world(pos.x, pos.y);
                    new_spawn_position[0] = world_pos[0];
                    new_spawn_position[1] = world_pos[1];
                    should_add_particle = true;
                }
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

        if should_add_particle {
            self.particle_system.add_particle_at(
                self.spawn_position[0],
                self.spawn_position[1],
                self.particle_mass,
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
