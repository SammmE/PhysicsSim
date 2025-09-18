use bytemuck::{Pod, Zeroable};

pub mod vector;
pub use vector::Vector2f;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Particle {
    pub position: Vector2f,     // x, y coordinates
    pub velocity: Vector2f,     // velocity for physics
    pub acceleration: Vector2f, // acceleration for physics
    pub mass: f32,              // particle mass
}

impl Particle {
    pub fn new(x: f32, y: f32, mass: f32) -> Self {
        Self {
            position: Vector2f::new(x, y),
            velocity: Vector2f::new(0.0, 0.0),
            acceleration: Vector2f::new(0.0, 0.0),
            mass,
        }
    }

    pub fn update(&mut self, dt: f32) {
        // Basic physics: update position based on velocity
        self.position.x += self.velocity.x * dt;
        self.position.y += self.velocity.y * dt;
    }

    pub fn apply_force(&mut self, force: Vector2f, dt: f32) {
        // F = ma, so a = F/m
        let acc_x = force.x / self.mass;
        let acc_y = force.y / self.mass;

        // Store acceleration for vector visualization
        self.acceleration.x = acc_x;
        self.acceleration.y = acc_y;

        // Update velocity: v = v0 + at
        self.velocity.x += acc_x * dt;
        self.velocity.y += acc_y * dt;
    }

    pub fn set_position(&mut self, x: f32, y: f32) {
        self.position.x = x;
        self.position.y = y;
    }

    pub fn set_velocity(&mut self, vx: f32, vy: f32) {
        self.velocity.x = vx;
        self.velocity.y = vy;
    }

    pub fn set_acceleration(&mut self, ax: f32, ay: f32) {
        self.acceleration.x = ax;
        self.acceleration.y = ay;
    }

    pub fn set_mass(&mut self, mass: f32) {
        self.mass = mass;
    }

    pub fn get_position(&self) -> Vector2f {
        self.position
    }

    pub fn get_velocity(&self) -> Vector2f {
        self.velocity
    }

    pub fn get_acceleration(&self) -> Vector2f {
        self.acceleration
    }

    pub fn get_mass(&self) -> f32 {
        self.mass
    }
}

pub struct ParticleSystem {
    pub particles: Vec<Particle>,
}

impl ParticleSystem {
    pub fn new() -> Self {
        Self {
            particles: Vec::with_capacity(1000), // Pre-allocate capacity
        }
    }

    pub fn add_particle_at(&mut self, x: f32, y: f32, mass: f32) {
        // Only add if we haven't reached reasonable limit for performance
        if self.particles.len() < 10000 {
            self.particles.push(Particle::new(x, y, mass));
        }
    }

    pub fn add_particle_with_velocity(&mut self, x: f32, y: f32, vx: f32, vy: f32, mass: f32) {
        if self.particles.len() < 10000 {
            let mut particle = Particle::new(x, y, mass);
            particle.velocity = Vector2f::new(vx, vy);
            self.particles.push(particle);
        }
    }

    pub fn add_particle_with_acceleration(&mut self, x: f32, y: f32, ax: f32, ay: f32, mass: f32) {
        if self.particles.len() < 10000 {
            let mut particle = Particle::new(x, y, mass);
            particle.acceleration = Vector2f::new(ax, ay);
            self.particles.push(particle);
        }
    }

    pub fn update(&mut self, dt: f32) {
        // Update all particles with optimized iteration
        for particle in self.particles.iter_mut() {
            particle.update(dt);
        }
    }

    pub fn clear(&mut self) {
        self.particles.clear();
        // Don't shrink capacity for better performance on future allocations
    }

    pub fn particle_count(&self) -> usize {
        self.particles.len()
    }
}
