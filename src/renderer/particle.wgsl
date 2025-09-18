struct Uniforms {
    view_proj: mat4x4<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec2<f32>,     // Quad vertex position
    @location(1) particle_pos: vec2<f32>, // Particle world position
    @location(2) particle_vel: vec2<f32>, // Particle velocity
    @location(3) particle_mass: f32,      // Particle mass
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(vertex: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    // Scale particle based on mass (optimized calculation)
    let scale = max(sqrt(vertex.particle_mass) * 2.0, 1.0);
    let scaled_pos = vertex.position * scale;
    
    // Transform to world position
    let world_pos = scaled_pos + vertex.particle_pos;
    
    // Apply projection
    out.clip_position = uniforms.view_proj * vec4<f32>(world_pos, 0.0, 1.0);
    
    // Optimized coloring: pre-compute constants and use faster math
    let vel_mag_sq = dot(vertex.particle_vel, vertex.particle_vel);
    let speed_factor = clamp(vel_mag_sq * 0.0001, 0.0, 1.0); // Use squared velocity to avoid sqrt
    let mass_factor = clamp(vertex.particle_mass * 0.2, 0.4, 1.0);
    
    // Use more vibrant colors for better visibility
    out.color = vec4<f32>(
        speed_factor + 0.2,     // Red: speed + base value
        mass_factor,            // Green: based on mass
        0.9 - speed_factor * 0.3, // Blue: inverse relationship with speed
        1.0                     // Full alpha
    );
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}