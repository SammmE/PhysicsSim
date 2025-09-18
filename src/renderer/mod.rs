use crate::engine::Particle;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LineVertex {
    pub position: [f32; 2],
    pub color: [f32; 4],
}

pub struct ParticleRenderer {
    pub render_pipeline: wgpu::RenderPipeline,
    pub instance_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub uniform_buffer: wgpu::Buffer,
    quad_vertex_buffer: wgpu::Buffer,
    quad_index_buffer: wgpu::Buffer,

    // Line rendering for vectors
    pub line_render_pipeline: wgpu::RenderPipeline,
    pub line_vertex_buffer: wgpu::Buffer,
    max_lines: usize,

    // Preview particle rendering
    pub preview_instance_buffer: wgpu::Buffer,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4], // 4x4 matrix for view-projection
}

impl ParticleRenderer {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat, max_instances: usize) -> Self {
        // Create quad vertices (triangle strip)
        let quad_vertices: &[f32] = &[
            -0.5, -0.5, // bottom-left
            0.5, -0.5, // bottom-right
            -0.5, 0.5, // top-left
            0.5, 0.5, // top-right
        ];

        let quad_indices: &[u16] = &[0, 1, 2, 1, 3, 2];

        let quad_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Quad Vertex Buffer"),
            contents: bytemuck::cast_slice(quad_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let quad_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Quad Index Buffer"),
            contents: bytemuck::cast_slice(quad_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        // Create instance buffer for particle data
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Buffer"),
            size: (max_instances * std::mem::size_of::<Particle>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create uniform buffer
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniform Buffer"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("Particle Bind Group Layout"),
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
            label: Some("Particle Bind Group"),
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Particle Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("particle.wgsl").into()),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Particle Render Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Particle Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[
                    // Quad vertices
                    wgpu::VertexBufferLayout {
                        array_stride: 2 * std::mem::size_of::<f32>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x2,
                        }],
                    },
                    // Instance data (particles)
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Particle>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[
                            // Position (Vector2f at offset 0)
                            wgpu::VertexAttribute {
                                offset: 0,
                                shader_location: 1,
                                format: wgpu::VertexFormat::Float32x2,
                            },
                            // Velocity (Vector2f at offset 8)
                            wgpu::VertexAttribute {
                                offset: 8,
                                shader_location: 2,
                                format: wgpu::VertexFormat::Float32x2,
                            },
                            // Skip acceleration (Vector2f at offset 16)
                            // Mass (f32 at offset 24)
                            wgpu::VertexAttribute {
                                offset: 24,
                                shader_location: 3,
                                format: wgpu::VertexFormat::Float32,
                            },
                        ],
                    },
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        // Create line vertex buffer for vector visualization
        let max_lines = max_instances * 2; // 2 vectors per particle max
        let line_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Line Vertex Buffer"),
            size: (max_lines * 2 * std::mem::size_of::<LineVertex>()) as u64, // 2 vertices per line
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create preview particle buffer for single particle preview
        let preview_instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Preview Instance Buffer"),
            size: std::mem::size_of::<Particle>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create line render pipeline
        let line_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Line Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("line.wgsl").into()),
        });

        let line_render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Line Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &line_shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<LineVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        // Position
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                        // Color
                        wgpu::VertexAttribute {
                            offset: 2 * std::mem::size_of::<f32>() as u64,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                    ],
                }],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &line_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        Self {
            render_pipeline,
            instance_buffer,
            bind_group,
            uniform_buffer,
            quad_vertex_buffer,
            quad_index_buffer,
            line_render_pipeline,
            line_vertex_buffer,
            max_lines,
            preview_instance_buffer,
        }
    }

    pub fn update_particles(&self, queue: &wgpu::Queue, particles: &[Particle]) {
        // Only update if we have particles and avoid unnecessary uploads
        if !particles.is_empty() && particles.len() <= 10000 {
            // Ensure we don't exceed buffer size
            queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(particles));
        }
    }

    pub fn update_uniforms(&self, queue: &wgpu::Queue, width: f32, height: f32) {
        // Simple orthographic projection
        let aspect = width / height;
        let scale = 0.01; // Adjust this to change the world scale

        let left = -aspect / scale;
        let right = aspect / scale;
        let bottom = -1.0 / scale;
        let top = 1.0 / scale;

        #[rustfmt::skip]
        let projection = [
            [2.0 / (right - left), 0.0, 0.0, -(right + left) / (right - left)],
            [0.0, 2.0 / (top - bottom), 0.0, -(top + bottom) / (top - bottom)],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        let uniforms = Uniforms {
            view_proj: projection,
        };

        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
    }

    pub fn update_vector_lines(
        &self,
        queue: &wgpu::Queue,
        particles: &[Particle],
        show_velocity: bool,
        show_acceleration: bool,
        velocity_scale: f32,
        acceleration_scale: f32,
    ) {
        if !show_velocity && !show_acceleration {
            return;
        }

        let mut line_vertices = Vec::new();

        for particle in particles {
            let start_pos = [particle.position.x, particle.position.y];

            // Add velocity vector line
            if show_velocity {
                let vel_end = [
                    particle.position.x + particle.velocity.x * velocity_scale,
                    particle.position.y + particle.velocity.y * velocity_scale,
                ];

                line_vertices.push(LineVertex {
                    position: start_pos,
                    color: [0.0, 1.0, 0.0, 0.8], // Green for velocity
                });
                line_vertices.push(LineVertex {
                    position: vel_end,
                    color: [0.0, 1.0, 0.0, 0.8],
                });
            }

            // Add acceleration vector line
            if show_acceleration {
                let acc_end = [
                    particle.position.x + particle.acceleration.x * acceleration_scale,
                    particle.position.y + particle.acceleration.y * acceleration_scale,
                ];

                line_vertices.push(LineVertex {
                    position: start_pos,
                    color: [1.0, 0.0, 0.0, 0.8], // Red for acceleration
                });
                line_vertices.push(LineVertex {
                    position: acc_end,
                    color: [1.0, 0.0, 0.0, 0.8],
                });
            }
        }

        // Update line buffer if we have vertices and don't exceed capacity
        if !line_vertices.is_empty() && line_vertices.len() <= self.max_lines * 2 {
            queue.write_buffer(
                &self.line_vertex_buffer,
                0,
                bytemuck::cast_slice(&line_vertices),
            );
        }
    }

    pub fn update_preview_particle(&self, queue: &wgpu::Queue, preview_particle: &Particle) {
        queue.write_buffer(
            &self.preview_instance_buffer,
            0,
            bytemuck::cast_slice(&[*preview_particle]),
        );
    }

    pub fn update_preview_vector_lines(
        &self,
        queue: &wgpu::Queue,
        preview_particle: &Particle,
        show_velocity: bool,
        show_acceleration: bool,
        velocity_scale: f32,
        acceleration_scale: f32,
    ) {
        if !show_velocity && !show_acceleration {
            return;
        }

        let mut line_vertices = Vec::new();
        let start_pos = [preview_particle.position.x, preview_particle.position.y];

        // Add velocity vector line
        if show_velocity {
            let vel_end = [
                preview_particle.position.x + preview_particle.velocity.x * velocity_scale,
                preview_particle.position.y + preview_particle.velocity.y * velocity_scale,
            ];

            line_vertices.push(LineVertex {
                position: start_pos,
                color: [0.0, 1.0, 0.0, 1.0], // Bright green for preview velocity
            });
            line_vertices.push(LineVertex {
                position: vel_end,
                color: [0.0, 1.0, 0.0, 1.0],
            });
        }

        // Add acceleration vector line
        if show_acceleration {
            let acc_end = [
                preview_particle.position.x + preview_particle.acceleration.x * acceleration_scale,
                preview_particle.position.y + preview_particle.acceleration.y * acceleration_scale,
            ];

            line_vertices.push(LineVertex {
                position: start_pos,
                color: [1.0, 0.0, 0.0, 1.0], // Bright red for preview acceleration
            });
            line_vertices.push(LineVertex {
                position: acc_end,
                color: [1.0, 0.0, 0.0, 1.0],
            });
        }

        // Update line buffer if we have vertices
        if !line_vertices.is_empty() {
            queue.write_buffer(
                &self.line_vertex_buffer,
                0,
                bytemuck::cast_slice(&line_vertices),
            );
        }
    }

    pub fn render_preview_particle(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
    ) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Preview Particle Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load, // Don't clear, draw over existing content
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.quad_vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.preview_instance_buffer.slice(..));
        render_pass.set_index_buffer(self.quad_index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        render_pass.draw_indexed(0..6, 0, 0..1); // Render 1 preview particle
    }

    pub fn render_preview_vector_lines(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        line_count: usize,
    ) {
        if line_count == 0 {
            return;
        }

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Preview Vector Lines Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load, // Don't clear, draw over existing content
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        render_pass.set_pipeline(&self.line_render_pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.line_vertex_buffer.slice(..));
        render_pass.draw(0..line_count as u32 * 2, 0..1); // 2 vertices per line
    }

    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        particle_count: usize,
    ) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Particle Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.1,
                        g: 0.1,
                        b: 0.1,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.quad_vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        render_pass.set_index_buffer(self.quad_index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        render_pass.draw_indexed(0..6, 0, 0..particle_count as u32);
    }

    pub fn render_vector_lines(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        line_count: usize,
    ) {
        if line_count == 0 {
            return;
        }

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Vector Lines Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load, // Don't clear, draw over particles
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        render_pass.set_pipeline(&self.line_render_pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.line_vertex_buffer.slice(..));
        render_pass.draw(0..line_count as u32 * 2, 0..1); // 2 vertices per line
    }
}
