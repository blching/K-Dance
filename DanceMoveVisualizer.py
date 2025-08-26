import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import os
import json

class DanceMoveVisualizer:
    def __init__(self, data_manager):
        self.data_manager = data_manager
        
    def create_skeleton_frame(self, angles, ax, frame_num=0, total_frames=1):
        """Create a single frame of the skeleton visualization on the provided axis"""
        ax.clear()
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Invert y-axis to match image coordinates
        ax.set_title(f'Dance Move - Frame {frame_num+1}/{total_frames}')
        
        # Extract angles
        left_elbow, right_elbow, left_shoulder, right_shoulder, \
        left_knee, right_knee, left_hip, right_hip = angles
        
        # Define body parts with relative positions
        # Pelvis (center point)
        pelvis = [0, 0]
        
        # Shoulders
        shoulder_width = 0.6
        left_shoulder_pos = [-shoulder_width/2, -0.2]
        right_shoulder_pos = [shoulder_width/2, -0.2]
        
        # Head
        head_radius = 0.15
        head = patches.Circle((0, -0.5), head_radius, fill=True, color='skyblue')
        ax.add_patch(head)
        
        # Draw torso
        torso = plt.Line2D([0, 0], [0, -0.35], lw=3, color='black')
        ax.add_line(torso)
        
        # Draw arms
        # Left arm
        left_elbow_angle_rad = np.radians(left_elbow)
        left_shoulder_angle_rad = np.radians(left_shoulder)
        
        left_upper_arm = plt.Line2D(
            [left_shoulder_pos[0], left_shoulder_pos[0] + 0.4 * np.cos(left_shoulder_angle_rad)],
            [left_shoulder_pos[1], left_shoulder_pos[1] + 0.4 * np.sin(left_shoulder_angle_rad)],
            lw=2, color='red'
        )
        ax.add_line(left_upper_arm)
        
        left_elbow_pos = [left_shoulder_pos[0] + 0.4 * np.cos(left_shoulder_angle_rad),
                         left_shoulder_pos[1] + 0.4 * np.sin(left_shoulder_angle_rad)]
        
        left_lower_arm = plt.Line2D(
            [left_elbow_pos[0], left_elbow_pos[0] + 0.3 * np.cos(left_elbow_angle_rad)],
            [left_elbow_pos[1], left_elbow_pos[1] + 0.3 * np.sin(left_elbow_angle_rad)],
            lw=2, color='red'
        )
        ax.add_line(left_lower_arm)
        
        # Right arm
        right_elbow_angle_rad = np.radians(right_elbow)
        right_shoulder_angle_rad = np.radians(right_shoulder)
        
        right_upper_arm = plt.Line2D(
            [right_shoulder_pos[0], right_shoulder_pos[0] + 0.4 * np.cos(right_shoulder_angle_rad)],
            [right_shoulder_pos[1], right_shoulder_pos[1] + 0.4 * np.sin(right_shoulder_angle_rad)],
            lw=2, color='blue'
        )
        ax.add_line(right_upper_arm)
        
        right_elbow_pos = [right_shoulder_pos[0] + 0.4 * np.cos(right_shoulder_angle_rad),
                          right_shoulder_pos[1] + 0.4 * np.sin(right_shoulder_angle_rad)]
        
        right_lower_arm = plt.Line2D(
            [right_elbow_pos[0], right_elbow_pos[0] + 0.3 * np.cos(right_elbow_angle_rad)],
            [right_elbow_pos[1], right_elbow_pos[1] + 0.3 * np.sin(right_elbow_angle_rad)],
            lw=2, color='blue'
        )
        ax.add_line(right_lower_arm)
        
        # Draw legs
        # Left leg
        left_hip_angle_rad = np.radians(left_hip)
        left_knee_angle_rad = np.radians(left_knee)
        
        left_upper_leg = plt.Line2D(
            [pelvis[0] - 0.2, pelvis[0] - 0.2 + 0.5 * np.cos(left_hip_angle_rad)],
            [pelvis[1], pelvis[1] + 0.5 * np.sin(left_hip_angle_rad)],
            lw=2, color='green'
        )
        ax.add_line(left_upper_leg)
        
        left_knee_pos = [pelvis[0] - 0.2 + 0.5 * np.cos(left_hip_angle_rad),
                        pelvis[1] + 0.5 * np.sin(left_hip_angle_rad)]
        
        left_lower_leg = plt.Line2D(
            [left_knee_pos[0], left_knee_pos[0] + 0.5 * np.cos(left_knee_angle_rad)],
            [left_knee_pos[1], left_knee_pos[1] + 0.5 * np.sin(left_knee_angle_rad)],
            lw=2, color='green'
        )
        ax.add_line(left_lower_leg)
        
        # Right leg
        right_hip_angle_rad = np.radians(right_hip)
        right_knee_angle_rad = np.radians(right_knee)
        
        right_upper_leg = plt.Line2D(
            [pelvis[0] + 0.2, pelvis[0] + 0.2 + 0.5 * np.cos(right_hip_angle_rad)],
            [pelvis[1], pelvis[1] + 0.5 * np.sin(right_hip_angle_rad)],
            lw=2, color='purple'
        )
        ax.add_line(right_upper_leg)
        
        right_knee_pos = [pelvis[0] + 0.2 + 0.5 * np.cos(right_hip_angle_rad),
                         pelvis[1] + 0.5 * np.sin(right_hip_angle_rad)]
        
        right_lower_leg = plt.Line2D(
            [right_knee_pos[0], right_knee_pos[0] + 0.5 * np.cos(right_knee_angle_rad)],
            [right_knee_pos[1], right_knee_pos[1] + 0.5 * np.sin(right_knee_angle_rad)],
            lw=2, color='purple'
        )
        ax.add_line(right_lower_leg)
        
        # Add joint markers
        joints = [
            pelvis, left_shoulder_pos, right_shoulder_pos,
            left_elbow_pos, right_elbow_pos,
            left_knee_pos, right_knee_pos
        ]
        
        for joint in joints:
            circle = patches.Circle(joint, 0.05, fill=True, color='orange')
            ax.add_patch(circle)
        
        return ax
    
    def visualize_dance_moves(self, song_name, sequence_index=0, output_path=None):
        """Create an animation of dance moves for a specific song"""
        # Load reference data for the song
        ref_data = self.data_manager.load_reference_data(song_name)
        
        if not ref_data:
            print(f"No reference data found for {song_name}")
            return None
        
        if sequence_index >= len(ref_data):
            print(f"Sequence index {sequence_index} out of range. Available sequences: {len(ref_data)}")
            return None
        
        sequence = ref_data[sequence_index]
        frames = len(sequence)
        
        # Create animation
        fig, ax = plt.subplots(figsize=(10, 8))
        
        def update(frame):
            ax.clear()
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.set_title(f'{song_name} - Frame {frame+1}/{frames}')
            
            angles = sequence[frame]
            self.create_skeleton_frame(angles, ax, frame, frames)
            
            return ax,
        
        ani = FuncAnimation(fig, update, frames=frames, blit=False, repeat=True)
        
        # Save to file if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            ani.save(output_path, writer='pillow', fps=10)
            print(f"Animation saved to {output_path}")
        
        plt.close()
        return ani
    
    def create_dance_move_sheet(self, song_name, output_path=None):
        """Create a sheet with key poses from the dance"""
        ref_data = self.data_manager.load_reference_data(song_name)
        
        if not ref_data:
            print(f"No reference data found for {song_name}")
            return None
        
        sequence = ref_data[0]  # Use the first sequence
        frames = len(sequence)
        
        # Select key frames to display
        key_frames = [0, frames//4, frames//2, 3*frames//4, frames-1]
        
        fig, axes = plt.subplots(1, len(key_frames), figsize=(15, 5))
        if len(key_frames) == 1:
            axes = [axes]
        
        fig.suptitle(f'Key Dance Poses for {song_name}', fontsize=16)
        
        for i, frame_idx in enumerate(key_frames):
            ax = axes[i]
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.set_title(f'Frame {frame_idx+1}')
            
            angles = sequence[frame_idx]
            self.create_skeleton_frame(angles, ax, frame_idx, frames)
        
        plt.tight_layout()
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path)
            print(f"Dance move sheet saved to {output_path}")
        
        return fig
    
class EnhancedDanceMoveVisualizer(DanceMoveVisualizer):
    def create_skeleton_frame_with_position(self, pose_data, ax, frame_num=0, total_frames=1):
        """Create skeleton frame with positional information"""
        angles = pose_data['angles']
        body_center = pose_data['body_center']
        body_scale = pose_data['body_scale']
        
        # Set the coordinate system based on body position and scale
        ax.set_xlim(body_center[0] - body_scale[0]*2, body_center[0] + body_scale[0]*2)
        ax.set_ylim(body_center[1] - body_scale[1]*2, body_center[1] + body_scale[1]*2)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title(f'Dance Move - Frame {frame_num+1}/{total_frames}')
        
        # Draw the skeleton using the enhanced data
        # (Implementation would be similar to before but using the enhanced data)
        
        return ax
    
    def visualize_spatial_movement(self, song_name, sequence_index=0):
        """Visualize how the dancer moves through space"""
        ref_data = self.data_manager.load_enhanced_reference_data(song_name)
        
        if not ref_data:
            print(f"No enhanced reference data found for {song_name}")
            return None
        
        sequence = ref_data[sequence_index]
        
        # Plot the trajectory of body center
        body_centers = [frame['body_center'] for frame in sequence]
        body_centers = np.array(body_centers)
        
        plt.figure(figsize=(10, 8))
        plt.plot(body_centers[:, 0], body_centers[:, 1], 'b-', alpha=0.5, label='Body trajectory')
        plt.scatter(body_centers[:, 0], body_centers[:, 1], c=range(len(body_centers)), 
                   cmap='viridis', s=20, label='Frame position')
        plt.colorbar(label='Frame number')
        
        # Mark start and end points
        plt.scatter(body_centers[0, 0], body_centers[0, 1], c='green', s=100, marker='o', label='Start')
        plt.scatter(body_centers[-1, 0], body_centers[-1, 1], c='red', s=100, marker='s', label='End')
        
        plt.title(f'Spatial Movement for {song_name}')
        plt.xlabel('X position (normalized)')
        plt.ylabel('Y position (normalized)')
        plt.legend()
        plt.grid(True)
        plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
        
        return plt