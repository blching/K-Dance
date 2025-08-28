import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import os
import json
import mediapipe as mp
    
class EnhancedDanceMoveVisualizer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.POSE_CONNECTIONS = self.mp_pose.POSE_CONNECTIONS
        
    def create_skeleton_frame_enhanced(self, pose_data, ax, frame_num=0, total_frames=1):
        """Create a skeleton frame using enhanced data that includes positional information"""
        ax.clear()
        
        # Extract data from enhanced pose data
        angles = pose_data['angles']
        body_center = pose_data['body_center']
        body_scale = pose_data['body_scale']
        body_orientation = pose_data['body_orientation']
        limb_lengths = pose_data['limb_lengths']
        raw_positions = pose_data['raw_positions']
        
        # Set the coordinate system based on body position and scale
        ax.set_xlim(body_center[0] - body_scale[0]*2, body_center[0] + body_scale[0]*2)
        ax.set_ylim(body_center[1] - body_scale[1]*2, body_center[1] + body_scale[1]*2)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title(f'Dance Move - Frame {frame_num+1}/{total_frames}\n'
                    f'Body Center: ({body_center[0]:.2f}, {body_center[1]:.2f})\n'
                    f'Orientation: {np.degrees(body_orientation):.1f}°')
        
        # Draw body center
        center_circle = patches.Circle(body_center, 0.05, fill=True, color='red', alpha=0.7)
        ax.add_patch(center_circle)
        
        # Draw orientation arrow
        arrow_length = body_scale[0] * 0.8
        dx = arrow_length * np.cos(body_orientation)
        dy = arrow_length * np.sin(body_orientation)
        ax.arrow(body_center[0], body_center[1], dx, dy, 
                head_width=0.05, head_length=0.05, fc='red', ec='red', alpha=0.7)
        
        # Draw skeleton using raw positions
        self.draw_skeleton_from_raw_positions(raw_positions, ax)
        
        # Draw limb length indicators
        self.draw_limb_length_indicators(raw_positions, limb_lengths, ax)
        
        # Draw joint angles
        self.draw_joint_angles(raw_positions, angles, ax)
        
        return ax
    
    def draw_skeleton_from_raw_positions(self, raw_positions, ax):
        """Draw the skeleton using raw landmark positions"""
        # Define colors for different body parts
        colors = {
            'left_arm': 'red',
            'right_arm': 'blue',
            'left_leg': 'green',
            'right_leg': 'purple',
            'torso': 'black',
            'head': 'skyblue'
        }
        
        # Draw connections
        for connection in self.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            start_landmark = self.mp_pose.PoseLandmark(start_idx)
            end_landmark = self.mp_pose.PoseLandmark(end_idx)
            
            if start_landmark.name in raw_positions and end_landmark.name in raw_positions:
                start_pos = raw_positions[start_landmark.name]
                end_pos = raw_positions[end_landmark.name]
                
                # Determine color based on body part
                if start_idx in [11, 13, 15] and end_idx in [11, 13, 15]:
                    color = colors['left_arm']
                elif start_idx in [12, 14, 16] and end_idx in [12, 14, 16]:
                    color = colors['right_arm']
                elif start_idx in [23, 25, 27, 29, 31] and end_idx in [23, 25, 27, 29, 31]:
                    color = colors['left_leg']
                elif start_idx in [24, 26, 28, 30, 32] and end_idx in [24, 26, 28, 30, 32]:
                    color = colors['right_leg']
                elif start_idx in [11, 12, 23, 24] and end_idx in [11, 12, 23, 24]:
                    color = colors['torso']
                else:
                    color = 'gray'
                
                # Draw the connection
                ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                    color=color, linewidth=3, alpha=0.8)
        
        # Draw landmarks
        for landmark_name, position in raw_positions.items():
            circle = patches.Circle((position[0], position[1]), 0.02, 
                                fill=True, color='orange', alpha=0.8)
            ax.add_patch(circle)
            
            # Add landmark label
            ax.text(position[0] + 0.03, position[1] + 0.03, 
                landmark_name.split('_')[-1], fontsize=8, alpha=0.7)
    
    def draw_limb_length_indicators(self, raw_positions, limb_lengths, ax):
        """Draw indicators showing limb lengths"""
        # Define limb endpoints
        limb_endpoints = {
            'left_arm': ['LEFT_SHOULDER', 'LEFT_WRIST'],
            'right_arm': ['RIGHT_SHOULDER', 'RIGHT_WRIST'],
            'left_leg': ['LEFT_HIP', 'LEFT_ANKLE'],
            'right_leg': ['RIGHT_HIP', 'RIGHT_ANKLE']
        }
        
        for limb_name, endpoints in limb_endpoints.items():
            if endpoints[0] in raw_positions and endpoints[1] in raw_positions:
                start_pos = raw_positions[endpoints[0]]
                end_pos = raw_positions[endpoints[1]]
                
                # Calculate midpoint for text
                mid_x = (start_pos[0] + end_pos[0]) / 2
                mid_y = (start_pos[1] + end_pos[1]) / 2
                
                # Add length text
                length = limb_lengths.get(limb_name, 0)
                ax.text(mid_x, mid_y, f'{length:.2f}', 
                    fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
    
    def draw_joint_angles(self, raw_positions, angles, ax):
        """Draw joint angles on the skeleton"""
        # Map angles to joints
        angle_mapping = [
            ('LEFT_ELBOW', 'LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST'),
            ('RIGHT_ELBOW', 'RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST'),
            ('LEFT_SHOULDER', 'LEFT_HIP', 'LEFT_SHOULDER', 'LEFT_ELBOW'),
            ('RIGHT_SHOULDER', 'RIGHT_HIP', 'RIGHT_SHOULDER', 'RIGHT_ELBOW'),
            ('LEFT_KNEE', 'LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE'),
            ('RIGHT_KNEE', 'RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE'),
            ('LEFT_HIP', 'LEFT_SHOULDER', 'LEFT_HIP', 'LEFT_KNEE'),
            ('RIGHT_HIP', 'RIGHT_SHOULDER', 'RIGHT_HIP', 'RIGHT_KNEE')
        ]
        
        for i, (angle_name, a, b, c) in enumerate(angle_mapping):
            if a in raw_positions and b in raw_positions and c in raw_positions:
                point_a = raw_positions[a]
                point_b = raw_positions[b]
                point_c = raw_positions[c]
                
                # Calculate angle position (slightly offset from joint)
                offset = 0.1
                text_x = point_b[0] + offset
                text_y = point_b[1] + offset
                
                # Add angle text
                ax.text(text_x, text_y, f'{angles[i]:.1f}°', 
                    fontsize=8, bbox=dict(facecolor='yellow', alpha=0.7))
                
                # Draw angle arc
                self.draw_angle_arc(point_a, point_b, point_c, angles[i], ax)
    
    def draw_angle_arc(self, point_a, point_b, point_c, angle, ax):
        """Draw an arc representing the joint angle"""
        # Convert points to numpy arrays
        a = np.array(point_a)
        b = np.array(point_b)
        c = np.array(point_c)
        
        # Calculate vectors
        ba = a - b
        bc = c - b
        
        # Calculate angle between vectors
        angle_rad = np.arctan2(bc[1], bc[0]) - np.arctan2(ba[1], ba[0])
        
        # Draw arc
        arc_radius = 0.1
        arc = patches.Arc(b, arc_radius, arc_radius, angle=0, 
                        theta1=np.degrees(np.arctan2(ba[1], ba[0])), 
                        theta2=np.degrees(np.arctan2(bc[1], bc[0])),
                        color='red', linewidth=2, alpha=0.7)
        ax.add_patch(arc)
    
    def visualize_enhanced_dance_moves(self, enhanced_data, output_path=None):
        """Create an animation of dance moves using enhanced data"""
        frames = len(enhanced_data)
        
        # Create animation
        fig, ax = plt.subplots(figsize=(12, 10))
        
        def update(frame):
            ax.clear()
            pose_data = enhanced_data[frame]
            self.create_skeleton_frame_enhanced(pose_data, ax, frame, frames)
            return ax,
        
        ani = FuncAnimation(fig, update, frames=frames, blit=False, repeat=True)
        
        # Save to file if output path is provided
        if output_path:
            import os
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            ani.save(output_path, writer='pillow', fps=10)
            print(f"Enhanced animation saved to {output_path}")
        
        plt.close()
        return ani
    
    def create_enhanced_dance_move_sheet(self, enhanced_data, output_path=None):
        """Create a sheet with key poses from the enhanced dance data"""
        frames = len(enhanced_data)
        
        # Select key frames to display
        key_frames = [0, frames//4, frames//2, 3*frames//4, frames-1]
        
        fig, axes = plt.subplots(1, len(key_frames), figsize=(20, 5))
        if len(key_frames) == 1:
            axes = [axes]
        
        fig.suptitle('Key Dance Poses with Enhanced Data', fontsize=16)
        
        for i, frame_idx in enumerate(key_frames):
            ax = axes[i]
            pose_data = enhanced_data[frame_idx]
            self.create_skeleton_frame_enhanced(pose_data, ax, frame_idx, frames)
        
        plt.tight_layout()
        
        if output_path:
            import os
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Enhanced dance move sheet saved to {output_path}")
        
        return fig
    
    def visualize_spatial_movement(self, enhanced_data, output_path=None):
        """Visualize how the dancer moves through space using enhanced data"""
        # Extract body centers and orientations
        body_centers = [frame['body_center'] for frame in enhanced_data]
        body_orientations = [frame['body_orientation'] for frame in enhanced_data]
        
        body_centers = np.array(body_centers)
        
        plt.figure(figsize=(12, 10))
        
        # Plot trajectory
        plt.plot(body_centers[:, 0], body_centers[:, 1], 'b-', alpha=0.5, label='Body trajectory')
        
        # Plot points with color indicating frame number
        scatter = plt.scatter(body_centers[:, 0], body_centers[:, 1], 
                            c=range(len(body_centers)), cmap='viridis', 
                            s=30, label='Body position')
        plt.colorbar(scatter, label='Frame number')
        
        # Add orientation arrows at regular intervals
        arrow_interval = max(1, len(body_centers) // 10)
        for i in range(0, len(body_centers), arrow_interval):
            center = body_centers[i]
            orientation = body_orientations[i]
            dx = 0.2 * np.cos(orientation)
            dy = 0.2 * np.sin(orientation)
            plt.arrow(center[0], center[1], dx, dy, 
                    head_width=0.05, head_length=0.05, fc='red', ec='red', alpha=0.7)
        
        # Mark start and end points
        plt.scatter(body_centers[0, 0], body_centers[0, 1], c='green', s=100, marker='o', label='Start')
        plt.scatter(body_centers[-1, 0], body_centers[-1, 1], c='red', s=100, marker='s', label='End')
        
        plt.title('Spatial Movement Analysis')
        plt.xlabel('X position (normalized)')
        plt.ylabel('Y position (normalized)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
        
        if output_path:
            import os
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Spatial movement visualization saved to {output_path}")
        
        return plt