import json
import os
import time
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DanceDataManager:
    def __init__(self, data_dir="dance_data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def save_session(self, session_data, song_name, performer_type="user"):
        """Save session data with error handling"""
        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"{song_name}_{performer_type}_{timestamp}.json"
            path = os.path.join(self.data_dir, filename)
            
            # Create a temporary file first to avoid corruption
            temp_path = path + ".tmp"
            with open(temp_path, "w") as f:
                json.dump(session_data, f, indent=2)
            
            # Rename to final location (atomic operation)
            os.rename(temp_path, path)
            
            logger.info(f"Successfully saved session data to {path}")
            return path
        except Exception as e:
            logger.error(f"Error saving session data: {e}")
            # Clean up temporary file if it exists
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return None
    
    def load_reference_data(self, song_name):
        """Load professional performance data with error handling"""
        ref_data = []
        try:
            for file in os.listdir(self.data_dir):
                if song_name in file and "pro" in file:
                    file_path = os.path.join(self.data_dir, file)
                    try:
                        with open(file_path, "r") as f:
                            # Check if file is empty
                            if os.path.getsize(file_path) == 0:
                                logger.warning(f"Skipping empty file: {file_path}")
                                continue
                                
                            data = json.load(f)
                            if data:  # Only add if data is not empty
                                ref_data.append(data)
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON in file {file_path}: {e}")
                    except Exception as e:
                        logger.error(f"Error reading file {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error accessing data directory: {e}")
        
        return ref_data
    
    def load_user_data(self, song_name):
        """Load user performance data with error handling"""
        user_data = []
        try:
            for file in os.listdir(self.data_dir):
                if song_name in file and "user" in file:
                    file_path = os.path.join(self.data_dir, file)
                    try:
                        with open(file_path, "r") as f:
                            # Check if file is empty
                            if os.path.getsize(file_path) == 0:
                                logger.warning(f"Skipping empty file: {file_path}")
                                continue
                                
                            data = json.load(f)
                            if data:  # Only add if data is not empty
                                user_data.append(data)
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON in file {file_path}: {e}")
                    except Exception as e:
                        logger.error(f"Error reading file {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error accessing data directory: {e}")
        
        return user_data
    
    def get_all_songs(self):
        """Get list of all available songs with error handling"""
        songs = set()
        try:
            for file in os.listdir(self.data_dir):
                if file.endswith(".json"):
                    try:
                        # Extract song name from filename (format: songname_type_timestamp.json)
                        parts = file.split("_")
                        if len(parts) >= 3:  # Ensure filename has expected format
                            song_name = parts[0]
                            songs.add(song_name)
                    except Exception as e:
                        logger.error(f"Error parsing filename {file}: {e}")
        except Exception as e:
            logger.error(f"Error accessing data directory: {e}")
        
        return list(songs)
    
    def cleanup_corrupted_files(self):
        """Remove any corrupted or empty JSON files"""
        cleaned_count = 0
        try:
            for file in os.listdir(self.data_dir):
                if file.endswith(".json"):
                    file_path = os.path.join(self.data_dir, file)
                    try:
                        # Check if file is empty
                        if os.path.getsize(file_path) == 0:
                            os.remove(file_path)
                            logger.info(f"Removed empty file: {file_path}")
                            cleaned_count += 1
                            continue
                            
                        # Try to parse the JSON to check if it's valid
                        with open(file_path, "r") as f:
                            json.load(f)
                    except json.JSONDecodeError:
                        # File contains invalid JSON, remove it
                        os.remove(file_path)
                        logger.info(f"Removed corrupted JSON file: {file_path}")
                        cleaned_count += 1
                    except Exception as e:
                        logger.error(f"Error checking file {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
        return cleaned_count
    
    def save_enhanced_session(self, session_data, song_name, performer_type="user"):
        """Save session data with enhanced pose information"""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{song_name}_{performer_type}_{timestamp}_enhanced.json"
        path = os.path.join(self.data_dir, filename)
        
        # Convert any numpy arrays to lists for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        with open(path, "w") as f:
            json.dump(convert_for_json(session_data), f, indent=2)
        return path
    
    def load_enhanced_reference_data(self, song_name):
        """Load enhanced professional performance data"""
        ref_data = []
        for file in os.listdir(self.data_dir):
            if song_name in file and "pro" in file and "enhanced" in file:
                with open(os.path.join(self.data_dir, file), "r") as f:
                    ref_data.append(json.load(f))
        return ref_data
    
    def save_enhanced_session(self, session_data, song_name, performer_type="user"):
        """Save session data with enhanced pose information"""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{song_name}_{performer_type}_{timestamp}_enhanced.json"
        path = os.path.join(self.data_dir, filename)
        
        # Convert any numpy arrays to lists for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        with open(path, "w") as f:
            json.dump(convert_for_json(session_data), f, indent=2)
        return path

    def load_enhanced_user_data(self, song_name):
        """Load enhanced user performance data"""
        user_data = []
        for file in os.listdir(self.data_dir):
            if song_name in file and "user" in file and "enhanced" in file:
                with open(os.path.join(self.data_dir, file), "r") as f:
                    user_data.append(json.load(f))
        return user_data