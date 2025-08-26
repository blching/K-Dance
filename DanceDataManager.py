import json
import os
import time


class DanceDataManager:
    def __init__(self, data_dir="dance_data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def save_session(self, session_data, song_name, performer_type="user"):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{song_name}_{performer_type}_{timestamp}.json"
        path = os.path.join(self.data_dir, filename)
        
        with open(path, "w") as f:
            json.dump(session_data, f, indent=2)
        return path
    
    def load_reference_data(self, song_name):
        """Load professional performance data for a song"""
        ref_data = []
        for file in os.listdir(self.data_dir):
            if song_name in file and "pro" in file:
                with open(os.path.join(self.data_dir, file), "r") as f:
                    ref_data.append(json.load(f))
        return ref_data
    
    def load_user_data(self, song_name):
        """Load user performance data for a song"""
        user_data = []
        for file in os.listdir(self.data_dir):
            if song_name in file and "user" in file:
                with open(os.path.join(self.data_dir, file), "r") as f:
                    user_data.append(json.load(f))
        return user_data
    
    def get_all_songs(self):
        """Get list of all available songs in the dataset"""
        songs = set()
        for file in os.listdir(self.data_dir):
            if file.endswith(".json"):
                # Extract song name from filename (format: songname_type_timestamp.json)
                song_name = file.split("_")[0]
                songs.add(song_name)
        return list(songs)
    
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

def load_enhanced_user_data(self, song_name):
    """Load enhanced user performance data"""
    user_data = []
    for file in os.listdir(self.data_dir):
        if song_name in file and "user" in file and "enhanced" in file:
            with open(os.path.join(self.data_dir, file), "r") as f:
                user_data.append(json.load(f))
    return user_data