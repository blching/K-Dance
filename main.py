import os
import sys
from DanceCoachApp import DanceCoachApp

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(title):
    """Print a formatted header"""
    #clear_screen()
    print("=" * 60)
    print(f"K-POP DANCE COACH - {title}")
    print("=" * 60)

def wait_for_input():
    """Wait for user input to continue"""
    input("\nPress Enter to continue...")

def main_menu():
    """Main menu for the application"""
    app = DanceCoachApp()
    
    while True:
        print_header("MAIN MENU")
        print("1. View Available Songs")
        print("2. Dance to a Song Chorus (Webcam)")
        print("3. Dance to a Song Chorus (Video File)")
        print("4. Add Professional Training Video")
        print("5. Batch Add Professional Videos")
        print("6. Train Model for a Song")
        print("7. Exit")
        
        choice = input("\nEnter your choice (1-7): ")
        
        if choice == '1':
            view_songs_menu(app)
        elif choice == '2':
            dance_to_chorus_menu(app, use_webcam=True)
        elif choice == '3':
            dance_to_chorus_menu(app, use_webcam=False)
        elif choice == '4':
            add_professional_video_menu(app)
        elif choice == '5':
            batch_add_videos_menu(app)
        elif choice == '6':
            train_model_menu(app)
        elif choice == '7':
            print("Thank you for using K-Pop Dance Coach!")
            break
        else:
            print("Invalid choice. Please try again.")
            wait_for_input()

def dance_to_chorus_menu(app, use_webcam=True):
    """Menu for dancing to a song chorus with webcam or video"""
    print_header("DANCE TO SONG CHORUS" + (" (WEBCAM)" if use_webcam else " (VIDEO FILE)"))
    
    songs = app.get_available_songs()
    if not songs:
        print("No songs available. Please add professional videos first.")
        wait_for_input()
        return
    
    print("Available songs:")
    for i, song in enumerate(songs, 1):
        print(f"{i}. {song}")
    
    try:
        choice = int(input("\nSelect a song (number): "))
        if 1 <= choice <= len(songs):
            song_name = songs[choice-1]
            
            # Check if model is trained
            status = app.get_song_status(song_name)
            if not status['has_trained_model']:
                print(f"No model trained for {song_name}. Would you like to train one now? (y/n)")
                train_choice = input().lower()
                if train_choice == 'y':
                    print("Training model...")
                    try:
                        app.select_song(song_name)
                    except Exception as e:
                        print(f"Error training model: {e}")
                        wait_for_input()
                        return
                else:
                    print("Cannot dance to a song without a trained model.")
                    wait_for_input()
                    return
            
            # Ask for audio source and chorus timing
            audio_source = None
            if use_webcam:
                audio_source = input("Enter the path to the audio/video file (or press Enter to use training data): ")
                if audio_source and not os.path.exists(audio_source):
                    print("File not found. Continuing without audio.")
                    audio_source = None
            else:
                # For video sessions, we'll use the video itself for audio
                pass
            
            chorus_start = 0
            chorus_end = 30  # Default 30-second chorus
            
            try:
                chorus_start = float(input("Enter chorus start time (seconds): "))
                chorus_end = float(input("Enter chorus end time (seconds): "))
            except ValueError:
                print("Invalid time format. Using default chorus timing (0-30s).")
            
            # Start the session
            if use_webcam:
                # For webcam, we need to provide audio source
                app.select_song(song_name, audio_source, chorus_start, chorus_end)
                print("Starting webcam chorus session...")
                app.start_webcam_session(chorus_start, chorus_end)
            else:
                # For video, we'll get the video path later
                video_path = input("Enter the path to your dance video: ")
                if os.path.exists(video_path):
                    # For video sessions, we'll pass the video path to select_song
                    # so it can extract audio from it
                    app.select_song(song_name, video_path, chorus_start, chorus_end)
                    print("Starting video chorus session...")
                    app.start_video_session(video_path, chorus_start, chorus_end)
                else:
                    print("Video file not found.")
                    wait_for_input()
        else:
            print("Invalid selection.")
            wait_for_input()
    except ValueError:
        print("Please enter a valid number.")
        wait_for_input()

def view_songs_menu(app):
    """Display available songs and their status"""
    print_header("AVAILABLE SONGS")
    
    songs = app.get_available_songs()
    if not songs:
        print("No songs available. Please add professional videos first.")
        wait_for_input()
        return
    
    for i, song in enumerate(songs, 1):
        status = app.get_song_status(song)
        print(f"{i}. {song}")
        print(f"   Reference Videos: {status['reference_count']}")
        print(f"   User Sessions: {status['user_count']}")
        print(f"   Model Trained: {'Yes' if status['has_trained_model'] else 'No'}")
        print()
    
    wait_for_input()

def dance_to_song_menu(app, use_webcam=True, with_audio=False):
    """Menu for dancing to a song with audio options"""
    print_header("DANCE TO A SONG" + (" WITH AUDIO" if with_audio else ""))
    
    songs = app.get_available_songs()
    if not songs:
        print("No songs available. Please add professional videos first.")
        wait_for_input()
        return
    
    print("Available songs:")
    for i, song in enumerate(songs, 1):
        print(f"{i}. {song}")
    
    try:
        choice = int(input("\nSelect a song (number): "))
        if 1 <= choice <= len(songs):
            song_name = songs[choice-1]
            
            # Check if model is trained
            status = app.get_song_status(song_name)
            if not status['has_trained_model']:
                print(f"No model trained for {song_name}. Would you like to train one now? (y/n)")
                train_choice = input().lower()
                if train_choice == 'y':
                    print("Training model...")
                    try:
                        app.select_song(song_name)
                    except Exception as e:
                        print(f"Error training model: {e}")
                        wait_for_input()
                        return
                else:
                    print("Cannot dance to a song without a trained model.")
                    wait_for_input()
                    return
            
            # Ask for audio file if audio synchronization is requested
            audio_path = None
            if with_audio:
                audio_path = input("Enter the path to the audio file (or press Enter to skip): ")
                if audio_path and not os.path.exists(audio_path):
                    print("Audio file not found. Continuing without audio.")
                    audio_path = None
            
            # Start the session
            app.select_song(song_name, audio_path)
            if use_webcam:
                print("Starting webcam session..." + (" Audio synchronized." if with_audio else ""))
                app.start_session(0, with_audio)  # 0 for webcam
            else:
                video_path = input("Enter the path to your dance video: ")
                if os.path.exists(video_path):
                    print("Processing video...")
                    app.start_session(video_path, with_audio)
                else:
                    print("Video file not found.")
                    wait_for_input()
        else:
            print("Invalid selection.")
            wait_for_input()
    except ValueError:
        print("Please enter a valid number.")
        wait_for_input()

def add_professional_video_menu(app):
    """Menu for adding a professional training video"""
    print_header("ADD PROFESSIONAL TRAINING VIDEO")
    
    song_name = input("Enter the song name: ")
    video_path = input("Enter the path to the professional video: ")
    
    if not os.path.exists(video_path):
        print("Video file not found.")
        wait_for_input()
        return
    
    show_preview = input("Show preview during processing? (y/n): ").lower() == 'y'
    
    print("Processing video...")
    success = app.add_professional_video(video_path, song_name, show_preview)
    
    if success:
        print("Video processed successfully!")
        
        # Ask if user wants to train the model
        train_now = input("Would you like to train the model now? (y/n): ").lower() == 'y'
        if train_now:
            print("Training model...")
            try:
                app.select_song(song_name)
                print("Model trained successfully!")
            except Exception as e:
                print(f"Error training model: {e}")
    else:
        print("Failed to process video.")
    
    wait_for_input()

def batch_add_videos_menu(app):
    """Menu for batch adding professional videos"""
    print_header("BATCH ADD PROFESSIONAL VIDEOS")
    
    song_name = input("Enter the song name: ")
    video_dir = input("Enter the path to the directory containing videos: ")
    
    if not os.path.exists(video_dir):
        print("Directory not found.")
        wait_for_input()
        return
    
    show_preview = input("Show preview during processing? (y/n): ").lower() == 'y'
    count = app.batch_add_professional_videos(video_dir, song_name, show_preview)
    print(f"Processed {count} videos.")
    
    if count > 0:
        # Ask if user wants to train the model
        train_now = input("Would you like to train the model now? (y/n): ").lower() == 'y'
        if train_now:
            print("Training model...")
            try:
                app.select_song(song_name)
                print("Model trained successfully!")
            except Exception as e:
                print(f"Error training model: {e}")
    
    wait_for_input()

def train_model_menu(app):
    """Menu for training a model for a song"""
    print_header("TRAIN MODEL")
    
    songs = app.get_available_songs()
    if not songs:
        print("No songs available. Please add professional videos first.")
        wait_for_input()
        return
    
    print("Available songs:")
    for i, song in enumerate(songs, 1):
        status = app.get_song_status(song)
        print(f"{i}. {song} (References: {status['reference_count']})")
    
    try:
        choice = int(input("\nSelect a song to train (number): "))
        if 1 <= choice <= len(songs):
            song_name = songs[choice-1]
            
            # Check if we have reference data
            status = app.get_song_status(song_name)
            if status['reference_count'] == 0:
                print("No reference data available for this song.")
                wait_for_input()
                return
            
            print("Training model...")
            try:
                app.select_song(song_name)
                print("Model trained successfully!")
            except Exception as e:
                print(f"Error training model: {e}")
        else:
            print("Invalid selection.")
    except ValueError:
        print("Please enter a valid number.")
    
    wait_for_input()

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\nExiting K-Pop Dance Coach...")
        sys.exit(0)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


''' if __name__ == "__main__":
    app = DanceCoachApp()
    
    # Example: Add professional videos for training
    # app.add_professional_video("magnetic_training.mp4", "Magnetic", show_preview=True)
    # app.batch_add_professional_videos("pro_videos/dynamite", "Dynamite")
    
    # Check available songs
    songs = app.get_available_songs()
    print("Available songs:", songs)
    
    if songs:
        # Select the first available song
        song_name = songs[0]
        print(f"Selecting song: {song_name}")
        
        # Check song status
        status = app.get_song_status(song_name)
        print("Song status:", status)
        
        # Train if needed
        if not status["has_trained_model"] and status["has_reference_data"]:
            print("Training model...")
            app.select_song(song_name)
        
        # Start session with webcam
        #app.start_session(video_source=0)
        app.start_session("magnetictest.mp4")
    else:
        print("No songs available. Please add professional videos first.")
        '''