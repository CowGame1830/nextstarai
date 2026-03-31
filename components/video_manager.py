"""
Video management module - handles video selection and retrieval
"""
import os


def get_available_videos(input_dir='input_videos'):
    """Get list of available video files from input directory.
    
    Supports: .mp4, .avi, .mov, .mkv, .flv, .wmv
    Returns list of video file names sorted alphabetically.
    """
    if not os.path.exists(input_dir):
        return []
    
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    videos = []
    
    for file in os.listdir(input_dir):
        if os.path.isfile(os.path.join(input_dir, file)):
            if os.path.splitext(file)[1].lower() in video_extensions:
                videos.append(file)
    
    return sorted(videos)


def select_videos_interactive(input_dir='input_videos'):
    """Interactively select one or more videos from available options.
    
    Returns list of selected video file names (full paths).
    """
    videos = get_available_videos(input_dir)
    
    if not videos:
        print(f"No videos found in {input_dir}/")
        return []
    
    print("\n" + "="*60)
    print("Available Videos:")
    print("="*60)
    for i, video in enumerate(videos, 1):
        video_path = os.path.join(input_dir, video)
        file_size = os.path.getsize(video_path) / (1024*1024)  # Convert to MB
        print(f"{i}. {video:<40} ({file_size:.1f} MB)")
    
    print("="*60)
    print("\nEnter video numbers to process (space or comma-separated):")
    print("Examples: 1 2 3  or  1,2,3  or  1")
    
    while True:
        try:
            user_input = input("Selection: ").strip()
            if not user_input:
                print("Please select at least one video.")
                continue
            
            # Parse input (support both space and comma separated)
            selections = []
            for token in user_input.replace(',', ' ').split():
                token = token.strip()
                if token.isdigit():
                    num = int(token)
                    if 1 <= num <= len(videos):
                        selections.append(num)
                    else:
                        print(f"Invalid selection: {num} (must be between 1 and {len(videos)})")
                        break
                else:
                    print(f"Invalid input: {token}")
                    break
            else:
                if selections:
                    selected_videos = [os.path.join(input_dir, videos[i-1]) for i in sorted(set(selections))]
                    print(f"\nSelected {len(selected_videos)} video(s):")
                    for video in selected_videos:
                        print(f"  - {video}")
                    return selected_videos
        except Exception as e:
            print(f"Error: {e}")
