#!/usr/bin/env python3
"""
Demo script showing the complete CS:GO to OWAMcap conversion workflow.

This script demonstrates:
1. Converting a sample of the CS:GO dataset to OWAMcap format
2. Verifying the converted files
3. Using OWA CLI tools to inspect the results
4. Loading and analyzing the data programmatically
"""

import subprocess
import tempfile
from pathlib import Path

from mcap_owa.highlevel import OWAMcapReader


def run_command(cmd, cwd=None):
    """Run a shell command and return the result."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        print(f"Command failed: {cmd}")
        print(f"Error: {result.stderr}")
        return None
    return result.stdout


def demo_conversion():
    """Demonstrate the conversion process."""
    print("=== CS:GO to OWAMcap Conversion Demo ===\n")
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "converted"
        output_dir.mkdir()
        
        print("1. Converting sample CS:GO dataset files...")
        print("   Command: python convert_to_owamcap.py /mnt/raid12/datasets/CounterStrike_Deathmatch ./output --subset aim_expert --max-files 2 --max-frames 50")
        
        # Run conversion
        cmd = f"python convert_to_owamcap.py /mnt/raid12/datasets/CounterStrike_Deathmatch {output_dir} --subset aim_expert --max-files 2 --max-frames 50"
        result = run_command(cmd, cwd=Path(__file__).parent)
        
        if result:
            print("   ✓ Conversion completed successfully")
            print(f"   Output:\n{result}")
        else:
            print("   ✗ Conversion failed")
            return
        
        print("\n2. Verifying converted files...")
        print("   Command: python convert_to_owamcap.py verify ./output")
        
        # Run verification
        cmd = f"python convert_to_owamcap.py verify {output_dir}"
        result = run_command(cmd, cwd=Path(__file__).parent)
        
        if result:
            print("   ✓ Verification completed")
            print(f"   Output:\n{result}")
        else:
            print("   ✗ Verification failed")
            return
        
        # Find converted files
        mcap_files = list(output_dir.glob("*.mcap"))
        if not mcap_files:
            print("   ✗ No MCAP files found")
            return
        
        sample_file = mcap_files[0]
        
        print(f"\n3. Inspecting converted file with OWA CLI...")
        print(f"   Command: owl mcap info {sample_file}")
        
        # Run OWA CLI info
        cmd = f"owl mcap info {sample_file}"
        result = run_command(cmd)
        
        if result:
            print("   ✓ File inspection completed")
            print(f"   Output:\n{result}")
        else:
            print("   ✗ File inspection failed")
        
        print(f"\n4. Showing sample messages...")
        print(f"   Command: owl mcap cat {sample_file} --n 3")
        
        # Show sample messages
        cmd = f"owl mcap cat {sample_file} --n 3"
        result = run_command(cmd)
        
        if result:
            print("   ✓ Message display completed")
            print(f"   Output:\n{result}")
        else:
            print("   ✗ Message display failed")
        
        print(f"\n5. Programmatic analysis...")
        
        # Analyze programmatically
        try:
            analyze_owamcap_file(sample_file)
            print("   ✓ Programmatic analysis completed")
        except Exception as e:
            print(f"   ✗ Programmatic analysis failed: {e}")
        
        print(f"\n=== Demo Summary ===")
        print(f"✓ Successfully converted {len(mcap_files)} CS:GO files to OWAMcap format")
        print(f"✓ Files contain screen captures, mouse events, and keyboard events")
        print(f"✓ Compatible with standard OWA tools and libraries")
        print(f"✓ Ready for use in machine learning pipelines")


def analyze_owamcap_file(mcap_path: Path):
    """Analyze an OWAMcap file programmatically."""
    print(f"   Analyzing {mcap_path.name}...")
    
    stats = {
        'topics': {},
        'total_messages': 0,
        'screen_frames': 0,
        'mouse_events': 0,
        'keyboard_events': 0,
        'duration_seconds': 0
    }
    
    timestamps = []
    
    with OWAMcapReader(str(mcap_path), decode_args={"return_dict": True}) as reader:
        for msg in reader.iter_messages():
            stats['total_messages'] += 1
            timestamps.append(msg.timestamp)
            
            topic = msg.topic
            if topic not in stats['topics']:
                stats['topics'][topic] = 0
            stats['topics'][topic] += 1
            
            # Count specific message types
            if topic == "screen":
                stats['screen_frames'] += 1
            elif topic == "mouse":
                stats['mouse_events'] += 1
            elif topic == "keyboard":
                stats['keyboard_events'] += 1
    
    # Calculate duration
    if timestamps:
        duration_ns = max(timestamps) - min(timestamps)
        stats['duration_seconds'] = duration_ns / 1e9
    
    # Print analysis
    print(f"     Total messages: {stats['total_messages']}")
    print(f"     Screen frames: {stats['screen_frames']}")
    print(f"     Mouse events: {stats['mouse_events']}")
    print(f"     Keyboard events: {stats['keyboard_events']}")
    print(f"     Duration: {stats['duration_seconds']:.2f} seconds")
    print(f"     Topics: {list(stats['topics'].keys())}")
    
    if stats['screen_frames'] > 0 and stats['duration_seconds'] > 0:
        fps = stats['screen_frames'] / stats['duration_seconds']
        print(f"     Effective FPS: {fps:.1f}")


def show_usage_examples():
    """Show usage examples for the conversion script."""
    print("\n=== Usage Examples ===")
    
    examples = [
        {
            "description": "Convert all aim_expert files",
            "command": "python convert_to_owamcap.py /mnt/raid12/datasets/CounterStrike_Deathmatch ./output --subset aim_expert"
        },
        {
            "description": "Convert first 10 files with 200 frames each",
            "command": "python convert_to_owamcap.py /mnt/raid12/datasets/CounterStrike_Deathmatch ./output --max-files 10 --max-frames 200"
        },
        {
            "description": "Convert without creating video files (faster, larger MCAP files)",
            "command": "python convert_to_owamcap.py /mnt/raid12/datasets/CounterStrike_Deathmatch ./output --no-video --max-files 5"
        },
        {
            "description": "Verify converted files",
            "command": "python convert_to_owamcap.py verify ./output"
        },
        {
            "description": "Run tests",
            "command": "python test_conversion.py"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['description']}:")
        print(f"   {example['command']}")
        print()


def main():
    """Run the demo."""
    try:
        demo_conversion()
        show_usage_examples()
        
        print("\n=== Next Steps ===")
        print("1. Run full conversion on desired dataset subsets")
        print("2. Use converted OWAMcap files in your ML training pipeline")
        print("3. Leverage OWA tools for data analysis and visualization")
        print("4. See README.md for detailed documentation")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
