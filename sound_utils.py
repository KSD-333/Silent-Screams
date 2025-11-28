"""
Sound Utilities Module
----------------------
Cross-platform audio playback for alert notifications.
Supports custom WAV files and programmatic beep generation.

Features:
- Play custom WAV alert sounds
- Generate programmatic beep if no sound file available
- Cross-platform support (Windows, macOS, Linux)
- Fallback mechanisms for different audio backends
"""

import os
import sys
import numpy as np
from typing import Optional


def generate_beep_sound(filename: str = "alarm.wav",
                       frequency: int = 1000,
                       duration: float = 0.5,
                       sample_rate: int = 44100,
                       amplitude: float = 0.3):
    """
    Generate a simple beep sound and save as WAV file.
    
    Args:
        filename: Output filename
        frequency: Beep frequency in Hz (default: 1000 Hz)
        duration: Duration in seconds (default: 0.5s)
        sample_rate: Audio sample rate (default: 44100 Hz)
        amplitude: Volume (0.0 to 1.0, default: 0.3)
    """
    try:
        from scipy.io import wavfile
        
        # Generate time array
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Generate sine wave
        wave = amplitude * np.sin(2 * np.pi * frequency * t)
        
        # Add envelope to avoid clicks
        envelope = np.ones_like(wave)
        fade_samples = int(0.01 * sample_rate)  # 10ms fade
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
        
        wave = wave * envelope
        
        # Convert to 16-bit PCM
        wave_int16 = np.int16(wave * 32767)
        
        # Save as WAV
        wavfile.write(filename, sample_rate, wave_int16)
        
        print(f"Generated beep sound: {filename}")
        return True
    
    except Exception as e:
        print(f"Warning: Could not generate beep sound: {e}")
        return False


def play_alert_sound(sound_file: str = "alarm.wav"):
    """
    Play alert sound with cross-platform support.
    
    Tries multiple methods in order:
    1. Custom sound file (if exists)
    2. Generated beep sound
    3. System beep (platform-specific)
    
    Args:
        sound_file: Path to sound file (WAV format)
    """
    # Check if sound file exists
    if not os.path.exists(sound_file):
        print(f"Sound file not found: {sound_file}")
        print("Generating default beep sound...")
        generate_beep_sound(sound_file)
    
    # Try to play sound
    success = False
    
    # Method 1: Try simpleaudio (cross-platform, recommended)
    if not success:
        success = _play_with_simpleaudio(sound_file)
    
    # Method 2: Try playsound (cross-platform)
    if not success:
        success = _play_with_playsound(sound_file)
    
    # Method 3: Platform-specific fallbacks
    if not success:
        success = _play_with_platform_specific(sound_file)
    
    # Method 4: System beep as last resort
    if not success:
        _system_beep()


def _play_with_simpleaudio(sound_file: str) -> bool:
    """Try playing sound with simpleaudio library."""
    try:
        import simpleaudio as sa
        
        wave_obj = sa.WaveObject.from_wave_file(sound_file)
        play_obj = wave_obj.play()
        # Don't wait for completion to avoid blocking
        
        return True
    
    except ImportError:
        return False
    except Exception as e:
        print(f"simpleaudio error: {e}")
        return False


def _play_with_playsound(sound_file: str) -> bool:
    """Try playing sound with playsound library."""
    try:
        from playsound import playsound
        
        # Play in non-blocking mode if possible
        playsound(sound_file, block=False)
        
        return True
    
    except ImportError:
        return False
    except Exception as e:
        print(f"playsound error: {e}")
        return False


def _play_with_platform_specific(sound_file: str) -> bool:
    """Try platform-specific audio playback."""
    try:
        if sys.platform == 'win32':
            # Windows: use winsound
            import winsound
            winsound.PlaySound(sound_file, winsound.SND_FILENAME | winsound.SND_ASYNC)
            return True
        
        elif sys.platform == 'darwin':
            # macOS: use afplay
            import subprocess
            subprocess.Popen(['afplay', sound_file], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            return True
        
        elif sys.platform.startswith('linux'):
            # Linux: try multiple options
            import subprocess
            
            # Try aplay (ALSA)
            try:
                subprocess.Popen(['aplay', sound_file],
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
                return True
            except FileNotFoundError:
                pass
            
            # Try paplay (PulseAudio)
            try:
                subprocess.Popen(['paplay', sound_file],
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
                return True
            except FileNotFoundError:
                pass
        
        return False
    
    except Exception as e:
        print(f"Platform-specific playback error: {e}")
        return False


def _system_beep():
    """Play system beep as last resort."""
    try:
        if sys.platform == 'win32':
            import winsound
            winsound.Beep(1000, 500)  # 1000 Hz for 500ms
        else:
            # Unix-like systems
            print('\a')  # ASCII bell character
    
    except Exception as e:
        print(f"System beep error: {e}")


def test_audio():
    """Test audio playback functionality."""
    print("Testing audio playback...")
    print("-" * 50)
    
    # Generate test beep
    print("\n1. Generating test beep sound...")
    success = generate_beep_sound("test_beep.wav")
    
    if success:
        print("   ✓ Beep generated successfully")
        
        # Test playback
        print("\n2. Testing playback...")
        print("   Playing sound (you should hear a beep)...")
        play_alert_sound("test_beep.wav")
        
        print("   ✓ Playback initiated")
        
        # Cleanup
        import time
        time.sleep(1)  # Wait for sound to finish
        
        if os.path.exists("test_beep.wav"):
            os.remove("test_beep.wav")
            print("\n3. Cleanup complete")
    
    else:
        print("   ✗ Failed to generate beep")
    
    print("\n" + "-" * 50)
    print("Audio test complete!")


if __name__ == "__main__":
    """
    Test sound utilities.
    """
    test_audio()
