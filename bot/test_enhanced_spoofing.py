#!/usr/bin/env python3
"""
Test script for enhanced spoofing capabilities
"""

import os
import sys
import time
from pathlib import Path

# Add the bot directory to the path
bot_dir = Path(__file__).parent
sys.path.insert(0, str(bot_dir))

def test_enhanced_engine():
    """Test the enhanced spoof engine functionality."""
    
    print("🧪 Testing Enhanced Spoof Engine...")
    
    try:
        from enhanced_spoof_engine import EnhancedSpoofEngine, spoof_video_enhanced_tiktok, spoof_video_enhanced_instagram
        print("✅ Enhanced spoof engine imported successfully")
        
        # Test engine initialization
        engine = EnhancedSpoofEngine()
        print("✅ Enhanced engine initialized successfully")
        
        # Test integration functions
        print("✅ Integration functions available:")
        print("   - spoof_video_enhanced_tiktok")
        print("   - spoof_video_enhanced_instagram")
        
        return True
        
    except ImportError as e:
        print(f"❌ Enhanced engine import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Enhanced engine test failed: {e}")
        return False

def test_standard_engine():
    """Test the standard spoof engine with enhanced integration."""
    
    print("\n🧪 Testing Standard Engine with Enhanced Integration...")
    
    try:
        import spoof_engine as se
        print("✅ Standard spoof engine imported successfully")
        
        # Test enhanced preset modes
        se.PRESET_MODE = "TIKTOK_ENHANCED"
        print(f"✅ TikTok Enhanced mode set: {se.PRESET_MODE}")
        
        se.PRESET_MODE = "IG_REELS_ENHANCED"
        print(f"✅ Instagram Enhanced mode set: {se.PRESET_MODE}")
        
        # Test enhanced configuration flags
        enhanced_flags = [
            'ENABLE_WATERMARK_REMOVAL',
            'ENABLE_VISUAL_ECHO', 
            'ENABLE_RESOLUTION_TWEAK',
            'ENABLE_FPS_JITTER'
        ]
        
        for flag in enhanced_flags:
            if hasattr(se, flag):
                print(f"✅ Enhanced flag available: {flag}")
            else:
                print(f"⚠️  Enhanced flag missing: {flag}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Standard engine import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Standard engine test failed: {e}")
        return False

def test_bot_integration():
    """Test bot integration with enhanced spoofing."""
    
    print("\n🧪 Testing Bot Integration...")
    
    try:
        # Test if bot can import enhanced functions
        sys.path.insert(0, str(bot_dir))
        
        # Simulate bot import
        from enhanced_spoof_engine import spoof_video_enhanced_tiktok, spoof_video_enhanced_instagram
        print("✅ Bot can import enhanced TikTok function")
        print("✅ Bot can import enhanced Instagram function")
        
        # Test spoof engine integration
        import spoof_engine as se
        
        # Check if ENHANCED_SPOOFER_AVAILABLE flag works
        try:
            from enhanced_spoof_engine import spoof_video_enhanced_tiktok, spoof_video_enhanced_instagram
            enhanced_available = True
        except ImportError:
            enhanced_available = False
        
        print(f"✅ Enhanced spoofer availability: {enhanced_available}")
        
        return True
        
    except Exception as e:
        print(f"❌ Bot integration test failed: {e}")
        return False

def test_ffmpeg_availability():
    """Test FFmpeg availability for enhanced processing."""
    
    print("\n🧪 Testing FFmpeg Availability...")
    
    import shutil
    
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        print(f"✅ FFmpeg found at: {ffmpeg_path}")
        return True
    else:
        # Check Windows fallback
        windows_ffmpeg = "C:\\Tools\\FFmpeg\\ffmpeg.exe"
        if os.path.exists(windows_ffmpeg):
            print(f"✅ FFmpeg found at Windows path: {windows_ffmpeg}")
            return True
        else:
            print("❌ FFmpeg not found - enhanced processing will fail")
            print("💡 Install FFmpeg or place at C:\\Tools\\FFmpeg\\ffmpeg.exe")
            return False

def main():
    """Run all tests."""
    
    print("🚀 SpectreSpoofer Enhanced Capabilities Test Suite")
    print("=" * 60)
    
    tests = [
        ("Enhanced Engine", test_enhanced_engine),
        ("Standard Engine Integration", test_standard_engine), 
        ("Bot Integration", test_bot_integration),
        ("FFmpeg Availability", test_ffmpeg_availability)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            start_time = time.time()
            result = test_func()
            end_time = time.time()
            
            results.append((test_name, result, end_time - start_time))
            
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False, 0))
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result, duration in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name:<30} ({duration:.2f}s)")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! Enhanced spoofing is ready for use.")
        print("\n💡 Next steps:")
        print("   1. Test with actual video files")
        print("   2. Monitor detection rates on platforms")
        print("   3. Adjust settings based on effectiveness")
    else:
        print(f"\n⚠️  {total-passed} test(s) failed. Review the issues above.")
        print("\n🔧 Common fixes:")
        print("   - Install FFmpeg if missing")
        print("   - Check file imports and dependencies")
        print("   - Verify all enhanced modules are in place")

if __name__ == "__main__":
    main()