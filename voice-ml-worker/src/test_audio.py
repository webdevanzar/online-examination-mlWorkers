"""
Infinite microphone monitoring with VAD detection (1 second display)
"""
from audio_detector import listen_and_detect
import time

def print_summary_stats(stats):
    """Print summary statistics"""
    print("\n" + "="*50)
    print("📊 REAL-TIME STATISTICS")
    print("="*50)
    print(f"   Total chunks processed: {stats['total_chunks']}")
    print(f"   Speech chunks detected: {stats['speech_chunks']}")
    print(f"   Speech detection rate: {(stats['speech_chunks']/stats['total_chunks']*100):.1f}%")
    print(f"   Maximum RMS level: {stats['max_rms']:.3f}")
    print(f"   Current speech probability: {stats['current_prob']:.3f}")
    print("="*50)

def main():
    print("🚀 INFINITE MICROPHONE MONITORING")
    print("=" * 55)
    print("💡 Speak into your microphone to test VAD detection")
    print("⏹️  Press Ctrl+C to stop the monitoring")
    print("=" * 55)
    
    gen = listen_and_detect()
    
    # Stats tracking
    stats = {
        'total_chunks': 0,
        'speech_chunks': 0,
        'max_rms': 0.0,
        'current_prob': 0.0,
        'start_time': time.time()
    }
    
    last_summary_time = time.time()
    last_display = time.time()       # 👈 For 1-second terminal update
    speech_activity = []
    
    try:
        while True:
            res = next(gen)
            stats['total_chunks'] += 1
            stats['current_prob'] = res.get("speech_probability", 0.0)

            rms = res.get("audio_rms", 0.0)
            stats['max_rms'] = max(stats['max_rms'], rms)

            if stats['current_prob'] > 0.1:
                stats['speech_chunks'] += 1
                speech_status = "🎤 SPEECH"
                speech_activity.append(1)
            else:
                speech_status = "🔇 silence"
                speech_activity.append(0)

            if len(speech_activity) > 20:
                speech_activity.pop(0)

            # 👉 Print only once every 1 second
            if time.time() - last_display >= 1:
                recent_speech_ratio = sum(speech_activity) / len(speech_activity) if speech_activity else 0

                if rms < 0.001:
                    status = "🟡 WEAK"
                elif rms < 0.1:
                    status = "🟢 GOOD"
                else:
                    status = "🟢 EXCELLENT"

                if recent_speech_ratio > 0.7:
                    activity_indicator = "🔊 ACTIVE"
                elif recent_speech_ratio > 0.3:
                    activity_indicator = "🔈 MODERATE"
                else:
                    activity_indicator = "🔈 QUIET"

                gain_applied = res.get("gain_applied", 1)

                print(f"[{time.strftime('%H:%M:%S')}] {status} {speech_status} {activity_indicator}")
                print(f"      prob={stats['current_prob']:.3f} vol={rms:.3f} gain={gain_applied}x")

                last_display = time.time()

            # Summary every 30 sec
            current_time = time.time()
            if current_time - last_summary_time >= 30:
                print_summary_stats(stats)
                last_summary_time = current_time
                speech_activity.clear()

            time.sleep(0.01)
            
    except KeyboardInterrupt:
        total_duration = time.time() - stats['start_time']
        print("\n" + "="*60)
        print("🏁 MONITORING STOPPED - FINAL SUMMARY")
        print("="*60)
        print(f"📈 Total monitoring time: {total_duration:.1f} seconds")
        print(f"🔢 Total chunks processed: {stats['total_chunks']}")
        print(f"🎤 Speech chunks detected: {stats['speech_chunks']}")
        print(f"📊 Overall speech detection rate: {(stats['speech_chunks']/stats['total_chunks']*100):.1f}%")
        print(f"📶 Maximum signal level: {stats['max_rms']:.3f} RMS")
        print(f"⚡ Gain applied: {res.get('gain_applied', 1)}x")

        chunks_per_second = stats['total_chunks'] / total_duration
        print(f"⚙️  Processing rate: {chunks_per_second:.1f} chunks/second")

        if stats['speech_chunks'] > 0:
            print("✅ SYSTEM STATUS: Microphone and VAD working perfectly!")
        else:
            print("⚠️  SYSTEM STATUS: No speech detected - check microphone")

        print("="*60)

if __name__ == "__main__":
    main()
