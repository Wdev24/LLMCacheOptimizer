#!/usr/bin/env python3
"""
Quick test runner for the semantic cache system
"""

import subprocess
import sys
import time
import requests
import os

def check_system_status():
    """Check if the semantic cache system is running"""
    try:
        response = requests.get("http://localhost:5000", timeout=3)
        return True
    except:
        return False

def start_cache_system():
    """Help user start the cache system"""
    print("❌ Semantic cache system is not running!")
    print("\n🚀 To start the system:")
    print("   1. Open a new terminal")
    print("   2. Navigate to your semantic_cache directory")
    print("   3. Run: python app.py")
    print("   4. Wait for 'Running on http://localhost:5000' message")
    print("   5. Come back here and run this test again")
    
    input("\nPress Enter when the system is running...")

def install_requirements():
    """Install pandas if not available"""
    try:
        import pandas
        return True
    except ImportError:
        print("📦 Installing pandas for test reports...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pandas'])
            return True
        except:
            print("⚠️ Could not install pandas. Test will run but no CSV export.")
            return False

def run_quick_test():
    """Run a quick 10-prompt test"""
    print("🧪 Running Quick Test (10 prompts)...")
    
    quick_prompts = [
        {"prompt": "What is machine learning?", "expected": "cache"},
        {"prompt": "Call Ramesh Kumar", "expected": "cache"},  # Should be in cache
        {"prompt": "Call Ramesh Gupta", "expected": "llm"},   # Should be LLM (different person)
        {"prompt": "Set alarm for 10:50", "expected": "cache"}, # Should be in cache
        {"prompt": "Set alarm for 10:51", "expected": "llm"},   # Should be LLM (different time)
        {"prompt": "Define artificial intelligence", "expected": "cache"},
        {"prompt": "How to cook pasta?", "expected": "llm"},
        {"prompt": "Explain machine learning", "expected": "cache"}, # Semantic match
        {"prompt": "Machine learning stocks", "expected": "llm"}, # Different intent
        {"prompt": "What is ML?", "expected": "cache"} # Abbreviation match
    ]
    
    correct = 0
    false_positives = 0
    
    for i, test in enumerate(quick_prompts, 1):
        print(f"\n[{i}/10] Testing: '{test['prompt']}'")
        
        try:
            response = requests.post(
                "http://localhost:5000/api/query",
                json={"query": test["prompt"]},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                actual_source = data.get("source", "error")
                confidence = data.get("confidence", 0)
                
                # Determine if correct based on confidence and source
                if test["expected"] == "cache":
                    is_correct = actual_source == "cache" and confidence >= 0.6
                else:
                    is_correct = actual_source == "llm" or confidence < 0.6
                
                status = "✅" if is_correct else "❌"
                print(f"   {status} Expected: {test['expected']}, Got: {actual_source} (Confidence: {confidence:.3f})")
                
                if is_correct:
                    correct += 1
                elif test["expected"] == "llm" and actual_source == "cache":
                    false_positives += 1
                    
            else:
                print(f"   ❌ Error: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        time.sleep(0.2)  # Brief pause
    
    accuracy = (correct / len(quick_prompts)) * 100
    fp_rate = (false_positives / len(quick_prompts)) * 100
    
    print(f"\n📊 QUICK TEST RESULTS:")
    print(f"   Accuracy: {accuracy:.1f}% ({correct}/{len(quick_prompts)})")
    print(f"   False Positives: {false_positives} ({fp_rate:.1f}%)")
    
    if accuracy >= 80 and fp_rate <= 10:
        print("   🎉 System performing well!")
    elif accuracy >= 70:
        print("   ⚠️ System needs some tuning")
    else:
        print("   🔧 System needs significant improvement")

def main():
    """Main function"""
    print("🧠 Semantic Cache Test Runner")
    print("=" * 50)
    
    # Check if pandas is available
    install_requirements()
    
    # Check if system is running
    if not check_system_status():
        start_cache_system()
        
        # Check again
        if not check_system_status():
            print("❌ System still not running. Exiting.")
            return
    
    print("✅ Semantic cache system is running!")
    
    # Choose test type
    print("\nChoose test type:")
    print("1. Quick Test (10 prompts, ~30 seconds)")
    print("2. Full Elite Test (100+ prompts, ~5-10 minutes)")
    print("3. Check Current Stats")
    
    try:
        choice = input("\nEnter choice (1, 2, or 3): ").strip()
    except KeyboardInterrupt:
        print("\nExiting...")
        return
    
    if choice == "1":
        run_quick_test()
        
    elif choice == "2":
        print("\n🚀 Starting Full Elite Test...")
        try:
            # Import and run the elite tester
            from elite_cache_tester import EliteCacheTester
            
            tester = EliteCacheTester()
            
            print("Choose mode:")
            print("1. Sequential (recommended)")
            print("2. Parallel (stress test)")
            
            mode = input("Enter choice (1 or 2): ").strip()
            
            start_time = time.time()
            
            if mode == "2":
                results = tester.run_parallel_test(max_workers=3)
            else:
                results = tester.run_sequential_test()
            
            analysis = tester.analyze_results(results)
            analysis['total_test_time'] = round(time.time() - start_time, 2)
            
            tester.generate_report(results, analysis)
            tester.save_results(results, analysis)
            
            print(f"\n✅ Full test completed in {analysis['total_test_time']}s")
            
        except ImportError:
            print("❌ Elite tester not found. Running basic test...")
            run_quick_test()
        except Exception as e:
            print(f"❌ Error running elite test: {e}")
            print("Running basic test instead...")
            run_quick_test()
            
    elif choice == "3":
        print("\n📊 Fetching current stats...")
        try:
            response = requests.get("http://localhost:5000/api/cache/stats", timeout=5)
            if response.status_code == 200:
                stats = response.json()["stats"]
                print(f"\n   Total Queries: {stats.get('total_queries', 0)}")
                print(f"   Cache Hits: {stats.get('cache_hits', 0)}")
                print(f"   LLM Calls: {stats.get('llm_calls', 0)}")
                print(f"   Cache Hit Rate: {stats.get('cache_hit_rate', 0)}%")
                print(f"   Avg Confidence: {stats.get('average_confidence', 0):.3f}")
                print(f"   Avg Response Time: {stats.get('average_response_time', 0):.3f}s")
                print(f"   Cached Vectors: {stats.get('index_stats', {}).get('total_vectors', 0)}")
            else:
                print("❌ Could not fetch stats")
        except Exception as e:
            print(f"❌ Error fetching stats: {e}")
    
    else:
        print("Invalid choice. Running quick test...")
        run_quick_test()

if __name__ == "__main__":
    main()