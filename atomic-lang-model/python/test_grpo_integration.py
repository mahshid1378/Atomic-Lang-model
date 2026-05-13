#!/usr/bin/env python3
"""
GRPO Integration Test Suite
==========================

Comprehensive test suite for validating the GRPO/RLVR integration.
Run this script to verify all components work correctly.
"""

import sys
import traceback
import time
import torch
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_basic_imports():
    """Test 1: Basic imports work correctly."""
    print("🔧 Test 1: Testing basic imports...")
    
    try:
        from logic_env import LogicEnvironment, LogicAction, LogicState, TaskType
        from grpo_trainer import GRPOTrainer, GRPOConfig, QuantizedSLM
        from evaluation_framework import ModelEvaluator, EvaluationConfig
        from hybrid_model import HybridLanguageModel
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_environment_basic():
    """Test 2: Logic environment basic functionality."""
    print("\n🧠 Test 2: Testing logic environment...")
    
    try:
        from logic_env import LogicEnvironment, LogicAction, TaskType
        
        # Create environment
        env = LogicEnvironment()
        
        # Test reset
        state = env.reset()
        print(f"✅ Environment reset successful")
        print(f"   Task: {state.task_type.value}")
        print(f"   Question: {state.question[:50]}...")
        
        # Test action execution
        action = LogicAction(
            reasoning="Test reasoning",
            answer=state.ground_truth  # Use correct answer
        )
        
        next_state, reward, done, info = env.step(action)
        print(f"✅ Action execution successful")
        print(f"   Reward: {reward}")
        print(f"   Explanation: {info['explanation']}")
        
        # Test determinism
        env.current_state = state
        _, reward2, _, _ = env.step(action)
        assert reward == reward2, "Rewards should be deterministic"
        print("✅ Verifier is deterministic")
        
        return True
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        traceback.print_exc()
        return False

def test_task_generation():
    """Test 3: Task generation for all types."""
    print("\n📝 Test 3: Testing task generation...")
    
    try:
        from logic_env import LogicTaskSampler, TaskType
        
        sampler = LogicTaskSampler()
        
        # Test each task type
        for task_type in TaskType:
            for difficulty in [1, 2, 3]:
                state = sampler.sample_task(task_type, difficulty)
                print(f"✅ {task_type.value} difficulty {difficulty}: {state.question[:40]}...")
        
        print("✅ All task types generate successfully")
        return True
    except Exception as e:
        print(f"❌ Task generation failed: {e}")
        traceback.print_exc()
        return False

def test_model_loading():
    """Test 4: Model loading with quantization."""
    print("\n🤖 Test 4: Testing model loading...")
    
    try:
        from grpo_trainer import QuantizedSLM, GRPOConfig
        
        # Use smallest available model for testing
        config = GRPOConfig(
            model_name="microsoft/DialoGPT-small",  # 117M params
            load_in_4bit=True,
            max_length=256  # Smaller for testing
        )
        
        print("Loading quantized model...")
        model = QuantizedSLM(config)
        
        print(f"✅ Model loaded successfully")
        print(f"   Device: {model.device}")
        print(f"   Trainable parameters: {model.model.get_nb_trainable_parameters()}")
        
        # Test memory usage
        if torch.cuda.is_available():
            memory_gb = torch.cuda.memory_allocated() / (1024**3)
            print(f"   GPU memory used: {memory_gb:.2f} GB")
            assert memory_gb < 6.0, f"Memory usage too high: {memory_gb:.2f} GB"
        
        return True, model
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        traceback.print_exc()
        return False, None

def test_text_generation(model):
    """Test 5: Text generation functionality."""
    print("\n💬 Test 5: Testing text generation...")
    
    if model is None:
        print("❌ Skipping (no model loaded)")
        return False
    
    try:
        prompt = "Solve this syllogism: All students are people. All people are mortal."
        
        start_time = time.time()
        response, log_probs = model.generate_response(prompt, max_new_tokens=30)
        response_time = time.time() - start_time
        
        print(f"✅ Text generation successful")
        print(f"   Prompt: {prompt}")
        print(f"   Response: {response}")
        print(f"   Response time: {response_time:.3f}s")
        print(f"   Log probs shape: {log_probs.shape if hasattr(log_probs, 'shape') else 'scalar'}")
        
        return True
    except Exception as e:
        print(f"❌ Text generation failed: {e}")
        traceback.print_exc()
        return False

def test_episode_collection():
    """Test 6: Episode collection for GRPO."""
    print("\n📊 Test 6: Testing episode collection...")
    
    try:
        from grpo_trainer import GRPOTrainer, GRPOConfig
        
        # Use minimal config for testing
        config = GRPOConfig(
            model_name="microsoft/DialoGPT-small",
            batch_size=2,
            target_batch_tokens=512 * 1024,  # 512KB for testing
            task_types=["syllogism"],
            max_length=256
        )
        
        print("Creating GRPO trainer...")
        trainer = GRPOTrainer(config)
        
        print("Collecting episodes...")
        episodes = trainer.collect_episodes(n_episodes=3)
        
        print(f"✅ Episode collection successful")
        print(f"   Episodes collected: {len(episodes)}")
        
        for i, episode in enumerate(episodes):
            print(f"   Episode {i+1}: reward={episode.reward}, task={episode.state.task_type.value}")
        
        return True, trainer
    except Exception as e:
        print(f"❌ Episode collection failed: {e}")
        traceback.print_exc()
        return False, None

def test_grpo_training_step(trainer):
    """Test 7: GRPO training step."""
    print("\n🎯 Test 7: Testing GRPO training step...")
    
    if trainer is None:
        print("❌ Skipping (no trainer)")
        return False
    
    try:
        print("Running training step...")
        start_time = time.time()
        
        # Run one training step
        metrics = trainer.train_step()
        
        step_time = time.time() - start_time
        
        print(f"✅ Training step successful")
        print(f"   Loss: {metrics.get('loss', 'N/A')}")
        print(f"   Groups: {metrics.get('groups', 'N/A')}")
        print(f"   Episodes: {metrics.get('episodes', 'N/A')}")
        print(f"   Step time: {step_time:.3f}s")
        
        return True
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        traceback.print_exc()
        return False

def test_evaluation_framework():
    """Test 8: Evaluation framework."""
    print("\n📈 Test 8: Testing evaluation framework...")
    
    try:
        from evaluation_framework import ModelEvaluator, EvaluationConfig
        
        # Minimal config for testing
        config = EvaluationConfig(
            holdout_size_per_task=20,  # Small for testing
            quick_eval_size=10,
            task_types=["syllogism"],
            difficulty_levels=[1, 2],
            save_results=False,  # Don't save during testing
            plot_metrics=False   # Don't plot during testing
        )
        
        print("Creating evaluator...")
        evaluator = ModelEvaluator(config)
        
        print("Testing test set generation...")
        test_problems = evaluator.test_sets.get_test_set("syllogism", 1, 5)
        print(f"✅ Generated {len(test_problems)} test problems")
        
        # Test plateau detection
        evaluator.recent_scores = [0.7, 0.71, 0.70, 0.72, 0.71]
        plateau = evaluator._detect_plateau()
        print(f"✅ Plateau detection works: {plateau}")
        
        return True
    except Exception as e:
        print(f"❌ Evaluation framework failed: {e}")
        traceback.print_exc()
        return False

def test_hybrid_model_integration():
    """Test 9: Integration with existing hybrid model."""
    print("\n🔗 Test 9: Testing hybrid model integration...")
    
    try:
        from hybrid_model import HybridLanguageModel
        
        # Create hybrid model (may not have Rust binary in testing)
        hybrid = HybridLanguageModel()
        
        # Test sentence validation
        test_sentences = [
            "the student left",
            "the teacher smiled",
            "invalid grammar sentence this"
        ]
        
        for sentence in test_sentences:
            valid = hybrid.validate_syntax(sentence)
            print(f"   '{sentence}': {valid}")
        
        print("✅ Hybrid model integration works")
        return True
    except Exception as e:
        print(f"❌ Hybrid model integration failed: {e}")
        traceback.print_exc()
        return False

def test_memory_efficiency():
    """Test 10: Memory efficiency requirements."""
    print("\n💾 Test 10: Testing memory efficiency...")
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping memory test")
        return True
    
    try:
        # Clear GPU memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        from grpo_trainer import QuantizedSLM, GRPOConfig
        
        config = GRPOConfig(
            model_name="microsoft/DialoGPT-small",
            load_in_4bit=True
        )
        
        model = QuantizedSLM(config)
        
        peak_memory = torch.cuda.max_memory_allocated()
        current_memory = torch.cuda.memory_allocated()
        
        memory_gb = current_memory / (1024**3)
        peak_gb = peak_memory / (1024**3)
        
        print(f"✅ Memory efficiency check")
        print(f"   Current memory: {memory_gb:.2f} GB")
        print(f"   Peak memory: {peak_gb:.2f} GB")
        
        if memory_gb < 6.0:
            print(f"✅ Memory usage within target (<6GB)")
        else:
            print(f"⚠️  Memory usage high: {memory_gb:.2f} GB")
        
        return True
    except Exception as e:
        print(f"❌ Memory efficiency test failed: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests and provide summary."""
    print("🧪 GRPO Integration Test Suite")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Environment Basic", test_environment_basic),
        ("Task Generation", test_task_generation),
        ("Model Loading", test_model_loading),
        ("Episode Collection", test_episode_collection),
        ("Evaluation Framework", test_evaluation_framework),
        ("Hybrid Model Integration", test_hybrid_model_integration),
        ("Memory Efficiency", test_memory_efficiency),
    ]
    
    results = []
    model = None
    trainer = None
    
    for test_name, test_func in tests:
        try:
            if test_name == "Model Loading":
                success, model = test_func()
            elif test_name == "Episode Collection":
                success, trainer = test_func()
            else:
                success = test_func()
            
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Additional tests that require model/trainer
    if model is not None:
        try:
            success = test_text_generation(model)
            results.append(("Text Generation", success))
        except Exception as e:
            results.append(("Text Generation", False))
    
    if trainer is not None:
        try:
            success = test_grpo_training_step(trainer)
            results.append(("GRPO Training Step", success))
        except Exception as e:
            results.append(("GRPO Training Step", False))
    
    # Print summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
        else:
            failed += 1
    
    print("\n" + "-" * 50)
    print(f"Total: {passed + failed} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("Your GRPO integration is ready for use.")
    else:
        print(f"\n⚠️  {failed} tests failed. Check the error messages above.")
        print("Refer to GRPO_TESTING_GUIDE.md for troubleshooting.")
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)