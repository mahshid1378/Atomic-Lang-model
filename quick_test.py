#!/usr/bin/env python3
"""
Quick Test Script
================

Simple test to verify the GRPO implementation works without heavy dependencies.
"""

import sys
from pathlib import Path

# Add the python directory to path
python_dir = Path(__file__).parent / "atomic-lang-model" / "python"
sys.path.append(str(python_dir))

def test_imports():
    """Test basic imports without heavy dependencies."""
    print("Testing imports...")
    
    try:
        # Test our logic environment
        from logic_env import LogicEnvironment, LogicAction, LogicState, TaskType
        print("Logic environment imports work")
        
        # Test basic functionality
        env = LogicEnvironment()
        state = env.reset()
        print(f"Environment created: {state.task_type.value}")
        print(f"   Question: {state.question}")
        print(f"   Ground truth: {state.ground_truth}")
        
        # Test action
        action = LogicAction(reasoning="Test", answer=state.ground_truth)
        next_state, reward, done, info = env.step(action)
        print(f"Action executed: reward={reward}, explanation={info['explanation']}")
        
        return True
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    except Exception as e:
        print(f"Runtime error: {e}")
        return False

def test_task_generation():
    """Test task generation without ML dependencies."""
    print("\nTesting task generation...")
    
    try:
        from logic_env import LogicTaskSampler, TaskType
        
        sampler = LogicTaskSampler()
        
        # Test each task type
        for task_type in TaskType:
            state = sampler.sample_task(task_type, difficulty=1)
            print(f"{task_type.value}: {state.question[:60]}...")
        
        return True
    except Exception as e:
        print(f"Task generation failed: {e}")
        return False

def test_verifier():
    """Test the deterministic verifier."""
    print("\nTesting verifier...")
    
    try:
        from logic_env import LogicVerifier, LogicState, LogicAction, TaskType
        
        verifier = LogicVerifier()
        
        # Test syllogism verification
        state = LogicState(
            question="All A are B. All B are C. Therefore, all A are C.",
            ground_truth="all A are C",
            task_type=TaskType.SYLLOGISM
        )
        
        # Correct answer
        correct_action = LogicAction(reasoning="Valid syllogism", answer="all A are C")
        reward1, explanation1 = verifier.verify(state, correct_action)
        
        # Incorrect answer  
        wrong_action = LogicAction(reasoning="Invalid", answer="all C are A")
        reward2, explanation2 = verifier.verify(state, wrong_action)
        
        print(f"Correct answer: reward={reward1}, explanation={explanation1}")
        print(f"Wrong answer: reward={reward2}, explanation={explanation2}")
        
        # Test determinism
        reward3, _ = verifier.verify(state, correct_action)
        assert reward1 == reward3, "Verifier should be deterministic"
        print("Verifier is deterministic")
        
        return True
    except Exception as e:
        print(f"Verifier test failed: {e}")
        return False

def test_hybrid_fallback():
    """Test hybrid model fallback mode."""
    print("\nTesting hybrid model fallback...")
    
    try:
        from hybrid_model import HybridLanguageModel
        
        # This should work even without Rust binary
        model = HybridLanguageModel()
        
        # Test validation (will use Python fallback)
        test_sentences = [
            "the student left",
            "students left", 
            "the teacher smiled"
        ]
        
        for sentence in test_sentences:
            valid = model.validate_syntax(sentence)
            print(f"   '{sentence}': {valid}")
        
        print("Hybrid model fallback works")
        return True
    except Exception as e:
        print(f"Hybrid model test failed: {e}")
        return False

def main():
    """Run quick tests."""
    print("Quick GRPO Test")
    print("=" * 30)
    
    tests = [
        ("Basic Imports", test_imports),
        ("Task Generation", test_task_generation), 
        ("Verifier", test_verifier),
        ("Hybrid Model", test_hybrid_fallback)
    ]
    
    results = []
    for test_name, test_func in tests:
        success = test_func()
        results.append((test_name, success))
    
    print("\n" + "=" * 30)
    print("QUICK TEST SUMMARY")
    print("=" * 30)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\nCore functionality works!")
        print("\nNext steps:")
        print("1. Install ML dependencies: pip install torch transformers peft bitsandbytes")
        print("2. Run full test: python3 atomic-lang-model/python/test_grpo_integration.py")
        print("3. Start training: python3 atomic-lang-model/python/grpo_trainer.py")
    else:
        print(f"\n{total - passed} tests failed")
        print("Check the error messages above for debugging.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
