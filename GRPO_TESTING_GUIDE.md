# GRPO Integration Testing Guide

This guide provides comprehensive testing procedures for the GRPO/RLVR integration with the atomic-lang-model, based on David Kypuros's recommendations.

## Overview

The implementation includes:
1. **Logic Environment** (`logic_env.py`) - Gym-style API wrapper for the logic verifier
2. **GRPO Trainer** (`grpo_trainer.py`) - Group Relative Policy Optimization with quantized SLM + LoRA
3. **Evaluation Framework** (`evaluation_framework.py`) - Hold-out test sets and plateau detection

## Prerequisites

```bash
# Install required dependencies
pip install torch transformers peft bitsandbytes accelerate
pip install numpy matplotlib datasets evaluate

# For optional external benchmarks
pip install openai-gym gsm8k  # If available
```

## Testing Procedures

### 1. Environment Setup Testing

**Test the Logic Environment basics:**

```bash
cd atomic-lang-model/python
python logic_env.py
```

**Expected Output:**
- Environment demo showing different task types
- Rewards (+1/-1) for correct/incorrect answers
- Verification explanations
- All task types working (syllogism, propositional, agreement, movement)

**Key Tests:**
```python
# Test individual components
from logic_env import LogicEnvironment, TaskType

env = LogicEnvironment()
state = env.reset()
print(f"Task: {state.task_type}, Question: {state.question}")

# Test verifier determinism
action = LogicAction(reasoning="test", answer=state.ground_truth)
reward1, _, _, _ = env.step(action)
env.current_state = state  # Reset
reward2, _, _, _ = env.step(action)
assert reward1 == reward2, "Verifier should be deterministic"
```

### 2. Model Loading and Quantization Testing

**Test quantized model loading:**

```bash
python -c "
from grpo_trainer import QuantizedSLM, GRPOConfig
config = GRPOConfig(model_name='microsoft/DialoGPT-small')
model = QuantizedSLM(config)
print('Model loaded successfully')
print(f'Trainable parameters: {model.model.get_nb_trainable_parameters()}')
"
```

**Expected Output:**
- Model loads without errors
- Trainable parameters < 1M (LoRA working)
- Memory usage < 6GB VRAM

**Memory Test:**
```python
import torch
print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```

### 3. GRPO Training Loop Testing

**Quick training test (1-2 updates):**

```bash
python -c "
from grpo_trainer import GRPOTrainer, GRPOConfig

config = GRPOConfig(
    model_name='microsoft/DialoGPT-small',
    target_batch_tokens=1024*1024,  # 1MB for quick test
    task_types=['syllogism']
)

trainer = GRPOTrainer(config)

# Test single training step
metrics = trainer.train_step()
print(f'Training step completed: {metrics}')
assert 'loss' in metrics, 'Loss should be computed'
assert metrics['groups'] > 0, 'Should have episode groups'
"
```

**Expected Output:**
- Training step completes without errors
- Loss value is computed (may be positive or negative)
- Episode groups > 0
- No CUDA out of memory errors

### 4. End-to-End Integration Testing

**Full pipeline test (5-10 updates):**

```bash
python grpo_trainer.py
```

**Monitor for:**
- Stable training without crashes
- Loss convergence (should not diverge)
- Success rate improvement over time
- Memory usage staying within bounds
- Checkpoint saving/loading

### 5. Evaluation Framework Testing

**Test evaluation on hold-out sets:**

```bash
python evaluation_framework.py
```

**Expected Output:**
- Test sets generated successfully
- Evaluation metrics computed
- Plots saved (if matplotlib available)
- Results saved to JSON files

**Key Metrics to Check:**
```python
from evaluation_framework import ModelEvaluator, EvaluationConfig

config = EvaluationConfig(holdout_size_per_task=100)
evaluator = ModelEvaluator(config)

# Test plateau detection
evaluator.recent_scores = [0.7, 0.71, 0.70, 0.72, 0.71]
plateau = evaluator._detect_plateau()
print(f"Plateau detected: {plateau}")
```

### 6. Formal Correctness Verification

**Test that rewards match formal verification:**

```python
from logic_env import LogicEnvironment, LogicAction
from hybrid_model import HybridLanguageModel

env = LogicEnvironment()
hybrid = HybridLanguageModel()

# Test agreement between verifier and hybrid model
for _ in range(10):
    state = env.reset()
    action = LogicAction(reasoning="test", answer=state.ground_truth)
    
    # Environment reward
    env_reward, _, _, info = env.step(action)
    
    # Hybrid model validation
    if hasattr(hybrid, 'validate_syntax'):
        syntax_valid = hybrid.validate_syntax(action.answer)
        
        print(f"Env reward: {env_reward}, Syntax valid: {syntax_valid}")
        print(f"Explanation: {info['explanation']}")
```

### 7. Performance Benchmarks

**Memory Efficiency Test:**
```python
# Test maximum batch size without OOM
for batch_size in [1, 2, 4, 8, 16]:
    try:
        config = GRPOConfig(batch_size=batch_size)
        trainer = GRPOTrainer(config)
        metrics = trainer.train_step()
        print(f"Batch size {batch_size}: OK")
    except torch.cuda.OutOfMemoryError:
        print(f"Batch size {batch_size}: OOM")
        break
```

**Response Time Test:**
```python
import time
from grpo_trainer import QuantizedSLM, GRPOConfig

config = GRPOConfig()
model = QuantizedSLM(config)

times = []
for _ in range(10):
    start = time.time()
    response, _ = model.generate_response("Test prompt", max_new_tokens=50)
    times.append(time.time() - start)

print(f"Average response time: {np.mean(times):.3f}s ± {np.std(times):.3f}s")
```

### 8. Edge Cases and Error Handling

**Test error recovery:**

```python
# Test with malformed inputs
from logic_env import LogicAction

malformed_actions = [
    LogicAction(reasoning="", answer=""),
    LogicAction(reasoning="test", answer="!@#$%"),
    LogicAction(reasoning="very " * 1000, answer="overflow"),
]

env = LogicEnvironment()
for action in malformed_actions:
    state = env.reset()
    try:
        reward, _, _, info = env.step(action)
        print(f"Handled malformed input: reward={reward}")
    except Exception as e:
        print(f"Error handling failed: {e}")
```

### 9. Curriculum Learning Validation

**Test difficulty progression:**

```python
from evaluation_framework import ModelEvaluator, EvaluationConfig

config = EvaluationConfig(difficulty_levels=[1, 2, 3, 4, 5])
evaluator = ModelEvaluator(config)

# Check that higher difficulty has lower success rates
for difficulty in [1, 2, 3, 4, 5]:
    problems = evaluator.test_sets.get_test_set("syllogism", difficulty, 20)
    # Manually verify that problems get harder
    print(f"Difficulty {difficulty} sample: {problems[0].question}")
```

### 10. Stopping Criteria Testing

**Test plateau detection:**

```python
from evaluation_framework import ModelEvaluator, EvaluationConfig

config = EvaluationConfig(plateau_patience=3, plateau_threshold=0.01)
evaluator = ModelEvaluator(config)

# Simulate plateau
evaluator.recent_scores = [0.85, 0.851, 0.849, 0.850]
assert evaluator._detect_plateau(), "Should detect plateau"

# Simulate improvement
evaluator.recent_scores = [0.80, 0.82, 0.85, 0.87]
assert not evaluator._detect_plateau(), "Should not detect plateau"
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `batch_size` and `target_batch_tokens`
   - Enable gradient checkpointing
   - Use smaller model (e.g., `distilgpt2`)

2. **LoRA Not Working**
   - Check `target_modules` match model architecture
   - Verify `peft` library version compatibility

3. **Slow Training**
   - Reduce `group_size` for faster updates
   - Increase `accumulation_steps` for larger effective batches
   - Use CPU for verifier (should be automatic)

4. **Poor Convergence**
   - Check that rewards are non-zero
   - Verify task difficulty is appropriate
   - Adjust learning rate (try 1e-6 to 1e-4)

### Performance Targets

**Minimum Acceptable Performance:**
- Memory usage: < 6GB VRAM
- Training speed: > 1 update/minute
- Success rate: > 0.7 on difficulty 1 problems
- Formal correctness: > 0.8 agreement with verifier

**Good Performance:**
- Memory usage: < 4GB VRAM
- Training speed: > 5 updates/minute  
- Success rate: > 0.9 on difficulty 1, > 0.7 on difficulty 3
- Formal correctness: > 0.95 agreement with verifier

## Validation Checklist

- [ ] Logic environment loads and runs demo
- [ ] Quantized model loads with <6GB VRAM
- [ ] LoRA adapters applied (trainable params < 1M)
- [ ] GRPO training step completes without errors
- [ ] Episode collection works with all task types
- [ ] Verifier provides deterministic rewards
- [ ] Evaluation framework generates test sets
- [ ] Plateau detection works correctly
- [ ] Memory usage stays within bounds during training
- [ ] Checkpoints save and load correctly
- [ ] Error handling works for malformed inputs
- [ ] Performance meets minimum targets

## Integration with Existing Code

**Update CLAUDE.md with new commands:**

```bash
# Add to CLAUDE.md build commands section:
npm run grpo-train    # Start GRPO training
npm run grpo-eval     # Run evaluation
npm run grpo-test     # Run test suite
```

**Create npm scripts in package.json:**

```json
{
  "scripts": {
    "grpo-train": "cd atomic-lang-model/python && python grpo_trainer.py",
    "grpo-eval": "cd atomic-lang-model/python && python evaluation_framework.py",
    "grpo-test": "cd atomic-lang-model/python && python -m pytest test_grpo.py"
  }
}
```

This testing guide ensures the GRPO integration works correctly with minimal errors and meets the performance targets for commodity hardware deployment.