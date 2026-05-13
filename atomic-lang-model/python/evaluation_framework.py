#!/usr/bin/env python3
"""
Evaluation Framework
===================

Comprehensive evaluation framework for the GRPO-trained atomic language model.
Implements David Kypuros's recommendations for evaluation and stopping criteria.

Key components:
- Hold-out test sets (5k problems per task type)
- Success rate tracking with plateau detection
- GSM-8K and MATH zero-shot evaluation
- Formal correctness verification
- Difficulty curriculum assessment
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import time
import random
from collections import defaultdict

from logic_env import LogicEnvironment, LogicAction, LogicState, TaskType
from grpo_trainer import GRPOTrainer, GRPOConfig, QuantizedSLM


@dataclass
class EvaluationConfig:
    """Configuration for evaluation framework."""
    # Test set sizes
    holdout_size_per_task: int = 5000
    quick_eval_size: int = 100
    
    # Difficulty levels to test
    difficulty_levels: List[int] = None
    task_types: List[str] = None
    
    # Plateau detection
    plateau_patience: int = 5
    plateau_threshold: float = 0.01
    min_success_rate: float = 0.8
    
    # External benchmarks
    use_gsm8k: bool = False  # Set to True if available
    use_math: bool = False   # Set to True if available
    
    # Output configuration
    save_results: bool = True
    results_dir: str = "evaluation_results"
    plot_metrics: bool = True
    
    def __post_init__(self):
        """Set default values."""
        if self.difficulty_levels is None:
            self.difficulty_levels = [1, 2, 3, 4, 5]
        if self.task_types is None:
            self.task_types = ["syllogism", "propositional", "agreement", "movement"]


@dataclass
class EvaluationResult:
    """Single evaluation result."""
    task_type: str
    difficulty: int
    question: str
    ground_truth: str
    model_answer: str
    model_reasoning: str
    reward: float
    is_correct: bool
    verification_explanation: str
    response_time: float


@dataclass
class EvaluationSummary:
    """Summary of evaluation results."""
    timestamp: str
    total_problems: int
    overall_success_rate: float
    success_by_task: Dict[str, float]
    success_by_difficulty: Dict[int, float]
    avg_response_time: float
    formal_correctness_rate: float
    plateau_detected: bool
    stopping_criteria_met: bool


class HoldoutTestSet:
    """
    Hold-out test set generator and manager.
    Generates consistent test problems for evaluation.
    """
    
    def __init__(self, config: EvaluationConfig, seed: int = 42):
        self.config = config
        self.seed = seed
        self.env = LogicEnvironment()
        
        # Pre-generate test sets for consistency
        self.test_sets = {}
        self._generate_test_sets()
    
    def _generate_test_sets(self):
        """Generate hold-out test sets for all task types and difficulties."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        print("Generating hold-out test sets...")
        
        for task_type_str in self.config.task_types:
            task_type = TaskType(task_type_str)
            self.test_sets[task_type_str] = {}
            
            for difficulty in self.config.difficulty_levels:
                problems = []
                
                for _ in range(self.config.holdout_size_per_task):
                    # Generate consistent problem
                    problem_state = self.env.sampler.sample_task(task_type, difficulty)
                    problems.append(problem_state)
                
                self.test_sets[task_type_str][difficulty] = problems
                
                print(f"Generated {len(problems)} problems for {task_type_str} difficulty {difficulty}")
        
        print(f"Total problems generated: {sum(len(diff_dict) for task_dict in self.test_sets.values() for diff_dict in task_dict.values())}")
    
    def get_test_set(self, task_type: str, difficulty: int, size: Optional[int] = None) -> List[LogicState]:
        """Get test set for specific task type and difficulty."""
        if task_type not in self.test_sets:
            raise ValueError(f"Unknown task type: {task_type}")
        
        if difficulty not in self.test_sets[task_type]:
            raise ValueError(f"Unknown difficulty: {difficulty}")
        
        problems = self.test_sets[task_type][difficulty]
        
        if size is None:
            return problems
        else:
            return random.sample(problems, min(size, len(problems)))
    
    def get_mixed_test_set(self, size: int) -> List[LogicState]:
        """Get mixed test set across all task types and difficulties."""
        all_problems = []
        
        for task_type in self.test_sets:
            for difficulty in self.test_sets[task_type]:
                all_problems.extend(self.test_sets[task_type][difficulty])
        
        return random.sample(all_problems, min(size, len(all_problems)))
    
    def save_test_sets(self, path: str):
        """Save test sets to file for reproducibility."""
        # Convert to serializable format
        serializable_sets = {}
        
        for task_type in self.test_sets:
            serializable_sets[task_type] = {}
            for difficulty in self.test_sets[task_type]:
                problems = []
                for state in self.test_sets[task_type][difficulty]:
                    problems.append({
                        "question": state.question,
                        "ground_truth": state.ground_truth,
                        "task_type": state.task_type.value,
                        "difficulty": state.difficulty
                    })
                serializable_sets[task_type][difficulty] = problems
        
        with open(path, 'w') as f:
            json.dump(serializable_sets, f, indent=2)
        
        print(f"Test sets saved to {path}")


class ModelEvaluator:
    """
    Model evaluator with comprehensive metrics.
    Tracks performance over time and detects plateaus.
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.test_sets = HoldoutTestSet(config)
        self.env = LogicEnvironment()
        
        # Results tracking
        self.evaluation_history: List[EvaluationSummary] = []
        self.detailed_results: List[EvaluationResult] = []
        
        # Plateau detection
        self.recent_scores = []
        
        # Create results directory
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(exist_ok=True)
    
    def evaluate_model(self, model: QuantizedSLM, 
                      quick_eval: bool = False) -> EvaluationSummary:
        """
        Comprehensive model evaluation.
        
        Args:
            model: The model to evaluate
            quick_eval: If True, use smaller test set for faster evaluation
        """
        print(f"Starting {'quick' if quick_eval else 'full'} evaluation...")
        start_time = time.time()
        
        # Get test problems
        if quick_eval:
            test_problems = self.test_sets.get_mixed_test_set(self.config.quick_eval_size)
        else:
            test_problems = []
            for task_type in self.config.task_types:
                for difficulty in self.config.difficulty_levels:
                    problems = self.test_sets.get_test_set(task_type, difficulty, 100)
                    test_problems.extend(problems)
        
        # Evaluate each problem
        results = []
        correct_count = 0
        response_times = []
        
        for i, problem in enumerate(test_problems):
            if i % 100 == 0:
                print(f"Evaluated {i}/{len(test_problems)} problems...")
            
            result = self._evaluate_single_problem(model, problem)
            results.append(result)
            
            if result.is_correct:
                correct_count += 1
            
            response_times.append(result.response_time)
        
        # Compute summary statistics
        overall_success_rate = correct_count / len(results) if results else 0.0
        
        success_by_task = defaultdict(list)
        success_by_difficulty = defaultdict(list)
        
        for result in results:
            success_by_task[result.task_type].append(result.is_correct)
            success_by_difficulty[result.difficulty].append(result.is_correct)
        
        # Average success rates
        success_by_task = {
            task: np.mean(successes) for task, successes in success_by_task.items()
        }
        success_by_difficulty = {
            diff: np.mean(successes) for diff, successes in success_by_difficulty.items()
        }
        
        # Formal correctness (using verifier)
        formal_correct = sum(1 for r in results if r.reward > 0)
        formal_correctness_rate = formal_correct / len(results) if results else 0.0
        
        # Plateau detection
        self.recent_scores.append(overall_success_rate)
        if len(self.recent_scores) > self.config.plateau_patience:
            self.recent_scores.pop(0)
        
        plateau_detected = self._detect_plateau()
        stopping_criteria_met = (
            plateau_detected and 
            overall_success_rate >= self.config.min_success_rate
        )
        
        # Create summary
        summary = EvaluationSummary(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            total_problems=len(results),
            overall_success_rate=overall_success_rate,
            success_by_task=success_by_task,
            success_by_difficulty=success_by_difficulty,
            avg_response_time=np.mean(response_times) if response_times else 0.0,
            formal_correctness_rate=formal_correctness_rate,
            plateau_detected=plateau_detected,
            stopping_criteria_met=stopping_criteria_met
        )
        
        # Store results
        self.evaluation_history.append(summary)
        self.detailed_results.extend(results)
        
        # Save results if configured
        if self.config.save_results:
            self._save_results(summary, results)
        
        # Plot metrics if configured
        if self.config.plot_metrics:
            self._plot_metrics()
        
        print(f"Evaluation completed in {time.time() - start_time:.2f}s")
        print(f"Overall success rate: {overall_success_rate:.3f}")
        print(f"Formal correctness rate: {formal_correctness_rate:.3f}")
        
        if plateau_detected:
            print("⚠️  Plateau detected!")
        
        if stopping_criteria_met:
            print("✅ Stopping criteria met!")
        
        return summary
    
    def _evaluate_single_problem(self, model: QuantizedSLM, 
                                 problem: LogicState) -> EvaluationResult:
        """Evaluate model on a single problem."""
        start_time = time.time()
        
        # Create prompt
        prompt = self._create_prompt(problem)
        
        # Generate response
        try:
            response, _ = model.generate_response(prompt, max_new_tokens=100)
        except Exception as e:
            response = f"Error: {str(e)}"
        
        response_time = time.time() - start_time
        
        # Parse action
        action = self._parse_action(response)
        
        # Verify with environment
        self.env.current_state = problem
        _, reward, _, info = self.env.step(action)
        
        return EvaluationResult(
            task_type=problem.task_type.value,
            difficulty=problem.difficulty,
            question=problem.question,
            ground_truth=problem.ground_truth,
            model_answer=action.answer,
            model_reasoning=action.reasoning,
            reward=reward,
            is_correct=reward > 0,
            verification_explanation=info.get("explanation", ""),
            response_time=response_time
        )
    
    def _create_prompt(self, state: LogicState) -> str:
        """Create prompt for evaluation (same as training)."""
        prompt_templates = {
            TaskType.SYLLOGISM: "Solve this syllogism:\n{question}\n\nReasoning:",
            TaskType.PROPOSITIONAL: "Evaluate this propositional argument:\n{question}\n\nAnswer:",
            TaskType.AGREEMENT: "Fix the agreement in this sentence:\n{question}\n\nCorrected:",
            TaskType.MOVEMENT: "Transform this sentence:\n{question}\n\nResult:",
        }
        
        template = prompt_templates.get(state.task_type, "Solve: {question}\nAnswer:")
        return template.format(question=state.question)
    
    def _parse_action(self, response: str) -> LogicAction:
        """Parse model response into action (same as training)."""
        lines = response.strip().split('\n')
        
        reasoning = ""
        answer = ""
        
        for line in lines:
            if line.strip():
                if not answer:
                    answer = line.strip()
                else:
                    reasoning += line.strip() + " "
        
        return LogicAction(
            reasoning=reasoning.strip(),
            answer=answer or response.strip()
        )
    
    def _detect_plateau(self) -> bool:
        """Detect if performance has plateaued."""
        if len(self.recent_scores) < self.config.plateau_patience:
            return False
        
        # Check if recent scores are within threshold
        score_range = max(self.recent_scores) - min(self.recent_scores)
        return score_range <= self.config.plateau_threshold
    
    def _save_results(self, summary: EvaluationSummary, results: List[EvaluationResult]):
        """Save evaluation results to files."""
        timestamp = summary.timestamp.replace(":", "-").replace(" ", "_")
        
        # Save summary
        summary_path = self.results_dir / f"summary_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(asdict(summary), f, indent=2)
        
        # Save detailed results
        results_path = self.results_dir / f"detailed_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        
        print(f"Results saved to {self.results_dir}")
    
    def _plot_metrics(self):
        """Plot evaluation metrics over time."""
        if len(self.evaluation_history) < 2:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Extract data
        timestamps = [i for i in range(len(self.evaluation_history))]
        success_rates = [s.overall_success_rate for s in self.evaluation_history]
        formal_rates = [s.formal_correctness_rate for s in self.evaluation_history]
        response_times = [s.avg_response_time for s in self.evaluation_history]
        
        # Plot overall success rate
        ax1.plot(timestamps, success_rates, 'b-', marker='o')
        ax1.set_title('Overall Success Rate')
        ax1.set_xlabel('Evaluation')
        ax1.set_ylabel('Success Rate')
        ax1.grid(True)
        
        # Plot formal correctness
        ax2.plot(timestamps, formal_rates, 'g-', marker='s')
        ax2.set_title('Formal Correctness Rate')
        ax2.set_xlabel('Evaluation')
        ax2.set_ylabel('Correctness Rate')
        ax2.grid(True)
        
        # Plot success by task type
        task_data = defaultdict(list)
        for summary in self.evaluation_history:
            for task, rate in summary.success_by_task.items():
                task_data[task].append(rate)
        
        for task, rates in task_data.items():
            ax3.plot(timestamps[:len(rates)], rates, marker='o', label=task)
        ax3.set_title('Success Rate by Task Type')
        ax3.set_xlabel('Evaluation')
        ax3.set_ylabel('Success Rate')
        ax3.legend()
        ax3.grid(True)
        
        # Plot response times
        ax4.plot(timestamps, response_times, 'r-', marker='^')
        ax4.set_title('Average Response Time')
        ax4.set_xlabel('Evaluation')
        ax4.set_ylabel('Time (seconds)')
        ax4.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / "evaluation_metrics.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Metrics plot saved to {plot_path}")
    
    def generate_report(self) -> str:
        """Generate comprehensive evaluation report."""
        if not self.evaluation_history:
            return "No evaluation data available."
        
        latest = self.evaluation_history[-1]
        
        report = f"""
Atomic Language Model - Evaluation Report
========================================
Generated: {latest.timestamp}

OVERALL PERFORMANCE
-------------------
Total Problems Evaluated: {latest.total_problems}
Overall Success Rate: {latest.overall_success_rate:.3f}
Formal Correctness Rate: {latest.formal_correctness_rate:.3f}
Average Response Time: {latest.avg_response_time:.3f}s

PERFORMANCE BY TASK TYPE
------------------------
"""
        
        for task, rate in latest.success_by_task.items():
            report += f"{task.capitalize()}: {rate:.3f}\n"
        
        report += f"""
PERFORMANCE BY DIFFICULTY
-------------------------
"""
        
        for diff in sorted(latest.success_by_difficulty.keys()):
            rate = latest.success_by_difficulty[diff]
            report += f"Difficulty {diff}: {rate:.3f}\n"
        
        report += f"""
TRAINING STATUS
---------------
Plateau Detected: {"Yes" if latest.plateau_detected else "No"}
Stopping Criteria Met: {"Yes" if latest.stopping_criteria_met else "No"}

EVALUATION HISTORY
------------------
Total Evaluations: {len(self.evaluation_history)}
"""
        
        if len(self.evaluation_history) > 1:
            first = self.evaluation_history[0]
            improvement = latest.overall_success_rate - first.overall_success_rate
            report += f"Success Rate Improvement: {improvement:+.3f}\n"
        
        return report


def demo_evaluation():
    """Demonstrate the evaluation framework."""
    print("📊 Evaluation Framework Demo")
    print("=" * 50)
    
    # Configuration
    config = EvaluationConfig(
        holdout_size_per_task=100,  # Smaller for demo
        quick_eval_size=20,
        task_types=["syllogism", "propositional"],
        difficulty_levels=[1, 2, 3]
    )
    
    # Create evaluator
    evaluator = ModelEvaluator(config)
    
    # Mock model for demo
    class MockModel:
        def generate_response(self, prompt: str, max_new_tokens: int = 100):
            # Simple mock responses
            if "syllogism" in prompt.lower():
                return "All A are C.", torch.tensor([0.0])
            else:
                return "valid", torch.tensor([0.0])
    
    # Run evaluation
    mock_model = MockModel()
    summary = evaluator.evaluate_model(mock_model, quick_eval=True)
    
    # Generate report
    report = evaluator.generate_report()
    print(report)
    
    # Save test sets
    test_sets_path = evaluator.results_dir / "holdout_test_sets.json"
    evaluator.test_sets.save_test_sets(test_sets_path)


if __name__ == "__main__":
    demo_evaluation()