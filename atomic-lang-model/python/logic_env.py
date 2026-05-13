#!/usr/bin/env python3
"""
Logic Environment
=================

Gym-style API wrapper for the atomic language model's logic verifier.
This implements David Kypuros's recommendation to treat the logic core
as a "verifier-environment" for GRPO/RLVR training.

Key components:
- State: {"question": <problem>, "ground_truth": <answer>}
- Action: {"reasoning": <text>, "answer": <text>}
- Reward: +1 if verifier(answer, ground_truth) else -1

The verifier runs on CPU and provides objective rewards for free.
"""

import json
import subprocess
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

from hybrid_model import HybridLanguageModel
from tiny_lm import ProbGrammar


class TaskType(Enum):
    """Types of logic tasks the environment can generate."""
    SYLLOGISM = "syllogism"
    PROPOSITIONAL = "propositional"
    AGREEMENT = "agreement"
    MOVEMENT = "movement"


@dataclass
class LogicState:
    """Environment state containing problem and ground truth."""
    question: str
    ground_truth: str
    task_type: TaskType
    difficulty: int = 1
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LogicAction:
    """Agent action containing reasoning and answer."""
    reasoning: str
    answer: str
    confidence: float = 1.0


class LogicVerifier:
    """
    Deterministic logic verifier that runs on CPU.
    Provides objective rewards for GRPO training.
    """
    
    def __init__(self, hybrid_model: Optional[HybridLanguageModel] = None):
        """Initialize verifier with optional hybrid model."""
        self.hybrid_model = hybrid_model or HybridLanguageModel()
        
        # Simple rule-based verifiers for different task types
        self.verifiers = {
            TaskType.SYLLOGISM: self._verify_syllogism,
            TaskType.PROPOSITIONAL: self._verify_propositional,
            TaskType.AGREEMENT: self._verify_agreement,
            TaskType.MOVEMENT: self._verify_movement,
        }
    
    def verify(self, state: LogicState, action: LogicAction) -> Tuple[float, str]:
        """
        Verify an action against ground truth.
        
        Returns:
            reward: +1.0 for correct, -1.0 for incorrect
            explanation: Human-readable verification result
        """
        verifier_fn = self.verifiers.get(state.task_type)
        if not verifier_fn:
            return -1.0, f"Unknown task type: {state.task_type}"
        
        return verifier_fn(state, action)
    
    def _verify_syllogism(self, state: LogicState, action: LogicAction) -> Tuple[float, str]:
        """Verify syllogistic reasoning."""
        # Extract conclusion from action
        answer = action.answer.strip().lower()
        ground_truth = state.ground_truth.strip().lower()
        
        # Simple string matching for now - would use formal logic in production
        if answer == ground_truth:
            return 1.0, "Correct syllogistic conclusion"
        
        # Check if answer is syntactically valid
        if self.hybrid_model.validate_syntax(action.answer):
            return -0.5, "Syntactically valid but incorrect conclusion"
        else:
            return -1.0, "Invalid syntax and incorrect conclusion"
    
    def _verify_propositional(self, state: LogicState, action: LogicAction) -> Tuple[float, str]:
        """Verify propositional logic entailment."""
        answer = action.answer.strip().lower()
        ground_truth = state.ground_truth.strip().lower()
        
        # Truth value matching
        if answer in ["true", "false", "valid", "invalid"]:
            if answer == ground_truth:
                return 1.0, "Correct propositional evaluation"
            else:
                return -1.0, "Incorrect propositional evaluation"
        
        return -1.0, "Invalid propositional answer format"
    
    def _verify_agreement(self, state: LogicState, action: LogicAction) -> Tuple[float, str]:
        """Verify grammatical agreement."""
        # Check if the answer satisfies agreement constraints
        if self.hybrid_model.validate_syntax(action.answer):
            # Additional agreement-specific checks would go here
            answer_tokens = action.answer.split()
            ground_truth_tokens = state.ground_truth.split()
            
            if answer_tokens == ground_truth_tokens:
                return 1.0, "Correct agreement and syntax"
            else:
                return 0.5, "Valid syntax but incorrect agreement"
        
        return -1.0, "Invalid syntax violates agreement"
    
    def _verify_movement(self, state: LogicState, action: LogicAction) -> Tuple[float, str]:
        """Verify movement transformation."""
        # Check if movement preserves meaning and syntax
        if self.hybrid_model.validate_syntax(action.answer):
            # Would check movement constraints in full implementation
            if action.answer.strip() == state.ground_truth.strip():
                return 1.0, "Correct movement transformation"
            else:
                return 0.0, "Valid syntax but incorrect movement"
        
        return -1.0, "Invalid movement violates syntax"


class LogicTaskSampler:
    """
    Procedural task generator for different types of logic problems.
    Generates unlimited training data on-the-fly.
    """
    
    def __init__(self, grammar: Optional[ProbGrammar] = None):
        """Initialize with probabilistic grammar."""
        self.grammar = grammar or ProbGrammar()
        
        # Templates for different task types
        self.templates = {
            TaskType.SYLLOGISM: [
                "All {A} are {B}. All {B} are {C}. Therefore, all {A} are {C}.",
                "Some {A} are {B}. All {B} are {C}. Therefore, some {A} are {C}.",
                "No {A} are {B}. All {C} are {A}. Therefore, no {C} are {B}.",
            ],
            TaskType.PROPOSITIONAL: [
                "If {P} then {Q}. {P}. Therefore, {Q}.",
                "If {P} then {Q}. Not {Q}. Therefore, not {P}.",
                "{P} or {Q}. Not {P}. Therefore, {Q}.",
            ],
            TaskType.AGREEMENT: [
                "The {noun} {verb}",
                "The {adj} {noun} {verb}",
                "{det} {noun} who {verb} {verb2}",
            ],
            TaskType.MOVEMENT: [
                "{obj} the {subj} {verb}",
                "Who did the {subj} {verb}?",
                "What {verb} the {subj}?",
            ]
        }
        
        # Vocabulary for different slots
        self.vocab = {
            "A": ["students", "teachers", "books"],
            "B": ["people", "objects", "things"],
            "C": ["mortal", "useful", "valuable"],
            "P": ["it rains", "it's sunny", "it's cold"],
            "Q": ["the ground is wet", "it's warm", "I wear a coat"],
            "noun": ["student", "teacher", "book", "class"],
            "verb": ["left", "arrived", "smiled", "praised"],
            "verb2": ["stayed", "departed", "laughed"],
            "adj": ["smart", "new", "old", "good"],
            "det": ["the", "a"],
            "subj": ["student", "teacher"],
            "obj": ["who", "what", "which book"],
        }
    
    def sample_task(self, task_type: TaskType, difficulty: int = 1) -> LogicState:
        """Sample a task of the given type and difficulty."""
        if task_type not in self.templates:
            raise ValueError(f"Unknown task type: {task_type}")
        
        # Choose template based on difficulty
        templates = self.templates[task_type]
        template_idx = min(difficulty - 1, len(templates) - 1)
        template = templates[template_idx]
        
        # Fill in template with vocabulary
        question, ground_truth = self._instantiate_template(template, task_type)
        
        return LogicState(
            question=question,
            ground_truth=ground_truth,
            task_type=task_type,
            difficulty=difficulty
        )
    
    def _instantiate_template(self, template: str, task_type: TaskType) -> Tuple[str, str]:
        """Fill template with concrete vocabulary."""
        import re
        import random
        
        # Find all placeholder variables
        placeholders = re.findall(r'\{(\w+)\}', template)
        substitutions = {}
        
        for placeholder in placeholders:
            if placeholder in self.vocab:
                substitutions[placeholder] = random.choice(self.vocab[placeholder])
            else:
                substitutions[placeholder] = f"<{placeholder}>"
        
        # Substitute placeholders
        question = template
        for placeholder, value in substitutions.items():
            question = question.replace(f"{{{placeholder}}}", value)
        
        # Generate ground truth based on task type
        ground_truth = self._generate_ground_truth(question, task_type, substitutions)
        
        return question, ground_truth
    
    def _generate_ground_truth(self, question: str, task_type: TaskType, 
                              substitutions: Dict[str, str]) -> str:
        """Generate correct answer for the question."""
        if task_type == TaskType.SYLLOGISM:
            # Extract conclusion from syllogism
            if "Therefore," in question:
                return question.split("Therefore, ")[1].strip()
            return "valid"
        
        elif task_type == TaskType.PROPOSITIONAL:
            # Extract conclusion from propositional argument
            if "Therefore," in question:
                return question.split("Therefore, ")[1].strip()
            return "valid"
        
        elif task_type == TaskType.AGREEMENT:
            # For agreement tasks, the question is the answer
            return question.strip()
        
        elif task_type == TaskType.MOVEMENT:
            # Generate canonical form for movement
            if "Who" in question:
                return f"The {substitutions.get('subj', 'person')} {substitutions.get('verb', 'acted')} {substitutions.get('obj', 'someone')}."
            return question.strip()
        
        return "unknown"


class LogicEnvironment:
    """
    Gym-style environment for logic training.
    
    This is the main interface that GRPO will interact with.
    """
    
    def __init__(self, task_types: Optional[List[TaskType]] = None,
                 difficulty_range: Tuple[int, int] = (1, 3)):
        """Initialize environment with task configuration."""
        self.task_types = task_types or list(TaskType)
        self.difficulty_range = difficulty_range
        
        self.verifier = LogicVerifier()
        self.sampler = LogicTaskSampler()
        
        self.current_state: Optional[LogicState] = None
        self.step_count = 0
    
    def reset(self) -> LogicState:
        """Reset environment and sample new task."""
        import random
        
        # Sample task type and difficulty
        task_type = random.choice(self.task_types)
        difficulty = random.randint(*self.difficulty_range)
        
        # Generate new task
        self.current_state = self.sampler.sample_task(task_type, difficulty)
        self.step_count = 0
        
        return self.current_state
    
    def step(self, action: LogicAction) -> Tuple[LogicState, float, bool, Dict[str, Any]]:
        """
        Execute action and return (state, reward, done, info).
        
        This is the core GRPO interaction point.
        """
        if self.current_state is None:
            raise ValueError("Environment not initialized. Call reset() first.")
        
        # Verify action against ground truth
        reward, explanation = self.verifier.verify(self.current_state, action)
        
        # Episode is done after one step (episodic tasks)
        done = True
        self.step_count += 1
        
        # Additional info for debugging/analysis
        info = {
            "explanation": explanation,
            "task_type": self.current_state.task_type.value,
            "difficulty": self.current_state.difficulty,
            "step_count": self.step_count,
            "ground_truth": self.current_state.ground_truth,
        }
        
        return self.current_state, reward, done, info
    
    def render(self, mode: str = "human") -> Optional[str]:
        """Render current state for debugging."""
        if self.current_state is None:
            return "Environment not initialized"
        
        output = f"""
Logic Environment State:
========================
Task Type: {self.current_state.task_type.value}
Difficulty: {self.current_state.difficulty}
Question: {self.current_state.question}
Ground Truth: {self.current_state.ground_truth}
Step Count: {self.step_count}
"""
        
        if mode == "human":
            print(output)
        return output
    
    def get_observation(self) -> Dict[str, Any]:
        """Get current observation for the agent."""
        if self.current_state is None:
            return {}
        
        return {
            "question": self.current_state.question,
            "task_type": self.current_state.task_type.value,
            "difficulty": self.current_state.difficulty,
        }


def demo_logic_environment():
    """Demonstrate the logic environment."""
    print("🧠 Logic Environment Demo")
    print("=" * 50)
    
    # Create environment
    env = LogicEnvironment()
    
    # Run a few episodes
    for episode in range(3):
        print(f"\n--- Episode {episode + 1} ---")
        
        # Reset and get initial state
        state = env.reset()
        env.render()
        
        # Sample action (simulate agent response)
        if state.task_type == TaskType.SYLLOGISM:
            action = LogicAction(
                reasoning="Following syllogistic logic...",
                answer=state.ground_truth  # Correct answer for demo
            )
        else:
            action = LogicAction(
                reasoning="Applying logical rules...",
                answer=state.ground_truth
            )
        
        # Execute action
        next_state, reward, done, info = env.step(action)
        
        print(f"Action: {action.answer}")
        print(f"Reward: {reward}")
        print(f"Explanation: {info['explanation']}")
        print(f"Done: {done}")


if __name__ == "__main__":
    demo_logic_environment()