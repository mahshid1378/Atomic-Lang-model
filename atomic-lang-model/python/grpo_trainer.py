#!/usr/bin/env python3
"""
GRPO Trainer
============

Group Relative Policy Optimization trainer for the atomic language model.
Implements David Kypuros's recommendations for GRPO/RLVR integration.

Key features:
- Quantized SLM with LoRA adapters for memory efficiency
- Group-based advantage computation (no value network needed)
- CPU-based verifier rewards (no GPU needed for rollouts)
- Batch accumulation for efficient updates
- Designed for commodity hardware (<6GB VRAM)

Architecture:
- Policy: 125-350M parameter SLM with 4-bit quantization
- LoRA: Low-rank adapters for efficient updates
- Verifier: Pure CPU logic engine (microsecond checks)
- Batching: Accumulate 8-16MB of token data per update
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    BitsAndBytesConfig, TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import json
import time
from pathlib import Path

from logic_env import LogicEnvironment, LogicAction, LogicState, TaskType


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""
    # Model configuration
    model_name: str = "microsoft/DialoGPT-small"  # 117M params
    max_length: int = 512
    
    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = None
    
    # Quantization configuration
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    
    # GRPO hyperparameters
    group_size: int = 6
    learning_rate: float = 1e-5
    clip_ratio: float = 0.2
    max_grad_norm: float = 1.0
    
    # Training configuration
    batch_size: int = 4
    accumulation_steps: int = 4
    max_episodes_per_batch: int = 64
    target_batch_tokens: int = 8 * 1024 * 1024  # 8MB of token data
    
    # Environment configuration
    task_types: List[str] = None
    difficulty_range: Tuple[int, int] = (1, 3)
    
    def __post_init__(self):
        """Set default values."""
        if self.target_modules is None:
            self.target_modules = ["c_attn", "c_proj", "c_fc"]
        if self.task_types is None:
            self.task_types = ["syllogism", "propositional", "agreement"]


@dataclass
class Episode:
    """Single training episode with GRPO data."""
    state: LogicState
    action: LogicAction
    reward: float
    log_prob: float
    tokens: torch.Tensor
    attention_mask: torch.Tensor
    
    
class EpisodeBuffer:
    """Buffer for collecting episodes before GRPO update."""
    
    def __init__(self, max_size: int = 1000):
        self.episodes: List[Episode] = []
        self.max_size = max_size
        
    def add(self, episode: Episode):
        """Add episode to buffer."""
        self.episodes.append(episode)
        if len(self.episodes) > self.max_size:
            self.episodes.pop(0)
    
    def get_groups(self, group_size: int) -> List[List[Episode]]:
        """Group episodes for GRPO computation."""
        groups = []
        for i in range(0, len(self.episodes), group_size):
            group = self.episodes[i:i + group_size]
            if len(group) >= 2:  # Need at least 2 for relative advantages
                groups.append(group)
        return groups
    
    def clear(self):
        """Clear all episodes."""
        self.episodes.clear()
        
    def size(self) -> int:
        """Get number of episodes."""
        return len(self.episodes)
    
    def total_tokens(self) -> int:
        """Get total number of tokens in buffer."""
        return sum(episode.tokens.numel() for episode in self.episodes)


class QuantizedSLM:
    """
    Quantized Small Language Model with LoRA adapters.
    Designed to fit in <6GB VRAM for edge deployment.
    """
    
    def __init__(self, config: GRPOConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Configure quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config.load_in_4bit,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype)
        )
        
        # Load base model with quantization
        self.base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            bias="none"
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.base_model, lora_config)
        self.model.train()
        
        print(f"Model loaded on {self.device}")
        print(f"Trainable parameters: {self.model.get_nb_trainable_parameters()}")
        
    def generate_response(self, prompt: str, max_new_tokens: int = 100) -> Tuple[str, torch.Tensor]:
        """
        Generate response and return text + log probabilities.
        
        Returns:
            response: Generated text
            log_probs: Log probabilities for generated tokens
        """
        # Tokenize prompt
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=self.config.max_length
        ).to(self.device)
        
        # Generate with log probabilities
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Extract generated tokens and compute log probs
        generated_tokens = outputs.sequences[0][inputs.input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Compute log probabilities
        scores = torch.stack(outputs.scores, dim=1)  # [1, seq_len, vocab_size]
        log_probs = F.log_softmax(scores, dim=-1)
        token_log_probs = log_probs[0, range(len(generated_tokens)), generated_tokens]
        
        return generated_text, token_log_probs
    
    def compute_log_probs(self, tokens: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute log probabilities for given tokens."""
        with torch.no_grad():
            outputs = self.model(input_ids=tokens, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = tokens[..., 1:].contiguous()
            
            log_probs = F.log_softmax(shift_logits, dim=-1)
            
            # Gather log probs for actual tokens
            token_log_probs = log_probs.gather(
                dim=-1, 
                index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)
            
            return token_log_probs.sum(dim=-1)  # Sum over sequence


class GRPOTrainer:
    """
    Group Relative Policy Optimization trainer.
    
    Implements the core GRPO algorithm with:
    - Group-based advantage computation
    - Trust region clipping
    - No value network required
    """
    
    def __init__(self, config: GRPOConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model and environment
        self.model = QuantizedSLM(config)
        self.env = LogicEnvironment(
            task_types=[TaskType(t) for t in config.task_types],
            difficulty_range=config.difficulty_range
        )
        
        # Training components
        self.optimizer = torch.optim.AdamW(
            self.model.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        self.episode_buffer = EpisodeBuffer()
        
        # Metrics tracking
        self.training_stats = {
            "episodes": 0,
            "updates": 0,
            "total_reward": 0.0,
            "avg_reward": 0.0,
            "success_rate": 0.0
        }
        
    def collect_episodes(self, n_episodes: int) -> List[Episode]:
        """Collect episodes from environment interaction."""
        episodes = []
        
        for _ in range(n_episodes):
            # Reset environment
            state = self.env.reset()
            
            # Create prompt for the task
            prompt = self._create_prompt(state)
            
            # Generate response
            response, log_probs = self.model.generate_response(prompt, max_new_tokens=50)
            
            # Parse action from response
            action = self._parse_action(response)
            
            # Execute action in environment
            next_state, reward, done, info = self.env.step(action)
            
            # Tokenize full sequence for training
            full_text = prompt + " " + response
            tokens = self.model.tokenizer(
                full_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length
            )
            
            # Create episode
            episode = Episode(
                state=state,
                action=action,
                reward=reward,
                log_prob=log_probs.sum().item(),
                tokens=tokens.input_ids.squeeze(0),
                attention_mask=tokens.attention_mask.squeeze(0)
            )
            
            episodes.append(episode)
            
            # Update stats
            self.training_stats["episodes"] += 1
            self.training_stats["total_reward"] += reward
            self.training_stats["avg_reward"] = (
                self.training_stats["total_reward"] / self.training_stats["episodes"]
            )
            
        return episodes
    
    def compute_grpo_loss(self, episodes: List[Episode]) -> torch.Tensor:
        """
        Compute GRPO loss for a group of episodes.
        
        Key insight: Use group-relative advantages instead of value network.
        """
        if len(episodes) < 2:
            return torch.tensor(0.0, device=self.device)
        
        # Extract rewards and compute advantages within group
        rewards = torch.tensor([ep.reward for ep in episodes], dtype=torch.float32)
        
        # Group-relative advantages (mean 0, std 1 within group)
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Compute log probabilities with current policy
        current_log_probs = []
        old_log_probs = []
        
        for i, episode in enumerate(episodes):
            # Current policy log prob
            current_lp = self.model.compute_log_probs(
                episode.tokens.unsqueeze(0).to(self.device),
                episode.attention_mask.unsqueeze(0).to(self.device)
            )
            current_log_probs.append(current_lp)
            
            # Old policy log prob (from collection)
            old_log_probs.append(torch.tensor(episode.log_prob, device=self.device))
        
        current_log_probs = torch.stack(current_log_probs)
        old_log_probs = torch.stack(old_log_probs)
        
        # Compute probability ratios
        log_ratios = current_log_probs - old_log_probs
        ratios = torch.exp(log_ratios)
        
        # GRPO loss with clipping
        advantages = advantages.to(self.device)
        
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * advantages
        
        # Take minimum (conservative update)
        policy_loss = -torch.min(surr1, surr2).mean()
        
        return policy_loss
    
    def train_step(self) -> Dict[str, float]:
        """Execute one GRPO training step."""
        # Collect episodes until we have enough token data
        all_episodes = []
        
        while self.episode_buffer.total_tokens() < self.config.target_batch_tokens:
            episodes = self.collect_episodes(self.config.batch_size)
            for ep in episodes:
                self.episode_buffer.add(ep)
        
        # Group episodes for GRPO computation
        groups = self.episode_buffer.get_groups(self.config.group_size)
        
        if not groups:
            return {"loss": 0.0, "groups": 0}
        
        # Compute loss over all groups
        total_loss = torch.tensor(0.0, device=self.device)
        
        for group in groups:
            loss = self.compute_grpo_loss(group)
            total_loss += loss
        
        avg_loss = total_loss / len(groups)
        
        # Backward pass with gradient accumulation
        avg_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.model.parameters(), 
            self.config.max_grad_norm
        )
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Clear buffer
        self.episode_buffer.clear()
        
        # Update stats
        self.training_stats["updates"] += 1
        
        return {
            "loss": avg_loss.item(),
            "groups": len(groups),
            "episodes": len(all_episodes)
        }
    
    def _create_prompt(self, state: LogicState) -> str:
        """Create prompt for the given task state."""
        prompt_templates = {
            TaskType.SYLLOGISM: "Solve this syllogism:\n{question}\n\nReasoning:",
            TaskType.PROPOSITIONAL: "Evaluate this propositional argument:\n{question}\n\nAnswer:",
            TaskType.AGREEMENT: "Fix the agreement in this sentence:\n{question}\n\nCorrected:",
            TaskType.MOVEMENT: "Transform this sentence:\n{question}\n\nResult:",
        }
        
        template = prompt_templates.get(state.task_type, "Solve: {question}\nAnswer:")
        return template.format(question=state.question)
    
    def _parse_action(self, response: str) -> LogicAction:
        """Parse model response into action format."""
        # Simple parsing - could be more sophisticated
        lines = response.strip().split('\n')
        
        reasoning = ""
        answer = ""
        
        for line in lines:
            if line.strip():
                if not answer:  # First non-empty line is the answer
                    answer = line.strip()
                else:  # Subsequent lines are reasoning
                    reasoning += line.strip() + " "
        
        return LogicAction(
            reasoning=reasoning.strip(),
            answer=answer or response.strip()
        )
    
    def evaluate(self, n_episodes: int = 100) -> Dict[str, float]:
        """Evaluate model performance."""
        correct = 0
        total = 0
        
        for _ in range(n_episodes):
            state = self.env.reset()
            prompt = self._create_prompt(state)
            response, _ = self.model.generate_response(prompt, max_new_tokens=50)
            action = self._parse_action(response)
            
            _, reward, _, _ = self.env.step(action)
            
            if reward > 0:
                correct += 1
            total += 1
        
        success_rate = correct / total if total > 0 else 0.0
        self.training_stats["success_rate"] = success_rate
        
        return {
            "success_rate": success_rate,
            "correct": correct,
            "total": total
        }
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_stats": self.training_stats,
            "config": self.config
        }
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_stats = checkpoint["training_stats"]
        
        print(f"Checkpoint loaded from {path}")


def main():
    """Main training loop."""
    print("🚀 GRPO Training for Atomic Language Model")
    print("=" * 60)
    
    # Configuration
    config = GRPOConfig(
        model_name="microsoft/DialoGPT-small",
        group_size=6,
        learning_rate=5e-6,
        batch_size=4,
        target_batch_tokens=4 * 1024 * 1024,  # 4MB for demo
        task_types=["syllogism", "propositional"]
    )
    
    # Initialize trainer
    trainer = GRPOTrainer(config)
    
    # Training loop
    max_updates = 100
    eval_every = 10
    
    print(f"Starting training for {max_updates} updates...")
    
    for update in range(max_updates):
        # Training step
        train_metrics = trainer.train_step()
        
        print(f"Update {update + 1}/{max_updates} - "
              f"Loss: {train_metrics['loss']:.4f}, "
              f"Groups: {train_metrics['groups']}")
        
        # Evaluation
        if (update + 1) % eval_every == 0:
            eval_metrics = trainer.evaluate(n_episodes=20)
            print(f"Evaluation - Success Rate: {eval_metrics['success_rate']:.3f}")
            
            # Save checkpoint
            checkpoint_path = f"grpo_checkpoint_{update + 1}.pt"
            trainer.save_checkpoint(checkpoint_path)
    
    print("\n✅ Training completed!")
    
    # Final evaluation
    final_eval = trainer.evaluate(n_episodes=100)
    print(f"Final Success Rate: {final_eval['success_rate']:.3f}")


if __name__ == "__main__":
    main()