#!/usr/bin/env python3
"""
Configuration class for activation analysis.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AnalysisConfig:
    """Configuration for activation analysis."""
    
    # Debug flags
    debug: bool = False
    verbose: bool = True
    
    # Analysis parameters
    threshold: float = 0.8
    max_samples: int = 50
    
    # Tokenizer settings
    tokenizer_path: str = "aimagelab/LLaVA_MORE-llama_3_1-8B-finetuning"
    
    # Output settings
    save_plots: bool = True
    plot_dpi: int = 150
    
    def log(self, message: str, level: str = "INFO"):
        """Log message if verbose is enabled."""
        if self.verbose:
            print(f"[{level}] {message}")
    
    def debug_log(self, message: str):
        """Log debug message if debug is enabled."""
        if self.debug:
            print(f"[DEBUG] {message}") 