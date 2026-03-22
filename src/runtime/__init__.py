"""Runtime bridge for live, playback, and synthetic EEG streams."""

from src.runtime.adaptation import AdaptationLayer, AdaptationProfile, default_profile
from src.runtime.baseline import BaselineInference
from src.runtime.contracts import DecisionOutput, DecisionScores, EEGChunk
from src.runtime.decision_engine import DecisionEngine
from src.runtime.eeg_frame import EEGFrame
from src.runtime.engine import EngineOutput, StreamingEngine
from src.runtime.postprocessor import DecisionPostprocessor
from src.runtime.router import SourceRouter
from src.runtime.window_buffer import WindowBuffer

__all__ = [
    "AdaptationLayer",
    "AdaptationProfile",
    "BaselineInference",
    "DecisionEngine",
    "DecisionOutput",
    "DecisionPostprocessor",
    "DecisionScores",
    "EEGChunk",
    "EEGFrame",
    "EngineOutput",
    "SourceRouter",
    "StreamingEngine",
    "WindowBuffer",
    "default_profile",
]
