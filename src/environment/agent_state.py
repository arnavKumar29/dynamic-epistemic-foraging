"""
agent_state.py — Core agent state dataclass.

Serializable to/from JSON for network broadcast.

Fields
------
agent_id  : Unique integer identifier (0..N-1).
x, y      : Continuous-space position (pixels, floats — NOT snapped to a grid).
heading   : Orientation in radians, measured from east (0) counterclockwise.
            Full 2π range so robots can move in any direction at any angle.
velocity  : Scalar speed (pixels per tick).
status    : FSM state string.
tick      : Simulation tick when this state was recorded.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
import json


@dataclass
class AgentState:
    agent_id: int
    x:        float
    y:        float
    heading:  float          # radians, 0 = east, full 2π range
    velocity: float          # pixels per tick
    status:   str            # "EXPLORE" | "APPROACH" | "COLLECT" | "RETURN" | "DEPOSIT"
    tick:     int = 0        # simulation tick (default 0 for construction)

    # ------------------------------------------------------------------ #
    # Serialization helpers                                                #
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict) -> "AgentState":
        # Support both 'id' (legacy) and 'agent_id'
        if "id" in data and "agent_id" not in data:
            data = dict(data)
            data["agent_id"] = data.pop("id")
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, json_str: str) -> "AgentState":
        return cls.from_dict(json.loads(json_str))

    # ------------------------------------------------------------------ #
    # Movement helpers                                                     #
    # ------------------------------------------------------------------ #
    def advance(self, dt: float = 1.0) -> None:
        """
        Move forward one step along current heading.
        Full continuous-angle motion — heading can be any value in [0, 2π).
        """
        import math
        self.x += self.velocity * math.cos(self.heading) * dt
        self.y += self.velocity * math.sin(self.heading) * dt

    def clamp_to_bounds(self, width: int, height: int) -> None:
        """Reflect heading off arena walls (elastic bounce)."""
        import math
        if self.x < 0 or self.x > width:
            self.heading = math.pi - self.heading
            self.x = max(0.0, min(float(width), self.x))
        if self.y < 0 or self.y > height:
            self.heading = -self.heading
            self.y = max(0.0, min(float(height), self.y))

    # ------------------------------------------------------------------ #
    # Derived properties                                                   #
    # ------------------------------------------------------------------ #
    @property
    def pos(self) -> tuple:
        """Convenience accessor returning (x, y)."""
        return (self.x, self.y)
