"""
arena.py — Pygame visualization engine for Dynamic Epistemic Foraging.

Renders (layered bottom to top)
---------------------------------
  1. Background grid + obstacles
  2. Resource nodes (green circles with animated glow)
  3. D* Lite trust hazard heatmap (semi-transparent red gradient overlay)
  4. Planned path per agent (dashed polyline in agent color)
  5. Trust aura around distrusted agents (expanding red pulse)
  6. Agent body circle (color-coded by trust: blue=trusted → red=distrusted)
  7. Heading arrow (continuous-angle omnidirectional)
  8. Intent arrow toward highest-posterior goal (faint dashed)
  9. Agent label: ID + FSM state + trust score
 10. HUD bar: tick counter, MITM stats, per-agent mini-bars, attack indicator
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import pygame
import pygame.gfxdraw


# ------------------------------------------------------------------ #
# Color palette                                                        #
# ------------------------------------------------------------------ #
BG_COLOR        = (12,  14,  24)
GRID_COLOR      = (22,  25,  40)
OBS_COLOR       = (55,  58,  75)
OBS_BORDER      = (75,  78,  98)
RES_COLOR       = (46, 204, 113)
RES_GLOW_INNER  = (39, 174,  96)
RES_SPENT_COLOR = (60,  62,  72)
HEADING_COLOR   = (255, 255, 255)
FONT_COLOR      = (210, 215, 235)
ATTACK_COLOR    = (231,  76,  60)
HUD_BG          = (8,   10,  18)
PATH_ALPHA      = 140


# ── Agent colors per ID ────────────────────────────────────────────── #
_AGENT_BASE_COLORS = [
    (52, 152, 219),    # A0 — sky blue
    (155,  89, 182),   # A1 — purple
    (230, 126,  34),   # A2 — orange
    (26, 188, 156),    # A3 — teal
]


def _trust_to_color(trust: float, agent_id: int = 0) -> Tuple[int,int,int]:
    """
    Interpolate from the agent's base color (trusted) toward red (distrusted).
    At trust=1.0 → agent base color.
    At trust=0.4 → yellow-orange.
    At trust=0.0 → deep red.
    """
    t = max(0.0, min(1.0, trust))
    base = _AGENT_BASE_COLORS[agent_id % len(_AGENT_BASE_COLORS)]
    if t >= 0.5:
        # Lerp: base → yellow
        f = (t - 0.5) / 0.5
        r = int(base[0] * f + 220 * (1 - f))
        g = int(base[1] * f + 200 * (1 - f))
        b = int(base[2] * f + 40  * (1 - f))
    else:
        # Lerp: yellow → red
        f = t / 0.5
        r = 220
        g = int(200 * f)
        b = int(40  * f)
    return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))


class Arena:
    """
    Pygame-based visualization engine with full research overlay.

    Parameters
    ----------
    bounds    : (width, height) in pixels.
    obstacles : List of obstacle rects [(x, y, w, h)].
    resources : List of resource circles [(x, y, radius)].
    """

    HUD_HEIGHT = 70

    def __init__(
        self,
        bounds:    Tuple[int,int],
        obstacles: List,
        resources: List,
    ) -> None:
        self.width, self.height = bounds
        self.obstacles = obstacles
        self.resources = resources   # mutable — reflects availability

        pygame.init()
        pygame.font.init()

        total_h = self.height + self.HUD_HEIGHT
        self.screen = pygame.display.set_mode((self.width, total_h))
        pygame.display.set_caption("Dynamic Epistemic Foraging — Research Simulation")
        self.clock  = pygame.time.Clock()

        self._font_xs = pygame.font.SysFont("Consolas", 10)
        self._font_sm = pygame.font.SysFont("Consolas", 11)
        self._font_md = pygame.font.SysFont("Consolas", 13)
        self._font_lg = pygame.font.SysFont("Consolas", 15, bold=True)

        # Cached static background (grid + obstacles + initial resources)
        self._bg_surface: Optional[pygame.Surface] = None
        self._build_bg()

        # Tick counter for animations
        self._render_tick = 0

    # ------------------------------------------------------------------ #
    # Static background (rebuilt on demand)                               #
    # ------------------------------------------------------------------ #
    def _build_bg(self) -> None:
        surf = pygame.Surface((self.width, self.height))
        surf.fill(BG_COLOR)
        # Grid
        for x in range(0, self.width, 40):
            pygame.draw.line(surf, GRID_COLOR, (x, 0), (x, self.height))
        for y in range(0, self.height, 40):
            pygame.draw.line(surf, GRID_COLOR, (0, y), (self.width, y))
        # Obstacles
        for obs in self.obstacles:
            rect = pygame.Rect(obs[0], obs[1], obs[2], obs[3])
            pygame.draw.rect(surf, OBS_COLOR,  rect)
            pygame.draw.rect(surf, OBS_BORDER, rect, 2)
        self._bg_surface = surf

    # ------------------------------------------------------------------ #
    # Main render                                                          #
    # ------------------------------------------------------------------ #
    def render(
        self,
        agents:            Dict,             # {agent_id: AgentState}
        trust_map:         Dict[str, float], # agent_label → trust [0,1]
        tick:              int   = 0,
        attack_active:     bool  = False,
        paths:             Optional[Dict[int, List[Tuple[int,int]]]] = None,
        hazard_fields:     Optional[Dict[int, Dict[str,float]]]      = None,
        intent_posteriors: Optional[Dict[int, Dict[str, Dict[str,float]]]] = None,
        goals:             Optional[Dict[str, Tuple[float,float]]]    = None,
        mitm_stats:        Optional[dict] = None,
        scores:            Optional[Dict[int, int]] = None,
        fsm_states:        Optional[Dict[int, str]] = None,
        resource_states:   Optional[List[bool]]     = None,
    ) -> bool:
        """
        Render one frame.  Returns False if the window was closed.

        Parameters
        ----------
        agents            : {agent_id: AgentState}
        trust_map         : Flat trust map (agent label → trust) from A0's view.
        tick              : Current simulation tick.
        attack_active     : Ground-truth attack window flag.
        paths             : Per-agent planned grid paths from D* Lite.
        hazard_fields     : Per-agent hazard snapshots {"x,y": cost}.
        intent_posteriors : Per-agent, per-peer intent posteriors from ToM.
        goals             : Named goal positions for intent arrow rendering.
        mitm_stats        : MITM proxy counters.
        scores            : Per-agent resource scores.
        fsm_states        : Per-agent FSM state strings.
        resource_states   : List of booleans: resources[i].available.
        """
        # ── Event handling ────────────────────────────────────────────── #
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False

        self._render_tick += 1

        # ── Blit static background ────────────────────────────────────── #
        self.screen.fill(BG_COLOR)
        self.screen.blit(self._bg_surface, (0, 0))

        # ── Layer 1: Hazard heatmap ───────────────────────────────────── #
        if hazard_fields:
            # Use agent 0's hazard view (primary observer)
            hf = hazard_fields.get(0, {})
            if hf:
                self._draw_hazard_heatmap(hf)

        # ── Layer 2: Resources ────────────────────────────────────────── #
        for i, res in enumerate(self.resources):
            available = True
            if resource_states and i < len(resource_states):
                available = resource_states[i]
            self._draw_resource(int(res[0]), int(res[1]), int(res[2]), available)

        # ── Layer 3: Planned paths ────────────────────────────────────── #
        if paths:
            for agent_id, path in paths.items():
                if len(path) > 1:
                    color = _AGENT_BASE_COLORS[agent_id % len(_AGENT_BASE_COLORS)]
                    self._draw_dashed_path(path, color)

        # ── Layer 4: Agents ───────────────────────────────────────────── #
        for agent_id, state in agents.items():
            agent_label = f"A{agent_id}"
            trust       = trust_map.get(agent_label, 1.0)
            color       = _trust_to_color(trust, agent_id)

            cx = int(state.x)
            cy = int(state.y)

            # Trust aura (animated pulse for distrusted agents)
            if trust < 0.5:
                self._draw_trust_aura(cx, cy, trust)

            # Agent body
            pygame.draw.circle(self.screen, color, (cx, cy), 12)
            pygame.draw.circle(self.screen, (255, 255, 255), (cx, cy), 12, 2)

            # Heading arrow — full continuous angle
            hx = cx + int(20 * math.cos(state.heading))
            hy = cy + int(20 * math.sin(state.heading))
            pygame.draw.line(self.screen, HEADING_COLOR, (cx, cy), (hx, hy), 3)
            # Arrow tip
            self._draw_arrowhead(cx, cy, hx, hy, color=(255, 255, 255), size=5)

            # Intent arrow (toward highest-posterior goal)
            if intent_posteriors and goals:
                agent_intents = intent_posteriors.get(agent_id, {})
                # Use the intent posteriors this agent has for all peers
                # Average them or show the current best intent
                best_goal_name, best_goal_p = self._best_intent(agent_intents)
                if best_goal_name and best_goal_name in goals and best_goal_p > 0.5:
                    gx, gy = goals[best_goal_name]
                    self._draw_intent_arrow(cx, cy, int(gx), int(gy), best_goal_p)

            # Label: ID | FSM state | trust
            fsm = (fsm_states or {}).get(agent_id, state.status)
            score = (scores or {}).get(agent_id, 0)
            label_str = f"A{agent_id} [{fsm[:3]}] {trust:.2f}"
            label_surf = self._font_sm.render(label_str, True, FONT_COLOR)
            # Shadow
            shadow_surf = self._font_sm.render(label_str, True, (0, 0, 0))
            self.screen.blit(shadow_surf, (cx + 15, cy - 9))
            self.screen.blit(label_surf,  (cx + 14, cy - 10))

            # Score chip
            score_surf = self._font_xs.render(f"★{score}", True, (255, 215, 0))
            self.screen.blit(score_surf, (cx + 14, cy + 3))

        # ── HUD bar ───────────────────────────────────────────────────── #
        self._draw_hud(tick, mitm_stats, attack_active, trust_map, agents, scores)

        pygame.display.flip()
        return True

    # ------------------------------------------------------------------ #
    # Drawing helpers                                                      #
    # ------------------------------------------------------------------ #
    def _draw_hazard_heatmap(self, hazard: Dict[str, float]) -> None:
        """
        Render the D* Lite hazard field as a semi-transparent red gradient.
        Higher hazard cost = more opaque red.  Uses a single SRCALPHA surface
        so the background grid remains visible.
        """
        if not hazard:
            return
        max_cost = max(hazard.values()) if hazard else 1.0
        heat_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        for cell_str, cost in hazard.items():
            try:
                cx_s, cy_s = cell_str.split(",")
                hx, hy = int(cx_s), int(cy_s)
            except ValueError:
                continue
            if not (0 <= hx < self.width and 0 <= hy < self.height):
                continue
            intensity = min(1.0, cost / max(max_cost, 0.001))
            alpha     = int(30 + intensity * 110)  # 30..140
            r = int(200 + intensity * 55)
            g = int(30  * (1 - intensity))
            b = 20
            pygame.draw.rect(heat_surf, (r, g, b, alpha), (hx - 2, hy - 2, 5, 5))
        self.screen.blit(heat_surf, (0, 0))

    def _draw_resource(self, cx: int, cy: int, radius: int, available: bool) -> None:
        if not available:
            # Spent resource — show faint ring
            pygame.draw.circle(self.screen, RES_SPENT_COLOR, (cx, cy), radius, 1)
            return
        # Animated glow — oscillates with render tick
        pulse = 0.5 + 0.5 * math.sin(self._render_tick * 0.08)
        glow_r = radius + 4 + int(pulse * 4)
        for r_off in (glow_r + 4, glow_r):
            pygame.draw.circle(self.screen, RES_GLOW_INNER, (cx, cy), r_off, 1)
        pygame.draw.circle(self.screen, RES_COLOR, (cx, cy), radius)

    def _draw_trust_aura(self, cx: int, cy: int, trust: float) -> None:
        """Animated pulsing red aura around distrusted agent."""
        pulse      = 0.5 + 0.5 * math.sin(self._render_tick * 0.15)
        base_r     = 35 + int(pulse * 12)
        intensity  = max(0.0, 1.0 - trust * 2.0)  # 0 at trust=0.5, 1 at trust=0
        alpha      = int(40 + intensity * 90)

        aura_surf = pygame.Surface((base_r * 2, base_r * 2), pygame.SRCALPHA)
        pygame.draw.circle(
            aura_surf, (231, 76, 60, alpha),
            (base_r, base_r), base_r
        )
        self.screen.blit(aura_surf, (cx - base_r, cy - base_r))

    def _draw_dashed_path(
        self,
        path:  List[Tuple[int,int]],
        color: Tuple[int,int,int],
        dash:  int = 6,
        gap:   int = 4,
    ) -> None:
        """Draw a dashed polyline for the D* Lite planned path."""
        surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        r, g, b = color
        draw_on = True
        counter = 0
        for i in range(len(path) - 1):
            x0, y0 = path[i]
            x1, y1 = path[i + 1]
            dx, dy = x1 - x0, y1 - y0
            seg_len = math.hypot(dx, dy)
            if seg_len < 0.01:
                continue
            ux, uy  = dx / seg_len, dy / seg_len
            pos     = 0.0
            while pos < seg_len:
                step   = dash if draw_on else gap
                end    = min(pos + step, seg_len)
                if draw_on:
                    px0 = int(x0 + ux * pos)
                    py0 = int(y0 + uy * pos)
                    px1 = int(x0 + ux * end)
                    py1 = int(y0 + uy * end)
                    pygame.draw.line(surf, (r, g, b, 160), (px0, py0), (px1, py1), 2)
                pos     += step
                draw_on = not draw_on
        self.screen.blit(surf, (0, 0))

    def _draw_intent_arrow(
        self, x0: int, y0: int, gx: int, gy: int, posterior: float
    ) -> None:
        """Draw a faint dashed arrow toward the inferred goal."""
        alpha = int(40 + posterior * 120)
        surf  = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        # Clamp arrow length
        dx, dy = gx - x0, gy - y0
        dist = math.hypot(dx, dy)
        if dist < 20:
            return
        scale = min(1.0, 60.0 / dist)
        ex    = int(x0 + dx * scale)
        ey    = int(y0 + dy * scale)
        pygame.draw.line(surf, (255, 220, 50, alpha), (x0, y0), (ex, ey), 1)
        self.screen.blit(surf, (0, 0))

    def _draw_arrowhead(
        self,
        x0: int, y0: int, x1: int, y1: int,
        color: Tuple[int,int,int],
        size:  int = 5,
    ) -> None:
        dx, dy = x1 - x0, y1 - y0
        dist = math.hypot(dx, dy)
        if dist < 0.01:
            return
        ux, uy = dx / dist, dy / dist
        px, py = -uy, ux   # perpendicular
        tip  = (x1, y1)
        l1   = (int(x1 - ux * size + px * size * 0.6), int(y1 - uy * size + py * size * 0.6))
        l2   = (int(x1 - ux * size - px * size * 0.6), int(y1 - uy * size - py * size * 0.6))
        pygame.draw.polygon(self.screen, color, [tip, l1, l2])

    def _best_intent(self, agent_intents: dict) -> Tuple[Optional[str], float]:
        """Find the goal with the highest intent posterior across all peers."""
        best_goal, best_p = None, 0.0
        for peer_id, posteriors in agent_intents.items():
            for goal, p in posteriors.items():
                if p > best_p:
                    best_p, best_goal = p, goal
        return best_goal, best_p

    # ------------------------------------------------------------------ #
    # HUD                                                                  #
    # ------------------------------------------------------------------ #
    def _draw_hud(
        self,
        tick:         int,
        mitm_stats:   Optional[dict],
        attack_active: bool,
        trust_map:    Dict[str, float],
        agents:       Dict,
        scores:       Optional[Dict[int, int]],
    ) -> None:
        hud_y = self.height
        pygame.draw.rect(self.screen, HUD_BG, (0, hud_y, self.width, self.HUD_HEIGHT))
        pygame.draw.line(self.screen, (40, 44, 66), (0, hud_y), (self.width, hud_y), 1)

        # Row 1: Tick counter
        tick_surf = self._font_md.render(f"Tick {tick:05d}", True, FONT_COLOR)
        self.screen.blit(tick_surf, (8, hud_y + 5))

        # Attack indicator
        if attack_active:
            atk_surf  = self._font_lg.render("⚠ MITM ATTACK ACTIVE", True, ATTACK_COLOR)
            atk_w     = atk_surf.get_width()
            self.screen.blit(atk_surf, (self.width - atk_w - 8, hud_y + 5))

        # MITM stats
        if mitm_stats:
            stat_str  = (
                f"MITM  F:{mitm_stats.get('forged',0)}  "
                f"R:{mitm_stats.get('replayed',0)}  "
                f"D:{mitm_stats.get('dropped',0)}"
            )
            c         = ATTACK_COLOR if attack_active else (130, 140, 170)
            stat_surf = self._font_sm.render(stat_str, True, c)
            self.screen.blit(stat_surf, (120, hud_y + 5))

        # Row 2: Per-agent mini trust bars
        bar_y   = hud_y + 26
        bar_x   = 8
        bar_w   = 80
        bar_h   = 10
        spacing = 100

        for agent_id, state in agents.items():
            label   = f"A{agent_id}"
            trust   = trust_map.get(label, 1.0)
            color   = _trust_to_color(trust, agent_id)
            score   = (scores or {}).get(agent_id, 0)

            # Background bar
            pygame.draw.rect(self.screen, (30, 35, 50), (bar_x, bar_y, bar_w, bar_h))
            # Trust fill
            fill_w = int(bar_w * trust)
            pygame.draw.rect(self.screen, color, (bar_x, bar_y, max(1, fill_w), bar_h))
            # Border
            pygame.draw.rect(self.screen, (60, 65, 90), (bar_x, bar_y, bar_w, bar_h), 1)

            # Label
            lbl  = self._font_xs.render(f"{label} {trust:.2f} ★{score}", True, FONT_COLOR)
            self.screen.blit(lbl, (bar_x, bar_y + bar_h + 2))

            bar_x += spacing

        # Legend
        legend_y = hud_y + 52
        for i, (label, color) in enumerate([
            ("Trusted", (52, 152, 219)),
            ("Distrusted", (220, 60, 40)),
            ("Hazard Zone", (180, 30, 20)),
        ]):
            lx = 8 + i * 120
            pygame.draw.rect(self.screen, color, (lx, legend_y, 10, 8))
            ls = self._font_xs.render(label, True, (150, 155, 175))
            self.screen.blit(ls, (lx + 12, legend_y))

    # ------------------------------------------------------------------ #
    # Clock                                                                #
    # ------------------------------------------------------------------ #
    def tick_clock(self, fps: int = 20) -> None:
        self.clock.tick(fps)

    def close(self) -> None:
        pygame.quit()
