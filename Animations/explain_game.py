from manim import *
from config import BACKGROUND_COLOR


class TronEnvironment(Scene):
    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR

        # --- LEFT SIDE: Gameplay Placeholder ---
        # Intentionally left completely blank for your video editor!

        # --- RIGHT SIDE: Setup ---
        right_center = RIGHT * 3.5

        title = (
            Text("The Game: Tron(ish)", font_size=36, weight=BOLD, color=TEAL)
            .to_edge(UP)
            .shift(RIGHT * 3.0)
        )
        self.play(Write(title))
        self.wait(1)

        # --- Section 1: Actions ---
        actions_title = (
            Text("Actions", font_size=24, color=YELLOW)
            .next_to(title, DOWN, buff=0.8)
            .align_to(title, LEFT)
        )

        # Draw 3 simple arrows to represent relative movement
        arrow_fwd = Arrow(start=DOWN, end=UP, color=WHITE, buff=0).scale(0.5)
        arrow_left = Arrow(start=RIGHT, end=LEFT, color=WHITE, buff=0).scale(0.5)
        arrow_right = Arrow(start=LEFT, end=RIGHT, color=WHITE, buff=0).scale(0.5)

        l_text = Text("Left", font_size=18).next_to(arrow_left, DOWN)
        f_text = Text("Forward", font_size=18).next_to(arrow_fwd, DOWN)
        r_text = Text("Right", font_size=18).next_to(arrow_right, DOWN)

        action_group = (
            VGroup(
                VGroup(arrow_left, l_text),
                VGroup(arrow_fwd, f_text),
                VGroup(arrow_right, r_text),
            )
            .arrange(RIGHT, buff=0.5)
            .next_to(actions_title, DOWN, buff=0.4)
            .align_to(actions_title, LEFT)
        )

        self.play(Write(actions_title))
        self.play(FadeIn(action_group, shift=UP))
        self.wait(1.5)

        # --- Section 2: Rewards ---
        rewards_title = (
            Text("Rewards", font_size=24, color=YELLOW)
            .next_to(action_group, DOWN, buff=0.8)
            .align_to(title, LEFT)
        )

        survive_text = Text("+0.05", font_size=24, color=GREEN)
        survive_desc = Text("Every step survived", font_size=18).next_to(
            survive_text, RIGHT, buff=0.3
        )
        survive_group = (
            VGroup(survive_text, survive_desc)
            .next_to(rewards_title, DOWN, buff=0.4)
            .align_to(rewards_title, LEFT)
        )

        death_text = Text("-1.00", font_size=24, color=RED)
        death_desc = Text("Hitting a wall/trail", font_size=18).next_to(
            death_text, RIGHT, buff=0.3
        )
        death_group = (
            VGroup(death_text, death_desc)
            .next_to(survive_group, DOWN, buff=0.3)
            .align_to(rewards_title, LEFT)
        )

        self.play(Write(rewards_title))
        self.play(Write(survive_group))
        self.play(Write(death_group))
        self.wait(2)

        # Clear space for the Observation explanation
        self.play(
            FadeOut(actions_title),
            FadeOut(action_group),
            FadeOut(rewards_title),
            FadeOut(survive_group),
            FadeOut(death_group),
        )

        # --- Section 3: Observation & Egocentric View ---
        obs_title = (
            Text("The Observation", font_size=24, color=YELLOW)
            .next_to(title, DOWN, buff=0.8)
            .align_to(title, LEFT)
        )

        # BIGGER CHANNELS: Stacked text inside larger boxes
        ch1_t = Text("Ch 1:\nHead", font_size=22, line_spacing=1)
        ch1 = SurroundingRectangle(ch1_t, color=GREEN, buff=0.3)
        ch1_group = VGroup(ch1, ch1_t)

        ch2_t = Text("Ch 2:\nTrail", font_size=22, line_spacing=1)
        ch2 = SurroundingRectangle(ch2_t, color=BLUE, buff=0.3)
        ch2_group = VGroup(ch2, ch2_t).next_to(ch1_group, RIGHT, buff=0.3)

        ch3_t = Text("Ch 3:\nWalls", font_size=22, line_spacing=1)
        ch3 = SurroundingRectangle(ch3_t, color=RED, buff=0.3)
        ch3_group = VGroup(ch3, ch3_t).next_to(ch2_group, RIGHT, buff=0.3)

        channels = (
            VGroup(ch1_group, ch2_group, ch3_group)
            .next_to(obs_title, DOWN, buff=0.5)
            .align_to(obs_title, LEFT)
        )

        self.play(Write(obs_title))
        self.play(FadeIn(channels, shift=UP))
        self.wait(1)

        # Explain the Egocentric trick
        ego_title = (
            Text("Egocentric Rotation:", font_size=20, weight=BOLD)
            .next_to(channels, DOWN, buff=0.8)
            .align_to(obs_title, LEFT)
        )
        ego_desc = (
            Text(
                "The board always rotates so\nthe head is facing UP.",
                font_size=18,
                color=GRAY,
            )
            .next_to(ego_title, DOWN, buff=0.2)
            .align_to(obs_title, LEFT)
        )

        self.play(Write(ego_title), Write(ego_desc))

        # Visualizing the rotation
        # Create a mini grid to represent the board
        grid = (
            NumberPlane(
                x_range=[-2, 2, 1],
                y_range=[-2, 2, 1],
                background_line_style={"stroke_opacity": 0.5, "stroke_width": 2},
            )
            .set_color(BLUE)
            .scale(0.4)
            .next_to(ego_desc, DOWN, buff=0.5)
            .align_to(obs_title, LEFT)
            .shift(RIGHT * 1.5)
        )

        # An arrow representing the Tron bike moving RIGHT
        bike = (
            Arrow(start=LEFT, end=RIGHT, color=GREEN, buff=0)
            .scale(0.5)
            .move_to(grid.get_center())
        )

        grid_group = VGroup(grid, bike)

        facing_right_text = Text("Moving Right...", font_size=14, color=RED).next_to(
            grid_group, DOWN, buff=0.2
        )

        self.play(FadeIn(grid_group), Write(facing_right_text))
        self.wait(1)

        # Rotate the board to face UP
        rotating_text = Text("Network sees: UP", font_size=14, color=GREEN).next_to(
            grid_group, DOWN, buff=0.2
        )

        self.play(
            grid_group.animate.rotate(PI / 2),  # Rotate 90 degrees counter-clockwise
            ReplacementTransform(facing_right_text, rotating_text),
            run_time=1.5,
        )
        self.wait(3)
