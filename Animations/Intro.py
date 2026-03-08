from manim import *


class Intro(Scene):

    def construct(self):
        self.camera.background_color = "#0A0B0D"

        title = Text(
            "Video Outline",
            font_size=32,
            weight=BOLD,
            color=WHITE,
        ).to_edge(UP)

        self.play(Write(title, run_time=1))
        self.wait(4)

        rl_text = Text(
            "Basics of Reinforcement Learning",
            font_size=26,
            weight=BOLD,
            color=BLUE,
        )

        muzero_text = Text(
            "MuZero Explained + Implementation",
            font_size=26,
            weight=BOLD,
            color=RED,
        )

        game_text = Text(
            "Results of MuZero on a custom game",
            font_size=26,
            weight=BOLD,
            color=YELLOW,
        )

        rl_text.shift(UP * 1.5)
        muzero_text.next_to(rl_text, DOWN, buff=1)
        game_text.next_to(muzero_text, DOWN, buff=1)

        self.play(Write(rl_text, run_time=1))
        self.wait(3)
        self.play(Write(muzero_text, run_time=1))
        self.wait(3)
        self.play(Write(game_text, run_time=1))
        self.wait(4)
        self.play(Write(Group(title, rl_text, muzero_text, game_text)))
