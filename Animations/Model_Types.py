from manim import *


class ModelTypes(Scene):
    def construct(self):
        self.camera.background_color = "#0A0B0D"

        self.show_big_title()
        self.show_model_types()

    def show_big_title(self):
        title = Text(
            "Model Types in Reinforcement Learning",
            font_size=48,
            weight=BOLD,
            color=WHITE,
        )

        subtitle = MarkupText(
            f'Model-Free <span fgcolor="{WHITE}">vs.</span> <span fgcolor="{BLUE}">Model-Based</span>',
            color=ORANGE,
            font_size=32,
            weight=BOLD,
        ).next_to(title, DOWN, buff=0.5)

        self.play(Write(title, run_time=1))
        self.wait(2)
        self.play(Write(subtitle, run_time=1))
        self.wait(4)
        self.play(FadeOut(title), FadeOut(subtitle))

    def show_model_types(self):
        # Left Side: Model-Free
        free_title = Text("Model-Free Methods", font_size=32, weight=BOLD, color=ORANGE)
        free_p1 = Text("Learns purely from trial and error", font_size=24)
        free_p2 = Text("No planning or simulation of future states", font_size=24)
        free_p3 = Text("Requires a lot of interactions, often millions", font_size=24)
        free_p4 = Text("Examples: Q-learning, DQN, PPO", font_size=24)
        jellyfish_image = ImageMobject("resources/jellyfish.png").scale(0.12)

        left_group = Group(
            free_title, free_p1, free_p2, free_p3, free_p4, jellyfish_image
        ).arrange(DOWN, buff=0.6)
        left_group.move_to(LEFT * 3.5).to_edge(UP, buff=0.5)

        # Right Side: Model-Based
        based_title = Text("Model-Based Methods", font_size=32, weight=BOLD, color=BLUE)
        based_p1 = Text("Figures out the rules of the environment", font_size=24)
        based_p2 = Text("Simulates and plans for future states", font_size=24)
        based_p3 = Text("Learns from fewer interactions", font_size=24)
        based_p4 = MarkupText(
            f"Examples: AlphaZero, Dyna-Q, <span fgcolor='{RED}'>MuZero</span>",
            font_size=24,
        )
        carlsen_image = ImageMobject("resources/Carlsen.png").scale(0.5)

        right_group = Group(
            based_title, based_p1, based_p2, based_p3, based_p4, carlsen_image
        ).arrange(DOWN, buff=0.6)
        right_group.move_to(RIGHT * 3.5).to_edge(UP, buff=0.5)

        # Divider
        line = Line(start=UP * 4, end=DOWN * 4, color=WHITE)

        # Animations
        self.play(Write(free_title), Write(based_title), Create(line))
        self.wait(2)
        self.play(Write(free_p1))
        self.wait(2)
        self.play(Write(based_p1))
        self.wait(2)
        self.play(Write(free_p2))
        self.wait(2)
        self.play(Write(based_p2))
        self.wait(2)
        self.play(Write(free_p3))
        self.wait(2)
        self.play(Write(based_p3))
        self.wait(4)
        self.play(Write(free_p4))
        self.wait(2)
        self.play(Write(based_p4))
        self.wait(2)
        jellyfish_image.shift(DOWN * 0.5)
        self.play(FadeIn(jellyfish_image))
        self.play(FadeIn(carlsen_image))
        self.wait(4)
