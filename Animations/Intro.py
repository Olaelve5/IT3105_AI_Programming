from manim import *


class Intro(Scene):

    def construct(self):
        self.camera.background_color = "#0A0B0D"

        title = Text(
            "Reinforcement Learning",
            font_size=48,
            weight=BOLD,
            color=BLUE,
        )

        self.play(Write(title))
        self.wait(4)
        self.play(FadeOut(title))


        