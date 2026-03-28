from manim import *
from config import BACKGROUND_COLOR


class Intro(Scene):
    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR

        # 1. The Subtitle (Course Info)
        course_text = Text(
            "AI Programming (IT-3105) Spring 2025 Main Project",
            font_size=24,
            color=GRAY,
        )

        # 2. The Main Title
        main_title = Text("A MuZero Knockoff", font_size=64, weight=BOLD, color=TEAL)

        # Arrange them vertically
        title_group = VGroup(course_text, main_title).arrange(DOWN, buff=0.5)

        # Animate them in cleanly
        self.play(FadeIn(course_text, shift=UP * 0.5), run_time=1)
        self.play(Write(main_title), run_time=1.5)
        self.wait(2.5)

        # Fade out
        self.play(FadeOut(title_group))
        self.wait(0.5)


class Credits(Scene):
    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR

        # 1. The Header
        header = Text("Credits", font_size=48, weight=BOLD, color=YELLOW).to_edge(
            UP, buff=1.0
        )

        # 2. The "Everything" section
        role_text = Text("Everything", font_size=24, color=GRAY)
        name_text = Text("Ola Johannes Elvedahl", font_size=40, color=WHITE)

        credits_group = VGroup(role_text, name_text).arrange(DOWN, buff=0.2)

        # Animate
        self.play(FadeIn(header, shift=DOWN * 0.5))
        self.play(FadeIn(credits_group, shift=UP * 0.5))

        self.wait(3)
        self.play(FadeOut(VGroup(header, credits_group)))
        self.wait(0.5)
