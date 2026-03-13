from manim import *


class ModelTypesExample(Scene):
    def construct(self):
        self.camera.background_color = "#1F2225"
        self.show_image()
        self.show_text()

    def show_image(self):
        pacman_image = ImageMobject("resources/pacman.png")
        pacman_image.set_height(config.frame_height * 0.75)
        pacman_image.to_edge(LEFT, buff=0.5)

        border = SurroundingRectangle(
            pacman_image, color=BLUE, buff=0.1, stroke_width=4, corner_radius=0.2
        )

        # Add both to the scene
        self.add(pacman_image, border)

    def show_text(self):
        title_free = Text("Model-Free Method:", font_size=32, weight=BOLD, color=ORANGE)
        title_based = Text("Model-Based Method:", font_size=32, weight=BOLD, color=BLUE)

        free_subtitle = Paragraph(
            "I know from my experience that going right in this position",
            " will probably lead to a game over.",
            "I don't know why, but I just know it.",
            font_size=24,
            color=WHITE,
        )

        based_subtitle = Paragraph(
            "I have a model of the game, and I can simulate what happens if I go right.",
            "I can see that it leads to a game over, so I won't go right.",
            font_size=24,
            color=WHITE,
        )

        free_group = VGroup(title_free, free_subtitle).arrange(DOWN, aligned_edge=LEFT)
        based_group = VGroup(title_based, based_subtitle).arrange(
            DOWN, aligned_edge=LEFT
        )

        free_group.to_edge(UP, buff=0.5).to_edge(RIGHT, buff=0.5)
        based_group.to_edge(DOWN, buff=0.5).to_edge(RIGHT, buff=0.5)

        self.play(Write(title_free))
        self.wait(1)
        self.play(Write(free_subtitle))
        self.wait(2)

        self.play(Write(title_based))
        self.wait(1)
        self.play(Write(based_subtitle))
        self.wait(2)
