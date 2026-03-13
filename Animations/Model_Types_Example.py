from manim import *
from config import BACKGROUND_COLOR


class ModelTypesExample(Scene):
    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR

        self.left_center = LEFT * config.frame_width / 4
        self.right_center = RIGHT * config.frame_width / 4

        self.show_image()
        self.show_text()

    def show_image(self):
        pacman_image = ImageMobject("resources/pacman.png")
        pacman_image.set_height(config.frame_height * 0.7)

        border = SurroundingRectangle(
            pacman_image, color=BLUE, buff=0.1, stroke_width=4, corner_radius=0.2
        )

        image_group = Group(pacman_image, border)
        image_group.move_to(self.left_center)

        self.play(FadeIn(image_group))
        self.wait(6)

    def show_text(self):
        title_free = Text("Model-Free Agent:", font_size=32, weight=BOLD, color=ORANGE)
        free_subtitle = MarkupText(
            'I know from my experience that turning <span foreground="yellow">right</span> in this\n'
            'position will probably lead to a <span foreground="red">game over</span>.\n'
            "\nI don't know why, but I just know it.",
            font_size=22,
            color=WHITE,
        )
        # Ensure the subtitle aligns to the LEFT of the title
        free_block = VGroup(title_free, free_subtitle).arrange(
            DOWN, aligned_edge=LEFT, buff=0.3
        )

        title_based = Text("Model-Based Agent:", font_size=32, weight=BOLD, color=BLUE)
        based_subtitle = MarkupText(
            "I have a model of the game, and I can simulate\n"
            'what happens if I turn <span foreground="yellow">right</span>. \n'
            '\nI can see that it leads to a <span foreground="red">game over</span>,\nso I won\'t turn right.',
            font_size=22,
            color=WHITE,
        )
        based_block = VGroup(title_based, based_subtitle).arrange(
            DOWN, aligned_edge=LEFT, buff=0.3
        )

        all_text = VGroup(free_block, based_block).arrange(
            DOWN, aligned_edge=LEFT, buff=1.2
        )
        all_text.move_to(self.right_center)

        # Animations
        self.play(Write(title_free))
        self.play(FadeIn(free_subtitle, shift=UP * 0.2))
        self.wait(6)
        self.play(Write(title_based))
        self.play(FadeIn(based_subtitle, shift=UP * 0.2))
        self.wait(6)
