from manim import *


class MuZero_Explained(Scene):
    def construct(self):
        self.camera.background_color = "#0A0B0D"

        self.show_title()
        self.show_architecture()

    def show_title(self):
        title = Text(
            "MuZero",
            font_size=48,
            weight=BOLD,
            color=RED,
        )

        self.play(Write(title))
        self.wait(4)
        self.play(FadeOut(title))

    def show_architecture(self):

        title = Text(
            "MuZero's Architecture",
            font_size=32,
            weight=BOLD,
            color=WHITE,
        ).to_edge(UP)

        circle_size = 1.2

        rep_network_circle = Circle(
            color=BLUE, fill_opacity=0.5, radius=circle_size
        ).shift(LEFT * 3)
        rep_label = Paragraph(
            "Representation",
            "Network",
            font_size=24,
            line_spacing=0.5,
            alignment="center",
        ).move_to(rep_network_circle.get_center())

        dyn_network_circle = Circle(color=GREEN, fill_opacity=0.5, radius=circle_size)
        dyn_label = Paragraph(
            "Dynamics",
            "Network",
            font_size=24,
            line_spacing=0.5,
            alignment="center",
        ).move_to(dyn_network_circle.get_center())

        pred_network_circle = Circle(
            color=YELLOW, fill_opacity=0.5, radius=circle_size
        ).shift(RIGHT * 3)
        pred_label = Paragraph(
            "Prediction",
            "Network",
            font_size=24,
            line_spacing=0.5,
            alignment="center",
        ).move_to(pred_network_circle.get_center())

        self.play(Write(title))
        self.play(Create(rep_network_circle), Write(rep_label))
        self.play(Create(dyn_network_circle), Write(dyn_label))
        self.play(Create(pred_network_circle), Write(pred_label))
        self.wait(4)

        self.play(
            FadeOut(
                Group(title, dyn_network_circle, dyn_label, pred_network_circle, pred_label)
            )
        )

        # Move representation network to the center
        self.play(
            rep_network_circle.animate.move_to(ORIGIN),
            rep_label.animate.move_to(ORIGIN),
        )
        self.wait(4)
