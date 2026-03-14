from manim import *
from config import BACKGROUND_COLOR


class Intro(Scene):
    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR

        # --- 1. CENTER TITLE (The "Hub") ---
        title = Text("IT3105 AI Programming", font_size=42, weight=BOLD)
        subtitle = Text("Project 2", font_size=32, color=ORANGE).next_to(
            title, DOWN, buff=0.2
        )

        center_group = VGroup(title, subtitle).move_to(ORIGIN).set_z_index(2)

        # A hidden background rectangle to ensure lines don't strike through the text
        center_bg = BackgroundRectangle(
            center_group, color=BACKGROUND_COLOR, fill_opacity=1, buff=0.3
        ).set_z_index(1)

        # Animate the center title in
        self.play(FadeIn(center_bg), FadeIn(center_group, shift=UP))

        # Speak: "In the course IT3105 AI Programming, I was tasked to implement a MuZero model..."
        self.wait(3.5)

        # --- 2. CREATE THE SCATTERED NODES ---
        # Helper function to generate identical circle nodes
        def create_node(text_str, position, ring_color):
            circle = Circle(radius=1.3, color=ring_color, stroke_width=4)
            circle.set_fill(ring_color, opacity=0.1)

            # Using \n to break the text into two lines so it fits the circle
            node_text = Text(text_str, font_size=20, line_spacing=1).move_to(
                circle.get_center()
            )

            node_group = VGroup(circle, node_text).move_to(position).set_z_index(2)

            # Create a line from the center to this node
            line = Line(
                ORIGIN, position, color=GRAY, stroke_opacity=0.4, stroke_width=2
            ).set_z_index(0)

            return node_group, line

        # Define the nodes (Scattered in a triangle shape around the center)
        node1, line1 = create_node("How MuZero\nWorks", LEFT * 4.5 + UP * 1.8, ORANGE)
        node2, line2 = create_node(
            "Model\nImplementation", RIGHT * 4.5 + UP * 1.8, BLUE
        )
        node3, line3 = create_node("Custom Game\nResults", DOWN * 2.8, GREEN)

        # --- 3. ANIMATE VIDEO GOALS ---

        # Speak: "...The goal of this video is to teach you how MuZero works..."
        self.play(Create(line1), run_time=0.5)
        self.play(FadeIn(node1, scale=0.5))
        self.wait(1.5)

        # Speak: "...and show you my implementation of the model..."
        self.play(Create(line2), run_time=0.5)
        self.play(FadeIn(node2, scale=0.5))
        self.wait(1.5)

        # Speak: "...along with the results on a custom game."
        self.play(Create(line3), run_time=0.5)
        self.play(FadeIn(node3, scale=0.5))
        self.wait(2.5)

        # --- 4. CLEAN FADE OUT ---
        self.play(
            FadeOut(center_group),
            FadeOut(center_bg),
            FadeOut(node1),
            FadeOut(line1),
            FadeOut(node2),
            FadeOut(line2),
            FadeOut(node3),
            FadeOut(line3),
        )
        self.wait(1)
