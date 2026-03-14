from manim import *
from config import BACKGROUND_COLOR


class ModelTypes(Scene):
    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR
        self.show_big_title()
        self.show_model_types()

    def show_big_title(self):
        title = Text(
            "Model-Free vs. Model-Based",
            font_size=44,
            weight=BOLD,
            t2c={"Model-Free": ORANGE, "Model-Based": BLUE},
        )
        self.play(Write(title))
        self.wait(2)
        self.play(FadeOut(title))

    def show_model_types(self):
        line = Line(start=UP * 4, end=DOWN * 4, color=GRAY, stroke_opacity=0.5)

        # --- TOP TEXT ---
        free_header = (
            VGroup(
                Text("Model-Free", font_size=32, color=ORANGE, weight=BOLD),
                Text(
                    "Decides purely based on past experience",
                    font_size=20,
                    color=GRAY_B,
                ),
            )
            .arrange(DOWN, buff=0.2)
            .move_to(LEFT * 3.5 + UP * 2.8)
        )

        based_header = (
            VGroup(
                Text("Model-Based", font_size=32, color=BLUE, weight=BOLD),
                Text("Decides based on future  planning", font_size=20, color=GRAY_B),
            )
            .arrange(DOWN, buff=0.2)
            .move_to(RIGHT * 3.5 + UP * 2.8)
        )

        self.play(Create(line), FadeIn(free_header), FadeIn(based_header))

        # --- 1. MODEL-FREE: VERTICAL FLOW ---
        mf_center = LEFT * 3.5 + UP * 1.2

        state_mf = Text("Environment State", font_size=20).move_to(mf_center)

        exp_box = RoundedRectangle(
            width=3.5, height=1.5, color=ORANGE, corner_radius=0.1, fill_opacity=0.3
        )
        exp_text = Text("Past Experiences", font_size=20).move_to(exp_box)
        exp_group = VGroup(exp_box, exp_text).next_to(state_mf, DOWN, buff=0.8)

        action_mf = Text("Action", font_size=20, color=ORANGE, weight=BOLD).next_to(
            exp_group, DOWN, buff=1.2
        )

        arrow_in_mf = Arrow(
            state_mf.get_bottom(),
            exp_group.get_top(),
            color=WHITE,
            buff=0.1,
            tip_length=0.15,
        )
        arrow_out_mf = Arrow(
            exp_group.get_bottom(),
            action_mf.get_top(),
            color=ORANGE,
            buff=0.1,
            tip_length=0.15,
        )

        # --- 2. MODEL-BASED: VERTICAL FLOW (MCTS) ---
        mb_center = RIGHT * 3.5 + UP * 1.2

        state_mb = Text("Environment State", font_size=20).move_to(mb_center)

        # Root node of the tree (Z-index 2 keeps it above the lines)
        root = Dot(
            state_mb.get_bottom() + DOWN * 0.9, radius=0.1, color=BLUE
        ).set_z_index(2)
        arrow_in_mb = Arrow(
            state_mb.get_bottom(),
            root.get_top(),
            color=WHITE,
            buff=0.1,
            tip_length=0.15,
        )

        # Tree Layers (Z-index 2 keeps them above the lines)
        l1 = VGroup(
            Dot(root.get_center() + DOWN * 0.8 + LEFT * 0.8, color=BLUE).set_z_index(2),
            Dot(root.get_center() + DOWN * 0.8 + RIGHT * 0.8, color=BLUE).set_z_index(
                2
            ),
        )

        l2 = VGroup(
            Dot(l1[0].get_center() + DOWN * 0.8 + LEFT * 0.4, color=BLUE).set_z_index(
                2
            ),
            Dot(l1[0].get_center() + DOWN * 0.8 + RIGHT * 0.4, color=BLUE).set_z_index(
                2
            ),
            Dot(l1[1].get_center() + DOWN * 0.8 + RIGHT * 0.4, color=BLUE).set_z_index(
                2
            ),
        )

        tree_lines = (
            VGroup(
                Line(root, l1[0], color=BLUE),
                Line(root, l1[1], color=BLUE),
                Line(l1[0], l2[0], color=BLUE),
                Line(l1[0], l2[1], color=BLUE),
                Line(l1[1], l2[2], color=BLUE),
            )
            .set_stroke(width=2)
            .set_z_index(1)
        )

        # --- ANIMATIONS ---

        # Model-Free Animation
        self.play(FadeIn(state_mf))
        self.play(GrowArrow(arrow_in_mf))
        self.play(FadeIn(exp_box), FadeIn(exp_text))

        # "Searching" effect
        self.play(
            exp_box.animate.set_fill(ORANGE, opacity=0.3),
            rate_func=there_and_back,
            run_time=0.8,
        )

        self.play(GrowArrow(arrow_out_mf))
        self.play(FadeIn(action_mf))

        self.wait(1)

        # Model-Based Animation
        self.play(FadeIn(state_mb))
        self.play(GrowArrow(arrow_in_mb))
        self.play(Create(root))

        # Grow tree
        self.play(Create(tree_lines[0:2]), FadeIn(l1), run_time=1)
        self.play(Create(tree_lines[2:]), FadeIn(l2), run_time=1)

        # Highlight path (Z-index 1 ensures it stays behind the dots)
        path = VGroup(
            Line(root, l1[0], color=WHITE, stroke_width=4),
            Line(l1[0], l2[1], color=WHITE, stroke_width=4),
        ).set_z_index(1)

        # Turn target node GREEN and force it strictly to the very front
        self.play(
            Create(path), l2[1].animate.set_color(GREEN).scale(1.5).set_z_index(3)
        )

        # Define Action text and arrow AFTER the dot scales to fix the arrow size bug
        action_mb = Text("Action", font_size=20, color=GREEN, weight=BOLD).next_to(
            l2[1], DOWN, buff=0.8
        )

        arrow_out_mb = Arrow(
            l2[1].get_bottom(),
            action_mb.get_top(),
            color=GREEN,
            buff=0.1,
            tip_length=0.15,
        )

        # Output action from the selected node
        self.play(GrowArrow(arrow_out_mb))
        self.play(FadeIn(action_mb))

        self.wait(5)
