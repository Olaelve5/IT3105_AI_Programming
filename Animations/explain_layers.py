from manim import *
from config import BACKGROUND_COLOR


class ExplainLayers(Scene):
    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR
        self.explain_representation_net()
        self.explain_dynamics_net()
        self.explain_prediction_net()

    def create_layer_block(self, main_text, sub_text, box_color, width=1.8, height=0.7):
        """Creates a horizontal layer block for network architectures (Scaled Down)."""
        box = RoundedRectangle(
            width=width,
            height=height,
            corner_radius=0.1,
            color=box_color,
            fill_opacity=0.2,
        )
        main_label = Text(main_text, font_size=12, weight=BOLD).move_to(
            box.get_center() + UP * 0.12
        )
        sub_label = Text(sub_text, font_size=10, color=GRAY).move_to(
            box.get_center() + DOWN * 0.15
        )
        return VGroup(box, main_label, sub_label)

    def explain_representation_net(self):
        """Visualizes the Representation Network architecture horizontally."""

        # Base Y position pushed further down
        y_pos = DOWN * 3.0

        input_text = Text("Observation", font_size=12, color=WHITE).move_to(
            LEFT * 5.0 + y_pos
        )

        conv_block = self.create_layer_block(
            "Conv 3x3 (64)", "LayerNorm + ReLU", BLUE
        ).next_to(input_text, RIGHT, buff=0.4)

        res_block = self.create_layer_block(
            "5x ResBlocks", "Conv + LN + ReLU", BLUE
        ).next_to(conv_block, RIGHT, buff=0.4)

        output_text = Text("Hidden State", font_size=12, color=WHITE).next_to(
            res_block, RIGHT, buff=0.4
        )
        scale_label = Text("(min-max scaled)", font_size=8, color=GRAY).next_to(
            output_text, DOWN, buff=0.1
        )
        output_group = VGroup(output_text, scale_label)

        # Create Arrows
        a1 = Arrow(input_text.get_right(), conv_block.get_left(), buff=0.1)
        a2 = Arrow(conv_block.get_right(), res_block.get_left(), buff=0.1)
        a3 = Arrow(res_block.get_right(), output_group.get_left(), buff=0.1)

        # Animate Pipeline
        self.play(FadeIn(input_text))
        self.play(Create(a1), FadeIn(conv_block))
        self.play(Create(a2), FadeIn(res_block))
        self.play(Create(a3), FadeIn(output_group))

        self.wait(3)

        # Cleanup for the next network (Removed title)
        self.play(
            FadeOut(VGroup(input_text, conv_block, res_block, output_group, a1, a2, a3))
        )

    def explain_dynamics_net(self):
        """Visualizes the Dynamics Network architecture horizontally."""

        y_pos = DOWN * 3.0

        # Inputs (Concatenated)
        state_in = Text("State", font_size=12, color=WHITE)
        action_in = Text("+ Action", font_size=12, color=YELLOW)
        input_group = (
            VGroup(state_in, action_in)
            .arrange(DOWN, buff=0.1)
            .move_to(LEFT * 5.0 + y_pos)
        )

        # Main Trunk
        conv_block = self.create_layer_block(
            "Conv 3x3 (64)", "LayerNorm + ReLU", GREEN
        ).next_to(input_group, RIGHT, buff=0.4)

        res_block = self.create_layer_block(
            "5x ResBlocks", "Conv + LN + ReLU", GREEN
        ).next_to(conv_block, RIGHT, buff=0.4)

        # Outputs (Branching from ResBlocks)
        next_state_text = (
            Text("Next State", font_size=12, color=WHITE)
            .next_to(res_block, RIGHT, buff=0.4)
            .shift(UP * 0.4)
        )
        scale_label = Text("(min-max scaled)", font_size=8, color=GRAY).next_to(
            next_state_text, RIGHT, buff=0.1
        )

        reward_head = (
            self.create_layer_block(
                "Reward/Discount",
                "Conv 16 -> Dense 256 -> Dense 1",
                RED,
                width=2.5,
            )
            .next_to(res_block, RIGHT, buff=0.4)
            .shift(DOWN * 0.4)
        )

        rd_outputs = Text("Reward & Discount", font_size=12, color=RED).next_to(
            reward_head, RIGHT, buff=0.2
        )

        # Arrows
        a1 = Arrow(input_group.get_right(), conv_block.get_left(), buff=0.1)
        a2 = Arrow(conv_block.get_right(), res_block.get_left(), buff=0.1)

        # Branching arrows
        a3_state = Arrow(res_block.get_right(), next_state_text.get_left(), buff=0.1)
        a3_head = Arrow(res_block.get_right(), reward_head.get_left(), buff=0.1)
        a4_rd = Arrow(reward_head.get_right(), rd_outputs.get_left(), buff=0.1)

        # Animate Pipeline
        self.play(FadeIn(input_group))
        self.play(Create(a1), FadeIn(conv_block))
        self.play(Create(a2), FadeIn(res_block))
        self.play(
            Create(a3_state),
            FadeIn(next_state_text),
            FadeIn(scale_label),
            Create(a3_head),
            FadeIn(reward_head),
        )
        self.play(Create(a4_rd), FadeIn(rd_outputs))

        self.wait(3)
        self.play(
            FadeOut(
                VGroup(
                    input_group,
                    conv_block,
                    res_block,
                    next_state_text,
                    scale_label,
                    reward_head,
                    rd_outputs,
                    a1,
                    a2,
                    a3_state,
                    a3_head,
                    a4_rd,
                )
            )
        )

    def explain_prediction_net(self):
        """Visualizes the Prediction Network architecture horizontally."""

        y_pos = DOWN * 3.0

        # Input
        input_text = Text("Hidden State", font_size=12, color=WHITE).move_to(
            LEFT * 4.5 + y_pos
        )

        # Branches
        # Policy Branch (Top)
        policy_head = (
            self.create_layer_block(
                "Policy Head", "Conv 32 -> Dense 256", YELLOW, width=2.0
            )
            .next_to(input_text, RIGHT, buff=0.6)
            .shift(UP * 0.45)
        )

        policy_out = Text("Policy (Probs)", font_size=12, color=YELLOW).next_to(
            policy_head, RIGHT, buff=0.3
        )

        # Value Branch (Bottom)
        value_head = (
            self.create_layer_block(
                "Value Head", "Conv 16 -> Dense 256", ORANGE, width=2.0
            )
            .next_to(input_text, RIGHT, buff=0.6)
            .shift(DOWN * 0.45)
        )

        value_out = Text("Value (Winner)", font_size=12, color=ORANGE).next_to(
            value_head, RIGHT, buff=0.3
        )

        # Arrows
        a_policy = Arrow(input_text.get_right(), policy_head.get_left(), buff=0.1)
        a_policy_out = Arrow(policy_head.get_right(), policy_out.get_left(), buff=0.1)

        a_value = Arrow(input_text.get_right(), value_head.get_left(), buff=0.1)
        a_value_out = Arrow(value_head.get_right(), value_out.get_left(), buff=0.1)

        # Animate Pipeline
        self.play(FadeIn(input_text))
        self.play(
            Create(a_policy), FadeIn(policy_head), Create(a_value), FadeIn(value_head)
        )
        self.play(
            Create(a_policy_out),
            FadeIn(policy_out),
            Create(a_value_out),
            FadeIn(value_out),
        )

        self.wait(3)
        self.play(
            FadeOut(
                VGroup(
                    input_text,
                    policy_head,
                    policy_out,
                    value_head,
                    value_out,
                    a_policy,
                    a_policy_out,
                    a_value,
                    a_value_out,
                )
            )
        )
