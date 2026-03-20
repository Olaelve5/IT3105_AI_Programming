from manim import *
from config import BACKGROUND_COLOR


class MuZero_Explained(Scene):
    """The first scene explaining the three neural networks."""

    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR

        self.show_architecture_overview()
        self.explain_representation_network()
        self.explain_dynamics_network()
        self.explain_prediction_network()
        self.recap_networks_together()

    # --- HELPER METHODS MOVED HERE ---
    def create_network_node(self, text_line1, text_line2, node_color):
        circle = Circle(color=node_color, fill_opacity=0.2, radius=1)
        label = Paragraph(
            text_line1, text_line2, font_size=18, line_spacing=0.5, alignment="center"
        ).next_to(circle, UP, buff=0.15)

        layers = [3, 4, 3]
        dots = VGroup()
        for i, num_nodes in enumerate(layers):
            layer = VGroup(*[Dot(radius=0.05, color=WHITE) for _ in range(num_nodes)])
            layer.arrange(DOWN, buff=0.25)
            layer.shift(RIGHT * (i - 1) * 0.6)
            dots.add(layer)

        lines = VGroup()
        for i in range(len(layers) - 1):
            for dot1 in dots[i]:
                for dot2 in dots[i + 1]:
                    lines.add(
                        Line(
                            dot1.get_center(),
                            dot2.get_center(),
                            stroke_width=1.5,
                            stroke_opacity=0.5,
                            color=WHITE,
                        )
                    )

        neural_net = VGroup(lines, dots)
        neural_net.move_to(circle.get_center())
        return VGroup(circle, neural_net, label)

    def create_boxed_text(self, text, box_color=WHITE):
        text_element = Text(text, font_size=18, color=WHITE)
        box = SurroundingRectangle(
            text_element,
            color=box_color,
            buff=0.2,
            corner_radius=0.1,
            stroke_width=2,
        )
        return VGroup(box, text_element)

    # ---------------------------------

    def show_architecture_overview(self):
        self.title = Text(
            "MuZero's Architecture", font_size=28, weight=BOLD, color=WHITE
        ).to_edge(UP)

        self.rep_node = self.create_network_node(
            "Representation", "Network", BLUE
        ).move_to(LEFT * 3.5)
        self.dyn_node = self.create_network_node("Dynamics", "Network", GREEN).move_to(
            ORIGIN
        )
        self.pred_node = self.create_network_node(
            "Prediction", "Network", YELLOW
        ).move_to(RIGHT * 3.5)

        self.play(Write(self.title))
        self.play(FadeIn(self.rep_node))
        self.play(FadeIn(self.dyn_node))
        self.play(FadeIn(self.pred_node))
        self.wait(3)

    def explain_representation_network(self):
        self.play(FadeOut(self.dyn_node), FadeOut(self.pred_node))
        self.play(self.rep_node.animate.move_to(ORIGIN))

        pacman_image = ImageMobject("resources/pacman.png")
        pacman_image.set_height(config.frame_height * 0.3)

        border = SurroundingRectangle(
            pacman_image, color=ORANGE, buff=0.1, stroke_width=2, corner_radius=0.2
        )
        image_text = Text("Board State", font_size=18).next_to(border, UP, buff=0.1)

        board_state = Group(pacman_image, border, image_text).next_to(
            self.rep_node[0], LEFT, buff=2.5
        )
        abstract_state = self.create_boxed_text(
            "Abstract State", box_color=ORANGE
        ).next_to(self.rep_node[0], RIGHT, buff=2.5)

        arrow_in = Arrow(
            start=board_state.get_right(),
            end=self.rep_node[0].get_left(),
            buff=0.3,
            stroke_width=2,
            tip_length=0.15,
        )
        arrow_out = Arrow(
            start=self.rep_node[0].get_right(),
            end=abstract_state.get_left(),
            buff=0.3,
            stroke_width=2,
            tip_length=0.15,
        )

        self.play(FadeIn(board_state))
        self.wait(1)
        self.play(GrowArrow(arrow_in))
        self.wait(0.5)
        self.play(GrowArrow(arrow_out))
        self.play(FadeIn(abstract_state))
        self.wait(3)

        self.play(
            FadeOut(
                Group(board_state, arrow_in, self.rep_node, arrow_out, abstract_state)
            )
        )

    def explain_dynamics_network(self):
        self.dyn_node.move_to(ORIGIN)
        self.play(FadeIn(self.dyn_node))

        abs_state_in = self.create_boxed_text("Abstract State", box_color=ORANGE)
        action_in = self.create_boxed_text("Hypothetical Action", box_color=TEAL)

        inputs_group = VGroup(abs_state_in, action_in).arrange(
            DOWN, buff=0.6, aligned_edge=RIGHT
        )
        inputs_group.next_to(self.dyn_node[0], LEFT, buff=2.5)

        abs_state_out = self.create_boxed_text("Next Abstract State", box_color=ORANGE)
        reward_out = self.create_boxed_text("Predicted Reward", box_color=RED)

        outputs_group = VGroup(abs_state_out, reward_out).arrange(
            DOWN, buff=0.6, aligned_edge=LEFT
        )
        outputs_group.next_to(self.dyn_node[0], RIGHT, buff=2.5)

        arrow_in_1 = Arrow(
            start=abs_state_in.get_right(),
            end=self.dyn_node[0].get_left(),
            buff=0.3,
            stroke_width=2,
            tip_length=0.15,
        )
        arrow_in_2 = Arrow(
            start=action_in.get_right(),
            end=self.dyn_node[0].get_left(),
            buff=0.3,
            stroke_width=2,
            tip_length=0.15,
        )
        arrow_out_1 = Arrow(
            start=self.dyn_node[0].get_right(),
            end=abs_state_out.get_left(),
            buff=0.3,
            stroke_width=2,
            tip_length=0.15,
        )
        arrow_out_2 = Arrow(
            start=self.dyn_node[0].get_right(),
            end=reward_out.get_left(),
            buff=0.3,
            stroke_width=2,
            tip_length=0.15,
        )

        self.play(FadeIn(inputs_group))
        self.wait(1)
        self.play(GrowArrow(arrow_in_1), GrowArrow(arrow_in_2))
        self.wait(0.5)
        self.play(GrowArrow(arrow_out_1), GrowArrow(arrow_out_2))
        self.play(FadeIn(outputs_group))
        self.wait(3)

        self.play(
            FadeOut(
                Group(
                    inputs_group,
                    arrow_in_1,
                    arrow_in_2,
                    self.dyn_node,
                    arrow_out_1,
                    arrow_out_2,
                    outputs_group,
                )
            )
        )

    def explain_prediction_network(self):
        self.pred_node.move_to(ORIGIN)
        self.play(FadeIn(self.pred_node))

        abs_state_in = self.create_boxed_text("Abstract State", box_color=ORANGE)
        abs_state_in.next_to(self.pred_node[0], LEFT, buff=2.5)

        policy_out = self.create_boxed_text("Policy (Action Probs)", box_color=TEAL)
        value_out = self.create_boxed_text("Predicted Value", box_color=GOLD)

        outputs_group = VGroup(policy_out, value_out).arrange(
            DOWN, buff=0.6, aligned_edge=LEFT
        )
        outputs_group.next_to(self.pred_node[0], RIGHT, buff=2.5)

        arrow_in = Arrow(
            start=abs_state_in.get_right(),
            end=self.pred_node[0].get_left(),
            buff=0.3,
            stroke_width=2,
            tip_length=0.15,
        )
        arrow_out_1 = Arrow(
            start=self.pred_node[0].get_right(),
            end=policy_out.get_left(),
            buff=0.3,
            stroke_width=2,
            tip_length=0.15,
        )
        arrow_out_2 = Arrow(
            start=self.pred_node[0].get_right(),
            end=value_out.get_left(),
            buff=0.3,
            stroke_width=2,
            tip_length=0.15,
        )

        self.play(FadeIn(abs_state_in))
        self.wait(1)
        self.play(GrowArrow(arrow_in))
        self.wait(0.5)
        self.play(GrowArrow(arrow_out_1), GrowArrow(arrow_out_2))
        self.play(FadeIn(outputs_group))
        self.wait(3)

        self.play(
            FadeOut(
                Group(abs_state_in, arrow_in, arrow_out_1, arrow_out_2, outputs_group)
            )
        )

    def recap_networks_together(self):
        self.rep_node.move_to(LEFT * 3.5)
        self.dyn_node.move_to(ORIGIN)
        self.play(self.pred_node.animate.move_to(RIGHT * 3.5))
        self.play(FadeIn(self.rep_node), FadeIn(self.dyn_node))
        self.wait(2)

        self.play(FadeOut(self.title))

        self.play(
            self.rep_node.animate.move_to(LEFT * 4 + UP * 2.5).scale(0.8),
            self.pred_node.animate.move_to(LEFT * 4).scale(0.8),
            self.dyn_node.animate.move_to(LEFT * 4 + DOWN * 2.5).scale(0.8),
            run_time=1.5,
        )
        self.wait(1)
