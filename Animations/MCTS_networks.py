from manim import *
from config import BACKGROUND_COLOR


class MuZeroMCTS(Scene):
    """The second scene explaining u-MCTS."""

    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR

        self.setup_networks_left()
        self.explain_mcts_integration()

    # --- HELPER METHODS ---
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

    def setup_networks_left(self):
        self.rep_node = (
            self.create_network_node("Representation", "Network", BLUE)
            .move_to(LEFT * 4 + UP * 2.5)
            .scale(0.8)
        )
        self.pred_node = (
            self.create_network_node("Prediction", "Network", YELLOW)
            .move_to(LEFT * 4)
            .scale(0.8)
        )
        self.dyn_node = (
            self.create_network_node("Dynamics", "Network", GREEN)
            .move_to(LEFT * 4 + DOWN * 2.5)
            .scale(0.8)
        )
        self.add(self.rep_node, self.pred_node, self.dyn_node)

    # ---------------------------------

    def explain_mcts_integration(self):
        title = Text("u-MCTS", font_size=24, weight=BOLD, color=WHITE)
        title.to_edge(UP).set_x(3.5)
        self.play(Write(title))

        # Root Node Setup (Text moved UP to keep the left side clear for arrows)
        root_center = RIGHT * 3.5 + UP * 1.5
        root_circle = Circle(radius=0.2, color=ORANGE, fill_opacity=0.5).move_to(
            root_center
        )
        root_text = Text("S0", font_size=16).next_to(root_circle, UP, buff=0.15)
        root_node = VGroup(root_circle, root_text)

        # 1. Rep Net -> S0
        arrow_rep_s0 = DashedLine(
            self.rep_node[0].get_right(),
            root_circle.get_left(),
            buff=0.2,
            color=BLUE,
            stroke_width=1.5,
        ).add_tip(tip_length=0.15, tip_width=0.15)

        label_rep_s0 = Text("Abstract State (S0)", font_size=14, color=BLUE).move_to(
            arrow_rep_s0.get_center() + UP * 0.25
        )

        self.play(Create(arrow_rep_s0), FadeIn(label_rep_s0))
        self.play(FadeIn(root_node))
        self.wait(0.5)
        self.play(FadeOut(arrow_rep_s0), FadeOut(label_rep_s0))

        # 2. S0 -> Pred Net
        arrow_s0_pred = DashedLine(
            root_circle.get_left(),
            self.pred_node[0].get_right(),
            buff=0.2,
            color=YELLOW,
            stroke_width=1.5,
        ).add_tip(tip_length=0.15, tip_width=0.15)

        self.play(Create(arrow_s0_pred))
        self.wait(0.5)

        # 3. Pred Net -> Produces Action a1 (Only 1 action spawned initially)
        action_box_1 = (
            self.create_boxed_text("Action a1", box_color=TEAL)
            .scale(0.7)
            .move_to(LEFT * 0.5 + UP * 0.0)
        )

        arrow_pred_act1 = DashedLine(
            self.pred_node[0].get_right(),
            action_box_1.get_left(),
            buff=0.2,
            color=TEAL,
            stroke_width=1.5,
        ).add_tip(tip_length=0.15, tip_width=0.15)

        self.play(Create(arrow_pred_act1))
        self.play(FadeIn(action_box_1))
        self.wait(0.5)
        self.play(FadeOut(arrow_pred_act1), FadeOut(arrow_s0_pred))

        # Setup S1 Node
        child1_circle = Circle(radius=0.2, color=ORANGE, fill_opacity=0.3).move_to(
            RIGHT * 2.3 + DOWN * 0.2
        )
        child1_text = Text("S1", font_size=16).next_to(child1_circle, UP, buff=0.15)
        child1_node = VGroup(child1_circle, child1_text)
        edge1 = Line(
            root_circle.get_bottom(), child1_circle.get_top(), buff=0.1, stroke_width=2
        )
        edge1_label = Text("a1", font_size=14, color=TEAL).next_to(
            edge1.get_center(), LEFT, buff=0.1
        )

        # 4. S0 + Action a1 -> Dynamics Net
        arrow_s0_dyn = DashedLine(
            root_circle.get_left(),
            self.dyn_node[0].get_right(),
            buff=0.2,
            color=GREEN,
            stroke_width=1.5,
        ).add_tip(tip_length=0.15, tip_width=0.15)

        arrow_act1_dyn = DashedLine(
            action_box_1.get_bottom(),
            self.dyn_node[0].get_right(),
            buff=0.2,
            color=GREEN,
            stroke_width=1.5,
        ).add_tip(tip_length=0.15, tip_width=0.15)

        self.play(Create(arrow_s0_dyn), Create(arrow_act1_dyn))
        self.wait(0.5)

        # 5. Dynamics Net -> Produces S1
        arrow_dyn_s1 = DashedLine(
            self.dyn_node[0].get_right(),
            child1_circle.get_left(),
            buff=0.2,
            color=ORANGE,
            stroke_width=1.5,
        ).add_tip(tip_length=0.15, tip_width=0.15)

        self.play(Create(arrow_dyn_s1))
        self.play(Create(edge1), Write(edge1_label), FadeIn(child1_node))
        self.wait(0.5)

        self.play(
            FadeOut(arrow_s0_dyn),
            FadeOut(arrow_act1_dyn),
            FadeOut(action_box_1),
            FadeOut(arrow_dyn_s1),
        )

        # 6. S1 -> Pred Net
        arrow_s1_pred = DashedLine(
            child1_circle.get_left(),
            self.pred_node[0].get_right(),
            buff=0.2,
            color=YELLOW,
            stroke_width=1.5,
        ).add_tip(tip_length=0.15, tip_width=0.15)

        self.play(Create(arrow_s1_pred))
        self.wait(0.5)

        # 7. Pred Net -> Produces Action a3
        action_box_3 = (
            self.create_boxed_text("Action a2", box_color=TEAL)
            .scale(0.7)
            .move_to(LEFT * 0.5 + DOWN * 1.0)
        )

        arrow_pred_act3 = DashedLine(
            self.pred_node[0].get_right(),
            action_box_3.get_left(),
            buff=0.2,
            color=TEAL,
            stroke_width=1.5,
        ).add_tip(tip_length=0.15, tip_width=0.15)

        self.play(Create(arrow_pred_act3))
        self.play(FadeIn(action_box_3))
        self.wait(0.5)
        self.play(FadeOut(arrow_pred_act3), FadeOut(arrow_s1_pred))

        # Setup S2 Node
        leaf_circle = Circle(radius=0.2, color=ORANGE, fill_opacity=0.2).move_to(
            child1_circle.get_center() + DOWN * 1.5 + RIGHT * 0.2
        )
        leaf_text = Text("S2", font_size=16).next_to(leaf_circle, RIGHT, buff=0.15)
        leaf_node = VGroup(leaf_circle, leaf_text)

        leaf_edge = Line(
            child1_circle.get_bottom(), leaf_circle.get_top(), buff=0.1, stroke_width=2
        )
        leaf_edge_label = Text("a2", font_size=14, color=TEAL).next_to(
            leaf_edge.get_center(), RIGHT, buff=0.1
        )

        # 8. S1 + Action a3 -> Dynamics Net
        arrow_s1_dyn = DashedLine(
            child1_circle.get_left(),
            self.dyn_node[0].get_right(),
            buff=0.2,
            color=GREEN,
            stroke_width=1.5,
        ).add_tip(tip_length=0.15, tip_width=0.15)

        arrow_act3_dyn = DashedLine(
            action_box_3.get_bottom(),
            self.dyn_node[0].get_right(),
            buff=0.2,
            color=GREEN,
            stroke_width=1.5,
        ).add_tip(tip_length=0.15, tip_width=0.15)

        self.play(Create(arrow_s1_dyn), Create(arrow_act3_dyn))
        self.wait(0.5)

        # 9. Dynamics Net -> Produces S2
        arrow_dyn_s2 = DashedLine(
            self.dyn_node[0].get_right(),
            leaf_circle.get_left(),
            buff=0.2,
            color=ORANGE,
            stroke_width=1.5,
        ).add_tip(tip_length=0.15, tip_width=0.15)

        self.play(Create(arrow_dyn_s2))
        self.play(Create(leaf_edge), Write(leaf_edge_label), FadeIn(leaf_node))
        self.wait(0.5)

        self.play(
            FadeOut(arrow_s1_dyn),
            FadeOut(arrow_act3_dyn),
            FadeOut(action_box_3),
            FadeOut(arrow_dyn_s2),
        )

        # 10. Revisit Root: S0 -> Pred Net
        arrow_s0_pred2 = DashedLine(
            root_circle.get_left(),
            self.pred_node[0].get_right(),
            buff=0.2,
            color=YELLOW,
            stroke_width=1.5,
        ).add_tip(tip_length=0.15, tip_width=0.15)

        self.play(Create(arrow_s0_pred2))
        self.wait(0.5)

        # 11. Pred Net -> Produces Action a2
        action_box_2 = (
            self.create_boxed_text("Action a2", box_color=TEAL)
            .scale(0.7)
            .move_to(LEFT * 0.5 + UP * 0.0)
        )

        arrow_pred_act2 = DashedLine(
            self.pred_node[0].get_right(),
            action_box_2.get_left(),
            buff=0.2,
            color=TEAL,
            stroke_width=1.5,
        ).add_tip(tip_length=0.15, tip_width=0.15)

        self.play(Create(arrow_pred_act2))
        self.play(FadeIn(action_box_2))
        self.wait(0.5)
        self.play(FadeOut(arrow_pred_act2), FadeOut(arrow_s0_pred2))

        # Setup S1' Node
        child2_circle = Circle(radius=0.2, color=ORANGE, fill_opacity=0.3).move_to(
            RIGHT * 4.7 + DOWN * 0.2
        )
        child2_text = Text("S1'", font_size=16).next_to(child2_circle, RIGHT, buff=0.15)
        child2_node = VGroup(child2_circle, child2_text)
        edge2 = Line(
            root_circle.get_bottom(), child2_circle.get_top(), buff=0.1, stroke_width=2
        )
        edge2_label = Text("a2", font_size=14, color=TEAL).next_to(
            edge2.get_center(), RIGHT, buff=0.1
        )

        # 12. S0 + Action a2 -> Dynamics Net
        arrow_s0_dyn2 = DashedLine(
            root_circle.get_left(),
            self.dyn_node[0].get_right(),
            buff=0.2,
            color=GREEN,
            stroke_width=1.5,
        ).add_tip(tip_length=0.15, tip_width=0.15)

        arrow_act2_dyn = DashedLine(
            action_box_2.get_bottom(),
            self.dyn_node[0].get_right(),
            buff=0.2,
            color=GREEN,
            stroke_width=1.5,
        ).add_tip(tip_length=0.15, tip_width=0.15)

        self.play(Create(arrow_s0_dyn2), Create(arrow_act2_dyn))
        self.wait(0.5)

        # 13. Dynamics Net -> Produces S1'
        arrow_dyn_s1alt = DashedLine(
            self.dyn_node[0].get_right(),
            child2_circle.get_left(),
            buff=0.2,
            color=ORANGE,
            stroke_width=1.5,
        ).add_tip(tip_length=0.15, tip_width=0.15)

        self.play(Create(arrow_dyn_s1alt))
        self.play(Create(edge2), Write(edge2_label), FadeIn(child2_node))
        self.wait(0.5)

        self.play(
            FadeOut(arrow_s0_dyn2),
            FadeOut(arrow_act2_dyn),
            FadeOut(action_box_2),
            FadeOut(arrow_dyn_s1alt),
        )
        self.wait(2)
