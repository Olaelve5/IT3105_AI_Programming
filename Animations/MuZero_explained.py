from manim import *
from config import BACKGROUND_COLOR


class MuZero_Explained(Scene):
    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR

        self.show_title()
        self.show_architecture_overview()
        self.explain_representation_network()
        self.explain_dynamics_network()
        self.explain_prediction_network()

        # New function added to the timeline
        self.show_full_architecture()

    def create_network_node(self, text_line1, text_line2, node_color):
        circle = Circle(color=node_color, fill_opacity=0.5, radius=1.2)
        label = Paragraph(
            text_line1, text_line2, font_size=24, line_spacing=0.5, alignment="center"
        ).move_to(circle.get_center())

        return VGroup(circle, label)

    def create_boxed_text(self, text):
        text_element = Text(text, font_size=24, color=WHITE)
        box = SurroundingRectangle(text_element, color=WHITE, buff=0.2)
        return VGroup(box, text_element)

    def show_title(self):
        title = Text("MuZero", font_size=48, weight=BOLD, color=RED)

        self.play(Write(title))
        self.wait(2)
        self.play(FadeOut(title))

    def show_architecture_overview(self):
        self.title = Text(
            "MuZero's Architecture", font_size=32, weight=BOLD, color=WHITE
        ).to_edge(UP)

        self.rep_node = self.create_network_node(
            "Representation", "Network", BLUE
        ).shift(LEFT * 3)
        self.dyn_node = self.create_network_node("Dynamics", "Network", GREEN)
        self.pred_node = self.create_network_node(
            "Prediction", "Network", YELLOW
        ).shift(RIGHT * 3)

        self.play(Write(self.title))
        self.play(FadeIn(self.rep_node, shift=UP))
        self.play(FadeIn(self.dyn_node, shift=UP))
        self.play(FadeIn(self.pred_node, shift=UP))
        self.wait(3)

    def explain_representation_network(self):
        self.play(FadeOut(self.title), FadeOut(self.dyn_node), FadeOut(self.pred_node))
        self.play(self.rep_node.animate.move_to(ORIGIN))

        board_state = self.create_boxed_text("Actual\nBoard State").next_to(
            self.rep_node, LEFT, buff=2.5
        )
        abstract_state = self.create_boxed_text("Abstract\nState").next_to(
            self.rep_node, RIGHT, buff=2.5
        )

        arrow_in = Arrow(
            start=board_state.get_right(),
            end=self.rep_node.get_left(),
            buff=0.1,
            stroke_width=2,
            tip_length=0.15,
        )
        arrow_out = Arrow(
            start=self.rep_node.get_right(),
            end=abstract_state.get_left(),
            buff=0.1,
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
                VGroup(board_state, arrow_in, self.rep_node, arrow_out, abstract_state)
            )
        )

    def explain_dynamics_network(self):
        self.dyn_node.move_to(ORIGIN)
        self.play(FadeIn(self.dyn_node))

        abstract_state = (
            self.create_boxed_text("Abstract\nState")
            .next_to(self.dyn_node, LEFT, buff=2.5)
            .shift(UP * 1)
        )
        action = (
            self.create_boxed_text("Action")
            .next_to(self.dyn_node, LEFT, buff=2.5)
            .shift(DOWN * 1)
        )

        next_state = (
            self.create_boxed_text("Next\nAbstract State")
            .next_to(self.dyn_node, RIGHT, buff=2.5)
            .shift(UP * 1)
        )
        reward = (
            self.create_boxed_text("Reward")
            .next_to(self.dyn_node, RIGHT, buff=2.5)
            .shift(DOWN * 1)
        )

        arrow_in_state = Arrow(
            start=abstract_state.get_right(),
            end=self.dyn_node.get_left(),
            buff=0.1,
            stroke_width=2,
            tip_length=0.15,
        )
        arrow_in_action = Arrow(
            start=action.get_right(),
            end=self.dyn_node.get_left(),
            buff=0.1,
            stroke_width=2,
            tip_length=0.15,
        )
        arrow_out_state = Arrow(
            start=self.dyn_node.get_right(),
            end=next_state.get_left(),
            buff=0.1,
            stroke_width=2,
            tip_length=0.15,
        )
        arrow_out_reward = Arrow(
            start=self.dyn_node.get_right(),
            end=reward.get_left(),
            buff=0.1,
            stroke_width=2,
            tip_length=0.15,
        )

        self.play(FadeIn(VGroup(abstract_state, action)))
        self.wait(1)
        self.play(GrowArrow(arrow_in_state), GrowArrow(arrow_in_action))
        self.wait(0.5)
        self.play(GrowArrow(arrow_out_state), GrowArrow(arrow_out_reward))
        self.play(FadeIn(VGroup(next_state, reward)))
        self.wait(3)

        self.play(
            FadeOut(
                VGroup(
                    abstract_state,
                    action,
                    arrow_in_state,
                    arrow_in_action,
                    self.dyn_node,
                    arrow_out_state,
                    arrow_out_reward,
                    next_state,
                    reward,
                )
            )
        )

    def explain_prediction_network(self):
        self.pred_node.move_to(ORIGIN)
        self.play(FadeIn(self.pred_node))

        abstract_state = self.create_boxed_text("Abstract\nState").next_to(
            self.pred_node, LEFT, buff=2.5
        )

        policy = (
            self.create_boxed_text("Policy\n(Moves)")
            .next_to(self.pred_node, RIGHT, buff=2.5)
            .shift(UP * 1)
        )
        value = (
            self.create_boxed_text("Value\n(Win %)")
            .next_to(self.pred_node, RIGHT, buff=2.5)
            .shift(DOWN * 1)
        )

        arrow_in = Arrow(
            start=abstract_state.get_right(),
            end=self.pred_node.get_left(),
            buff=0.1,
            stroke_width=2,
            tip_length=0.15,
        )
        arrow_out_policy = Arrow(
            start=self.pred_node.get_right(),
            end=policy.get_left(),
            buff=0.1,
            stroke_width=2,
            tip_length=0.15,
        )
        arrow_out_value = Arrow(
            start=self.pred_node.get_right(),
            end=value.get_left(),
            buff=0.1,
            stroke_width=2,
            tip_length=0.15,
        )

        self.play(FadeIn(abstract_state))
        self.wait(1)
        self.play(GrowArrow(arrow_in))
        self.wait(0.5)
        self.play(GrowArrow(arrow_out_policy), GrowArrow(arrow_out_value))
        self.play(FadeIn(VGroup(policy, value)))
        self.wait(3)

        self.play(
            FadeOut(
                VGroup(
                    abstract_state,
                    arrow_in,
                    self.pred_node,
                    arrow_out_policy,
                    arrow_out_value,
                    policy,
                    value,
                )
            )
        )

    def show_full_architecture(self):
        title = Text("The Full Loop", font_size=32, weight=BOLD, color=WHITE).to_edge(
            UP
        )
        self.play(Write(title))

        # 1. Setup Nodes (Scaled down slightly to fit everything on screen)
        rep_node = (
            self.create_network_node("Rep", "Net", BLUE).scale(0.7).move_to(LEFT * 4.5)
        )
        pred_node = (
            self.create_network_node("Pred", "Net", YELLOW).scale(0.7).move_to(UP * 2)
        )
        dyn_node = (
            self.create_network_node("Dyn", "Net", GREEN).scale(0.7).move_to(DOWN * 2)
        )

        # 2. Central State
        state = self.create_boxed_text("Abstract\nState").scale(0.7).move_to(ORIGIN)

        # 3. Representation Flow (Left to Center)
        board = (
            self.create_boxed_text("Board State")
            .scale(0.7)
            .next_to(rep_node, LEFT, buff=0.5)
        )
        a_board_to_rep = Arrow(
            board.get_right(),
            rep_node.get_left(),
            buff=0.1,
            stroke_width=2,
            tip_length=0.15,
        )
        a_rep_to_state = Arrow(
            rep_node.get_right(),
            state.get_left(),
            buff=0.1,
            stroke_width=2,
            tip_length=0.15,
        )

        self.play(FadeIn(board), FadeIn(rep_node), FadeIn(state))
        self.play(GrowArrow(a_board_to_rep), GrowArrow(a_rep_to_state))
        self.wait(1)

        # 4. Prediction Flow (Center to Top)
        a_state_to_pred = Arrow(
            state.get_top(),
            pred_node.get_bottom(),
            buff=0.1,
            stroke_width=2,
            tip_length=0.15,
        )
        policy = (
            self.create_boxed_text("Policy")
            .scale(0.7)
            .next_to(pred_node, UP, buff=0.5)
            .shift(LEFT * 1.5)
        )
        value = (
            self.create_boxed_text("Value")
            .scale(0.7)
            .next_to(pred_node, UP, buff=0.5)
            .shift(RIGHT * 1.5)
        )
        a_pred_to_policy = Arrow(
            pred_node.get_top(),
            policy.get_bottom(),
            buff=0.1,
            stroke_width=2,
            tip_length=0.15,
        )
        a_pred_to_value = Arrow(
            pred_node.get_top(),
            value.get_bottom(),
            buff=0.1,
            stroke_width=2,
            tip_length=0.15,
        )

        self.play(GrowArrow(a_state_to_pred), FadeIn(pred_node))
        self.play(GrowArrow(a_pred_to_policy), GrowArrow(a_pred_to_value))
        self.play(FadeIn(policy), FadeIn(value))
        self.wait(1)

        # 5. Dynamics Flow (Center to Bottom/Right)
        action = (
            self.create_boxed_text("Action")
            .scale(0.7)
            .next_to(dyn_node, LEFT, buff=1.5)
        )
        a_state_to_dyn = Arrow(
            state.get_bottom(),
            dyn_node.get_top(),
            buff=0.1,
            stroke_width=2,
            tip_length=0.15,
        )
        a_action_to_dyn = Arrow(
            action.get_right(),
            dyn_node.get_left(),
            buff=0.1,
            stroke_width=2,
            tip_length=0.15,
        )

        next_state = (
            self.create_boxed_text("Next State")
            .scale(0.7)
            .next_to(dyn_node, RIGHT, buff=1.5)
            .shift(UP * 0.5)
        )
        reward = (
            self.create_boxed_text("Reward")
            .scale(0.7)
            .next_to(dyn_node, RIGHT, buff=1.5)
            .shift(DOWN * 0.5)
        )
        a_dyn_to_next = Arrow(
            dyn_node.get_right(),
            next_state.get_left(),
            buff=0.1,
            stroke_width=2,
            tip_length=0.15,
        )
        a_dyn_to_reward = Arrow(
            dyn_node.get_right(),
            reward.get_left(),
            buff=0.1,
            stroke_width=2,
            tip_length=0.15,
        )

        self.play(FadeIn(action))
        self.play(
            GrowArrow(a_state_to_dyn), GrowArrow(a_action_to_dyn), FadeIn(dyn_node)
        )
        self.play(GrowArrow(a_dyn_to_next), GrowArrow(a_dyn_to_reward))
        self.play(FadeIn(next_state), FadeIn(reward))
        self.wait(1)

        # 6. The Recurrent Loop
        loop_arrow = CurvedArrow(
            start_point=next_state.get_top(),
            end_point=state.get_right(),
            angle=-TAU / 4,
            stroke_width=2,
            tip_length=0.15,
            color=WHITE,
        )
        loop_text = Text("Unroll K times", font_size=16, color=YELLOW).next_to(
            loop_arrow, RIGHT, buff=0.2
        )

        self.play(Create(loop_arrow))
        self.play(Write(loop_text))
        self.wait(4)
