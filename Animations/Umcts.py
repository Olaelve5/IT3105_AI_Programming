from manim import *
from config import BACKGROUND_COLOR


class UMCTS_explained(Scene):
    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR

        self.explain_node_stats()
        self.build_initial_tree()
        self.explain_selection()
        self.explain_evaluation_and_backprop()
        self.simulation_2()
        self.finish_simulations()

    def explain_node_stats(self):
        """Phase 1: Explain how a single node tracks its data."""
        self.demo_q_tracker = ValueTracker(0.0)
        self.demo_n_tracker = ValueTracker(0)

        # UPDATED: Smaller circle since text is outside
        circle = Circle(radius=0.4, color=ORANGE, fill_opacity=0.3, stroke_width=2)

        # Q-Value
        q_label = Text("Q = ", font_size=20)
        q_number = (
            DecimalNumber(
                self.demo_q_tracker.get_value(), num_decimal_places=2, font_size=26
            )
            .add_updater(lambda m: m.set_value(self.demo_q_tracker.get_value()))
            .next_to(q_label, RIGHT, buff=0.1)
        )
        q_group = VGroup(q_label, q_number)

        # Visits
        n_label = Text("V = ", font_size=20)
        n_number = (
            Integer(self.demo_n_tracker.get_value(), font_size=26)
            .add_updater(lambda m: m.set_value(self.demo_n_tracker.get_value()))
            .next_to(n_label, RIGHT, buff=0.1)
        )
        n_group = VGroup(n_label, n_number)

        # Prior
        p_text = Text("P = 0.75", font_size=20, color=TEAL)

        # UPDATED: Stack all stats vertically and put them to the right of the circle
        stats_group = VGroup(q_group, n_group, p_text).arrange(
            DOWN, aligned_edge=LEFT, buff=0.15
        )
        stats_group.next_to(circle, RIGHT, buff=0.3)

        self.demo_node_group = VGroup(circle, stats_group).move_to(ORIGIN)

        self.play(FadeIn(self.demo_node_group))
        self.wait(1)

        # Quick demonstration of the values updating
        self.play(
            self.demo_q_tracker.animate.set_value(0.85),
            run_time=3,
        )
        self.play(
            self.demo_n_tracker.animate.set_value(5),
            run_time=3,
        )
        self.wait(1)

        # Clear the screen for the tree
        self.play(FadeOut(self.demo_node_group))
        self.wait(0.5)

    def create_dynamic_node(self, q_init, n_init, prior_val, position, node_label=""):
        """Helper to instantly generate a trackable MCTS node with a sleek layout."""
        q_tracker = ValueTracker(q_init)
        n_tracker = ValueTracker(n_init)

        # UPDATED: Shrunk circle radius to 0.2
        circle = Circle(
            radius=0.2, color=ORANGE, fill_opacity=0.3, stroke_width=2
        ).move_to(position)

        # Q Value
        q_label = Text("Q = ", font_size=14)
        q_number = (
            DecimalNumber(q_tracker.get_value(), num_decimal_places=2, font_size=18)
            .add_updater(lambda m, qt=q_tracker: m.set_value(qt.get_value()))
            .next_to(q_label, RIGHT, buff=0.05)
        )
        q_group = VGroup(q_label, q_number)

        # Visits Value
        n_label = Text("V = ", font_size=14)
        n_number = (
            Integer(n_tracker.get_value(), font_size=18)
            .add_updater(lambda m, nt=n_tracker: m.set_value(nt.get_value()))
            .next_to(n_label, RIGHT, buff=0.05)
        )
        n_group = VGroup(n_label, n_number)

        # Prior
        p_text = Text(f"P = {prior_val:.2f}", font_size=14, color=TEAL)

        # UPDATED: Stack stats and pin to the right
        stats_group = VGroup(q_group, n_group, p_text).arrange(
            DOWN, aligned_edge=LEFT, buff=0.08
        )
        stats_group.next_to(circle, RIGHT, buff=0.15)

        if node_label:
            title = Text(node_label, font_size=14, weight=BOLD).next_to(
                circle, UP, buff=0.1
            )
            full_group = VGroup(circle, stats_group, title)
        else:
            full_group = VGroup(circle, stats_group)

        return full_group, q_tracker, n_tracker

    def build_initial_tree(self):
        """Phase 2: Establish the Root and sprout the initial empty child nodes."""

        root_pos = LEFT * 3.0 + UP * 2.8

        self.root_q_tracker = ValueTracker(0.20)
        self.root_n_tracker = ValueTracker(1)

        root_circle = Circle(
            radius=0.2, color=GRAY, fill_opacity=0.5, stroke_width=2
        ).move_to(root_pos)

        root_q_label = Text("Q = ", font_size=14)
        root_q_number = (
            DecimalNumber(
                self.root_q_tracker.get_value(), num_decimal_places=2, font_size=18
            )
            .add_updater(lambda m: m.set_value(self.root_q_tracker.get_value()))
            .next_to(root_q_label, RIGHT, buff=0.05)
        )
        root_q_group = VGroup(root_q_label, root_q_number)

        root_n_label = Text("V = ", font_size=14)
        root_n_number = (
            Integer(self.root_n_tracker.get_value(), font_size=18)
            .add_updater(lambda m: m.set_value(self.root_n_tracker.get_value()))
            .next_to(root_n_label, RIGHT, buff=0.05)
        )
        root_n_group = VGroup(root_n_label, root_n_number)

        stats_group = VGroup(root_q_group, root_n_group).arrange(
            DOWN, aligned_edge=LEFT, buff=0.08
        )
        stats_group.next_to(root_circle, RIGHT, buff=0.15)

        root_title = Text("S0 (Root)", font_size=14, weight=BOLD).next_to(
            root_circle, UP, buff=0.1
        )

        self.root_node = VGroup(root_circle, stats_group, root_title)

        self.play(FadeIn(self.root_node))
        self.wait(1)

        child1_pos = root_pos + DOWN * 2.0 + LEFT * 2
        child2_pos = root_pos + DOWN * 2.0 + RIGHT * 2

        # UPDATED: Priors are 0.55 and 0.45. q_init visually inherits 0.20
        self.child1_group, self.c1_q_tracker, self.c1_n_tracker = (
            self.create_dynamic_node(
                q_init=0.00, n_init=0, prior_val=0.55, position=child1_pos
            )
        )

        self.child2_group, self.c2_q_tracker, self.c2_n_tracker = (
            self.create_dynamic_node(
                q_init=0.00, n_init=0, prior_val=0.45, position=child2_pos
            )
        )

        self.edge1 = Line(
            self.root_node[0].get_bottom(),
            self.child1_group[0].get_top(),
            buff=0.1,
            stroke_width=2,
        )
        self.edge2 = Line(
            self.root_node[0].get_bottom(),
            self.child2_group[0].get_top(),
            buff=0.1,
            stroke_width=2,
        )

        self.edge1_label = Text("a1", font_size=14, color=TEAL).move_to(
            self.edge1.get_center() + LEFT * 0.2 + UP * 0.1
        )
        self.edge2_label = Text("a2", font_size=14, color=TEAL).move_to(
            self.edge2.get_center() + RIGHT * 0.2 + UP * 0.1
        )

        self.play(
            Create(self.edge1),
            Write(self.edge1_label),
            Create(self.edge2),
            Write(self.edge2_label),
            FadeIn(self.child1_group),
            FadeIn(self.child2_group),
        )
        self.wait(2)

    def explain_selection(self):
        """Phase 3: Introduce the 4 phases, then calculate the PUCT score."""

        phases_title = (
            Text("The 4 Phases of MCTS", font_size=24, weight=BOLD)
            .to_edge(UP)
            .shift(RIGHT * 3.5)
        )

        phase1 = Text("1. Selection", font_size=20)
        phase2 = Text("2. Expansion", font_size=20)
        phase3 = Text("3. Evaluation (Simulation)", font_size=20)
        phase4 = Text("4. Backpropagation", font_size=20)

        phases_group = (
            VGroup(phase1, phase2, phase3, phase4)
            .arrange(DOWN, aligned_edge=LEFT, buff=0.4)
            .next_to(phases_title, DOWN, buff=0.8)
        )

        self.play(Write(phases_title))
        self.play(FadeIn(phases_group, shift=RIGHT, lag_ratio=0.2))
        self.wait(2)

        self.play(
            phase1.animate.set_color(YELLOW).scale(1.1),
            FadeOut(VGroup(phase2, phase3, phase4)),
            run_time=1,
        )
        self.wait(0.5)

        self.selection_title = Text(
            "Phase 1: Selection", font_size=24, weight=BOLD, color=YELLOW
        ).move_to(phases_title.get_center())

        self.play(
            FadeOut(phases_title), ReplacementTransform(phase1, self.selection_title)
        )
        self.wait(0.5)

        formula = Text(
            "Score = Q + C * P * sqrt(V_parent) / (V_child + 1)", font_size=16
        ).next_to(self.selection_title, DOWN, buff=0.5)

        c_note = Text("(Simplified here as C = 2)", font_size=14, color=GRAY).next_to(
            formula, DOWN, buff=0.1
        )

        q_note = Text(
            "*Unvisited nodes inherit the parent's Q-value (0.20)",
            font_size=14,
            color=GRAY,
        ).next_to(c_note, DOWN, buff=0.1)

        true_c_formula = Text(
            "explo_rate = c1 + math.log((parent_visits + c2 + 1) / c2)",
            font_size=18,
            color=WHITE,
        )
        true_c_note = Text(
            "*(How C is actually calculated dynamically in code)",
            font_size=14,
            color=GRAY,
        )
        true_c_group = (
            VGroup(true_c_formula, true_c_note)
            .arrange(DOWN, aligned_edge=LEFT, buff=0.1)
            .to_edge(DOWN, buff=2.5)
        )

        self.play(Write(formula), Write(c_note), Write(q_note))
        self.wait(1)

        self.play(Write(true_c_group))
        self.wait(2)
        self.play(FadeOut(true_c_group))
        self.wait(0.5)

        a1_text = (
            Text("Action a1:", font_size=18, color=TEAL)
            .next_to(q_note, DOWN, buff=0.5)
            .align_to(formula, LEFT)
        )

        # UPDATED MATH: Uses the new 0.55 prior
        a1_calc = Text("0.20 + 2 * 0.55 * 1 / 1 = 1.30", font_size=18).next_to(
            a1_text, RIGHT, buff=0.2
        )

        self.play(Write(a1_text), Write(a1_calc))
        self.wait(1)

        a2_text = (
            Text("Action a2:", font_size=18, color=TEAL)
            .next_to(a1_text, DOWN, buff=0.4)
            .align_to(a1_text, LEFT)
        )

        # UPDATED MATH: Uses the new 0.45 prior
        a2_calc = Text("0.20 + 2 * 0.45 * 1 / 1 = 1.10", font_size=18).next_to(
            a2_text, RIGHT, buff=0.2
        )

        self.play(Write(a2_text), Write(a2_calc))
        self.wait(1)

        conclusion = (
            Text("a1 has the highest score", font_size=20, color=YELLOW)
            .next_to(a2_calc, DOWN, buff=0.4)
            .align_to(self.selection_title, LEFT)
        )

        self.play(Write(conclusion))

        self.selection_math_group = VGroup(
            formula, c_note, q_note, a1_text, a1_calc, a2_text, a2_calc, conclusion
        )

        self.play(
            self.edge1.animate.set_color(YELLOW).set_stroke(width=4),
            self.edge1_label.animate.set_color(YELLOW),
            self.child1_group[0].animate.set_color(YELLOW),
        )
        self.wait(2)

    def explain_evaluation_and_backprop(self):
        """Phases 2, 3 & 4: Expand, Evaluate, and Backpropagate."""

        self.play(FadeOut(self.selection_math_group))

        # --- Phase 2 (Expansion) ---
        phase2_title = Text(
            "Phase 2: Expansion", font_size=24, weight=BOLD, color=YELLOW
        ).move_to(self.selection_title)

        self.play(ReplacementTransform(self.selection_title, phase2_title))

        exp_text1 = (
            Text("Dynamics Net. generates new state S1", font_size=18)
            .next_to(phase2_title, DOWN, buff=0.6)
            .align_to(phase2_title, LEFT)
        )

        self.play(Write(exp_text1))
        self.wait(1)

        exp_text2 = (
            Text("Prediction Net. generates new child nodes", font_size=18)
            .next_to(exp_text1, DOWN, buff=0.4)
            .align_to(phase2_title, LEFT)
        )
        self.play(Write(exp_text2))
        self.wait(1)

        a1_c1_pos = self.child1_group[0].get_center() + DOWN * 1.8 + LEFT * 1.2
        a1_c2_pos = self.child1_group[0].get_center() + DOWN * 1.8 + RIGHT * 1.2

        # FIXED: Real internal values start at 0.00!
        self.a1_child1_group, self.a1_c1_q, self.a1_c1_n = self.create_dynamic_node(
            q_init=0.00, n_init=0, prior_val=0.60, position=a1_c1_pos
        )
        self.a1_child2_group, self.a1_c2_q, self.a1_c2_n = self.create_dynamic_node(
            q_init=0.00, n_init=0, prior_val=0.40, position=a1_c2_pos
        )

        # FIXED: Saved to self so we can dim them in the finale
        self.a1_edge1 = Line(
            self.child1_group[0].get_bottom(),
            self.a1_child1_group[0].get_top(),
            buff=0.1,
            stroke_width=2,
        )
        self.a1_edge2 = Line(
            self.child1_group[0].get_bottom(),
            self.a1_child2_group[0].get_top(),
            buff=0.1,
            stroke_width=2,
        )

        self.play(
            Create(self.a1_edge1),
            Create(self.a1_edge2),
            FadeIn(self.a1_child1_group),
            FadeIn(self.a1_child2_group),
            self.child1_group[0].animate.set_color(GRAY),
        )
        self.wait(1.5)

        # --- Phase 3 (Evaluation) ---
        phase3_title = Text(
            "Phase 3: Evaluation", font_size=24, weight=BOLD, color=YELLOW
        ).move_to(phase2_title)

        self.play(
            ReplacementTransform(phase2_title, phase3_title),
            FadeOut(exp_text1),
            FadeOut(exp_text2),
        )

        eval_text1 = (
            Text("Prediction Net. evaluates S1:", font_size=18)
            .next_to(phase3_title, DOWN, buff=0.6)
            .align_to(phase3_title, LEFT)
        )

        value_box = SurroundingRectangle(
            Text(" Value (v) = +0.80 ", font_size=20), color=GOLD, buff=0.2
        )
        value_text = Text(" Value (v) = +0.80 ", font_size=20, color=GOLD).move_to(
            value_box.get_center()
        )
        value_group = (
            VGroup(value_box, value_text)
            .next_to(eval_text1, DOWN, buff=0.4)
            .align_to(phase3_title, LEFT)
        )

        self.play(Write(eval_text1))
        self.play(Create(value_box), Write(value_text))
        self.wait(1.5)

        # --- Phase 4 (Backpropagation) ---
        phase4_title = (
            Text("Phase 4: Backpropagation", font_size=24, weight=BOLD, color=YELLOW)
            .next_to(value_group, DOWN, buff=0.8)
            .align_to(phase3_title, LEFT)
        )

        backprop_text = (
            Text("Pass the value back up to update Q and V.", font_size=18)
            .next_to(phase4_title, DOWN, buff=0.4)
            .align_to(phase4_title, LEFT)
        )

        self.play(
            Write(phase4_title),
            FadeIn(backprop_text),
        )
        self.wait(1)

        math_1 = (
            Text("Value is Discounted (discount = 0.99)", font_size=18, color=TEAL)
            .next_to(backprop_text, DOWN, buff=0.4)
            .align_to(backprop_text, LEFT)
        )
        math_2 = (
            Text("a1 Q: 0.00 + (0.99 * 0.80) / 1 visits = 0.79", font_size=18)
            .next_to(math_1, DOWN, buff=0.2)
            .align_to(math_1, LEFT)
        )
        math_3 = (
            Text("Root Q: (0.20 + 0.79) / 2 visits = 0.50", font_size=18)
            .next_to(math_2, DOWN, buff=0.2)
            .align_to(math_1, LEFT)
        )

        self.play(Write(math_1))

        payload = Dot(color=GOLD, radius=0.15).move_to(
            self.child1_group[0].get_center()
        )

        self.play(
            self.c1_q_tracker.animate.set_value(0.79),
            self.c1_n_tracker.animate.set_value(1),
            self.child1_group[0].animate.set_color(GRAY).set_stroke(width=4),
            self.edge1.animate.set_color(WHITE).set_stroke(width=2),
            self.edge1_label.animate.set_color(TEAL),
            Write(math_2),
            run_time=1,
        )
        self.wait(0.5)

        self.play(FadeIn(payload, scale=0.5))

        self.play(payload.animate.move_to(self.root_node[0].get_center()), run_time=1)

        self.play(
            self.root_q_tracker.animate.set_value(0.50),
            self.root_n_tracker.animate.set_value(2),
            FadeOut(payload, scale=1.5),
            FadeOut(value_box),
            Write(math_3),
            run_time=1,
        )

        self.play(self.child1_group[0].animate.set_stroke(width=2), run_time=0.5)
        self.wait(2)

    def simulation_2(self):
        """Phase 5: Show the second simulation exploring the other branch."""

        # FIXED: Bulletproof right-side wipe (Ignores all tree nodes because they are x < 0.2)
        right_side_mobjects = [
            m
            for m in self.mobjects
            if m.get_center()[0] > 1.0 and not isinstance(m, Dot)
        ]
        self.play(FadeOut(Group(*right_side_mobjects)))

        sim2_title = (
            Text("Next Iteration", font_size=32, color=BLUE)
            .to_edge(UP)
            .shift(RIGHT * 3.5)
        )
        self.play(Write(sim2_title))

        # 2. Selection Rematch
        formula = (
            Text("Score = Q + C * P * sqrt(V_p) / (V_c + 1)", font_size=16)
            .next_to(sim2_title, DOWN, buff=0.8)
            .align_to(sim2_title, LEFT)
        )

        # FIXED: Removed the fake visual update. a2 naturally displays 0.00!

        a1_calc = (
            Text("a1: 0.79 + 2*0.55*sqrt(2)/2 = 1.57", font_size=16, color=GRAY)
            .next_to(formula, DOWN, buff=0.5)
            .align_to(sim2_title, LEFT)
        )

        # a2 calculates using virtual inheritance (0.50), but its node stays visual 0.00
        a2_calc = (
            Text("a2: 0.50 + 2*0.45*sqrt(2)/1 = 1.77", font_size=16, color=YELLOW)
            .next_to(a1_calc, DOWN, buff=0.3)
            .align_to(sim2_title, LEFT)
        )

        self.play(Write(formula))
        self.play(Write(a1_calc), Write(a2_calc))

        self.play(
            self.edge2.animate.set_color(YELLOW).set_stroke(width=4),
            self.child2_group[0].animate.set_color(YELLOW),
            self.edge1.animate.set_color(WHITE).set_stroke(width=2),
            self.child1_group[0].animate.set_color(GRAY),
        )
        self.wait(1)

        # 3. Expansion & Evaluation
        eval_title = (
            Text("Expansion & Evaluation", font_size=24, color=YELLOW)
            .move_to(sim2_title)
            .align_to(sim2_title, LEFT)
        )

        self.play(
            ReplacementTransform(sim2_title, eval_title),
            FadeOut(formula),
            FadeOut(a1_calc),
            FadeOut(a2_calc),
        )

        a2_c1_pos = self.child2_group[0].get_center() + DOWN * 1.8 + LEFT * 1.2
        a2_c2_pos = self.child2_group[0].get_center() + DOWN * 1.8 + RIGHT * 1.2

        # FIXED: Real internal values start at 0.00, saved to self
        self.c2_child1_group, self.c2_c1_q, self.c2_c1_n = self.create_dynamic_node(
            0.00, 0, 0.5, a2_c1_pos
        )
        self.c2_child2_group, self.c2_c2_q, self.c2_c2_n = self.create_dynamic_node(
            0.00, 0, 0.5, a2_c2_pos
        )

        self.e2_1 = Line(
            self.child2_group[0].get_bottom(),
            self.c2_child1_group[0].get_top(),
            buff=0.1,
            stroke_width=2,
        )
        self.e2_2 = Line(
            self.child2_group[0].get_bottom(),
            self.c2_child2_group[0].get_top(),
            buff=0.1,
            stroke_width=2,
        )

        val_text = (
            Text("Dynamics Net. -> S2\nPrediction Net. -> v: +0.10", font_size=18)
            .next_to(eval_title, DOWN, buff=0.6)
            .align_to(eval_title, LEFT)
        )

        self.play(
            Write(val_text),
            Create(self.e2_1),
            Create(self.e2_2),
            FadeIn(self.c2_child1_group),
            FadeIn(self.c2_child2_group),
            self.child2_group[0].animate.set_color(GRAY),
        )
        self.wait(1)

        # 4. Backpropagation
        back_title = (
            Text("Backpropagation", font_size=24, color=YELLOW)
            .move_to(eval_title)
            .align_to(eval_title, LEFT)
        )

        back_math = (
            Text("Root Q: (0.99 + 0.10) / 3 = 0.36", font_size=18)
            .next_to(back_title, DOWN, buff=0.8)
            .align_to(back_title, LEFT)
        )

        self.play(
            ReplacementTransform(eval_title, back_title),
            FadeOut(val_text),
            Write(back_math),
        )

        payload = Dot(color=GOLD, radius=0.15).move_to(
            self.child2_group[0].get_center()
        )

        self.play(FadeIn(payload, scale=0.5))
        self.play(
            self.c2_q_tracker.animate.set_value(0.10),
            self.c2_n_tracker.animate.set_value(1),
            self.child2_group[0].animate.set_stroke(width=4),
            self.edge2.animate.set_color(WHITE).set_stroke(width=2),
            run_time=0.5,
        )

        self.play(payload.animate.move_to(self.root_node[0].get_center()), run_time=0.5)

        self.play(
            self.root_q_tracker.animate.set_value(0.36),
            self.root_n_tracker.animate.set_value(3),
            FadeOut(payload, scale=1.5),
        )

        self.play(self.child2_group[0].animate.set_stroke(width=2), run_time=0.5)
        self.wait(1)

    def finish_simulations(self):
        """Phase 6: Fast-forward to 100 simulations and choose the final action."""

        # Clean up the right side again
        right_ui = [
            m
            for m in self.mobjects
            if m.get_center()[0] > 1.0 and not isinstance(m, Dot)
        ]
        self.play(FadeOut(Group(*right_ui)))

        final_title = (
            Text("Continue until 100 simulations", font_size=32, color=BLUE)
            .to_edge(UP)
            .shift(RIGHT * 3.5)
        )
        self.play(Write(final_title))

        # Helper to create 3 vertical dots to show the tree continuing
        def create_vdots(node_group):
            return (
                VGroup(*[Dot(radius=0.04, color=GRAY) for _ in range(3)])
                .arrange(DOWN, buff=0.1)
                .next_to(node_group[0], DOWN, buff=0.2)
            )

        vdots1 = create_vdots(self.a1_child1_group)
        vdots2 = create_vdots(self.a1_child2_group)
        vdots3 = create_vdots(self.c2_child1_group)
        vdots4 = create_vdots(self.c2_child2_group)

        # Rapidly update the trackers to show time passing
        # Note: (a1 visits + a2 visits) = 99. Plus 1 initial visit at root = 100 root visits.
        # Level 2 visits = Parent visits - 1 (since the first visit is the expansion of the parent)
        self.play(
            self.root_n_tracker.animate.set_value(100),
            self.root_q_tracker.animate.set_value(0.65),
            # a1 (72 total visits -> 71 distributed to children)
            self.c1_n_tracker.animate.set_value(72),
            self.c1_q_tracker.animate.set_value(0.68),
            self.a1_c1_n.animate.set_value(50),
            self.a1_c1_q.animate.set_value(0.75),
            self.a1_c2_n.animate.set_value(21),
            self.a1_c2_q.animate.set_value(0.45),
            # a2 (27 total visits -> 26 distributed to children)
            self.c2_n_tracker.animate.set_value(27),
            self.c2_q_tracker.animate.set_value(0.42),
            self.c2_c1_n.animate.set_value(18),
            self.c2_c1_q.animate.set_value(0.55),
            self.c2_c2_n.animate.set_value(8),
            self.c2_c2_q.animate.set_value(0.25),
            # Turn Level 2 nodes GRAY as they get expanded during the 100 simulations
            self.a1_child1_group[0].animate.set_color(GRAY),
            self.a1_child2_group[0].animate.set_color(GRAY),
            self.c2_child1_group[0].animate.set_color(GRAY),
            self.c2_child2_group[0].animate.set_color(GRAY),
            # Fade in the vertical dots to show growth!
            FadeIn(vdots1),
            FadeIn(vdots2),
            FadeIn(vdots3),
            FadeIn(vdots4),
            run_time=3.0,
            rate_func=linear,
        )
        self.wait(1)

        rule_text1 = (
            Text("How do we pick the final move?", font_size=20, weight=BOLD)
            .next_to(final_title, DOWN, buff=0.8)
            .align_to(final_title, LEFT)
        )
        rule_text2 = (
            Text("We do not pick the highest Q-value.", font_size=18, color=RED)
            .next_to(rule_text1, DOWN, buff=0.2)
            .align_to(final_title, LEFT)
        )
        rule_text3 = (
            Text(
                "We pick the node with the most Visits (V).", font_size=18, color=GREEN
            )
            .next_to(rule_text2, DOWN, buff=0.2)
            .align_to(final_title, LEFT)
        )

        self.play(Write(rule_text1))
        self.play(Write(rule_text2), Write(rule_text3))
        self.wait(1)

        winner_text = (
            Text("a1 wins! (72 > 27)", font_size=24, color=YELLOW)
            .next_to(rule_text3, DOWN, buff=0.8)
            .align_to(final_title, LEFT)
        )
        self.play(Write(winner_text))

        # Highlight the final winning action
        self.play(
            self.child1_group[0].animate.set_color(GREEN).set_stroke(width=6),
            self.edge1.animate.set_color(GREEN).set_stroke(width=6),
            self.edge1_label.animate.set_color(GREEN),
        )

        # Dim the rest of the tree to emphasize the chosen path
        self.play(
            self.child2_group.animate.set_opacity(0.3),
            self.edge2.animate.set_opacity(0.3),
            self.edge2_label.animate.set_opacity(0.3),
            self.a1_child1_group.animate.set_opacity(0.3),
            self.a1_child2_group.animate.set_opacity(0.3),
            self.c2_child1_group.animate.set_opacity(0.3),
            self.c2_child2_group.animate.set_opacity(0.3),
            self.a1_edge1.animate.set_opacity(0.3),
            self.a1_edge2.animate.set_opacity(0.3),
            self.e2_1.animate.set_opacity(0.3),
            self.e2_2.animate.set_opacity(0.3),
            # Dim the depth dots too!
            vdots1.animate.set_opacity(0.3),
            vdots2.animate.set_opacity(0.3),
            vdots3.animate.set_opacity(0.3),
            vdots4.animate.set_opacity(0.3),
        )

        self.wait(3)
