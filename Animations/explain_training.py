from manim import *
from config import BACKGROUND_COLOR


class TrainingLoopExplained(Scene):
    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR

        self.show_overview()
        self.explain_self_play()
        self.explain_sampling()
        self.explain_unrolling()
        self.explain_loss_and_update()

    def show_overview(self):
        """Introduce the 4 main phases of the MuZero training loop."""

        title = Text(
            "The Training Loop", font_size=36, weight=BOLD, color=TEAL
        ).to_edge(UP)
        self.play(Write(title))

        # The 4 Phases
        phase1 = Text("1. Self-Play & Replay Buffer", font_size=24)
        phase2 = Text("2. Sample a Batch", font_size=24)
        phase3 = Text("3. Network Unrolling (BPTT)", font_size=24)
        phase4 = Text("4. Calculate Loss & Update", font_size=24)

        self.phases_group = (
            VGroup(phase1, phase2, phase3, phase4)
            .arrange(DOWN, aligned_edge=LEFT, buff=0.6)
            .move_to(ORIGIN)
        )

        # Animate them in sequentially
        self.play(FadeIn(self.phases_group, shift=UP, lag_ratio=0.2))
        self.wait(2)

        # Isolate Phase 1 to transition smoothly into the next method
        self.phase1_title = Text(
            "Phase 1: Self-Play & Replay Buffer",
            font_size=28,
            weight=BOLD,
            color=YELLOW,
        ).to_edge(UP)

        self.play(
            FadeOut(title),
            FadeOut(VGroup(phase2, phase3, phase4)),
            ReplacementTransform(phase1, self.phase1_title),
        )
        self.wait(1)

    def explain_self_play(self):
        """Phase 1: Visualizing data flow from Self-Play into the Replay Buffer."""
        self.add(self.phase1_title)

        # --- LEFT SIDE: Gameplay Footage Placeholder ---
        game_area = (
            Rectangle(width=5.5, height=5.5).to_edge(LEFT, buff=0.8).shift(DOWN * 0.5)
        )

        game_text = Text("Gameplay Overlay", font_size=20, color=GRAY).move_to(
            game_area
        )
        self.play(Write(game_text))
        self.wait(1)

        # Concept 1: Self-Play generates MCTS and Reward data
        concept_text1 = (
            Text("AI plays moves using MCTS.", font_size=18)
            .next_to(self.phase1_title, DOWN, buff=0.6)
            .align_to(game_area, LEFT)
        )

        concept_text2 = (
            Text(
                "Every step saves (Observation, Action,\nMCTS Policy, MCTS Value, Real Reward).",
                font_size=18,
                color=GRAY,
            )
            .next_to(concept_text1, DOWN, buff=0.2)
            .align_to(game_area, LEFT)
        )

        self.play(Write(concept_text1), Write(concept_text2))
        self.wait(1)

        # --- RIGHT SIDE: Stylized Replay Buffer ---
        buffer_width = 3.0
        buffer_height = 3.0
        buffer_bottom = game_area.get_bottom()[1]
        buffer_pos = RIGHT * 3.5 + UP * (buffer_bottom + buffer_height / 2) + UP * 1.0

        self.buffer_container = RoundedRectangle(
            width=buffer_width,
            height=buffer_height,
            corner_radius=0.3,
            color=BLUE,
            fill_opacity=0.1,
            stroke_width=3,
        ).move_to(buffer_pos)

        buffer_title = Text("Replay Buffer", font_size=22, color=WHITE).next_to(
            self.buffer_container, UP, buff=0.45
        )
        total_cap_text = Text(
            "(Capacity: 25,000 steps)", font_size=14, color=GRAY
        ).next_to(buffer_title, DOWN, buff=0.1)

        # Visualizing the stack with lines
        divider1 = Line(
            start=self.buffer_container.get_left() + UP * 0.8 + RIGHT * 0.1,
            end=self.buffer_container.get_right() + UP * 0.8 + LEFT * 0.1,
            color=BLUE,
            stroke_opacity=0.5,
        )
        divider2 = divider1.copy().shift(DOWN * 0.8)
        divider3 = divider1.copy().shift(DOWN * 1.6)
        buffer_lines = VGroup(divider1, divider2, divider3)

        # --- Capacity Indicator ---
        self.capacity_tracker = ValueTracker(0)

        cap_label_text = Text("Stored Steps:", font_size=18).next_to(
            self.buffer_container, DOWN, buff=0.3
        )
        cap_num = (
            Integer(0, font_size=22, color=GREEN)
            .add_updater(lambda m: m.set_value(self.capacity_tracker.get_value()))
            .next_to(cap_label_text, RIGHT, buff=0.2)
        )

        cap_total_text = Text(
            "Target each generation: 1,000 steps", font_size=14, color=GRAY
        ).next_to(cap_label_text, DOWN, buff=0.1)

        cap_group = VGroup(cap_label_text, cap_num, cap_total_text)

        self.play(
            FadeIn(buffer_title),
            FadeIn(total_cap_text),
            FadeIn(VGroup(self.buffer_container, buffer_lines)),
            FadeIn(cap_group),
        )
        self.wait(1)

        # --- The Data Flow Animation ---
        self.rapid_packets = VGroup(
            *[
                Dot(radius=0.1, color=ORANGE).move_to(
                    game_area.get_center()
                    + np.array(
                        [np.random.uniform(-1, 1), np.random.uniform(-1, 1), 0.0]
                    )
                    * 1.2
                )
                for _ in range(20)
            ]
        )

        self.play(FadeIn(self.rapid_packets, lag_ratio=0.1))
        self.wait(0.5)

        # Create a grid layout that is guaranteed to be inside the box
        # We'll center the 4x5 grid inside the buffer_container
        animations = []
        rows, cols = 5, 4
        x_spacing = 0.5
        y_spacing = 0.4

        for i, packet in enumerate(self.rapid_packets):
            # Calculate row and column
            row = i // cols  # 0 to 4
            col = i % cols  # 0 to 3

            # Calculate position relative to the center of the buffer
            # Offset centers the grid (subtracting half the total grid width/height)
            x_pos = (col - (cols - 1) / 2) * x_spacing
            y_pos = (row - (rows - 1) / 2) * y_spacing

            target_pos = self.buffer_container.get_center() + np.array(
                [x_pos, y_pos, 0]
            )

            animations.append(
                packet.animate(
                    rate_func=squish_rate_func(linear, 0.0 + i * 0.03, 0.4 + i * 0.03)
                ).move_to(target_pos)
            )

        self.play(
            LaggedStart(*animations, lag_ratio=0.05),
            self.capacity_tracker.animate.set_value(1000),
            run_time=3.0,
        )
        self.wait(1)

        # Cleanup LEFT side only for Phase 2 transition
        self.play(FadeOut(game_text), FadeOut(concept_text1), FadeOut(concept_text2))

    def explain_sampling(self):
        """Phase 2: Pulling a sequence from the buffer."""

        # Change the main title to Phase 2
        self.phase2_title = Text(
            "Phase 2: Sample a Batch", font_size=28, weight=BOLD, color=YELLOW
        ).to_edge(UP)
        self.play(ReplacementTransform(self.phase1_title, self.phase2_title))

        # TWEAK: Centered below the title
        desc1 = Text(
            "MuZero doesn't just train on single, isolated frames.", font_size=18
        ).next_to(self.phase2_title, DOWN, buff=0.6)

        desc2 = Text(
            "It samples an initial observation and a sequence of future actions.",
            font_size=18,
            color=GRAY,
        ).next_to(desc1, DOWN, buff=0.2)

        self.play(Write(desc1), Write(desc2))
        self.wait(1.5)

        # Grab a copy of a packet from the filled buffer so the buffer stays full!
        sampled_packet = self.rapid_packets[-1].copy()

        self.play(
            sampled_packet.animate.move_to(LEFT * 2.5 + DOWN * 0.5).scale(1.5),
            run_time=1,
        )

        # Expand it into a large "Sequence Box" on the left side
        self.sequence_box = RoundedRectangle(
            width=7.5, height=2.5, corner_radius=0.2, color=BLUE, fill_opacity=0.1
        ).move_to(LEFT * 2.5 + DOWN * 0.5)

        self.play(ReplacementTransform(sampled_packet, self.sequence_box))

        # 1. Show the Initial Observation
        self.obs_box = Rectangle(
            width=1.0, height=1.0, color=TEAL, fill_opacity=0.2
        ).move_to(self.sequence_box.get_left() + RIGHT * 0.7 + UP * 0.3)
        self.obs_text = Text("Obs", font_size=16).move_to(self.obs_box)

        self.play(FadeIn(self.obs_box), Write(self.obs_text))

        # 2. Show the sequence of Actions Unrolling (TWEAK: 6 steps, smaller arrows)
        actions_start = self.obs_box.get_right() + RIGHT * 0.1
        self.sampled_arrows = VGroup()
        self.sampled_actions = VGroup()

        for i in range(6):
            arrow = (
                Arrow(start=LEFT, end=RIGHT, color=WHITE, buff=0)
                .scale(0.4)
                .next_to(actions_start + RIGHT * i * 0.95, RIGHT, buff=0)
            )
            act_box = SurroundingRectangle(
                Text(f"A {i+1}", font_size=12), color=YELLOW, buff=0.1
            ).next_to(arrow, UP, buff=0.1)
            act_text = Text(f"A {i+1}", font_size=12, color=YELLOW).move_to(act_box)

            self.sampled_arrows.add(arrow)
            self.sampled_actions.add(VGroup(act_box, act_text))

        self.play(
            Create(self.sampled_arrows), FadeIn(self.sampled_actions), lag_ratio=0.2
        )
        self.wait(1)

        # 3. Show that Targets come with it
        target_text = Text(
            "Includes Targets: Policy, Value, Reward",
            font_size=16,
            color=ORANGE,
        ).move_to(self.sequence_box.get_bottom() + UP * 0.4)

        target_lines = VGroup()
        for i in range(6):
            line = DashedLine(
                start=self.sampled_arrows[i].get_center() + DOWN * 0.1,
                end=self.sampled_arrows[i].get_center() + DOWN * 0.7,
                color=GRAY,
            )
            target_lines.add(line)

        self.play(FadeIn(target_text), Create(target_lines), lag_ratio=0.1)
        self.wait(2)

        # Cleanup text for Phase 3, but keep the core Sequence Box and Buffer on screen!
        self.play(
            FadeOut(desc1), FadeOut(desc2), FadeOut(target_lines), FadeOut(target_text)
        )

        # 1. Fade out the unrolled actions/arrows to visually "pack" the sequence away
        self.play(FadeOut(self.sampled_arrows), FadeOut(self.sampled_actions))

        # 2. Create a target shape for the sequence box to shrink into (wrapping the obs_box)
        shrunk_sequence_box = RoundedRectangle(
            width=1.4, height=1.4, corner_radius=0.2, color=BLUE, fill_opacity=0.1
        ).move_to(self.obs_box.get_center())

        # 3. Create the text to sit next to it
        self.batch_text = Text(
            "× 256 (Batch Size)", font_size=20, color=YELLOW
        ).next_to(shrunk_sequence_box, RIGHT, buff=0.4)

        # 4. Animate the large sequence box shrinking, and write the batch text
        self.play(
            self.sequence_box.animate.become(shrunk_sequence_box),
            Write(self.batch_text),
        )
        self.wait(2)

    def explain_unrolling(self):
        """Phase 3: Passing the data through the networks."""

        # 1. Update the Main Title
        self.phase3_title = Text(
            "Phase 3: Network Unrolling (BPTT)", font_size=28, weight=BOLD, color=YELLOW
        ).to_edge(UP)
        self.play(ReplacementTransform(self.phase2_title, self.phase3_title))

        # 2. The Great Cleanup
        keep_mobs = [self.phase3_title, self.sequence_box, self.obs_box, self.obs_text]
        fade_out_mobs = [m for m in self.mobjects if m not in keep_mobs]

        self.play(*[FadeOut(m) for m in fade_out_mobs], run_time=1)

        # 3. Move the starting Observation to the far left to make room for the chain
        start_group = VGroup(self.sequence_box, self.obs_box, self.obs_text)
        self.play(start_group.animate.move_to(LEFT * 5.5 + UP * 0.5))

        # ==========================================
        # STEP 0: The Representation Network
        # ==========================================

        self.rep_node = self.create_simple_node("Rep.", BLUE).next_to(
            start_group, RIGHT, buff=0.8
        )
        arrow_obs_rep = Arrow(
            start_group.get_right(), self.rep_node.get_left(), buff=0.1
        )

        # Hidden State s_0
        s0_box = Rectangle(width=0.6, height=0.6, color=WHITE).next_to(
            self.rep_node, RIGHT, buff=0.8
        )
        s0_text = MathTex("s_0", font_size=24).move_to(s0_box)
        s0_group = VGroup(s0_box, s0_text)
        arrow_rep_s0 = Arrow(self.rep_node.get_right(), s0_group.get_left(), buff=0.1)

        # Prediction Network for s_0
        self.pred_node0 = self.create_simple_node("Pred.", YELLOW).next_to(
            s0_group, DOWN, buff=0.8
        )
        arrow_s0_pred0 = Arrow(
            s0_group.get_bottom(), self.pred_node0.get_top(), buff=0.1
        )

        pred_text0 = MathTex(r"\pi_0, v_0", font_size=20, color=ORANGE).next_to(
            self.pred_node0, DOWN, buff=0.3
        )
        arrow_pred0_out = Arrow(
            self.pred_node0.get_bottom(), pred_text0.get_top(), buff=0.1
        )

        self.play(Create(arrow_obs_rep), FadeIn(self.rep_node), run_time=0.8)
        self.play(Create(arrow_rep_s0), FadeIn(s0_group), run_time=0.8)
        self.play(
            Create(arrow_s0_pred0),
            FadeIn(self.pred_node0),
            Create(arrow_pred0_out),
            Write(pred_text0),
            run_time=0.8,
        )

        # Initialize lists to track nodes for Phase 4 (BPTT)
        self.dyn_nodes = []
        self.pred_nodes = [self.pred_node0]

        # ==========================================
        # STEPS 1 & 2: Dynamics and Prediction Loop
        # ==========================================

        s_groups = [s0_group]
        pred_texts = [pred_text0]
        reward_texts = []  # List to track rewards for the bounding box

        for i in range(1, 3):
            prev_s = s_groups[i - 1]

            # 1. Bring in the Real Action
            a_box = SurroundingRectangle(
                MathTex(f"a_{i}", font_size=20), color=YELLOW, buff=0.1
            )
            a_text = MathTex(f"a_{i}", font_size=20, color=YELLOW).move_to(a_box)
            a_group = VGroup(a_box, a_text).next_to(prev_s, UP, buff=0.8)

            # 2. Dynamics Network (Fixed variable name)
            dyn_node = self.create_simple_node("Dyn.", GREEN).next_to(
                prev_s, RIGHT, buff=0.8
            )

            arrow_s_dyn = Arrow(prev_s.get_right(), dyn_node.get_left(), buff=0.1)
            arrow_a_dyn = Arrow(
                a_group.get_bottom(), dyn_node.get_center(), buff=0.4, color=YELLOW
            )

            # 3. New Hidden State (s_i)
            s_box = Rectangle(width=0.6, height=0.6, color=WHITE)
            s_text = MathTex(f"s_{i}", font_size=24).move_to(s_box)
            s_group = VGroup(s_box, s_text).next_to(dyn_node, RIGHT, buff=0.8)
            arrow_dyn_s = Arrow(dyn_node.get_right(), s_group.get_left(), buff=0.1)

            # 4. Predicted Reward (r_i)
            r_text = (
                MathTex(f"r_{i}", font_size=20, color=RED)
                .next_to(dyn_node, UP, buff=0.5)
                .shift(RIGHT * 0.3)
            )
            arrow_dyn_r = Arrow(
                dyn_node.get_center(), r_text.get_center(), buff=0.3, color=RED
            )

            # 5. Prediction Network (Fixed variable name)
            pred_node = self.create_simple_node("Pred.", YELLOW).next_to(
                s_group, DOWN, buff=0.8
            )
            arrow_s_pred = Arrow(s_group.get_bottom(), pred_node.get_top(), buff=0.1)

            pred_text = MathTex(
                r"\pi_{" + str(i) + r"}, v_{" + str(i) + r"}",
                font_size=20,
                color=ORANGE,
            ).next_to(pred_node, DOWN, buff=0.3)
            arrow_pred_out = Arrow(
                pred_node.get_bottom(), pred_text.get_top(), buff=0.1
            )

            # Track elements for Phase 4
            s_groups.append(s_group)
            pred_texts.append(pred_text)
            reward_texts.append(r_text)
            self.dyn_nodes.append(dyn_node)
            self.pred_nodes.append(pred_node)

            # Animate
            self.play(FadeIn(a_group, shift=DOWN), run_time=0.5)
            self.play(
                Create(arrow_s_dyn),
                Create(arrow_a_dyn),
                FadeIn(dyn_node),
                run_time=0.5,
            )
            self.play(
                Create(arrow_dyn_s),
                FadeIn(s_group),
                Create(arrow_dyn_r),
                Write(r_text),
                run_time=0.5,
            )
            self.play(
                Create(arrow_s_pred),
                FadeIn(pred_node),
                Create(arrow_pred_out),
                Write(pred_text),
                run_time=0.5,
            )

        # Add continuation dots (...) to show it keeps unrolling
        continuation_dots = VGroup(
            *[Dot(radius=0.04, color=GRAY) for _ in range(3)]
        ).arrange(RIGHT, buff=0.15)
        continuation_dots.next_to(s_groups[-1], RIGHT, buff=0.4)
        self.play(FadeIn(continuation_dots))
        self.wait(1)

        # ==========================================
        # PREP FOR PHASE 4: Grouping ALL Outputs
        # ==========================================

        # Include both the policy/value predictions AND the reward predictions
        predictions_group = VGroup(*pred_texts, *reward_texts)
        self.pred_bounding_box = SurroundingRectangle(
            predictions_group, color=ORANGE, buff=0.5, corner_radius=0.2
        )

        prep_text = Text(
            "Predictions ready for Loss calculation", font_size=18, color=ORANGE
        ).next_to(self.pred_bounding_box, DOWN, buff=0.2)

        self.play(Create(self.pred_bounding_box), Write(prep_text))
        self.wait(2)

        self.prep_text = prep_text

    def explain_loss_and_update(self):
        """Phase 4: Comparing predictions to targets and updating weights."""

        # 1. The Great Slate Wipe
        # Keep the Phase 3 title briefly so we can transform it
        keep_mobs = [self.phase3_title]
        fade_out_mobs = [m for m in self.mobjects if m not in keep_mobs]
        self.play(*[FadeOut(m) for m in fade_out_mobs], run_time=1)

        # 2. Update the Main Title
        self.phase4_title = Text(
            "Phase 4: Calculate Loss & Update", font_size=28, weight=BOLD, color=YELLOW
        ).to_edge(UP)
        self.play(ReplacementTransform(self.phase3_title, self.phase4_title))

        # ==========================================
        # STEP 1: The Three Loss Functions
        # ==========================================

        # 1. Policy Loss (Cross Entropy)
        policy_title = Text("1. Policy Loss (Cross-Entropy)", font_size=16, color=BLUE)
        policy_math = MathTex(
            r"\mathcal{L}_{\pi} = -\sum \pi_{target} \cdot \log(\pi_{pred})",
            font_size=24,
        )
        policy_group = VGroup(policy_title, policy_math).arrange(
            DOWN, aligned_edge=LEFT
        )

        # 2. Value Loss (MSE scaled by 0.25)
        value_title = Text("2. Value Loss (Scaled MSE)", font_size=16, color=GREEN)
        value_math = MathTex(
            r"\mathcal{L}_{v} = 0.25 \cdot (v_{pred} - v_{target})^2", font_size=24
        )
        value_group = VGroup(value_title, value_math).arrange(DOWN, aligned_edge=LEFT)

        # 3. Reward Loss (MSE)
        reward_title = Text("3. Reward Loss (MSE)", font_size=16, color=RED)
        reward_math = MathTex(
            r"\mathcal{L}_{r} = (r_{pred} - r_{target})^2", font_size=24
        )
        reward_group = VGroup(reward_title, reward_math).arrange(
            DOWN, aligned_edge=LEFT
        )

        # Arrange all three horizontally
        equations_group = (
            VGroup(policy_group, value_group, reward_group)
            .arrange(RIGHT, buff=1.0)
            .move_to(UP * 0.5)
        )

        self.play(FadeIn(policy_group, shift=UP), run_time=0.8)
        self.play(FadeIn(value_group, shift=UP), run_time=0.8)
        self.play(FadeIn(reward_group, shift=UP), run_time=0.8)
        self.wait(1)

        # ==========================================
        # STEP 2: Backpropagation Through Time (BPTT) Summation
        # ==========================================

        bptt_text = Text(
            "Backpropagation Through Time (BPTT)", font_size=20, color=YELLOW
        ).next_to(equations_group, DOWN, buff=1.0)

        # The total loss equation summing over steps K
        total_loss_math = MathTex(
            r"\mathcal{L}_{total} = \sum_{t=0}^{K} \Big( \mathcal{L}_{\pi}^t + \mathcal{L}_{v}^t + \mathcal{L}_{r}^t \Big)",
            font_size=32,
        ).next_to(bptt_text, DOWN, buff=0.3)

        total_loss_box = SurroundingRectangle(
            total_loss_math, color=YELLOW, corner_radius=0.2
        )
        total_group = VGroup(bptt_text, total_loss_math, total_loss_box)

        self.play(FadeIn(bptt_text))
        self.play(Write(total_loss_math), Create(total_loss_box))
        self.wait(1)

        # ==========================================
        # STEP 3: The Discount Masking Detail
        # ==========================================

        # Highlight the special MuZero masking mechanic
        mask_text = Text(
            "* Future losses are masked out if the episode terminates early.",
            font_size=14,
            color=GRAY,
        ).next_to(total_loss_box, DOWN, buff=0.3)

        self.play(FadeIn(mask_text))
        self.wait(2)

        # ==========================================
        # STEP 4: The Final Update
        # ==========================================

        self.play(
            FadeOut(equations_group),
            total_group.animate.move_to(ORIGIN),
            FadeOut(mask_text),
        )

        update_text = Text(
            "Optimizer updates weights (optax.apply_updates)", font_size=24, color=GREEN
        ).next_to(total_group, DOWN, buff=0.5)

        self.play(Write(update_text))
        self.wait(2)

        # Clean ending
        final_text = Text(
            "Training Step Complete!", font_size=36, weight=BOLD, color=WHITE
        )
        self.play(FadeOut(Group(*self.mobjects)), FadeIn(final_text))
        self.wait(2)

    def create_simple_node(self, label_text, node_color, radius=0.45):
        """Creates a minimal, clean network node for the unrolling chain."""
        circle = Circle(color=node_color, fill_opacity=0.2, radius=radius)
        label = Text(label_text, font_size=16, weight=BOLD).move_to(circle.get_center())
        return VGroup(circle, label)
