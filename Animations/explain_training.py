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
        # We use an invisible rectangle just to anchor our coordinates, no border!
        game_area = (
            Rectangle(width=5.5, height=5.5).to_edge(LEFT, buff=0.8).shift(DOWN * 0.5)
        )

        game_text = Text("Gameplay Overlay", font_size=20, color=GRAY).move_to(
            game_area
        )
        source_label = Text("Data Source", font_size=16, color=TEAL).next_to(
            game_area, UP, buff=0.1
        )

        self.play(Write(game_text), FadeIn(source_label))
        self.wait(1)

        # --- Helper: Create small labeled "Data Packets" ---
        def create_data_packet(label_text, color=ORANGE):
            circle = Circle(radius=0.15, color=color, fill_opacity=0.8)
            label = Text(label_text, font_size=14, color=WHITE, weight=BOLD).move_to(
                circle
            )
            return VGroup(circle, label)

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
        self.wait(2)

        # Spawning example packets on the left
        pi_packet = create_data_packet("π").move_to(
            game_area.get_center() + UP * 0.5 + RIGHT * 0.5
        )
        v_packet = create_data_packet("v").move_to(
            game_area.get_center() + UP * 0.5 + LEFT * 0.5
        )
        u_packet = create_data_packet("u", color=TEAL).move_to(
            game_area.get_center() + DOWN * 0.5
        )

        data_group = VGroup(pi_packet, v_packet, u_packet)

        self.play(FadeIn(data_group, scale=0.5))
        self.wait(1.5)

        # --- RIGHT SIDE: Stylized Replay Buffer ---
        buffer_title = (
            Text("Replay Buffer", font_size=24, color=YELLOW)
            .to_edge(UP)
            .shift(RIGHT * 3.5)
        )

        # A container representing the stacked experience
        buffer_width = 3.5
        buffer_height = 4.0
        buffer_bottom = game_area.get_bottom()[1]  # Align bottoms visually
        buffer_pos = RIGHT * 3.5 + UP * (buffer_bottom + buffer_height / 2)

        buffer_container = Rectangle(
            width=buffer_width, height=buffer_height, color=GRAY_A, stroke_width=2
        ).move_to(buffer_pos)

        # Visualizing the stack/wrap-around nature (we can add some divider lines)
        divider1 = Line(
            start=buffer_container.get_left() + UP * 1.0,
            end=buffer_container.get_right() + UP * 1.0,
            color=GRAY_A,
            stroke_opacity=0.3,
        )
        divider2 = divider1.copy().shift(DOWN * 1.0)
        divider3 = divider1.copy().shift(DOWN * 2.0)
        buffer_lines = VGroup(divider1, divider2, divider3)

        # --- Capacity Indicator ---
        self.capacity_tracker = ValueTracker(0)

        cap_label_text = Text("Stored Steps:", font_size=18).next_to(
            buffer_container, DOWN, buff=0.3
        )
        cap_num = (
            Integer(0, font_size=22, color=GREEN)
            .add_updater(lambda m: m.set_value(self.capacity_tracker.get_value()))
            .next_to(cap_label_text, RIGHT, buff=0.2)
        )

        cap_group = VGroup(cap_label_text, cap_num)

        total_cap_text = Text(
            "(Capacity: 25,000 steps)", font_size=14, color=GRAY
        ).next_to(cap_group, DOWN, buff=0.15)

        self.play(
            ReplacementTransform(
                VGroup(self.phase1_title, concept_text1, concept_text2), buffer_title
            ),
            FadeIn(VGroup(buffer_container, buffer_lines)),
            FadeIn(VGroup(cap_group, total_cap_text), shift=UP),
        )
        self.wait(1)

        # --- The Data Flow Animation ---
        # Move the first data group into the buffer
        self.play(
            data_group.animate.scale(0.8).move_to(
                buffer_container.get_bottom() + UP * 0.5
            ),
            self.capacity_tracker.animate.set_value(1),
            run_time=1.5,
        )
        self.wait(0.5)

        # Spawning more generic, rapid packets to simulate "Fast-forward self-play"
        # FIXED: Added a Z-axis of 0.0 to the random array to fix the broadcast error!
        rapid_packets = VGroup(
            *[
                create_data_packet(
                    "Data"
                    if np.random.rand() > 0.3
                    else ("u" if np.random.rand() > 0.5 else "v")
                )
                .scale(0.7)
                .move_to(
                    game_area.get_center()
                    + np.array([np.random.randn(), np.random.randn(), 0.0]) * 1.0
                )
                for _ in range(12)
            ]
        )

        # Sequence of flow and accumulation
        animations = []
        for i, packet in enumerate(rapid_packets):
            # Target positions in the buffer stack
            stack_y_offset = (i // 3) * 0.6 + 1.2  # stack them in rows
            stack_x_offset = (i % 3) * 0.7 - 0.7

            target_pos = (
                buffer_container.get_bottom()
                + UP * stack_y_offset
                + RIGHT * stack_x_offset
            )

            animations.append(
                # Path animation: Spawns, travels across, gets added to buffer VGroup
                packet.animate(
                    rate_func=squish_rate_func(
                        linear, 0.0 + i * 0.05, 0.6 + i * 0.05
                    )  # Staggered flow
                )
                .scale(0.8)
                .move_to(target_pos)
            )

        # The fast-play flow animation
        flow_text = Text("Filling buffer...", font_size=18, color=TEAL).next_to(
            buffer_container, UP, buff=0.2
        )

        self.play(
            LaggedStart(*animations, lag_ratio=0.15),
            self.capacity_tracker.animate.set_value(25000),  # Counter goes high!
            Write(flow_text),
            run_time=4,
        )
        self.wait(2)

        # Cleanup and prepare for Phase 2 transition
        self.play(
            FadeOut(
                VGroup(game_text, source_label, rapid_packets, data_group, flow_text)
            ),
            buffer_container.animate.scale(0.6).to_corner(UL).shift(DOWN * 1.5),
            buffer_lines.animate.scale(0.6).to_corner(UL).shift(DOWN * 1.5),
            cap_group.animate.scale(0.6).next_to(buffer_container, RIGHT, buff=0.2),
            total_cap_text.animate.scale(0.6).next_to(cap_group, DOWN, buff=0.15),
            buffer_title.animate.set_color(GRAY)
            .scale(0.8)
            .next_to(buffer_container, UP, buff=0.2),
        )

    def explain_sampling(self):
        """Phase 2: Pulling a sequence from the buffer."""
        # TODO: Show a batch being extracted from the Replay Buffer
        # containing an observation and a sequence of subsequent actions.
        pass

    def explain_unrolling(self):
        """Phase 3: Passing the data through the networks."""
        # TODO: Show the Representation Network processing the initial state,
        # followed by the Dynamics Network unrolling step-by-step using the sampled actions.
        pass

    def explain_loss_and_update(self):
        """Phase 4: Comparing predictions to targets."""
        # TODO: Visually compare the Network outputs (p, v, r) to the
        # MCTS/Environment targets (pi, z, u) to calculate the loss.
        pass
