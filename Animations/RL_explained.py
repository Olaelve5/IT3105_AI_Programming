from manim import *


class RL_Explained(Scene):

    def construct(self):
        self.camera.background_color = "#0A0B0D"

        self.show_intro_title()
        self.show_rl_loop()

    def show_intro_title(self):
        text = Text(
            f"How does Reinforcement Learning work?",
            font_size=48,
            weight=BOLD,
            color=BLUE,
        )

        self.play(Write(text))
        self.wait(4)
        self.play(FadeOut(text))

    def show_rl_loop(self):
        # Title ------------------------->
        # title = Text("High Level Overview of RL", font_size=30, weight=BOLD)
        # title.to_edge(UP)

        # self.play(FadeIn(title))
        # self.wait(2)

        # Environment ------------------------->
        env_box = Rectangle(width=3, height=2, color=RED).shift(RIGHT * 3)
        env_label = Text("Environment", font_size=24).move_to(env_box)
        self.play(Create(env_box), Write(env_label))
        self.wait(2)

        # Agent ------------------------->
        agent_box = Rectangle(width=3, height=2, color=GREEN).shift(LEFT * 3)

        image = ImageMobject("resources/agent.png").scale(0.2)
        agent_text = Text("Agent", font_size=24)

        agent_group = Group(image, agent_text).arrange(RIGHT, buff=0.2)
        agent_group.move_to(agent_box.get_center())

        self.play(Create(agent_box), FadeIn(agent_group))
        self.wait(2)

        # Obs Arrow ------------------------->
        obs_path_points = [
            env_box.get_bottom() + DOWN * 0.2,
            env_box.get_bottom() + DOWN * 1,
            agent_box.get_bottom() + DOWN * 1,
            agent_box.get_bottom() + DOWN * 0.2,
        ]

        obs_path = VMobject()
        obs_path.set_points_as_corners(obs_path_points)
        obs_path.set_stroke(color=YELLOW, width=3)

        tip_segment = Arrow(
            start=obs_path_points[2], end=obs_path_points[3], color=YELLOW, buff=0
        )
        tip_segment.set_stroke(width=3)

        self.play(Create(obs_path), run_time=1.5)
        self.play(Create(tip_segment), run_time=0.5)
        self.wait(1)

        # Obs label
        obs_label = Text("Observation and Reward", font_size=18, color=WHITE).next_to(
            obs_path, DOWN
        )
        self.play(Write(obs_label), run_time=1)
        self.wait(2)

        # Action Arrow ------------------------->
        path_points = [
            agent_box.get_top() + UP * 0.2,
            agent_box.get_top() + UP * 1,
            env_box.get_top() + UP * 1,
            env_box.get_top() + UP * 0.2,
        ]

        action_path = VMobject()
        action_path.set_points_as_corners(path_points)
        action_path.set_stroke(color=BLUE, width=3)

        tip_segment = Arrow(
            start=path_points[2], end=path_points[3], color=BLUE, buff=0
        )
        tip_segment.set_stroke(width=3)

        self.play(Create(action_path), run_time=1.5)
        self.play(Create(tip_segment), run_time=0.5)
        self.wait(1)

        action_label = Text("Agent Action", font_size=18, color=WHITE).next_to(
            action_path, UP
        )
        self.play(Write(action_label), run_time=1)
        self.wait(2)

        # Transition Arrow ------------------------->
        transition_path = VMobject()
        transition_path.set_points_as_corners(
            [
                env_box.get_right() + UP * 0.5 + RIGHT * 0.2,
                env_box.get_right() + UP * 0.5 + RIGHT * 1,
                env_box.get_right() + DOWN * 0.5 + RIGHT * 1,
                env_box.get_right() + DOWN * 0.5 + RIGHT * 0.2,
            ]
        )

        transition_path.set_stroke(color=RED, width=3)
        tip_segment = Arrow(
            start=transition_path.get_points()[-2],
            end=transition_path.get_points()[-1],
            color=RED,
            buff=0,
        )
        tip_segment.set_stroke(width=3)

        self.play(Create(transition_path), run_time=1.5)
        self.play(Create(tip_segment), run_time=0.5)
        self.wait(1)

        transition_label = Text("Transition", font_size=18, color=WHITE).next_to(
            transition_path, RIGHT
        )
        self.play(Write(transition_label), run_time=1)
        self.wait(2)

        # Agent Learning Arrow ------------------------->
        learning_path = VMobject()
        learning_path.set_points_as_corners(
            [
                agent_box.get_left() + UP * 0.5 + LEFT * 0.2,
                agent_box.get_left() + UP * 0.5 + LEFT * 1,
                agent_box.get_left() + DOWN * 0.5 + LEFT * 1,
                agent_box.get_left() + DOWN * 0.5 + LEFT * 0.2,
            ]
        )

        learning_path.set_stroke(color=GREEN, width=3)
        tip_segment = Arrow(
            start=learning_path.get_points()[-2],
            end=learning_path.get_points()[-1],
            color=GREEN,
            buff=0,
        )
        tip_segment.set_stroke(width=3)

        self.play(Create(learning_path), run_time=1.5)
        self.play(Create(tip_segment), run_time=0.5)
        self.wait(1)

        learning_label = Text("Improve", font_size=18, color=WHITE).next_to(
            learning_path, LEFT
        )
        self.play(Write(learning_label), run_time=1)
        self.wait(2)
