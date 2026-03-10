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
        # Environment ------------------------->
        env_box = Rectangle(width=3, height=2, color=RED).shift(RIGHT * 3)
        env_label = Text("Environment", font_size=24).move_to(env_box)

        # Agent ------------------------->
        agent_box = Rectangle(width=3, height=2, color=GREEN).shift(LEFT * 3)

        image = ImageMobject("resources/agent.png").scale(0.2)
        agent_text = Text("Agent", font_size=24)

        agent_group = Group(image, agent_text).arrange(RIGHT, buff=0.2)
        agent_group.move_to(agent_box.get_center())

        self.play(Create(agent_box), FadeIn(agent_group))
        self.wait(8)

        self.play(Create(env_box), Write(env_label))
        self.wait(8)

        obs_path = self.show_observation_arrow(agent_box, env_box)
        action_path = self.show_action_arrow(agent_box, env_box)
        transition_path = self.show_transition_arrow(env_box)
        reward_path = self.show_reward_arrow(agent_box, env_box)
        learning_path = self.show_learning_arrow(agent_box)

        paths = [obs_path, action_path, transition_path, reward_path, learning_path]
        for path in paths:
            self.animate_dot_along_path(path, color=path.get_stroke_color(), run_time=2)
            self.wait(1)

        for _ in range(5):
            for path in paths:
                self.animate_dot_along_path(
                    path, color=path.get_stroke_color(), run_time=0.6
                )

    def show_observation_arrow(self, agent_box, env_box):
        obs_path_points = [
            env_box.get_bottom() + DOWN * 0.2,
            env_box.get_bottom() + DOWN * 1,
            agent_box.get_bottom() + DOWN * 1,
            agent_box.get_bottom() + DOWN * 0.3,
        ]

        obs_path = VMobject()
        obs_path.set_points_as_corners(obs_path_points)
        obs_path.set_stroke(color=YELLOW, width=3)

        tip_segment = Arrow(
            start=obs_path_points[2],
            end=obs_path_points[3],
            color=YELLOW,
            buff=0,
        )
        tip_segment.set_stroke(width=3)

        self.play(Create(obs_path), run_time=1.5)
        self.play(Create(tip_segment), run_time=0.5)
        self.wait(1)

        obs_label = Text("State Observation", font_size=18, color=WHITE).next_to(
            obs_path, DOWN
        )
        self.play(Write(obs_label), run_time=1)
        self.wait(2)

        return obs_path

    def show_action_arrow(self, agent_box, env_box):
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
            start=path_points[2],
            end=path_points[3],
            color=BLUE,
            buff=0,
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

        return action_path

    def show_reward_arrow(self, agent_box, env_box):
        points = [
            env_box.get_left() + LEFT * 0.2,
            agent_box.get_right() + RIGHT * 0.2,
        ]

        reward_path = VMobject()
        reward_path.set_points_as_corners(points)
        reward_path.set_stroke(color=ORANGE, width=3)

        tip_segment = Arrow(
            start=agent_box.get_right() + RIGHT * 0.9,
            end=agent_box.get_right() + RIGHT * 0.2,
            color=ORANGE,
            buff=0,
            tip_length=0.2,
        )
        tip_segment.set_stroke(width=3)

        self.play(Create(reward_path), run_time=1.2)
        self.play(Create(tip_segment), run_time=0.4)
        self.wait(1)

        reward_label = Text("Reward", font_size=18, color=WHITE).next_to(
            reward_path, DOWN
        )
        self.play(Write(reward_label), run_time=1)
        self.wait(2)

        return reward_path

    def show_transition_arrow(self, env_box):
        points = [
            env_box.get_right() + UP * 0.5 + RIGHT * 0.2,
            env_box.get_right() + UP * 0.5 + RIGHT * 1,
            env_box.get_right() + DOWN * 0.5 + RIGHT * 1,
            env_box.get_right() + DOWN * 0.5 + RIGHT * 0.2,
        ]

        transition_path = VMobject()
        transition_path.set_points_as_corners(points)
        transition_path.set_stroke(color=RED, width=3)

        tip_segment = Arrow(
            start=points[2],
            end=points[3],
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

        return transition_path

    def show_learning_arrow(self, agent_box):
        points = [
            agent_box.get_left() + UP * 0.5 + LEFT * 0.2,
            agent_box.get_left() + UP * 0.5 + LEFT * 1,
            agent_box.get_left() + DOWN * 0.5 + LEFT * 1,
            agent_box.get_left() + DOWN * 0.5 + LEFT * 0.2,
        ]

        learning_path = VMobject()
        learning_path.set_points_as_corners(points)
        learning_path.set_stroke(color=GREEN, width=3)

        tip_segment = Arrow(
            start=points[2],
            end=points[3],
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

        return learning_path

    def animate_dot_along_path(self, path, color=WHITE, dot_radius=0.12, run_time=1.5):
        short_path = path.copy().pointwise_become_partial(path, 0, 0.92)

        moving_dot = Dot(color=color, radius=dot_radius)
        moving_dot.move_to(short_path.get_start())
        self.add(moving_dot)

        self.play(MoveAlongPath(moving_dot, short_path), run_time=run_time)
        self.play(
            moving_dot.animate.scale(0.1).set_opacity(0),
            run_time=0.2,
            rate_func=linear,
        )
        self.remove(moving_dot)
