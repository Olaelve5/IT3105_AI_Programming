from manim import *
from config import BACKGROUND_COLOR


class MuZero_vs_AlphaZero(Scene):
    def construct(self):
        self.camera.background_color = BACKGROUND_COLOR

        az_title = Text("AlphaZero:", font_size=32, weight=BOLD, color=BLUE)
        az_sub = Text("Knows the rules of the game", font_size=22)
        az_header = VGroup(az_title, az_sub).arrange(RIGHT, buff=0.3, aligned_edge=DOWN)
        az_sub.shift(0.03 * UP)

        az_graph = self.get_horizontal_flow(
            "Perfect Simulator\n(The Rules)", color=BLUE
        )

        az_section = VGroup(az_header, az_graph).arrange(
            DOWN, buff=0.7, aligned_edge=LEFT
        )
        az_section.set_x(0).set_y(2)

        mz_title = Text("MuZero:", font_size=32, weight=BOLD, color=RED)
        mz_sub = Text("Learns the rules of the game", font_size=22)
        mz_header = VGroup(mz_title, mz_sub).arrange(RIGHT, buff=0.3, aligned_edge=DOWN)
        mz_sub.shift(0.05 * DOWN)

        mz_graph = self.get_horizontal_flow(
            "Learned Model\n(Abstract State)", color=RED
        )

        mz_section = VGroup(mz_header, mz_graph).arrange(
            DOWN, buff=0.7, aligned_edge=LEFT
        )
        mz_section.set_x(0).set_y(-1.5)

        self.play(Write(az_header), run_time=1.5)
        self.wait(3)

        self.animate_flow(az_graph)
        self.wait(5)

        self.play(Write(mz_header), run_time=1.5)
        self.wait(3)

        self.animate_flow(mz_graph)
        self.wait(8)

        self.play(FadeOut(az_section), FadeOut(mz_section), run_time=2)
        self.wait(1)

    def get_horizontal_flow(self, sim_text, color):
        input_node = Text("State + Action", font_size=18)

        sim_box = RoundedRectangle(
            height=0.9, width=2.8, corner_radius=0.1, color=color
        )
        sim_label = Text(sim_text, font_size=16, color=color, line_spacing=0.8).move_to(
            sim_box.get_center()
        )
        sim_node = VGroup(sim_box, sim_label)

        output_node = Text("Next State", font_size=18)

        flow_elements = VGroup(input_node, sim_node, output_node).arrange(
            RIGHT, buff=1.5
        )

        a1 = Arrow(
            input_node.get_right(),
            sim_box.get_left(),
            buff=0.2,
            color=WHITE,
            tip_length=0.15,
            stroke_width=3,
        )
        a2 = Arrow(
            sim_box.get_right(),
            output_node.get_left(),
            buff=0.2,
            color=WHITE,
            tip_length=0.15,
            stroke_width=3,
        )

        return VGroup(input_node, a1, sim_node, a2, output_node)

    def animate_flow(self, graph_group):
        self.play(
            Succession(
                FadeIn(graph_group[0]),
                Create(graph_group[1]),
                FadeIn(graph_group[2]),
                Create(graph_group[3]),
                FadeIn(graph_group[4]),
            ),
            run_time=4,
        )
