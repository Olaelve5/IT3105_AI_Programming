from manim import *

class MuZeroExplained(Scene):
    def construct(self):
        """
        This is the main timeline of your video. 
        Everything happens in the exact order you write it here.
        """
        
        # --- SECTION 1: Introduction ---
        self.show_intro()
        
        # --- SECTION 2: The Representation Network ---
        self.show_representation()
        
        # --- SECTION 3: The Dynamics & Prediction Networks ---
        self.show_unrolling()
        
        # --- SECTION 4: MCTS ---
        self.show_mcts()

    # ==========================================
    # Helper Methods (Where you build the visuals)
    # ==========================================

    def show_intro(self):
        # 1. Create a Mobject (Text)
        title = Text("How MuZero Thinks", font_size=48, weight=BOLD)
        subtitle = Text("Without knowing the rules", font_size=32, color=BLUE)
        
        # Position the subtitle below the title
        subtitle.next_to(title, DOWN)

        # Group them together so we can animate them as one unit
        intro_group = VGroup(title, subtitle)

        # 2. Animate them onto the screen
        self.play(Write(intro_group))
        self.wait(2) # Pause for 2 seconds so the viewer can read it

        # 3. Clear the screen for the next scene
        self.play(FadeOut(intro_group))


    def show_representation(self):
        # Let's draw the Representation function: h = R(o)
        
        # Create the visual elements
        obs_box = Rectangle(width=2, height=2, color=WHITE).shift(LEFT * 4)
        obs_label = Text("Observation\n(Board)", font_size=24).move_to(obs_box)
        
        network_circle = Circle(radius=1, color=PURPLE).shift(LEFT * 1)
        network_label = Text("Rep Net", font_size=24).move_to(network_circle)
        
        hidden_box = Rectangle(width=1.5, height=1.5, color=GREEN).shift(RIGHT * 2)
        # MathTex is how you write beautiful LaTeX math in Manim!
        hidden_label = Text("s_0", font_size=24).move_to(hidden_box)

        # Create arrows pointing between them
        arrow1 = Arrow(start=obs_box.get_right(), end=network_circle.get_left())
        arrow2 = Arrow(start=network_circle.get_right(), end=hidden_box.get_left())

        # Animate step-by-step
        self.play(Create(obs_box), Write(obs_label))
        self.wait(1)
        self.play(GrowArrow(arrow1), Create(network_circle), Write(network_label))
        self.wait(1)
        self.play(GrowArrow(arrow2), FadeIn(hidden_box), Write(hidden_label))
        self.wait(2)
        
        # Clear scene
        self.play(FadeOut(VGroup(obs_box, obs_label, network_circle, network_label, hidden_box, hidden_label, arrow1, arrow2)))

    def show_unrolling(self):
        title = Text("The Dynamics Network (Unrolling)", font_size=36).to_edge(UP)
        self.play(FadeIn(title))

        # TODO: This is where you can practice!
        # Try creating a starting hidden state (s_0), an Action arrow (a), 
        # and a new hidden state (s_1) using the concepts from the last method.
        
        instruction = Text("Your turn to code the Dynamics unroll here!", font_size=24, color=YELLOW)
        self.play(Write(instruction))
        self.wait(2)
        
        self.play(FadeOut(title), FadeOut(instruction))

    def show_mcts(self):
        # A quick example of how to draw a tree structure
        root = Circle(radius=0.4, color=GREEN).shift(UP * 2)
        root_text = Text("s_0", font_size=24).move_to(root)
        
        child_left = Circle(radius=0.4, color=BLUE).shift(DOWN * 1 + LEFT * 2)
        child_right = Circle(radius=0.4, color=BLUE).shift(DOWN * 1 + RIGHT * 2)
        
        edge_left = Line(root.get_bottom(), child_left.get_top())
        edge_right = Line(root.get_bottom(), child_right.get_top())

        self.play(Create(root), Write(root_text))
        self.play(Create(edge_left), Create(child_left))
        self.play(Create(edge_right), Create(child_right))
        
        self.wait(3)
        self.play(FadeOut(Group(*self.mobjects))) # Clears literally everything on screen