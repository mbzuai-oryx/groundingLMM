import json
from prompt_template import get_simple_scene_graph

LEVEL_2_PROCESSED_JSON_PATH = "in_context_examples"

image_name_1 = "sa_310343.jpg"
scene_graph_1 = (
    get_simple_scene_graph(image_name_1, json_contents=json.load(
        open(f"{LEVEL_2_PROCESSED_JSON_PATH}/{image_name_1[:-4]}.json", 'r'))))
dense_caption_1 = ("A hot air balloon flies high above a city over a river in a beautiful urban landscape. The balloon "
                   "is multi-colored. It is shaped like a beach umbrella. The sky is a blend of blue and white.")

image_name_2 = "sa_302406.jpg"
scene_graph_2 = (
    get_simple_scene_graph(image_name_2, json_contents=json.load(
        open(f"{LEVEL_2_PROCESSED_JSON_PATH}/{image_name_2[:-4]}.json", 'r'))))
dense_caption_2 = ("Dining tables with red chairs line a narrow busy european street in front of colorful buildings. "
                   "On the tables, wine glasses, cell phones, and napkins are set out. Nearby, there is a blackboard. "
                   "In the midground, a group of people walk down the street, including a girl in a skirt, a man in a "
                   "blue shirt, and another man in brown pants. A signboard is also in view. In the background, multiple"
                   " awnings are visible. There is a group of people present, one of whom is wearing a backpack.")

string = f"""
The provided prompt is a scene graph, which is a structured representation of a scene detailing its various elements and their relationships. The scene graph consists of: 
1. Layers of Depth: The scene is divided into different layers based on proximity - 'Immediate Foreground', 'Foreground', 'Midground', and 'Background'. Each layer depicts objects or entities at that depth in the scene.
2. Groups: Within each depth layer, objects and entities are clustered into groups, sometimes with specific attributes.
3. Relationships: This section illustrates the interactions or spatial relationships between various objects or groups.
4. Landmarks: It gives a broader view or categorization of the scene, defining its overarching theme or environment.
"""


def get_level_3_chat_prompt(scene_graph):
    prompt = f"""
The provided prompt is a scene graph, which is a structured representation of a scene detailing its various elements and their relationships. The scene graph consists of: 
1. Layers of Depth: The scene is divided into different layers based on proximity - 'Immediate Foreground', 'Foreground', 'Midground', and 'Background'. Each layer depicts objects or entities at that depth in the scene.
2. Groups: Within each depth layer, objects and entities are clustered into groups, sometimes with specific attributes.
3. Relationships: This section illustrates the interactions or spatial relationships between various objects or groups.
4. Landmarks: It gives a broader view or categorization of the scene, defining its overarching theme or environment.
---

##Example - 1:
Prompt: {scene_graph_1}
Desired caption: {dense_caption_1}
------
##Example - 2:
Prompt: {scene_graph_2}
Desired caption: {dense_caption_2}
------
Please provide a simple and straightforward 2-4 sentence image caption based on the following scene graph details:
{scene_graph}. 
Create the caption as if you are directly observing the image. Do not mention the use of any source data like 'The relationship indicates ...' or 'No relations specified'. Provide only the caption without any other explanatory text.
"""

    return prompt
