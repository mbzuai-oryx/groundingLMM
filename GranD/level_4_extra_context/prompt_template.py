import math
import json


def compute_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def euclidean_distance(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def correct_attribute(attribute):
    if ":" in attribute:
        attribute = ' '.join(attribute.split(':')[1:])
        attribute = attribute.strip()

    return attribute


def sample_object(objects, N=30):
    # Add all the objects with attributes
    sampled_objects, remaining_objects = [], []
    for obj in objects:
        if obj['attributes'] is not None:
            sampled_objects.append(obj)
        else:
            remaining_objects.append(obj)
    sampled_objects = sampled_objects + remaining_objects[:N - len(sampled_objects)]

    return sampled_objects


def get_simple_level_4_scene_graph(image_name, json_contents):
    scene_graph_content = {
        'objects': []
    }

    # Extract objects and landmark data for the given image_name from the json_contents
    objects = json_contents[image_name]['objects']
    landmark = json_contents[image_name]['landmark']

    # If no objects found, look for "floating_objects"
    if len(objects) == 0:
        objects = json_contents[image_name]['floating_objects']

    if len(objects) > 30:
        objects = sample_object(objects, N=30)

    # List objects
    for object in objects:
        scene_graph_content['objects'].append(object)

    # Construct the scene graph string representation
    scene_graph_string = "Objects:\n"
    for object in scene_graph_content['objects']:
        label = object.get('label', [])
        if label:
            labels = [label]
        else:
            labels = object.get('labels', [])
        attributes = object.get('attributes', [])
        if labels:
            labels = [labels[0]]
            scene_graph_string += f"{'/'.join(labels)}"
            if attributes:
                attributes = [attributes] if isinstance(attributes, str) else [attributes[0]]
                scene_graph_string += f"/{'/'.join(correct_attribute(attribute) for attribute in attributes)}"
            scene_graph_string += "\n"

    # Add landmark information
    scene_graph_string += f"\nLandmarks:\n{landmark['category']}, {landmark['fine_category']}\n"
    scene_graph_string = scene_graph_string.replace('_', " ")

    return scene_graph_string


LEVEL_2_PROCESSED_JSON_PATH = "in_context_examples"

image_name_1 = "sa_310343.jpg"
scene_graph_1 = (
    get_simple_level_4_scene_graph(image_name_1, json_contents=json.load(
        open(f"{LEVEL_2_PROCESSED_JSON_PATH}/{image_name_1[:-4]}.json", 'r'))))
caption_1 = ("Hot air balloons are big bags filled with hot air that carry people in a basket underneath. "
             "They started flying in France over 200 years ago. Flying in a hot air balloon lets people see cities and "
             "rivers from high up in the sky, giving a special view that they can't get from the ground. But it can be "
             "risky because of bad weather or if something breaks. These balloons are often seen during festivals, "
             "adding fun and bright colors to the celebration. It's a fun way for people to explore and enjoy a slow, "
             "peaceful trip in the sky.")

image_name_2 = "sa_302406.jpg"
scene_graph_2 = (
    get_simple_level_4_scene_graph(image_name_2, json_contents=json.load(
        open(f"{LEVEL_2_PROCESSED_JSON_PATH}/{image_name_2[:-4]}.json", 'r'))))
caption_2 = ("Red chairs are a popular choice for outdoor dining areas, as they add a vibrant touch and are generally "
             "made from materials that resist weather damage. Blackboards near restaurants often indicate a place where "
             "the menu is displayed, allowing for easy changes to the daily specials; they are usually made from a wooden "
             "frame holding a black slate. Awnings, often constructed with a metal frame and fabric covering, provide "
             "shade and protection from the weather, enhancing the comfort of diners in outdoor settings. However, "
             "outdoor dining can sometimes be disrupted by sudden changes in weather, and the close proximity to the "
             "street might expose diners to street noise and dust. Amid the urban setup, dining tables stand as central "
             "fixtures in outdoor settings, hosting a variety of activities including dining, socializing, or even "
             "working on a laptop in some cases, pointing to the versatile use of such spaces in modern urban landscapes.")


def get_prompt(scene_graph):
    prompt = f"""
##Example - 1:
Prompt: {scene_graph_1}
Additional context: {caption_1}

------
##Example - 2:
Prompt: {scene_graph_2}
Additional context: {caption_2}

------
Provide context based on the typical usage, history, potential dangers, and other interesting aspects surrounding the general theme presented by the objects and elements in the following scene graph:
{scene_graph} 

Limit the response to one paragraph with 5-7 sentences. 
DO NOT mention, refer to, or hint about "objects", "scene", or "scene graph". 
ONLY focus on explaining use cases, history, potential dangers, etc.
"""
    return prompt
