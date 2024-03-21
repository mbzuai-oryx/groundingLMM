import json


def get_scene_graph(image_name, json_contents):
    gpt_prompt_content = {"objects_and_attributes": [], 'relationships': []}
    objects = json_contents[image_name]['objects']
    relations = json_contents[image_name]['relationships']
    landmark = json_contents[image_name]['landmark']

    object_id_to_labels = {}
    for object in objects:
        object_id_to_labels[object['id']] = object['labels']
        gpt_prompt_content['objects_and_attributes'].append({f"obj_{object['id']}": object['labels'],
                                                             'attribute': object['attributes'],
                                                             'depth': object['depth'], 'bbox': object['bbox']})
    for caption in relations.keys():
        relation = {}
        for grounding in relations[caption]['grounding']:
            labels = []
            for id in grounding['object_ids']:
                if id in object_id_to_labels.keys():
                    labels.append([f"obj_{id}"] + object_id_to_labels[id])
            if len(labels) > 0:
                relation[caption] = {}
                relation[caption][grounding['phrase']] = list(labels)
        if relation:
            gpt_prompt_content['relationships'].append(relation)

    scene_graph_string = ""
    for key in gpt_prompt_content.keys():
        scene_graph_string += f"{key} = [\n"
        for item in gpt_prompt_content[key]:
            scene_graph_string += json.dumps(item) + "\n"
        scene_graph_string += "]\n"
    scene_graph_string += f"landmark = [{landmark['category'], landmark['fine_category']}]\n"

    return scene_graph_string


import math


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


def sample_object(objects, N=100):
    # Add all the objects with attributes
    sampled_objects, remaining_objects = [], []
    for obj in objects:
        if obj['attributes'] is not None:
            sampled_objects.append(obj)
        else:
            remaining_objects.append(obj)
    sampled_objects = sampled_objects + remaining_objects[:N - len(sampled_objects)]

    return sampled_objects


def get_simple_scene_graph(image_name, json_contents, proximity_threshold=200):
    scene_graph_content = {
        'objects_by_depth': {
            'Immediate Foreground': [],
            'Foreground': [],
            'Midground': [],
            'Background': []
        },
        'relationships': []
    }

    objects = json_contents[image_name]['objects']
    if len(objects) == 0:
        objects = json_contents[image_name]['floating_objects']
    elif len(objects) > 100:
        # Consider maximum 100 objects for constructing scene graph to avoid context len error
        objects = sample_object(objects)
    relations = json_contents[image_name]['relationships']
    short_captions = json_contents[image_name]['captions']
    landmark = json_contents[image_name]['landmark']

    # Remove objects with no depth information
    objects = [obj for obj in objects if "depth" in obj.keys()]

    # Determine the min and max depth for normalization
    min_depth = min([obj['depth'] for obj in objects])
    max_depth = max([obj['depth'] for obj in objects])

    # Preliminary group by depth
    depth_objects = {
        'Immediate Foreground': [],
        'Foreground': [],
        'Midground': [],
        'Background': []
    }

    for object in objects:
        normalized_depth = 1 - (
                (object['depth'] - min_depth) / (max_depth - min_depth)) if max_depth != min_depth else 0.5

        if normalized_depth < 0.1:
            depth_objects['Immediate Foreground'].append(object)
        elif normalized_depth < 0.6:
            depth_objects['Foreground'].append(object)
        elif normalized_depth < 0.85:
            depth_objects['Midground'].append(object)
        else:
            depth_objects['Background'].append(object)

    # Group objects based on distance between bboxes within each depth category and assign group IDs
    group_id = 0
    for category, objs in depth_objects.items():
        while objs:
            current_obj = objs.pop(0)
            current_obj['group_id'] = group_id
            scene_graph_content['objects_by_depth'][category].append(current_obj)

            current_center = compute_center(current_obj['bbox'])
            for obj in list(objs):
                obj_center = compute_center(obj['bbox'])
                if euclidean_distance(current_center, obj_center) < proximity_threshold:
                    obj['group_id'] = group_id
                    scene_graph_content['objects_by_depth'][category].append(obj)
                    objs.remove(obj)
            group_id += 1

    group_id_to_objects = {}
    object_id_to_group_id = {}
    for depth_category in scene_graph_content['objects_by_depth'].keys():
        group_id_to_objects[depth_category] = {}
        for object in scene_graph_content['objects_by_depth'][depth_category]:
            group_id = object['group_id']
            object_id = object['id']
            object_id_to_group_id[object_id] = group_id
            if not group_id in group_id_to_objects[depth_category].keys():
                group_id_to_objects[depth_category][group_id] = []

            group_id_to_objects[depth_category][group_id].append(object)

    for grounding in relations['grounding']:
        obj_ids = grounding['object_ids']
        if obj_ids:
            scene_graph_content['relationships'].append({
                'phrase': grounding['phrase'],
                'object_ids': obj_ids,
                'group_ids': [object_id_to_group_id[id] if id in object_id_to_group_id.keys() else -1 for id in
                              obj_ids]
            })

    scene_graph_content['relationships'] = []
    for grounding in relations['grounding']:
        obj_ids = grounding['object_ids']
        if obj_ids:
            scene_graph_content['relationships'].append({
                'phrase': grounding['phrase'],
                'object_ids': obj_ids,
                'group_ids': [object_id_to_group_id[id] if id in object_id_to_group_id.keys() else -1 for id in
                              obj_ids]
            })

    scene_graph_string = ""
    for depth_cat in group_id_to_objects.keys():
        if len(group_id_to_objects[depth_cat]) > 0:
            scene_graph_string += f"{depth_cat}:\n"
            for group_id in group_id_to_objects[depth_cat].keys():
                scene_graph_string += f"Group {group_id}: "
                for i, object in enumerate(group_id_to_objects[depth_cat][group_id]):
                    label = object.get('label', [])
                    if label:
                        labels = [label]
                    else:
                        labels = object.get('labels', [])
                    attributes = object.get('attributes', [])
                    if labels:
                        labels = [labels[0]]
                        if len(labels) == 1:
                            if i == 0:
                                scene_graph_string += f"{labels[0]}"
                            else:
                                scene_graph_string += f", {labels[0]}"
                        else:
                            if i == 0:
                                scene_graph_string += f"{'/'.join(labels)}"
                            else:
                                scene_graph_string += f", {'/'.join(labels)}"
                    if attributes:
                        attributes = [attributes] if isinstance(attributes, str) else [attributes[0]]
                        scene_graph_string += f"/{'/'.join(correct_attribute(attribute) for attribute in attributes)}"

                scene_graph_string += f"\n"

    scene_graph_string += "\nShort Captions:\n"
    for caption in short_captions:
        scene_graph_string += f"{caption}\n"

    scene_graph_string += "\nRelationships:\n"
    for i, relation in enumerate(scene_graph_content['relationships']):
        group_ids_str = ', '.join([f"{gid}" for gid in relation['group_ids'] if gid != -1])
        scene_graph_string += f"{relation['phrase']}, Group Ids: ({group_ids_str})\n"

    scene_graph_string += f"\nLandmarks:\n{landmark['category']}, {landmark['fine_category']}\n"
    scene_graph_string = scene_graph_string.replace('_', " ")

    return scene_graph_string
