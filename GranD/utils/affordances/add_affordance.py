def convert_label_to_category(categories):
    # Create a dictionary mapping each label to its category
    label_to_category = {}
    for category in categories:
        for label in category[2]:
            if label not in label_to_category:
                label_to_category[label] = []
            label_to_category[label].append({
                'category': category[0],
                'description': category[1]
            })
    return label_to_category


def get_affordances(image_dict, label_to_category):
    # Initialize the affordances dictionary
    affordances_dict = {}

    for obj_list_name in ['objects', 'floating_objects']:
        if obj_list_name in image_dict:
            for obj in image_dict[obj_list_name]:
                for label in obj['labels']:
                    if label in label_to_category:
                        for category in label_to_category[label]:
                            key = category['category']
                            if key not in affordances_dict:
                                affordances_dict[key] = {
                                    'description': category['description'],
                                    'object_ids': [obj['id']],
                                    'labels': [label]
                                }
                            else:
                                if obj['id'] not in affordances_dict[key]['object_ids']:
                                    affordances_dict[key]['object_ids'].append(obj['id'])
                                if label not in affordances_dict[key]['labels']:
                                    affordances_dict[key]['labels'].append(label)

    image_dict['affordances'] = affordances_dict
    return image_dict
