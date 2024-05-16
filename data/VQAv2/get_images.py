import json
import os
import shutil

# Define file paths
questions_file_path = '/home/adamchun/cs231n-project/data/VQAv2/v2_OpenEnded_mscoco_val2014_questions.json'
annotations_file_path = '/home/adamchun/cs231n-project/data/VQAv2/v2_mscoco_val2014_annotations.json'
images_dir = '/home/adamchun/cs231n-project/data/VQAv2/val2014'
output_images_dir = '/home/adamchun/cs231n-project/data/VQAv2/selected_images'
combined_mapping_file_path = '/home/adamchun/cs231n-project/data/VQAv2/combined_mapping.json'

# Create output directory if it doesn't exist
os.makedirs(output_images_dir, exist_ok=True)

# Load the JSON data
with open(questions_file_path, 'r') as f:
    questions_data = json.load(f)

with open(annotations_file_path, 'r') as f:
    annotations_data = json.load(f)

# Ensure that 'questions' and 'annotations' keys exist
if 'questions' in questions_data and 'annotations' in annotations_data:
    questions = questions_data['questions']
    annotations = annotations_data['annotations']

    # Extract image paths and corresponding questions for questions that start with "Is there"
    entries = [{'image_path': os.path.join(images_dir, f"COCO_val2014_{str(entry['image_id']).zfill(12)}.jpg"),
                'question': entry['question'], 'question_id': entry['question_id']}
               for entry in questions if entry.get('question', '').lower().startswith("is there")]

    # Remove duplicates and limit to 1800 entries
    unique_entries = list({entry['question_id']: entry for entry in entries}.values())[:1800]

    # Create a mapping of question_id to correct answers
    question_id_to_answer = {entry['question_id']: entry['multiple_choice_answer'] for entry in annotations}

    # Combine frame_to_entry and question_id_to_answer mappings
    combined_mapping = {}
    for frame_number, entry in enumerate(unique_entries):
        image_path = entry['image_path']
        correct_answer = question_id_to_answer.get(entry['question_id'])
        if correct_answer:  # Ensure correct_answer is not empty
            dest_path = os.path.join(output_images_dir, os.path.basename(image_path))
            if os.path.exists(image_path):
                shutil.copy(image_path, dest_path)
                combined_mapping[frame_number] = {
                    'image_path': image_path,
                    'question': entry['question'],
                    'question_id': entry['question_id'],
                    'correct_answer': correct_answer
                }
            else:
                print(f"Image {image_path} does not exist")
        else:
            raise Exception(f"Correct answer not found for question_id {entry['question_id']}")

    # Save the combined mapping to a JSON file
    with open(combined_mapping_file_path, 'w') as f:
        json.dump(combined_mapping, f, indent=2)
else:
    print("'questions' or 'annotations' key not found in the JSON files")



