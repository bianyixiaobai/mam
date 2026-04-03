import argparse
import json
import os

from pipeline.modality_selection import modality_selection
from pipeline.type_classification import type_classification
from pipeline.role_generation import generate_role
from pipeline.web_search_check import WebSearchCheck
from pipeline.meeting import roles_meeting
from pipeline.diagnosis import final_diagnosis
from pipeline.review import review_all
from pipeline.memory import memory


HISTORY_DIR = './history'
HISTORY_FILES = {
    'text':  os.path.join(HISTORY_DIR, 'text_history.json'),
    'image': os.path.join(HISTORY_DIR, 'image_history.json'),
    'video': os.path.join(HISTORY_DIR, 'video_history.json'),
    'audio': os.path.join(HISTORY_DIR, 'audio_history.json'),
}


def ensure_history_dir():
    os.makedirs(HISTORY_DIR, exist_ok=True)


def load_history(modality_type):
    """Load relevant history records as context string."""
    history_file = HISTORY_FILES.get(modality_type)
    if history_file and os.path.exists(history_file):
        with open(history_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if data:
            # Return last 3 records as reference context
            recent = data[-3:]
            items = []
            for item in recent:
                items.append(f"Q: {item.get('question', '')} A: {item.get('answer', '')}")
            return '\n'.join(items)
    return ''


def parse_type_result(type_result):
    """Normalize type_classification() return value to a single string."""
    if isinstance(type_result, tuple):
        # image returns (modality_type, body_part), e.g. ('CT', 'Lung')
        return ', '.join(str(t) for t in type_result)
    return str(type_result)


def run_pipeline(step_id, question, file_name):
    ensure_history_dir()

    modality_type = None
    type_name = None
    roles_generated = None
    search_result = None
    history_item = ''
    meeting_record = None
    diagnosis = None
    review_result = None

    print(f"\n========== MAM Inference Pipeline ==========")
    print(f"step_id   : {step_id}")
    print(f"question  : {question}")
    print(f"file_name : {file_name}")
    print(f"============================================\n")

    # ------------------------------------------------------------------
    # Step 1: Modality Selection
    # ------------------------------------------------------------------
    if step_id >= 1:
        print("[Step 1] Modality Selection...")
        modality_type = modality_selection(question, file_name)
        print(f"  => modality_type: {modality_type}")

    # ------------------------------------------------------------------
    # Step 2: Type Classification
    # ------------------------------------------------------------------
    if step_id >= 2:
        print("[Step 2] Type Classification...")
        type_result = type_classification(modality_type, question, file_name)
        type_name = parse_type_result(type_result)
        print(f"  => type_name: {type_name}")

    # ------------------------------------------------------------------
    # Step 3: Role Generation
    # ------------------------------------------------------------------
    if step_id >= 3:
        print("[Step 3] Role Generation...")
        roles_generated = generate_role(type_name, modality_type, question, file_name)
        print(f"  => roles_generated:\n{roles_generated}")

    # ------------------------------------------------------------------
    # Step 4: Web Search
    # ------------------------------------------------------------------
    if step_id >= 4:
        print("[Step 4] Web Search Check...")
        try:
            search_result = WebSearchCheck(question, file_name, modality_type)
            print(f"  => search_result:\n{search_result}")
        except Exception as e:
            print(f"  [Warning] Web search failed: {e}. Skipping.")
            search_result = ''

    # ------------------------------------------------------------------
    # Step 5: Load History
    # ------------------------------------------------------------------
    if step_id >= 5:
        print("[Step 5] Loading History...")
        history_item = load_history(modality_type)
        if search_result:
            history_item = f"Web search reference:\n{search_result}\n\nHistory reference:\n{history_item}"
        print(f"  => history_item (truncated): {history_item[:200]}")

    # ------------------------------------------------------------------
    # Step 6: Multi-Agent Meeting
    # ------------------------------------------------------------------
    if step_id >= 6:
        print("[Step 6] Multi-Agent Meeting...")
        meeting_record = roles_meeting(
            question, file_name, modality_type, type_name,
            roles_generated, history_item
        )
        print(f"\n  => meeting_record (truncated): {meeting_record[:300]}")

    # ------------------------------------------------------------------
    # Step 7: Final Diagnosis
    # ------------------------------------------------------------------
    if step_id >= 7:
        print("[Step 7] Final Diagnosis...")
        diagnosis = final_diagnosis(
            question, file_name, modality_type, type_name, meeting_record
        )
        print(f"  => diagnosis: {diagnosis}")

    # ------------------------------------------------------------------
    # Step 8: Review
    # ------------------------------------------------------------------
    if step_id >= 8:
        print("[Step 8] Review...")
        review_result = review_all(
            question, file_name, modality_type, type_name, diagnosis
        )
        print(f"  => review_result: {review_result}")

    # ------------------------------------------------------------------
    # Step 9: Save to Memory
    # ------------------------------------------------------------------
    if step_id >= 9:
        print("[Step 9] Saving to Memory...")
        memory(step_id, question, file_name, modality_type, diagnosis)
        print(f"  => Saved to {HISTORY_FILES.get(modality_type)}")

    print("\n========== Pipeline Complete ==========")
    print(f"Final Diagnosis:\n{diagnosis}")
    print(f"Review Result  : {review_result}")
    print("=======================================\n")

    return diagnosis


def main():
    parser = argparse.ArgumentParser(
        description='MAM: Modular Multi-Agent Framework Inference'
    )
    parser.add_argument('--step_id', type=int, default=9,
                        help='Run pipeline up to this step (1-9, default: 9 = full pipeline)')
    parser.add_argument('--question', type=str, required=True,
                        help='Medical question to answer')
    parser.add_argument('--file_name', type=str, default='',
                        help='Path to input file (image/audio/video); leave empty for text-only')
    args = parser.parse_args()

    run_pipeline(args.step_id, args.question, args.file_name)


if __name__ == '__main__':
    main()
