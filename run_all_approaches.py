"""
Master script to run all three approaches and compare results
"""
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data_loader import DroneDataset, annotations_to_dict_format, save_predictions
from src.evaluation.st_iou import evaluate_dataset
from src.evaluation.compare import ResultsComparator
from src.traditional.pipeline import TraditionalCVPipeline
from src.models.pipeline import DeepLearningPipeline
from src.hybrid.pipeline import HybridPipeline
from src.config import DATA_ROOT, OUTPUT_DIR


def run_traditional_cv(dataset, frame_skip=2):
    """Run traditional CV approach"""
    print("\n" + "="*60)
    print("APPROACH B: TRADITIONAL COMPUTER VISION")
    print("="*60)
    
    pipeline = TraditionalCVPipeline(
        feature_type="SIFT",
        use_tracking=True,
        confidence_threshold=0.3
    )
    
    # Process dataset
    results = pipeline.process_dataset(
        dataset.samples,
        frame_skip=frame_skip,
        show_progress=True
    )
    
    # Convert to submission format
    predictions_dict = {}
    submissions = []
    
    for video_id, sequences in results.items():
        predictions_dict[video_id] = sequences
        submission = annotations_to_dict_format(sequences, video_id)
        submissions.append(submission)
    
    # Save predictions
    output_path = OUTPUT_DIR / "predictions_traditional.json"
    save_predictions(submissions, str(output_path))
    print(f"Predictions saved to {output_path}")
    
    return predictions_dict


def run_deep_learning(dataset, frame_skip=2):
    """Run deep learning approach"""
    print("\n" + "="*60)
    print("APPROACH A: DEEP LEARNING (SOTA)")
    print("="*60)
    
    pipeline = DeepLearningPipeline(
        yolo_model="yolov8x.pt",
        encoder_model="dinov2",
        use_tracking=True,
        use_multiscale=False,
        similarity_threshold=0.7,
        confidence_threshold=0.3
    )
    
    # Process dataset
    results = pipeline.process_dataset(
        dataset.samples,
        frame_skip=frame_skip,
        show_progress=True
    )
    
    # Convert to submission format
    predictions_dict = {}
    submissions = []
    
    for video_id, sequences in results.items():
        predictions_dict[video_id] = sequences
        submission = annotations_to_dict_format(sequences, video_id)
        submissions.append(submission)
    
    # Save predictions
    output_path = OUTPUT_DIR / "predictions_deep_learning.json"
    save_predictions(submissions, str(output_path))
    print(f"Predictions saved to {output_path}")
    
    return predictions_dict


def run_hybrid(dataset, frame_skip=2):
    """Run hybrid approach"""
    print("\n" + "="*60)
    print("APPROACH C: HYBRID (CV + DL)")
    print("="*60)
    
    pipeline = HybridPipeline(
        feature_type="SIFT",
        encoder_model="dinov2",
        cv_confidence=0.2,
        similarity_threshold=0.7,
        use_tracking=True
    )
    
    # Process dataset
    results = pipeline.process_dataset(
        dataset.samples,
        frame_skip=frame_skip,
        show_progress=True
    )
    
    # Convert to submission format
    predictions_dict = {}
    submissions = []
    
    for video_id, sequences in results.items():
        predictions_dict[video_id] = sequences
        submission = annotations_to_dict_format(sequences, video_id)
        submissions.append(submission)
    
    # Save predictions
    output_path = OUTPUT_DIR / "predictions_hybrid.json"
    save_predictions(submissions, str(output_path))
    print(f"Predictions saved to {output_path}")
    
    return predictions_dict


def compare_all_approaches(dataset, predictions_dict):
    """Compare all approaches"""
    print("\n" + "="*60)
    print("COMPARING ALL APPROACHES")
    print("="*60)
    
    # Prepare ground truth
    ground_truth = {}
    for sample in dataset.samples:
        ground_truth[sample.video_id] = sample.annotations
    
    # Create comparator
    comparator = ResultsComparator()
    
    # Add results
    for approach_name, predictions in predictions_dict.items():
        comparator.add_results(approach_name, predictions, ground_truth)
    
    # Print summary
    comparator.print_summary()
    
    # Save report
    comparator.save_report(str(OUTPUT_DIR / "comparison_report"))
    
    # Plot comparison
    try:
        comparator.plot_comparison()
        comparator.plot_per_category()
    except:
        print("Warning: Could not generate plots (may not be in interactive environment)")
    
    return comparator


def main():
    parser = argparse.ArgumentParser(description="Run all approaches and compare")
    parser.add_argument("--frame-skip", type=int, default=2,
                       help="Process every N frames (default: 2)")
    parser.add_argument("--approaches", nargs="+", 
                       choices=["traditional", "deep_learning", "hybrid", "all"],
                       default=["all"],
                       help="Which approaches to run")
    parser.add_argument("--data-root", type=str, default=str(DATA_ROOT),
                       help="Path to data root")
    
    args = parser.parse_args()
    
    # Load dataset
    print("Loading dataset...")
    dataset = DroneDataset(args.data_root)
    print(f"Loaded {len(dataset)} videos")
    
    # Determine which approaches to run
    if "all" in args.approaches:
        run_approaches = ["traditional", "deep_learning", "hybrid"]
    else:
        run_approaches = args.approaches
    
    # Run approaches
    all_predictions = {}
    
    if "traditional" in run_approaches:
        try:
            predictions = run_traditional_cv(dataset, args.frame_skip)
            all_predictions["Traditional CV"] = predictions
        except Exception as e:
            print(f"Error running traditional CV: {e}")
            import traceback
            traceback.print_exc()
    
    if "deep_learning" in run_approaches:
        try:
            predictions = run_deep_learning(dataset, args.frame_skip)
            all_predictions["Deep Learning"] = predictions
        except Exception as e:
            print(f"Error running deep learning: {e}")
            import traceback
            traceback.print_exc()
    
    if "hybrid" in run_approaches:
        try:
            predictions = run_hybrid(dataset, args.frame_skip)
            all_predictions["Hybrid"] = predictions
        except Exception as e:
            print(f"Error running hybrid: {e}")
            import traceback
            traceback.print_exc()
    
    # Compare results
    if len(all_predictions) > 0:
        comparator = compare_all_approaches(dataset, all_predictions)
    else:
        print("No predictions to compare")
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == "__main__":
    main()

