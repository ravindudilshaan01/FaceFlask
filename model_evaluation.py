import cv2
import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support, roc_curve, auc
from datetime import datetime

# Constants - Updated paths to use system folder
MODEL_PATH = "system/models/face_model.yml"
EMBEDDINGS_FILE = "system/models/face_embeddings.pkl"
RESULTS_DIR = "evaluation_results"
KNOWN_FACES_DIR = "system/known_faces"

# Challenging test modes
TEST_MODES = {
    "standard": "Standard testing",
    "noise": "Add random noise to test images",
    "brightness": "Vary brightness of test images",
    "rotation": "Apply slight rotation to test images"
}

def load_model_and_mappings():
    """Load the trained face recognizer model and label mappings."""
    face_recognizer = cv2.face_LBPHFaceRecognizer.create()
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    
    face_recognizer.read(MODEL_PATH)
    
    with open(EMBEDDINGS_FILE, 'rb') as f:
        data = pickle.load(f)
        label_to_id = data.get('label_to_id', {})
        id_to_label = data.get('id_to_label', {})
    
    return face_recognizer, label_to_id, id_to_label

def load_face_data():
    """Load face images and their labels from known_faces directory."""
    # Group faces by person
    face_groups = {}
    
    for reg_number in os.listdir(KNOWN_FACES_DIR):
        dir_path = os.path.join(KNOWN_FACES_DIR, reg_number)
        if os.path.isdir(dir_path):
            face_groups[reg_number] = []
            for img_name in os.listdir(dir_path):
                img_path = os.path.join(dir_path, img_name)
                try:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        face_groups[reg_number].append(img)
                except Exception as e:
                    print(f"Error loading image {img_path}: {str(e)}")
    
    return face_groups

def apply_transformation(image, mode):
    """Apply various transformations to test images to make recognition harder."""
    if mode == "noise":
        # Add random noise
        noise = np.random.normal(0, 15, image.shape).astype(np.uint8)
        transformed = cv2.add(image, noise)
    elif mode == "brightness":
        # Adjust brightness
        factor = random.uniform(0.7, 1.3)
        transformed = cv2.convertScaleAbs(image, alpha=factor, beta=0)
    elif mode == "rotation":
        # Apply slight rotation
        angle = random.uniform(-10, 10)
        h, w = image.shape
        center = (w/2, h/2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        transformed = cv2.warpAffine(image, M, (w, h))
    else:
        # Standard mode - no transformation
        transformed = image.copy()
    
    return transformed

def evaluate_model(face_recognizer, X_test, y_test, label_to_id, id_to_label, test_mode="standard"):
    """Evaluate model performance on test set with optional transformations."""
    predictions = []
    confidences = []
    
    for i, face in enumerate(X_test):
        # Apply transformation based on test mode
        test_face = apply_transformation(face, test_mode)
        
        # Predict
        label, confidence = face_recognizer.predict(test_face)
        confidence = (100 - confidence) / 100
        predicted_reg_number = id_to_label[label]
        predictions.append(predicted_reg_number)
        confidences.append(confidence)
    
    return np.array(predictions), np.array(confidences)

def plot_confusion_matrix(y_true, y_pred, labels, title="Confusion Matrix"):
    """Plot confusion matrix using seaborn."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    return plt.gcf()

def plot_confidence_distribution(confidences, predictions, y_test):
    """Plot confidence score distribution for correct and incorrect predictions."""
    correct = confidences[predictions == y_test]
    incorrect = confidences[predictions != y_test]
    
    plt.figure(figsize=(10, 6))
    plt.hist(correct, alpha=0.5, label='Correct Predictions', bins=20, color='green')
    plt.hist(incorrect, alpha=0.5, label='Incorrect Predictions', bins=20, color='red')
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.title('Confidence Score Distribution')
    plt.legend()
    return plt.gcf()

def generate_evaluation_report():
    """Generate comprehensive evaluation report with visualizations."""
    # Create results directory if it doesn't exist (with exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load model and data
    print("Loading model and data...")
    try:
        face_recognizer, label_to_id, id_to_label = load_model_and_mappings()
        face_groups = load_face_data()
        
        if len(face_groups) == 0:
            print(f"No face data found in the {KNOWN_FACES_DIR} directory!")
            return
        
        # Timestamp for this evaluation run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # K-fold cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Store results for each test mode
        results = {}
        
        # Run evaluation for each test mode
        for test_mode in TEST_MODES:
            print(f"\nRunning evaluation with test mode: {test_mode} ({TEST_MODES[test_mode]})")
            
            # Create combined arrays of all faces and labels
            all_faces = []
            all_labels = []
            for reg_number, faces in face_groups.items():
                all_faces.extend(faces)
                all_labels.extend([reg_number] * len(faces))
            
            all_faces = np.array(all_faces)
            all_labels = np.array(all_labels)
            
            # Run cross-validation
            fold_results = []
            for fold, (train_idx, test_idx) in enumerate(kf.split(all_faces)):
                X_train, X_test = all_faces[train_idx], all_faces[test_idx]
                y_train, y_test = all_labels[train_idx], all_labels[test_idx]
                
                print(f"  Fold {fold+1}: {len(X_train)} training samples, {len(X_test)} test samples")
                
                # Train a new model for this fold
                model = cv2.face_LBPHFaceRecognizer.create()
                model.train(X_train, np.array([label_to_id[label] for label in y_train]))
                
                # Evaluate
                predictions, confidences = evaluate_model(model, X_test, y_test, label_to_id, id_to_label, test_mode)
                accuracy = accuracy_score(y_test, predictions)
                precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average='weighted')
                
                fold_results.append({
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'y_test': y_test,
                    'predictions': predictions,
                    'confidences': confidences
                })
                
                print(f"  Fold {fold+1} Accuracy: {accuracy:.4f}")
            
            # Aggregate results across folds
            mean_accuracy = np.mean([r['accuracy'] for r in fold_results])
            mean_precision = np.mean([r['precision'] for r in fold_results])
            mean_recall = np.mean([r['recall'] for r in fold_results])
            mean_f1 = np.mean([r['f1'] for r in fold_results])
            
            results[test_mode] = {
                'mean_accuracy': mean_accuracy,
                'mean_precision': mean_precision,
                'mean_recall': mean_recall,
                'mean_f1': mean_f1,
                'fold_results': fold_results
            }
            
            print(f"  Average Accuracy: {mean_accuracy:.4f}")
        
        # Generate report
        report_path = os.path.join(RESULTS_DIR, f"evaluation_report_{timestamp}.txt")
        
        with open(report_path, 'w') as f:
            f.write("Face Recognition Model Evaluation Report (Cross-Validation)\n")
            f.write("====================================================\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total faces: {sum(len(faces) for faces in face_groups.values())}\n")
            f.write(f"Number of individuals: {len(face_groups)}\n")
            f.write(f"Cross-validation: 5-fold with shuffling\n\n")
            
            # Overall summary for each test mode
            f.write("SUMMARY OF RESULTS\n")
            f.write("=================\n\n")
            
            for mode in results:
                f.write(f"Test Mode: {mode} ({TEST_MODES[mode]})\n")
                f.write(f"  Mean Accuracy: {results[mode]['mean_accuracy']:.4f}\n")
                f.write(f"  Mean Precision: {results[mode]['mean_precision']:.4f}\n")
                f.write(f"  Mean Recall: {results[mode]['mean_recall']:.4f}\n")
                f.write(f"  Mean F1-Score: {results[mode]['mean_f1']:.4f}\n\n")
        
        # Generate and save plots
        print("\nGenerating visualizations...")
        
        # Plot comparison of accuracy across test modes
        plt.figure(figsize=(12, 6))
        mode_names = list(results.keys())
        mode_accuracies = [results[mode]['mean_accuracy'] for mode in mode_names]
        mode_display_names = [f"{m} ({TEST_MODES[m]})" for m in mode_names]
        
        plt.bar(mode_display_names, mode_accuracies, color='skyblue')
        plt.xlabel('Test Mode')
        plt.ylabel('Mean Accuracy')
        plt.title('Recognition Accuracy by Test Mode')
        plt.ylim(0, 1.05)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(os.path.join(RESULTS_DIR, f"test_mode_comparison_{timestamp}.png"))
        plt.close()
        
        # Generate confusion matrix and confidence distribution for standard mode
        best_fold_idx = np.argmax([r['accuracy'] for r in results['standard']['fold_results']])
        best_fold = results['standard']['fold_results'][best_fold_idx]
        
        cm_fig = plot_confusion_matrix(
            best_fold['y_test'], 
            best_fold['predictions'], 
            sorted(set(best_fold['y_test'])),
            title=f"Confusion Matrix (Standard Mode, Fold {best_fold_idx+1})"
        )
        cm_fig.savefig(os.path.join(RESULTS_DIR, f"confusion_matrix_{timestamp}.png"))
        plt.close()
        
        conf_fig = plot_confidence_distribution(
            best_fold['confidences'], 
            best_fold['predictions'], 
            best_fold['y_test']
        )
        conf_fig.savefig(os.path.join(RESULTS_DIR, f"confidence_distribution_{timestamp}.png"))
        plt.close()
        
        print(f"\nEvaluation complete! Results saved in {RESULTS_DIR}")
        print(f"Report file: evaluation_report_{timestamp}.txt")
        print(f"Test mode comparison: test_mode_comparison_{timestamp}.png")
        print(f"Confusion matrix: confusion_matrix_{timestamp}.png")
        print(f"Confidence distribution: confidence_distribution_{timestamp}.png")
        
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        print(f"Please make sure you have trained the model and have face data in the {KNOWN_FACES_DIR} directory.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generate_evaluation_report()