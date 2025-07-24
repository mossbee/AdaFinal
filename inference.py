import net
import torch
import os
from face_alignment import align
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import json


adaface_models = {
    'ir_50':"pretrained/adaface_ir50_ms1mv2.ckpt",
}

def load_pretrained_model(architecture='ir_50', device='cpu'):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    statedict = torch.load(adaface_models[architecture], map_location=device, weights_only=False)['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model

def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    bgr_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
    tensor = torch.tensor(bgr_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) # torch.Size([1, 3, 112, 112])
    return tensor

def get_feature(image_path, model):
    aligned_rgb_img = align.get_aligned_face(image_path)
    bgr_input = to_input(aligned_rgb_img)
    with torch.no_grad():
        feature, _ = model(bgr_input) # feature shape: torch.Size([1, 512])
    return feature

def get_feature_from_pil(pil_rgb_image, model):
    bgr_input = to_input(pil_rgb_image)
    with torch.no_grad():
        feature, _ = model(bgr_input)  # feature shape: torch.Size([1, 512])
    return feature

def to_input_batch(pil_rgb_images):
    batch_tensors = []
    for pil_rgb_image in pil_rgb_images:
        np_img = np.array(pil_rgb_image)
        bgr_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
        tensor = torch.tensor(bgr_img, dtype=torch.float32).permute(2, 0, 1)  # Shape: (3, 112, 112)
        batch_tensors.append(tensor)
    
    return torch.stack(batch_tensors, dim=0)  # Shape: (batch_size, 3, 112, 112)

def get_features_from_aligned_images(pil_rgb_images, model, device='cpu', batch_size=32):
    # Ensure model is on correct device (only if needed)
    if next(model.parameters()).device != torch.device(device):
        model = model.to(device)
    
    all_features = []
    
    with torch.no_grad():
        for i in range(0, len(pil_rgb_images), batch_size):
            batch_images = pil_rgb_images[i:i + batch_size]
            batch_tensor = to_input_batch(batch_images).to(device)
            features, _ = model(batch_tensor)
            
            # Keep on GPU if possible, transfer at end
            if device == 'cpu':
                all_features.append(features)
            else:
                all_features.append(features.cpu())
            
            # Clean up batch tensor
            del batch_tensor
            if device != 'cpu':
                torch.cuda.empty_cache()
    
    feature_matrix = torch.cat(all_features, dim=0)
    return feature_matrix

def align_folder_images(source_folder, target_folder, max_workers=4, overwrite=False, image_extensions=None):
    if image_extensions is None:
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    os.makedirs(target_folder, exist_ok=True)
    
    source_files = []
    for fname in sorted(os.listdir(source_folder)):
        if os.path.splitext(fname.lower())[1] in image_extensions:
            source_path = os.path.join(source_folder, fname)
            # Save as png for high quality
            target_path = os.path.join(target_folder, os.path.splitext(fname)[0] + '.png')
            
            # Skip if target exists and overwrite is False
            if not overwrite and os.path.exists(target_path):
                continue
                
            source_files.append((source_path, target_path, fname))
    
    if not source_files:
        print(f"No images to process in {source_folder}")
        return {"successful": 0, "failed": 0, "skipped": 0, "failed_files": []}
    
    print(f"Processing {len(source_files)} images with {max_workers} workers...")
    
    # Statistics tracking
    stats = {
        "successful": 0,
        "failed": 0,
        "skipped": 0,
        "failed_files": []
    }
    stats_lock = threading.Lock()
    
    def align_single_image(file_info):
        """Align a single image and save it"""
        source_path, target_path, fname = file_info
        
        try:
            # Get aligned face
            aligned_rgb_img = align.get_aligned_face(source_path)
            
            # Save aligned image
            aligned_rgb_img.save(target_path)  # High quality JPEG
            
            with stats_lock:
                stats["successful"] += 1
                
            return True, fname, None
            
        except Exception as e:
            with stats_lock:
                stats["failed"] += 1
                stats["failed_files"].append(fname)
                
            return False, fname, str(e)
    
    # Process images in parallel
    successful_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(align_single_image, file_info): file_info 
                         for file_info in source_files}
        
        # Process completed tasks
        for future in as_completed(future_to_file):
            file_info = future_to_file[future]
            try:
                success, fname, error = future.result()
                if success:
                    successful_count += 1
                    if successful_count % 10 == 0:  # Progress update every 10 files
                        print(f"Processed {successful_count}/{len(source_files)} images...")
                else:
                    print(f"Failed to align {fname}: {error}")
                    
            except Exception as e:
                fname = file_info[2]
                print(f"Unexpected error processing {fname}: {e}")
                with stats_lock:
                    stats["failed"] += 1
                    stats["failed_files"].append(fname)
    
    # Print summary
    print(f"\nAlignment complete!")
    print(f"Successfully aligned: {stats['successful']} images")
    if stats['failed'] > 0:
        print(f"Failed to align: {stats['failed']} images")
    if stats['skipped'] > 0:
        print(f"Skipped (already exist): {stats['skipped']} images")
    
    return stats

def calculate_folder_similarity(folder_path, aligned_folder, model, device='cpu', batch_size=32):
    align_folder_images(folder_path, aligned_folder, max_workers=4, overwrite=True)

    aligned_images = []
    for fname in sorted(os.listdir(aligned_folder)):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
            img_path = os.path.join(aligned_folder, fname)
            aligned_images.append(Image.open(img_path).convert('RGB'))
    
    features = get_features_from_aligned_images(aligned_images, model, device=device, batch_size=batch_size)
    similarity_matrix = torch.mm(features, features.t())
    return similarity_matrix

def align_twin_pairs_images(id_to_images_json, test_twin_pairs_json, dest_folder, max_workers=4, overwrite=False):
    """
    Align all images of IDs specified in test_twin_id_pairs.json and save to destination folder.
    Creates a folder structure where each ID has its own subfolder containing aligned images.
    
    Args:
        id_to_images_json (str): Path to id_to_images.json file
        test_twin_pairs_json (str): Path to test_twin_id_pairs.json file 
        dest_folder (str): Destination folder to save aligned images
        max_workers (int): Number of parallel workers for alignment
        overwrite (bool): Whether to overwrite existing aligned images
        
    Returns:
        dict: Contains statistics and aligned_id_to_images mapping
            - successful: Number of successfully aligned images
            - failed: Number of failed alignments
            - skipped: Number of skipped images
            - failed_files: List of filenames that failed
            - missing_ids: List of IDs not found in id_to_images.json
            - aligned_id_to_images: Dictionary mapping ID to list of aligned image paths
    """
    # Load JSON files
    with open(id_to_images_json, 'r') as f:
        id_to_images = json.load(f)
    
    with open(test_twin_pairs_json, 'r') as f:
        twin_pairs = json.load(f)
    
    # Create destination folder
    os.makedirs(dest_folder, exist_ok=True)
    
    # Collect all unique IDs from twin pairs
    unique_ids = set()
    for pair in twin_pairs:
        unique_ids.add(pair[0])
        unique_ids.add(pair[1])
    
    print(f"Found {len(unique_ids)} unique IDs in twin pairs: {sorted(unique_ids)}")
    
    # Collect all image paths for these IDs, organized by ID
    id_to_source_images = {}
    missing_ids = []
    
    for id_name in unique_ids:
        if id_name in id_to_images:
            valid_images = []
            for img_path in id_to_images[id_name]:
                if os.path.exists(img_path):
                    valid_images.append(img_path)
                else:
                    print(f"Warning: Image not found: {img_path}")
            
            if valid_images:
                id_to_source_images[id_name] = valid_images
                # Create subfolder for this ID
                id_folder = os.path.join(dest_folder, id_name)
                os.makedirs(id_folder, exist_ok=True)
        else:
            missing_ids.append(id_name)
            print(f"Warning: ID '{id_name}' not found in id_to_images.json")
    
    if missing_ids:
        print(f"Missing IDs: {missing_ids}")
    
    if not id_to_source_images:
        print("No valid image paths found!")
        return {
            "successful": 0, "failed": 0, "skipped": 0, "failed_files": [], 
            "missing_ids": missing_ids, "aligned_id_to_images": {}
        }
    
    total_images = sum(len(images) for images in id_to_source_images.values())
    print(f"Processing {total_images} images across {len(id_to_source_images)} IDs...")
    
    # Prepare file info tuples for alignment
    source_files = []
    
    for id_name, img_paths in id_to_source_images.items():
        id_folder = os.path.join(dest_folder, id_name)
        
        for img_path in img_paths:
            # Keep original filename but change extension to PNG
            original_name = os.path.splitext(os.path.basename(img_path))[0]
            target_filename = f"{original_name}.png"
            target_path = os.path.join(id_folder, target_filename)
            
            # Skip if target exists and overwrite is False
            if not overwrite and os.path.exists(target_path):
                continue
                
            source_files.append((img_path, target_path, os.path.basename(img_path), id_name))
    
    if not source_files:
        print(f"No images to process (all may already exist in {dest_folder})")
        # Still build the aligned_id_to_images from existing files
        aligned_id_to_images = {}
        for id_name in id_to_source_images.keys():
            id_folder = os.path.join(dest_folder, id_name)
            if os.path.exists(id_folder):
                aligned_images = []
                for fname in sorted(os.listdir(id_folder)):
                    if fname.lower().endswith('.png'):
                        aligned_images.append(os.path.join(id_folder, fname))
                if aligned_images:
                    aligned_id_to_images[id_name] = aligned_images
        
        return {
            "successful": 0, "failed": 0, "skipped": total_images, "failed_files": [], 
            "missing_ids": missing_ids, "aligned_id_to_images": aligned_id_to_images
        }
    
    print(f"Aligning {len(source_files)} images with {max_workers} workers...")
    
    # Statistics tracking
    stats = {
        "successful": 0,
        "failed": 0,
        "skipped": total_images - len(source_files),
        "failed_files": [],
        "missing_ids": missing_ids
    }
    stats_lock = threading.Lock()
    
    # Track successful alignments by ID
    aligned_images_by_id = {id_name: [] for id_name in id_to_source_images.keys()}
    aligned_images_lock = threading.Lock()
    
    def align_single_image(file_info):
        """Align a single image and save it"""
        source_path, target_path, fname, id_name = file_info
        
        try:
            # Get aligned face
            aligned_rgb_img = align.get_aligned_face(source_path)
            
            # Save aligned image as PNG for high quality
            aligned_rgb_img.save(target_path)
            
            with stats_lock:
                stats["successful"] += 1
            
            with aligned_images_lock:
                aligned_images_by_id[id_name].append(target_path)
                
            return True, fname, None
            
        except Exception as e:
            with stats_lock:
                stats["failed"] += 1
                stats["failed_files"].append(fname)
                
            return False, fname, str(e)
    
    # Process images in parallel
    successful_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(align_single_image, file_info): file_info 
                         for file_info in source_files}
        
        # Process completed tasks
        for future in as_completed(future_to_file):
            file_info = future_to_file[future]
            try:
                success, fname, error = future.result()
                if success:
                    successful_count += 1
                    if successful_count % 10 == 0:  # Progress update every 10 files
                        print(f"Processed {successful_count}/{len(source_files)} images...")
                else:
                    print(f"Failed to align {fname}: {error}")
                    
            except Exception as e:
                fname = file_info[2]
                print(f"Unexpected error processing {fname}: {e}")
                with stats_lock:
                    stats["failed"] += 1
                    stats["failed_files"].append(fname)
    
    # Sort the aligned images for each ID
    for id_name in aligned_images_by_id:
        aligned_images_by_id[id_name].sort()
    
    # Add existing aligned images to the mapping (for skipped files)
    for id_name in id_to_source_images.keys():
        id_folder = os.path.join(dest_folder, id_name)
        if os.path.exists(id_folder):
            existing_images = []
            for fname in sorted(os.listdir(id_folder)):
                if fname.lower().endswith('.png'):
                    img_path = os.path.join(id_folder, fname)
                    if img_path not in aligned_images_by_id[id_name]:
                        existing_images.append(img_path)
            # Merge and sort
            all_images = aligned_images_by_id[id_name] + existing_images
            aligned_images_by_id[id_name] = sorted(all_images)
    
    # Print summary
    print(f"\nTwin pairs alignment complete!")
    print(f"Successfully aligned: {stats['successful']} images")
    if stats['failed'] > 0:
        print(f"Failed to align: {stats['failed']} images")
        print(f"Failed files: {stats['failed_files']}")
    if stats['skipped'] > 0:
        print(f"Skipped (already exist): {stats['skipped']} images")
    if stats['missing_ids']:
        print(f"Missing IDs in id_to_images.json: {stats['missing_ids']}")
    
    # Print folder structure summary
    print(f"\nFolder structure created in {dest_folder}:")
    for id_name, aligned_paths in aligned_images_by_id.items():
        if aligned_paths:
            print(f"  {id_name}/: {len(aligned_paths)} aligned images")
    
    # Add the aligned mapping to stats
    stats["aligned_id_to_images"] = aligned_images_by_id
    
    return stats

def save_aligned_id_mapping(aligned_id_to_images, output_json_path):
    """
    Save the aligned ID-to-images mapping to a JSON file.
    
    Args:
        aligned_id_to_images (dict): Dictionary mapping ID to list of aligned image paths
        output_json_path (str): Path where to save the JSON file
    """
    import json
    
    with open(output_json_path, 'w') as f:
        json.dump(aligned_id_to_images, f, indent=4)
    
    print(f"Saved aligned ID mapping to: {output_json_path}")
    print(f"Total IDs: {len(aligned_id_to_images)}")
    total_images = sum(len(images) for images in aligned_id_to_images.values())
    print(f"Total aligned images: {total_images}")

if __name__ == '__main__':
    # Example 1: Calculate similarity for a folder of images
    images_folder = '/home/mossbee/Work/AdaFace/face_alignment/test_images'
    aligned_folder = '/home/mossbee/Work/AdaFace/face_alignment/aligned_images'
    model = load_pretrained_model('ir_50', device='cpu')
    similarity_scores = calculate_folder_similarity(images_folder, aligned_folder, model, device='cpu', batch_size=32)
    print("Similarity matrix:")
    print(similarity_scores)
    
    # Example 2: Align twin pairs images with organized folder structure
    id_to_images_json = '/home/mossbee/Work/AdaFace/id_to_images.json'
    test_twin_pairs_json = '/home/mossbee/Work/AdaFace/test_twin_id_pairs.json'
    dest_folder = '/home/mossbee/Work/AdaFace/aligned_twin_images'
    
    print("\n" + "="*50)
    print("Aligning twin pairs images...")
    stats = align_twin_pairs_images(
        id_to_images_json=id_to_images_json,
        test_twin_pairs_json=test_twin_pairs_json,
        dest_folder=dest_folder,
        max_workers=4,
        overwrite=True
    )
    
    print(f"Final stats: {stats}")
    
    # Save the aligned ID mapping to a JSON file
    if stats["aligned_id_to_images"]:
        aligned_mapping_json = '/home/mossbee/Work/AdaFace/aligned_id_to_images.json'
        save_aligned_id_mapping(stats["aligned_id_to_images"], aligned_mapping_json)