import net
import torch
import os
from face_alignment import align
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


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

def align_image_paths_with_structure(image_paths, target_base_dir, max_workers=4, overwrite=False):
    """
    Align a list of image paths preserving original directory structure
    Returns: dict mapping original paths to aligned paths, and failed paths list
    """
    os.makedirs(target_base_dir, exist_ok=True)
    
    # Prepare file info list
    file_infos = []
    for img_path in image_paths:
        if not os.path.exists(img_path):
            continue
            
        # Preserve relative path structure
        img_name = os.path.basename(img_path)
        img_name_no_ext = os.path.splitext(img_name)[0]
        
        # Create the same relative directory structure in target_base_dir
        relative_dir = os.path.dirname(os.path.relpath(img_path, '/'))
        if relative_dir == '.' or relative_dir == '':
            relative_dir = ''
        
        aligned_subdir = os.path.join(target_base_dir, relative_dir) if relative_dir else target_base_dir
        aligned_path = os.path.join(aligned_subdir, f"{img_name_no_ext}.png")
        
        # Skip if target exists and overwrite is False
        if not overwrite and os.path.exists(aligned_path):
            continue
            
        file_infos.append((img_path, aligned_path, aligned_subdir))
    
    if not file_infos:
        print(f"No images to process")
        return {}, []
    
    print(f"Processing {len(file_infos)} images with {max_workers} workers...")
    
    # Statistics tracking
    alignable_images = {}
    failed_images = []
    stats_lock = threading.Lock()
    
    def align_single_image_with_structure(file_info):
        """Align a single image preserving directory structure"""
        source_path, target_path, target_dir = file_info
        
        try:
            # Create target directory if needed
            os.makedirs(target_dir, exist_ok=True)
            
            # Get aligned face
            aligned_rgb_img = align.get_aligned_face(source_path)
            
            # Save aligned image
            aligned_rgb_img.save(target_path)
            
            with stats_lock:
                alignable_images[source_path] = target_path
                
            return True, source_path, None
            
        except Exception as e:
            with stats_lock:
                failed_images.append(source_path)
                
            return False, source_path, str(e)
    
    # Process images in parallel
    successful_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(align_single_image_with_structure, file_info): file_info 
                         for file_info in file_infos}
        
        # Process completed tasks
        for future in as_completed(future_to_file):
            file_info = future_to_file[future]
            try:
                success, img_path, error = future.result()
                if success:
                    successful_count += 1
                    if successful_count % 10 == 0:  # Progress update every 10 files
                        print(f"Processed {successful_count}/{len(file_infos)} images...")
                else:
                    print(f"Failed to align {img_path}: {error}")
                    
            except Exception as e:
                img_path = file_info[0]
                print(f"Unexpected error processing {img_path}: {e}")
                with stats_lock:
                    failed_images.append(img_path)
    
    # Print summary
    print(f"\nAlignment complete!")
    print(f"Successfully aligned: {len(alignable_images)} images")
    print(f"Failed to align: {len(failed_images)} images")
    
    return alignable_images, failed_images

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

if __name__ == '__main__':
    images_folder = '/home/mossbee/Work/AdaFace/face_alignment/test_images'
    aligned_folder = '/home/mossbee/Work/AdaFace/face_alignment/aligned_images'
    model = load_pretrained_model('ir_50', device='cpu')
    similarity_scores = calculate_folder_similarity(images_folder, aligned_folder, model, device='cpu', batch_size=32)
    print(similarity_scores)