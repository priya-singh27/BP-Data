import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image_path, resize_factor=0.5, blur_kernel=(51, 51), clip_limit=2.0):
    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize image to speed up processing (downscale by resize_factor)
    small_img = cv2.resize(img, (0, 0), fx=resize_factor, fy=resize_factor)

    # Apply CLAHE to improve contrast
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    img_clahe = clahe.apply(small_img)

    # Apply Gaussian Blur to approximate the background (using smaller blur on downscaled image)
    bg_blur = cv2.GaussianBlur(img_clahe, blur_kernel, 0)

    # Resize back to original size
    bg_blur = cv2.resize(bg_blur, (img.shape[1], img.shape[0]))

    # Subtract blurred background from original CLAHE image to remove background
    img_clahe_full = clahe.apply(img)
    img_bg_subtracted = cv2.subtract(img_clahe_full, bg_blur)

    # Sharpen the image to enhance edges
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img_sharpened = cv2.filter2D(img_bg_subtracted, -1, kernel)

    # Adaptive Thresholding to binarize the image
    img_thresh = cv2.adaptiveThreshold(
        img_sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Morphological closing to reduce small noise
    kernel_morph = np.ones((5, 5), np.uint8)
    img_morph = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel_morph)

    # Apply morphological opening to remove isolated small noise spots
    kernel_open = np.ones((3, 3), np.uint8)
    img_open = cv2.morphologyEx(img_morph, cv2.MORPH_OPEN, kernel_open)
    

    # Morphological closing to reduce small noise
    kernel_morph = np.ones((5, 5), np.uint8)
    img_morph = cv2.morphologyEx(img_open, cv2.MORPH_CLOSE, kernel_morph)

    # Apply morphological opening to remove isolated small noise spots
    kernel_open = np.ones((3, 3), np.uint8)
    img_open = cv2.morphologyEx(img_morph, cv2.MORPH_OPEN, kernel_open)
    img_clean = cv2.medianBlur(img_open, 3)  # Kernel size of 3 is generally enough for small spots

    return img_clean

def enhance_contrast(image_path, mask):
    # Read the original grayscale image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Increase overall contrast by scaling pixel values
    enhanced_img = cv2.convertScaleAbs(img, alpha=1.8, beta=0)  # Increase contrast

    # Normalize the mask to [0, 1] range for blending
    mask_norm = mask / 255.0  # Normalize mask to [0, 1] range

    # Element-wise blending based on mask
    contrast_blended = (enhanced_img * mask_norm + img * (1 - mask_norm)).astype(np.uint8)

    return contrast_blended

def pad_and_resize(image, target_aspect_ratio=2, padding_color=255):
    # Calculate padding to increase height by 10%
    h, w = image.shape
    pad_height = int(h * 0.1)
    padded_img = cv2.copyMakeBorder(image, pad_height, pad_height, 0, 0, cv2.BORDER_CONSTANT, value=padding_color)
    
    # Resize to achieve the 2:1 aspect ratio
    new_h, new_w = padded_img.shape
    desired_width = new_h * target_aspect_ratio
    if desired_width > new_w:
        # Pad width to reach desired aspect ratio
        width_padding = int((desired_width - new_w) / 2)
        final_img = cv2.copyMakeBorder(padded_img, 0, 0, width_padding, width_padding, cv2.BORDER_CONSTANT, value=padding_color)
    else:
        # Resize to desired width if aspect ratio is close to 2:1
        final_img = cv2.resize(padded_img, (int(2 * new_h), new_h), interpolation=cv2.INTER_LINEAR)
    
    return final_img



def final_prep(image_path):
    # Get preprocessed mask
    mask = preprocess_image(image_path)

    # Enhance contrast with mask
    enhanced_img = enhance_contrast(image_path, mask)

    # Pad and resize to achieve the 2:1 aspect ratio
    final_img = pad_and_resize(enhanced_img)

    return final_img


if __name__ == "__main__":
    # File paths for the uploaded images
    image_paths = [
        "./crop_SYS.jpg",
        "./crop_DIA.jpg",
        "./crop_PUL.jpg",
    ]

    # Preprocess, enhance contrast, pad, and resize images
    for image_path in image_paths:
        # Pad and resize to achieve the 2:1 aspect ratio
        final_img = final_prep(image_path)  

        # Display the final padded and resized image
        plt.imshow(final_img, cmap='gray')
        plt.title(f'Padded and Resized Image: {image_path.split("/")[-1]}')
        plt.axis('off')
        plt.show()
        
        # Save the final image
        cv2.imwrite(f'./padded_resized_{image_path.split("/")[-1]}', final_img)
