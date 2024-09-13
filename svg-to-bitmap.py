import svgpathtools
import cairosvg
from PIL import Image
import io
import warnings
import numpy as np

# Suppress DecompressionBombWarning
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

def get_svg_dimensions(svg_file):
    paths, attributes = svgpathtools.svg2paths(svg_file)
    if not paths:
        raise ValueError("No paths found in the SVG file")
    
    xmin, xmax, ymin, ymax = float('inf'), float('-inf'), float('inf'), float('-inf')
    for path in paths:
        bbox = path.bbox()
        xmin = min(xmin, bbox[0])
        xmax = max(xmax, bbox[1])
        ymin = min(ymin, bbox[2])
        ymax = max(ymax, bbox[3])
    
    width = xmax - xmin
    height = ymax - ymin
    return width, height, xmin, ymin

def crop_svg(svg_file, output_png_file, dpi=1000, padding_mm=0.1):
    width, height, xmin, ymin = get_svg_dimensions(svg_file)

    print(f"Detected SVG dimensions: {width:.2f}mm x {height:.2f}mm")
    print(f"Offset: ({xmin:.2f}, {ymin:.2f})")

    # Add padding
    width += 2 * padding_mm
    height += 2 * padding_mm
    xmin -= padding_mm
    ymin -= padding_mm

    # Calculate pixel dimensions at a higher resolution
    scale_factor = 4  # Render at 4x the final size
    pixel_width = int(width * dpi * scale_factor / 25.4)  # Convert mm to pixels
    pixel_height = int(height * dpi * scale_factor / 25.4)

    print(f"Rendering image at: {pixel_width} x {pixel_height} pixels")

    # Convert to PNG using cairosvg with high resolution
    png_data = cairosvg.svg2png(
        url=svg_file,
        output_width=pixel_width,
        output_height=pixel_height,
        dpi=dpi * scale_factor,
        scale=1,
        background_color='white'
    )
    
    # Open as image using PIL
    with Image.open(io.BytesIO(png_data)) as image:
        # Convert to grayscale
        gray_image = image.convert('L')

        # Convert to numpy array
        img_array = np.array(gray_image)

        # Find non-white pixels
        non_white = np.where(img_array < 255)
        top, left = np.min(non_white[0]), np.min(non_white[1])
        bottom, right = np.max(non_white[0]), np.max(non_white[1])

        # Crop the image
        cropped_image = gray_image.crop((left, top, right, bottom))

        # Resize to the final resolution
        final_width = int(width * dpi / 25.4)
        final_height = int(height * dpi / 25.4)
        resized_image = cropped_image.resize((final_width, final_height), Image.LANCZOS)

        # Convert to 1-bit pixel format with high threshold
        bw_image = resized_image.convert('1', dither=Image.NONE)

        # Invert the image (black becomes white and vice versa)
        bw_image = Image.eval(bw_image, lambda x: 255 - x)

        # Save the image as PNG with correct DPI information
        bw_image.save(output_png_file, 'PNG', dpi=(dpi, dpi))

    # Print final dimensions
    final_width, final_height = bw_image.size
    print(f"Final image dimensions: {final_width} x {final_height} pixels")
    print(f"Final image size: {width:.2f}mm x {height:.2f}mm")
    print(f"DPI: {dpi}")

# Example usage
crop_svg('interface-board-F_Cu.svg', 'output_high_res.png', dpi=5000, padding_mm=0.1)
