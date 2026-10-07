"""
Draw particle outlines and index labels on the (cropped) original image --
the JPG equivalent of the old ImageJ overlay + ParticleLabeler step.
"""
from PIL import Image, ImageDraw, ImageFont
from skimage import measure


class ParticleOverlay:
    """Draws particle contours + index numbers onto a copy of the source image."""

    def __init__(self, font_size=14):
        self.font_size = font_size
        self._load_font()

    def _load_font(self):
        try:
            self.font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", self.font_size)
        except Exception:
            try:
                self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", self.font_size)
            except Exception:
                self.font = ImageFont.load_default()

    def draw(self, original_image_path, particle_mask, df, output_path, scale_um_per_pixel=None):
        """
        original_image_path: the (cropped) image to draw on top of
        particle_mask: boolean mask from particle_detector.segment_particles
        df: DataFrame from particle_detector.measure_particles (needs CentroidX/Y)
        output_path: where to save the labeled overlay PNG
        scale_um_per_pixel: if given, prints the scale in the corner for reference
        """
        img = Image.open(original_image_path).convert('RGB')
        draw = ImageDraw.Draw(img)

        # Outline each particle in red
        contours = measure.find_contours(particle_mask.astype(float), 0.5)
        for contour in contours:
            points = [(x, y) for y, x in contour]  # find_contours gives (row, col) = (y, x)
            if len(points) > 1:
                draw.line(points + [points[0]], fill='red', width=1)

        # Index label at each particle's centroid
        for idx, row in df.iterrows():
            x, y = int(row['CentroidX']), int(row['CentroidY'])
            text = str(idx)
            for adj_x in (-1, 0, 1):
                for adj_y in (-1, 0, 1):
                    if adj_x or adj_y:
                        draw.text((x + adj_x, y + adj_y), text, fill='black', font=self.font)
            draw.text((x, y), text, fill='white', font=self.font)

        if scale_um_per_pixel is not None:
            self.font_size = 22
            self._load_font()
            draw.text((10, 10), f"Scale: {scale_um_per_pixel:.4f} um/pixel", fill='yellow', font=self.font)

        img.save(output_path)
        return output_path
