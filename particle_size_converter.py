"""
Convert particle measurements from pixels to micrometers using a manually
supplied scale, instead of parsing a Leica XML metadata file (the new
microscope's JPGs don't carry that metadata).
"""


class ParticleSizeConverter:
    """Convert particle measurements from pixels to micrometers."""

    def __init__(self, scale_um_per_pixel):
        self.scale = scale_um_per_pixel

    def convert(self, df):
        """Return a copy of df with extra *_um columns and equivalent diameter."""
        df = df.copy()
        df['Area_um2'] = df['Area'] * (self.scale ** 2)
        df['Major_um'] = df['Major'] * self.scale
        df['Minor_um'] = df['Minor'] * self.scale
        df['Feret_um'] = df['Feret'] * self.scale
        df['MinFeret_um'] = df['MinFeret'] * self.scale
        df['Diameter_um'] = 2 * (df['Area_um2'] / 3.14159) ** 0.5
        return df

    def image_area_cm2(self, width_px, height_px):
        """Total (already-cropped) image area in cm^2, for normalizing particle counts."""
        width_um = width_px * self.scale
        height_um = height_px * self.scale
        area_um2 = width_um * height_um
        return area_um2 / 1e8
