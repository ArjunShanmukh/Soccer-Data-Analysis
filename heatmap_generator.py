import numpy as np
import cv2

class HeatmapGenerator:
    """
    Build a heatmap in field (transformed) coordinates using
    tracks[*]['position_transformed'], then warp it back into
    pixel/frame coordinates using the inverse perspective transform
    computed from ViewTransformer.target_vertices and .pixel_vertices.
    """

    def __init__(self, view_transformer, grid_size=(340, 200)):
        """
        view_transformer: instance of your ViewTransformer (has pixel_vertices, target_vertices)
        grid_size: (width, height) of the heatmap in field-space pixels (w, h)
        """
        self.view_transformer = view_transformer
        self.grid_w = int(grid_size[0])
        self.grid_h = int(grid_size[1])

        # compute field bounds from the transformer's target_vertices
        tv = self.view_transformer.target_vertices
        # target_vertices shape: (4,2). compute min/max
        self.min_x = float(np.min(tv[:, 0]))
        self.max_x = float(np.max(tv[:, 0]))
        self.min_y = float(np.min(tv[:, 1]))
        self.max_y = float(np.max(tv[:, 1]))

        # accumulator for counts in field-space
        self.accumulator = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)

        # precompute inverse perspective to warp heatmap -> pixel space
        # pixel_vertices and target_vertices must be float32
        pv = self.view_transformer.pixel_vertices.astype(np.float32)
        tv = self.view_transformer.target_vertices.astype(np.float32)
        # get inverse: target -> pixel
        self.inverse_perspective = cv2.getPerspectiveTransform(tv, pv)

    def _field_to_grid(self, pt):
        """Map a point in field coordinates to heatmap grid indices (col, row)."""
        x, y = pt[0], pt[1]
        # normalize x,y into [0,1]
        # handle degenerate ranges defensively
        x_norm = 0.0 if self.max_x == self.min_x else (x - self.min_x) / (self.max_x - self.min_x)
        y_norm = 0.0 if self.max_y == self.min_y else (y - self.min_y) / (self.max_y - self.min_y)
        col = int(np.clip(x_norm * (self.grid_w - 1), 0, self.grid_w - 1))
        row = int(np.clip(y_norm * (self.grid_h - 1), 0, self.grid_h - 1))
        return col, row

    def accumulate_from_tracks(self, tracks, ignore_ball=True):
        """
        Go through tracks['players'] and increment accumulator for every valid
        position_transformed point. If ignore_ball is True, will skip the 'ball' object.
        """
        # Reset accumulator (call once per run or keep cumulative across runs)
        self.accumulator[:] = 0.0

        # tracks["players"] is a list of per-frame dicts
        for frame_tracks in tracks.get("players", []):
            for track_id, info in frame_tracks.items():
                pos_t = info.get("position_transformed", None)
                if pos_t is None:
                    continue
                # pos_t may be [x,y] or nested; handle both
                try:
                    x, y = float(pos_t[0]), float(pos_t[1])
                except Exception:
                    continue
                col, row = self._field_to_grid((x, y))
                self.accumulator[row, col] += 1.0

        # optional: apply a small gaussian blur to smooth hotspots
        self.accumulator = cv2.GaussianBlur(self.accumulator, (0, 0), sigmaX=3, sigmaY=3, borderType=cv2.BORDER_REPLICATE)

    def build_heatmap_image(self, apply_cmap=True):
        """
        Convert accumulator to a heatmap RGB image in field-space (grid_w x grid_h)
        Returns the BGR heatmap image (uint8) and the normalized accumulator.
        """
        if np.max(self.accumulator) == 0:
            norm = self.accumulator.copy()
        else:
            norm = (self.accumulator / np.max(self.accumulator) * 255.0).astype(np.uint8)

        # resize to ensure width/height orientation consistent (rows->h, cols->w)
        heat_gray = cv2.resize(norm, (self.grid_w, self.grid_h), interpolation=cv2.INTER_LINEAR)

        if not apply_cmap:
            heat_bgr = cv2.cvtColor(heat_gray, cv2.COLOR_GRAY2BGR)
            return heat_bgr, norm

        # use a colormap to produce visually pleasing heatmap (applyColorMap expects 0..255)
        heat_color = cv2.applyColorMap(heat_gray, cv2.COLORMAP_JET)  # returns BGR
        return heat_color, norm

    def warp_heatmap_to_frame(self, heatmap_bgr, frame_shape):
        """
        Warp the field-space heatmap (heatmap_bgr) back to original frame pixel coordinates
        using the precomputed inverse_perspective. frame_shape = (h, w, channels)
        """
        h_frame, w_frame = frame_shape[0], frame_shape[1]
        # because heatmap_bgr is in grid coords, we need to warp from that image's coordinate system
        # to pixel coordinates. cv2.warpPerspective maps source image coordinates -> destination image coords
        # so we pass source = heatmap_bgr and M = inverse_perspective (target->pixel)
        warped = cv2.warpPerspective(heatmap_bgr, self.inverse_perspective, (w_frame, h_frame), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
        return warped

    def overlay_heatmap_on_frame(self, frame, alpha=0.5, apply_cmap=True):
        """
        Return a copy of frame with the heatmap overlayed.
        Make sure to call accumulate_from_tracks(...) first, then build_heatmap_image(...)
        """
        heat_bgr, _ = self.build_heatmap_image(apply_cmap=apply_cmap)
        warped = self.warp_heatmap_to_frame(heat_bgr, frame.shape)
        overlay = frame.copy().astype(np.float32)
        warped_f = warped.astype(np.float32)

        # Weighted overlay (only where warped has non-zero values)
        mask = (warped.sum(axis=2) > 0).astype(np.uint8)[:, :, None]
        # blend only at mask locations
        overlay = np.where(mask, cv2.addWeighted(frame.astype(np.float32), 1.0 - alpha, warped_f, alpha, 0), frame.astype(np.float32))
        return overlay.astype(np.uint8)

    def overlay_heatmap_on_frames(self, frames, alpha=0.5, apply_cmap=True):
        """
        Convenience: returns a new list of frames where the heatmap overlay is applied to each frame.
        (same overlay used for every frame, since accumulator is global across frames)
        """
        if len(frames) == 0:
            return frames
        self.build_heatmap_image(apply_cmap=apply_cmap)
        out_frames = []
        for frame in frames:
            out_frames.append(self.overlay_heatmap_on_frame(frame, alpha=alpha, apply_cmap=apply_cmap))
        return out_frames

    def draw_heatmap(self, frames, alpha=0.5, apply_cmap=True):
        """
        Builds accumulator → builds heatmap → overlays as a bottom-center inset
        with 1:1 heatmap:rectangle size ratio.
        """
        if len(frames) == 0:
            return frames

        # Build heatmap once (field-space)
        heatmap_bgr, _ = self.build_heatmap_image(apply_cmap=apply_cmap)

        # Fixed rectangle size (matches heatmap grid)
        rect_w, rect_h = self.grid_w, self.grid_h

        out_frames = []

        for frame in frames:
            h_frame, w_frame = frame.shape[:2]

            # Resize heatmap to rectangle size
            resized_heatmap = cv2.resize(
                heatmap_bgr,
                (rect_w, rect_h),
                interpolation=cv2.INTER_LINEAR
            )

            # bottom-center position
            x_offset = (w_frame - rect_w) // 2
            y_offset = h_frame - rect_h - 10  # 10 px margin from bottom

            overlay = frame.copy().astype(np.float32)
            canvas = overlay.copy()
            canvas[y_offset:y_offset + rect_h, x_offset:x_offset + rect_w] = resized_heatmap.astype(np.float32)

            # mask for blending only the heatmap area
            mask = np.zeros((h_frame, w_frame, 1), dtype=np.uint8)
            mask[y_offset:y_offset + rect_h, x_offset:x_offset + rect_w] = 1

            # blend heatmap onto frame
            blended = np.where(
                mask,
                cv2.addWeighted(overlay, 1.0 - alpha, canvas, alpha, 0),
                overlay
            )

            out_frames.append(blended.astype(np.uint8))


        return out_frames



