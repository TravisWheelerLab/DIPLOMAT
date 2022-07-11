from decimal import Decimal
from typing import Optional
import numpy as np
from diplomat.predictors.fpe.frame_pass import FramePass
from diplomat.predictors.fpe.skeleton_structures import Histogram
from diplomat.predictors.fpe.sparse_storage import ForwardBackwardData, ForwardBackwardFrame, AttributeDict
from diplomat.processing import ProgressBar, ConfigSpec
import diplomat.processing.type_casters as tc


class OptimizeStandardDeviation(FramePass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._histogram = None
        self._current_frame = None
        self._prior_max_locations = None
        self._max_locations = None
        self._ignore_below = None

    def run_pass(
        self,
        fb_data: ForwardBackwardData,
        prog_bar: Optional[ProgressBar] = None,
        in_place: bool = True,
        reset_bar: bool = True
    ) -> ForwardBackwardData:
        d_scale = fb_data.metadata.down_scaling
        bin_off = self.config.ignore_bins_below / Decimal(d_scale)
        self._ignore_below = bin_off

        self._histogram = Histogram(
            self.config.bin_size / Decimal(d_scale),
            bin_off
        )
        self._current_frame = 0
        self._prior_max_locations = None
        self._max_locations = [None] * fb_data.num_bodyparts

        result = super().run_pass(fb_data, prog_bar, in_place, reset_bar)

        std = self._histogram.get_std_using_mean(0)
        result.metadata.optimal_std = Histogram.to_floats(
            (*self._histogram.get_bin_for_value(std)[:2], std)
        )

        if(self.config.DEBUG):
            print(f"Optimal STD: {result.metadata.optimal_std}")

        return result

    def run_step(
        self,
        prior: Optional[ForwardBackwardFrame],
        current: ForwardBackwardFrame,
        frame_index: int,
        bodypart_index: int,
        metadata: AttributeDict
    ) -> Optional[ForwardBackwardFrame]:
        if(self._current_frame != frame_index):
            if(self._prior_max_locations is not None):
                for bp_gi in range(len(self._max_locations) // metadata.num_outputs):
                    for ci in range(metadata.num_outputs):
                        cp, cx, cy = self._max_locations[bp_gi * metadata.num_outputs + ci]

                        if(cp is None):
                            continue

                        min_dist = np.inf

                        for pi in range(metadata.num_outputs):
                            pp, px, py = self._prior_max_locations[bp_gi * metadata.num_outputs + pi]

                            if(pp is None):
                                continue

                            min_dist = min(((cx - px) ** 2 + (cy - py) ** 2) ** 0.5, min_dist)

                        if(min_dist != np.inf and min_dist >= self._ignore_below):
                            self._histogram.add(min_dist)

            self._prior_max_locations = self._max_locations
            self._max_locations = [None] * self.fb_data.num_bodyparts
            self._current_frame = frame_index

        y, x, probs, ox, oy = current.src_data.unpack()

        if(y is None):
            self._max_locations[bodypart_index] = (None, 0, 0)
            return None

        max_loc = np.argmax(probs)

        self._max_locations[bodypart_index] = (
            probs[max_loc],
            x[max_loc] + 0.5 + ox[max_loc] / metadata.down_scaling,
            y[max_loc] + 0.5 + oy[max_loc] / metadata.down_scaling
        )

        return None

    @classmethod
    def get_config_options(cls) -> ConfigSpec:
        return {
            "bin_size": (2, tc.RoundedDecimal(5), "A decimal, the size of each bin used in the histogram for computing the mode, in pixels."),
            "ignore_bins_below": (
                1, tc.RoundedDecimal(5),
                "A decimal, the offset of the first bin used in the histogram for computing "
                "the mode, in pixels. Defaults to 1."
            ),
            "DEBUG": (False, bool, "Set to True to print the optimal standard deviation found...")
        }
