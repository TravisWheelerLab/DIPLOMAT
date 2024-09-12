Changes for this version of DIPLOMAT:
- implemented a geometric normalizer for frame score, based on patterns that give approximations to the packing circles / spreading points problem by Szabo/Csendes/Casado/Garcia (Packing Equal Circles in a Square, 2001)
- using the frame score normalizer, the fix frame of a segment now augments the skeleton weight constant. fix frames with poor separability will have reduced skeleton weight. fix frames with missing limbs will have a minimal (1e-4) skeleton weight. testing indicates this reduces rapid switching to some extent.
